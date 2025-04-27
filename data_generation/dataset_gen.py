# File: data_generation/dataset_gen.py
# Generates individual map layouts (obstacles) and full problem instances (input.yaml).
# Solves instances using CBS to generate solution.yaml.
# Includes timeout handling for the CBS solver.

import sys
import os
import yaml
import argparse # Keep for potential future standalone use
import numpy as np
from tqdm import tqdm # For progress bars
import signal # For timeout handling
import errno # For checking specific errors
import shutil # For removing failed directories
from pathlib import Path # Use pathlib for cleaner path handling
import logging

logger = logging.getLogger(__name__)
# Basic config, will inherit level from main_data if run via that,
# or use this default if run standalone.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CBS/Environment Import ---
# Try to import assuming standard project structure where cbs/ is a sibling directory
try:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cbs_module_path = project_root / 'cbs'
    if str(project_root) not in sys.path:
         sys.path.insert(0, str(project_root))
    if str(cbs_module_path) not in sys.path:
         sys.path.insert(1, str(cbs_module_path)) # Add cbs dir itself maybe?

    from cbs.cbs import Environment as CBSEnvironment # Rename to avoid clash
    from cbs.cbs import CBS
    logger.debug("Successfully imported CBS Environment/CBS.")
except ImportError as e:
    logger.error(f"Could not import CBS Environment/CBS: {e}", exc_info=True)
    logger.error(f"Current sys.path: {sys.path}")
    logger.error(f"Attempted import from: {cbs_module_path}")
    raise ImportError("Could not import CBS Environment/CBS. Ensure cbs/cbs.py exists and is accessible from the project root.")


# --- Timeout Handling ---
class TimeoutError(Exception):
    """Custom exception for timeouts."""
    pass

def handle_timeout(signum, frame):
    """Signal handler that raises our custom TimeoutError."""
    logger.warning("CBS Search Timeout Triggered!")
    raise TimeoutError("CBS search timed out")

# Check if SIGALRM is available ONCE at the module level
can_use_alarm = hasattr(signal, 'SIGALRM')
if not can_use_alarm:
    logger.warning("Signal alarms (SIGALRM) not available on this OS (e.g., Windows). CBS timeout may not be strictly enforced.")
# --- End Timeout Handling ---


def generate_obstacles_for_map(dimensions: tuple[int, int], nb_obs: int, max_placement_attempts=100) -> set[tuple[int, int]] | None:
    """
    Generates a set of unique obstacle coordinates (x, y) for a given map size.

    Args:
        dimensions (tuple): (width, height) of the map.
        nb_obs (int): Number of obstacles to generate.
        max_placement_attempts (int): Max attempts to place each obstacle.

    Returns:
        set[tuple[int, int]]: A set of (x, y) tuples for obstacles, or None if failed.
    """
    map_width, map_height = map(int, dimensions)
    total_cells = map_width * map_height
    if nb_obs < 0: nb_obs = 0
    if nb_obs >= total_cells:
         logger.warning(f"Cannot place {nb_obs} obstacles in a {map_width}x{map_height} grid. Reducing to {total_cells - 1}.")
         nb_obs = total_cells - 1 # Leave at least one cell free

    rng = np.random.default_rng() # Use default NumPy RNG
    obstacle_coords_set = set()
    attempts_total = 0
    # More robust attempt limit: try more if grid is large or many obstacles needed
    max_total_attempts = max(nb_obs * max_placement_attempts, total_cells * 2)

    while len(obstacle_coords_set) < nb_obs and attempts_total < max_total_attempts:
        attempts_total += 1
        # Generate (x, y) -> (col, row) consistent with CBS input
        obstacle_pos = (rng.integers(0, map_width), rng.integers(0, map_height))
        if obstacle_pos not in obstacle_coords_set:
            obstacle_coords_set.add(obstacle_pos)

    if len(obstacle_coords_set) < nb_obs:
         logger.warning(f"Could only place {len(obstacle_coords_set)}/{nb_obs} unique obstacles after {attempts_total} attempts.")
         # Fail if significantly fewer obstacles were placed than requested
         if len(obstacle_coords_set) < nb_obs * 0.8: # Example threshold
              logger.error("Failed to place sufficient unique obstacles.")
              return None
    return obstacle_coords_set

def gen_input(
    dimensions: tuple[int, int],
    nb_agents: int,
    fixed_obstacles_xy_set: set[tuple[int, int]], # Expects set of (x, y) tuples
    max_placement_attempts=100
) -> dict | None:
    """
    Generates a dictionary defining agents (random start/goal) for a map
    with PRE-DEFINED obstacles.

    Args:
        dimensions (tuple): (width, height) of the map for CBS.
        nb_agents (int): Number of agents.
        fixed_obstacles_xy_set (set): Set of (x, y) tuples representing obstacles.
        max_placement_attempts (int): Max attempts to place each start/goal.

    Returns:
        dict: The input dictionary for CBS, or None if placement failed.
    """
    map_width, map_height = map(int, dimensions)
    # Convert obstacle set to list of lists [[x,y], ...] for YAML output
    # Sort for deterministic output (useful for comparing inputs)
    obstacles_list_for_yaml = sorted([list(obs) for obs in fixed_obstacles_xy_set])
    input_dict = {
        "agents": [],
        "map": {"dimensions": list(dimensions), "obstacles": obstacles_list_for_yaml}
    }

    # Use Python's default RNG for scenario generation randomness
    rng = np.random.default_rng()

    # occupied includes the fixed obstacles
    occupied_xy_tuples = set(fixed_obstacles_xy_set) # Start with obstacles

    # --- Generate Agent Starts and Goals ---
    agent_data_list = []
    # Keep track of starts/goals placed *within this function call* to avoid self-collision
    temp_agent_placements_xy = set()

    for agent_id in range(nb_agents):
        start_pos_xy, goal_pos_xy = None, None

        # --- Assign Start Position ---
        attempts_start = 0
        start_placed = False
        while attempts_start < max_placement_attempts:
            # Generate potential start: [x, y]
            potential_start = [rng.integers(0, map_width), rng.integers(0, map_height)]
            start_tuple = tuple(potential_start)

            # Check if the potential start is valid (not an obstacle and not another agent's temp start/goal)
            if start_tuple not in occupied_xy_tuples and start_tuple not in temp_agent_placements_xy:
                start_pos_xy = potential_start
                temp_agent_placements_xy.add(start_tuple) # Add to temp set
                start_placed = True
                break
            attempts_start += 1

        if not start_placed:
            logger.error(f"Failed to place start position for agent {agent_id} on map with {len(fixed_obstacles_xy_set)} obstacles after {max_placement_attempts} attempts. Input generation failed.")
            return None # Abort generation for this entire case

        # --- Assign Goal Position ---
        attempts_goal = 0
        goal_placed = False
        while attempts_goal < max_placement_attempts:
            # Generate potential goal: [x, y]
            potential_goal = [rng.integers(0, map_width), rng.integers(0, map_height)]
            goal_tuple = tuple(potential_goal)

            # Check if valid: not obstacle, not another agent's temp pos, AND not this agent's start pos
            if goal_tuple not in occupied_xy_tuples and \
               goal_tuple not in temp_agent_placements_xy and \
               goal_tuple != tuple(start_pos_xy):
                goal_pos_xy = potential_goal
                temp_agent_placements_xy.add(goal_tuple) # Add to temp set
                goal_placed = True
                break
            attempts_goal += 1

        if not goal_placed:
            logger.error(f"Failed to place goal position for agent {agent_id} (start was {start_pos_xy}) on map with {len(fixed_obstacles_xy_set)} obstacles after {max_placement_attempts} attempts. Input generation failed.")
            # No need to backtrack occupied set, just fail the case
            return None # Abort generation for this entire case

        # Append successfully placed agent data
        agent_data_list.append({
            "start": [int(start_pos_xy[0]), int(start_pos_xy[1])], # Store [x, y]
            "goal": [int(goal_pos_xy[0]), int(goal_pos_xy[1])],   # Store [x, y]
            "name": f"agent{agent_id}"
        })

    input_dict["agents"] = agent_data_list
    return input_dict


def data_gen(input_dict: dict | None, output_dir: Path, cbs_timeout_seconds=60) -> tuple[bool, str]:
    """
    Generates input.yaml and runs CBS to generate solution.yaml for a single instance.
    Cleans up the output directory on failure (CBS timeout or error).

    Args:
        input_dict (dict | None): Dictionary from gen_input, or None if generation failed.
        output_dir (Path): Path object for the specific case directory (e.g., .../case_1).
        cbs_timeout_seconds (int): Timeout in seconds for the CBS search.

    Returns:
        tuple[bool, str]: (success_status, reason_string)
    """
    if input_dict is None:
        return False, "input_gen_failed"

    try:
        # Ensure the specific case directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating output directory {output_dir}: {e}", exc_info=True)
        return False, "io_error" # Cannot proceed if directory creation fails

    param = input_dict
    # Write input file first
    parameters_path = output_dir / "input.yaml"
    try:
        with open(parameters_path, "w") as parameters_file:
            # Use standard flow style, allow unicode, don't sort keys
            yaml.dump(param, parameters_file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.error(f"Error writing input file {parameters_path}: {e}", exc_info=True)
        # Cleanup: Remove the directory if input writing failed
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "io_error"

    # --- Extract CBS parameters ---
    try:
        dimension = param["map"]["dimensions"] # Expect [width, height]
        # Obstacles should be list of lists [[x,y],..] as saved in input_dict by gen_input
        obstacles_list_of_lists = param["map"]["obstacles"]
        agents = param["agents"] # List of agent dicts
    except KeyError as e:
        logger.error(f"Input dictionary missing expected key: {e} in {parameters_path}")
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "input_format_error"

    if not agents:
        logger.warning(f"Case {output_dir.name} has no agents defined in input.yaml. Skipping CBS.")
        # Decide if this is a failure or just skippable. Let's treat as failure requiring cleanup.
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "no_agents"

    solution_cbs_format = None # To store the result from cbs_solver.search()
    search_failed_reason = "unknown"
    cbs_env = None
    cbs_solver = None

    # --- Setup signal alarm ---
    original_handler = None
    alarm_set_successfully = False # Flag to track if alarm was properly set FOR THIS CALL

    # Check the global availability flag first
    if can_use_alarm:
        try:
            # Attempt to set the signal handler and alarm
            original_handler = signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(cbs_timeout_seconds) # Use the timeout value passed to the function
            alarm_set_successfully = True # Mark success
            logger.debug(f"SIGALRM handler set for {output_dir.name} with timeout {cbs_timeout_seconds}s.")
        except (ValueError, OSError, AttributeError, TypeError) as e: # Catch various potential errors
            logger.warning(f"Cannot set SIGALRM handler for {output_dir.name}: {e}. Timeout may not work.")
            original_handler = None # Ensure handler is None if setup failed
            alarm_set_successfully = False # Ensure flag reflects failure

    # --- Execute CBS ---
    try:
        # Initialize CBS Environment and Solver
        # Pass obstacles as list of lists [[x,y], ...]
        cbs_env = CBSEnvironment(dimension, agents, obstacles_list_of_lists)
        cbs_solver = CBS(cbs_env, verbose=False) # Keep verbose False during generation

        # Call CBS search
        logger.debug(f"Starting CBS search for {output_dir.name}...")
        solution_cbs_format = cbs_solver.search()
        logger.debug(f"CBS search finished for {output_dir.name}.")

        # --- Disable the alarm ---
        if alarm_set_successfully: # Only disable if it was successfully set
            signal.alarm(0)
            logger.debug(f"SIGALRM disabled for {output_dir.name}.")

        # Check if CBS returned a valid solution (not empty dict {})
        if not solution_cbs_format:
            search_failed_reason = "no_solution_found"
            logger.debug(f"CBS failed for {output_dir.name}: {search_failed_reason}")
            try: shutil.rmtree(output_dir) # Cleanup on failure
            except OSError as e: logger.error(f"Error removing failed directory {output_dir}: {e}")
            return False, search_failed_reason

    except TimeoutError:
        # This exception is raised by our signal handler
        search_failed_reason = "timeout"
        logger.debug(f"CBS failed for {output_dir.name}: {search_failed_reason}")
        try: shutil.rmtree(output_dir) # Cleanup on timeout
        except OSError as e: logger.error(f"Error removing timed-out directory {output_dir}: {e}")
        return False, search_failed_reason
    except Exception as e:
        # Catch any other unexpected error during CBS execution
        logger.error(f"Error during CBS processing for {output_dir.name}: {type(e).__name__} - {e}", exc_info=True)
        search_failed_reason = f"cbs_error:{type(e).__name__}"
        try: shutil.rmtree(output_dir) # Cleanup on other errors
        except OSError as e: logger.error(f"Error removing error directory {output_dir}: {e}")
        return False, search_failed_reason
    finally:
        # --- Restore original signal handler ---
        # Ensure restoration only happens if the alarm was set and we captured the original handler
        if alarm_set_successfully and original_handler is not None:
             try:
                 signal.signal(signal.SIGALRM, original_handler)
                 signal.alarm(0) # Ensure alarm is off just in case
                 logger.debug(f"SIGALRM handler restored for {output_dir.name}.")
             except (ValueError, OSError) as e:
                 # Log error but continue, as the main task is done or failed already
                 logger.warning(f"Error restoring signal handler for {output_dir.name}: {e}")

    # --- Process Successful CBS Result ---
    try:
        # Calculate cost based on the returned plan (sum of individual path lengths - 1)
        cost = sum(len(path) - 1 for path in solution_cbs_format.values() if path and len(path) > 0)

        # Prepare the dictionary to be saved to solution.yaml
        output_data_to_save = {
            "schedule": solution_cbs_format, # The dict returned by CBS
            "cost": int(cost), # Ensure cost is an integer if possible
            "status": "Success"
        }

        # Save the successful result to solution.yaml
        solution_path = output_dir / "solution.yaml"
        with open(solution_path, "w") as solution_file:
            # Use standard flow style, do not sort keys to preserve agent order
            yaml.dump(output_data_to_save, solution_file, default_flow_style=False, sort_keys=False)

        logger.debug(f"Successfully generated solution for {output_dir.name}")
        return True, "success" # Indicate success

    except Exception as e:
        # Catch errors during cost calculation or YAML writing AFTER successful CBS
        logger.error(f"Error processing successful CBS result or writing solution file {output_dir / 'solution.yaml'}: {e}", exc_info=True)
        # Cleanup even if CBS succeeded but saving failed
        try: shutil.rmtree(output_dir)
        except OSError as e_rm: logger.error(f"Error removing directory {output_dir} after save failure: {e_rm}")
        return False, "io_error"


# --- Standalone execution block (commented out as main_data.py drives the process) ---
"""
# This block is typically run when executing `python data_generation/dataset_gen.py ...` directly.
# In the IROS 2020 setup, main_data.py orchestrates the generation, so this is not used.

# def create_solutions(dataset_path: Path, num_target_cases: int, config: dict):
#     ''' Generates CBS problems and solves them up to num_target_cases. (OLD LOGIC) '''
#     # ... (previous implementation generating map *per case*) ...
#     pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ONE CBS instance (input.yaml and solution.yaml). Primarily for testing.")
    parser.add_argument("--output_dir", type=str, default="temp_case", help="Directory to store the generated case files.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--width", type=int, default=10, help="Map width.")
    parser.add_argument("--height", type=int, default=10, help="Map height.")
    parser.add_argument("--obstacles", type=int, default=10, help="Number of obstacles.")
    parser.add_argument("--timeout", type=int, default=60, help="CBS search timeout in seconds.")

    args = parser.parse_args()

    case_dir = Path(args.output_dir)
    map_dims = (args.width, args.height)

    print(f"--- Generating single test case in: {case_dir.resolve()} ---")
    print(f"Config: Agents={args.agents}, Size={args.width}x{args.height}, Obstacles={args.obstacles}, Timeout={args.timeout}s")

    # 1. Generate obstacles for this single test case
    obstacle_set = generate_obstacles_for_map(map_dims, args.obstacles)

    if obstacle_set is None:
        print("Failed to generate obstacles. Exiting.")
    else:
        # 2. Generate input (start/goal) for these obstacles
        input_d = gen_input(
            dimensions=map_dims,
            nb_agents=args.agents,
            fixed_obstacles_xy_set=obstacle_set
        )

        if input_d is None:
            print("Failed to generate start/goal inputs. Exiting.")
        else:
            # 3. Run data_gen (which runs CBS)
            success, reason = data_gen(
                input_dict=input_d,
                output_dir=case_dir,
                cbs_timeout_seconds=args.timeout
            )

            if success:
                print(f"Successfully generated case in {case_dir}")
                # You can inspect input.yaml and solution.yaml here
            else:
                print(f"Failed to generate case: {reason}")
                if case_dir.exists():
                     print(f"(Directory {case_dir} should have been removed)")

    print(f"--- Single case generation test finished ---")
"""