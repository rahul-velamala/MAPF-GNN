# File: data_generation/dataset_gen.py
# (Includes fix for YAML serialization, timeout handling, path handling, cleanup)
import numpy as np
import logging
import sys
import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm # For progress bars
import signal # For timeout handling
import errno # For checking specific errors
import shutil # For removing failed directories
from pathlib import Path # Use pathlib for cleaner path handling
import logging

logger = logging.getLogger(__name__)
# Basic config, will inherit level from main_data if run via that
logging.basicConfig(level=logging.INFO)

# Use relative import assuming called from main_data.py
try:
    from cbs.cbs import Environment as CBSEnvironment # Rename to avoid clash
    from cbs.cbs import CBS
except ImportError:
    # Fallback for direct execution or different structure
    try:
        from ..cbs.cbs import Environment as CBSEnvironment
        from ..cbs.cbs import CBS
    except ImportError as e:
        logger.error(f"Could not import CBS Environment/CBS: {e}", exc_info=True)
        raise ImportError("Could not import CBS Environment/CBS. Ensure cbs/cbs.py exists and is accessible.")

# --- Timeout Handling ---
class TimeoutError(Exception):
    """Custom exception for timeouts."""
    pass

def handle_timeout(signum, frame):
    """Signal handler that raises our custom TimeoutError."""
    raise TimeoutError("CBS search timed out")

can_use_alarm = hasattr(signal, 'SIGALRM')
if not can_use_alarm:
    logger.warning("Signal alarms (SIGALRM) not available on this OS. CBS timeout may not be strictly enforced.")
# --- End Timeout Handling ---


def gen_input(dimensions: tuple[int, int], nb_obs: int, nb_agents: int, max_placement_attempts=100) -> dict | None:
    """
    Generates a dictionary defining agents (random start/goal) and
    map (dimensions, random obstacles) for a CBS problem instance.

    Args:
        dimensions (tuple): (width, height) of the map for CBS.
        nb_obs (int): Number of obstacles to generate.
        nb_agents (int): Number of agents.
        max_placement_attempts (int): Max attempts to place each item randomly.

    Returns:
        dict: The input dictionary for CBS, or None if placement failed.
    """
    map_width, map_height = map(int, dimensions)
    input_dict = {"agents": [], "map": {"dimensions": list(dimensions), "obstacles": []}}

    # Use Python's default RNG for scenario generation randomness
    rng = np.random.default_rng()

    generated_obstacles_coords = [] # Store as simple lists [x,y]
    occupied = set() # Keep track of all occupied cells (obstacles, starts, goals) as tuples (x, y)

    def is_valid(pos_xy, current_occupied):
        x, y = pos_xy
        if not (0 <= x < map_width and 0 <= y < map_height): return False
        if tuple(pos_xy) in current_occupied: return False
        return True

    # --- Generate Obstacles ---
    num_placed_obstacles = 0
    for _ in range(nb_obs):
        attempts = 0
        while attempts < max_placement_attempts:
            obstacle_pos = [rng.integers(0, map_width), rng.integers(0, map_height)]
            if is_valid(obstacle_pos, occupied):
                generated_obstacles_coords.append([int(obstacle_pos[0]), int(obstacle_pos[1])]) # Store list [x,y]
                occupied.add(tuple(obstacle_pos))
                num_placed_obstacles += 1
                break
            attempts += 1
        if attempts == max_placement_attempts:
            logger.warning(f"Could not place obstacle {num_placed_obstacles+1}/{nb_obs} after {max_placement_attempts} attempts. Continuing with {num_placed_obstacles} obstacles.")
            break # Stop trying to place more obstacles if one fails

    input_dict["map"]["obstacles"] = generated_obstacles_coords # Save as list of lists

    # --- Generate Agent Starts and Goals ---
    agent_data_list = []
    temp_occupied_agent_placements = set() # Track starts/goals added in this loop

    for agent_id in range(nb_agents):
        start_pos, goal_pos = None, None

        # Assign Start
        attempts = 0
        while attempts < max_placement_attempts:
            potential_start = [rng.integers(0, map_width), rng.integers(0, map_height)]
            # Check against base obstacles AND temp agent placements
            if is_valid(potential_start, occupied | temp_occupied_agent_placements):
                start_pos = potential_start
                temp_occupied_agent_placements.add(tuple(start_pos))
                break
            attempts += 1
        if start_pos is None:
            logger.error(f"Failed to place start position for agent {agent_id} after {max_placement_attempts} attempts. Input generation failed.")
            return None # Abort generation for this case

        # Assign Goal
        attempts = 0
        while attempts < max_placement_attempts:
            potential_goal = [rng.integers(0, map_width), rng.integers(0, map_height)]
            # Ensure goal is different from start and not occupied
            if tuple(potential_goal) != tuple(start_pos) and \
               is_valid(potential_goal, occupied | temp_occupied_agent_placements):
                goal_pos = potential_goal
                temp_occupied_agent_placements.add(tuple(goal_pos))
                break
            attempts += 1
        if goal_pos is None:
            logger.error(f"Failed to place goal position for agent {agent_id} after {max_placement_attempts} attempts. Input generation failed.")
            # No need to backtrack occupied, just fail the case
            return None # Abort generation for this case

        agent_data_list.append({
    "start": [int(start_pos[0]), int(start_pos[1])],
    "goal": [int(goal_pos[0]), int(goal_pos[1])],
    "name": f"agent{agent_id}"
})

    input_dict["agents"] = agent_data_list
    return input_dict


def data_gen(input_dict: dict | None, output_dir: Path, cbs_timeout_seconds=60) -> tuple[bool, str]:
    """
    Generates input.yaml and solution.yaml for a single CBS instance.
    Cleans up the output directory on failure.

    Args:
        input_dict (dict | None): Dictionary from gen_input, or None if generation failed.
        output_dir (Path): Path object for the case directory (e.g., .../case_1).
        cbs_timeout_seconds (int): Timeout for the CBS search.

    Returns:
        tuple[bool, str]: (success_status, reason_string)
    """
    if input_dict is None:
        return False, "input_gen_failed"

    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    except OSError as e:
        logger.error(f"Error creating output directory {output_dir}: {e}", exc_info=True)
        return False, "io_error"

    param = input_dict
    # Write input file first
    parameters_path = output_dir / "input.yaml"
    try:
        with open(parameters_path, "w") as parameters_file:
            # Use standard flow style, allow unicode, don't sort keys
            yaml.dump(param, parameters_file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.error(f"Error writing input file {parameters_path}: {e}", exc_info=True)
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "io_error"

    # Extract CBS parameters
    dimension = param["map"]["dimensions"] # [width, height]
    obstacles_list_of_lists = param["map"]["obstacles"] # List of lists [x, y]
    agents = param["agents"] # List of dicts

    if not agents:
        logger.warning(f"Case {output_dir.name} has no agents defined in input.yaml. Skipping.")
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "no_agents"

    solution_cbs_format = None # Store the raw CBS solution format
    search_failed_reason = "unknown"
    cbs_env = None
    cbs_solver = None

    # --- Setup signal alarm ---
    original_handler = None
    if can_use_alarm:
        try:
            original_handler = signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(cbs_timeout_seconds)
        except ValueError as e:
            logger.warning(f"Cannot set SIGALRM handler for {output_dir.name} (maybe in thread?): {e}")
            original_handler = None

    try:
        # --- Initialize CBS Environment and Solver ---
        cbs_env = CBSEnvironment(dimension, agents, obstacles_list_of_lists)
        cbs_solver = CBS(cbs_env, verbose=False) # Keep verbose False

        # --- Call CBS search ---
        # search() now returns dict in output format {'agent': [{'t':..,'x':..,'y':..}]} or {}
        solution_cbs_format = cbs_solver.search()

        # --- Disable the alarm ---
        if can_use_alarm and original_handler: signal.alarm(0)

        if not solution_cbs_format: # Check if solution is empty {}
            search_failed_reason = "no_solution_found"
            try: shutil.rmtree(output_dir)
            except OSError: pass
            return False, search_failed_reason

    except TimeoutError:
        search_failed_reason = "timeout"
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, search_failed_reason
    except Exception as e:
        logger.error(f"Error during CBS processing for {output_dir.name}: {type(e).__name__} - {e}", exc_info=True)
        search_failed_reason = f"cbs_error:{type(e).__name__}"
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, search_failed_reason
    finally:
        # --- Restore original signal handler ---
        if can_use_alarm and original_handler is not None:
            try:
                signal.signal(signal.SIGALRM, original_handler)
                signal.alarm(0) # Ensure alarm is off
            except ValueError: pass

    # --- If we got here, search was successful ---
    try:
        # Need to calculate cost again based on the returned plan
        # This requires temporarily converting back to internal format or calculating from output format
        # Let's recalculate from output format for simplicity here:
        cost = sum(len(path) - 1 for path in solution_cbs_format.values() if path and len(path) > 0)

        # Prepare the dictionary to be saved to YAML
        output_data_to_save = {
            "schedule": solution_cbs_format,
            "cost": cost,
            "status": "Success"
        }

        # Save the converted data to solution.yaml
        solution_path = output_dir / "solution.yaml"
        with open(solution_path, "w") as solution_file:
            yaml.dump(output_data_to_save, solution_file, default_flow_style=False, sort_keys=False)

        return True, "success" # Indicate success
    except Exception as e:
        logger.error(f"Error calculating cost or writing solution file {output_dir / 'solution.yaml'}: {e}", exc_info=True)
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "io_error"

def generate_obstacles_for_map(dimensions: tuple[int, int], nb_obs: int, max_placement_attempts=100) -> set[tuple[int, int]] | None:
    """
    Generates a set of obstacle coordinates (x, y) for CBS.

    Args:
        dimensions (tuple): (width, height) of the map.
        nb_obs (int): Number of obstacles to generate.
        max_placement_attempts (int): Max attempts to place each obstacle.

    Returns:
        set: A set of obstacle tuples (x, y), or None if placement failed.
    """
    map_width, map_height = map(int, dimensions)
    generated_obstacles_set = set()
    rng = np.random.default_rng() # Use default RNG

    if nb_obs < 0: nb_obs = 0
    if nb_obs >= map_width * map_height:
        logger.warning(f"Requested obstacles ({nb_obs}) >= total cells. Cannot place.")
        return set() # Return empty set if impossible

    num_placed_obstacles = 0
    for _ in range(nb_obs):
        attempts = 0
        while attempts < max_placement_attempts:
            # Generate position (x, y) for CBS format
            obstacle_pos_xy = (rng.integers(0, map_width), rng.integers(0, map_height))
            if obstacle_pos_xy not in generated_obstacles_set:
                generated_obstacles_set.add(obstacle_pos_xy)
                num_placed_obstacles += 1
                break
            attempts += 1
        if attempts == max_placement_attempts:
            logger.warning(f"Could not place obstacle {num_placed_obstacles+1}/{nb_obs} after {max_placement_attempts} attempts. Returning {num_placed_obstacles} obstacles.")
            # Return the successfully placed obstacles so far
            return generated_obstacles_set

    return generated_obstacles_set

def create_solutions(dataset_path: Path, num_target_cases: int, config: dict):
    """
    Generates CBS problem instances (input.yaml) and solves them (solution.yaml)
    up to the target number of successful cases in the specified directory.

    Args:
        dataset_path (Path): Path object for the dataset split (e.g., .../train).
        num_target_cases (int): The desired number of directories with successful solutions.
        config (dict): Configuration containing 'map_shape' (width, height for CBS),
                       'nb_obstacles', 'nb_agents', 'cbs_timeout_seconds'.
    """
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create dataset directory {dataset_path}: {e}", exc_info=True)
        return

    # --- Determine existing and needed cases ---
    existing_successful_cases = 0
    highest_existing_index = -1
    if dataset_path.exists():
        try:
            for item in dataset_path.iterdir():
                if item.is_dir() and item.name.startswith("case_"):
                    # Check for successful completion marker (solution.yaml)
                    if (item / "solution.yaml").exists():
                        existing_successful_cases += 1
                    try:
                        index = int(item.name.split('_')[-1])
                        highest_existing_index = max(highest_existing_index, index)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse index from directory name: {item.name}")
        except Exception as e:
            logger.warning(f"Error analyzing existing cases in {dataset_path}: {e}. Starting generation index from 0.", exc_info=True)
            highest_existing_index = -1
    else:
         logger.warning(f"Dataset path {dataset_path} does not exist yet.")


    needed_cases = num_target_cases - existing_successful_cases
    start_index = highest_existing_index + 1

    if needed_cases <= 0:
        logger.info(f"Target of {num_target_cases} successful cases already met or exceeded in {dataset_path} ({existing_successful_cases} found). Skipping generation.")
        return

    logger.info(f"Found {existing_successful_cases} existing successful cases. Highest index: {highest_existing_index}.")
    logger.info(f"Generating {needed_cases} new successful solutions to reach target of {num_target_cases}...")

    # --- Get Generation Parameters ---
    try:
        cbs_map_shape = config["map_shape"] # Expected [width, height]
        nb_obstacles = int(config["nb_obstacles"])
        nb_agents = int(config["nb_agents"])
        cbs_timeout = int(config.get("cbs_timeout_seconds", 60))
        if len(cbs_map_shape) != 2: raise ValueError("'map_shape' must be [width, height]")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Config missing or invalid required key for generation ('map_shape', 'nb_obstacles', 'nb_agents'): {e}", exc_info=True)
        return
    logger.info(f"(Using CBS map shape {cbs_map_shape}, {nb_obstacles} obstacles, {nb_agents} agents, timeout {cbs_timeout}s)")

    # --- Generation Loop ---
    failure_counts = { "input_gen_failed": 0, "timeout": 0, "no_solution_found": 0, "cbs_error": 0, "io_error": 0, "no_agents": 0, "unknown": 0 }
    generated_this_run = 0
    current_case_index = start_index
    # Limit attempts to prevent infinite loops if generation is very difficult
    max_generation_attempts = max(needed_cases * 10, 200) # Try up to 10x needed, min 200 attempts

    pbar = tqdm(total=needed_cases, desc=f"Generating ({dataset_path.name})", unit="case")
    attempts_this_run = 0
    while generated_this_run < needed_cases and attempts_this_run < max_generation_attempts:
        attempts_this_run += 1
        case_path = dataset_path / f"case_{current_case_index}"

        # Skip if this case index somehow already exists (e.g. from partial previous run)
        if case_path.exists():
             logger.debug(f"Directory {case_path} already exists, incrementing index.")
             current_case_index += 1
             continue

        # --- Generate Input ---
        input_data = gen_input(cbs_map_shape, nb_obstacles, nb_agents)

        # --- Generate Solution (data_gen handles cleanup on failure) ---
        success, reason = data_gen(input_data, case_path, cbs_timeout_seconds=cbs_timeout)

        if success:
            generated_this_run += 1
            pbar.update(1)
            current_case_index += 1 # Only increment index on success
        else:
            # Don't increment case_index, retry with the same index but potentially different random scenario
            reason_key = reason.split(":")[0] # Get base reason if specific error included
            failure_counts[reason_key] = failure_counts.get(reason_key, 0) + 1
            # data_gen should have cleaned up, but double check
            if case_path.exists() and case_path.is_dir():
                logger.warning(f"data_gen failed but directory {case_path} still exists. Attempting removal.")
                try: shutil.rmtree(case_path)
                except OSError as e: logger.error(f"Error removing failed case dir {case_path}: {e}")

        pbar.set_postfix({"Success": generated_this_run, "Fails": attempts_this_run - generated_this_run})

    pbar.close()

    # --- Final Summary ---
    if attempts_this_run >= max_generation_attempts and generated_this_run < needed_cases:
        logger.warning(f"Reached maximum generation attempts ({max_generation_attempts}) but only generated {generated_this_run}/{needed_cases} new cases for {dataset_path.name}.")

    final_successful_cases = sum(1 for item in dataset_path.iterdir() if item.is_dir() and item.name.startswith("case_") and (item / "solution.yaml").exists())
    logger.info(f"\n--- Generation Finished for: {dataset_path.name} ---")
    logger.info(f"Total successful cases in directory: {final_successful_cases}")
    logger.info(f"Generated {generated_this_run} new successful cases in this run.")

    total_failed = sum(failure_counts.values())
    if total_failed > 0:
        logger.info(f"Failures during this run ({total_failed} total attempts failed):")
        sorted_failures = sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_failures:
            if count > 0:
                logger.info(f"  - {reason}: {count}")


# --- Main execution block for running standalone ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CBS datasets (input.yaml and solution.yaml).")
    parser.add_argument("--path", type=str, default="dataset/cbs_generated", help="Base directory to store generated case subdirectories.")
    parser.add_argument("--num_cases", type=int, default=100, help="Total number of *successful* cases desired in the directory.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents per case.")
    parser.add_argument("--width", type=int, default=10, help="Map width (for CBS).")
    parser.add_argument("--height", type=int, default=10, help="Map height (for CBS).")
    parser.add_argument("--obstacles", type=int, default=10, help="Number of obstacles per case.")
    parser.add_argument("--timeout", type=int, default=30, help="CBS search timeout in seconds per case.")

    args = parser.parse_args()

    dataset_dir = Path(args.path)

    generation_config = {
        "map_shape": [args.width, args.height], # CBS uses [width, height]
        "nb_obstacles": args.obstacles,
        "nb_agents": args.agents,
        "cbs_timeout_seconds": args.timeout
    }
    print(f"--- Starting CBS Solution Generation ---")
    print(f"Target Path: {dataset_dir.resolve()}")
    print(f"Target Successful Cases: {args.num_cases}")
    print(f"Config per case: Agents={args.agents}, Size={args.width}x{args.height}, Obstacles={args.obstacles}, Timeout={args.timeout}s")

    create_solutions(dataset_dir, args.num_cases, generation_config)

    print(f"\n--- CBS Solution Generation Script Finished ---")