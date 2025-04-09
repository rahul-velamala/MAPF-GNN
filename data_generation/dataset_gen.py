# File: data_generation/dataset_gen.py
# (Complete Code - Includes fix for YAML serialization of solution)

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

# Use relative import assuming called from main_data.py
try:
    from cbs.cbs import Environment as CBSEnvironment # Rename to avoid clash
    from cbs.cbs import CBS, State, Location # Import State/Location if needed? No, CBS returns internal format.
except ImportError:
     # Fallback for direct execution or different structure
    try:
        from ..cbs.cbs import Environment as CBSEnvironment
        from ..cbs.cbs import CBS
    except ImportError:
        raise ImportError("Could not import CBS Environment/CBS. Ensure cbs/cbs.py exists and is accessible.")

# --- Timeout Handling ---
class TimeoutError(Exception):
    """Custom exception for timeouts."""
    pass

def handle_timeout(signum, frame):
    """Signal handler that raises our custom TimeoutError."""
    raise TimeoutError("CBS search timed out")
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
    map_width, map_height = dimensions
    input_dict = {"agents": [], "map": {"dimensions": list(dimensions), "obstacles": []}}

    generated_obstacles = []
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
            obstacle_pos = [
                np.random.randint(0, map_width),
                np.random.randint(0, map_height),
            ]
            if is_valid(obstacle_pos, occupied):
                generated_obstacles.append(tuple(obstacle_pos))
                occupied.add(tuple(obstacle_pos))
                num_placed_obstacles += 1
                break
            attempts += 1
        if attempts == max_placement_attempts:
             print(f"Warning: Could not place obstacle {num_placed_obstacles+1}/{nb_obs} after {max_placement_attempts} attempts. Continuing with fewer obstacles.")

    input_dict["map"]["obstacles"] = generated_obstacles

    # --- Generate Agent Starts and Goals ---
    for agent_id in range(nb_agents):
        start_pos, goal_pos = None, None

        # Assign Start
        attempts = 0
        while attempts < max_placement_attempts:
            potential_start = [np.random.randint(0, map_width), np.random.randint(0, map_height)]
            if is_valid(potential_start, occupied):
                start_pos = potential_start
                occupied.add(tuple(start_pos))
                break
            attempts += 1
        if start_pos is None:
            print(f"Error: Failed to place start position for agent {agent_id} after {max_placement_attempts} attempts. Input generation failed.")
            return None

        # Assign Goal
        attempts = 0
        while attempts < max_placement_attempts:
            potential_goal = [np.random.randint(0, map_width), np.random.randint(0, map_height)]
            if tuple(potential_goal) != tuple(start_pos) and is_valid(potential_goal, occupied):
                goal_pos = potential_goal
                occupied.add(tuple(goal_pos))
                break
            attempts += 1
        if goal_pos is None:
            occupied.remove(tuple(start_pos)) # Backtrack
            print(f"Error: Failed to place goal position for agent {agent_id} after {max_placement_attempts} attempts. Input generation failed.")
            return None

        input_dict["agents"].append(
            {"start": list(start_pos), "goal": list(goal_pos), "name": f"agent{agent_id}"}
        )

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
        print(f"Error creating output directory {output_dir}: {e}")
        return False, "io_error"

    param = input_dict
    # Write input file first
    parameters_path = output_dir / "input.yaml"
    try:
        with open(parameters_path, "w") as parameters_file:
             yaml.safe_dump(param, parameters_file)
    except Exception as e:
         print(f"Error writing input file {parameters_path}: {e}")
         try: shutil.rmtree(output_dir)
         except OSError: pass
         return False, "io_error"

    # Extract CBS parameters
    dimension = param["map"]["dimensions"] # [width, height]
    obstacles = param["map"]["obstacles"] # List of tuples [(x, y), ...]
    agents = param["agents"] # List of dicts

    if not agents:
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "no_agents"

    solution_internal = None # Store the internal CBS solution format (list of State objects)
    search_failed_reason = "unknown"
    cbs_env = None
    cbs_solver = None

    # --- Setup signal alarm ---
    can_use_alarm = hasattr(signal, 'SIGALRM')
    original_handler = None
    if can_use_alarm:
        original_handler = signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(cbs_timeout_seconds) # Set the alarm

    try:
        # --- Initialize CBS Environment and Solver ---
        obstacles_list_of_lists = [list(obs) for obs in obstacles]
        cbs_env = CBSEnvironment(dimension, agents, obstacles_list_of_lists)
        cbs_solver = CBS(cbs_env, verbose=False) # Keep verbose False

        # --- Call CBS search ---
        solution_internal = cbs_solver.search() # Get internal solution

        # --- Disable the alarm ---
        if can_use_alarm:
             signal.alarm(0)

        if not solution_internal: # Check if solution is empty {} or None
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
        print(f"Error during CBS processing for {output_dir.name}: {type(e).__name__} - {e}")
        search_failed_reason = f"cbs_error:{type(e).__name__}"
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, search_failed_reason
    finally:
        # --- Restore original signal handler ---
        if can_use_alarm and original_handler is not None:
            signal.signal(signal.SIGALRM, original_handler)
            signal.alarm(0) # Ensure alarm is off

    # --- If we got here, search was successful ---
    try:
        # !!! CONVERT internal solution (list of States) to serializable format !!!
        solution_output_format = cbs_solver.generate_plan_from_solution(solution_internal)

        # Prepare the dictionary to be saved to YAML
        output_data_to_save = dict()
        output_data_to_save["schedule"] = solution_output_format
        # Calculate cost based on the internal solution format
        cost = cbs_env.compute_solution_cost(solution_internal)
        output_data_to_save["cost"] = cost
        output_data_to_save["status"] = "Success" # Add status

        # Save the converted data to solution.yaml
        solution_path = output_dir / "solution.yaml"
        with open(solution_path, "w") as solution_file:
            # Dump the dictionary containing the serializable schedule format
            yaml.safe_dump(output_data_to_save, solution_file, default_flow_style=None, sort_keys=False)

        return True, "success" # Indicate success
    except Exception as e:
        # Catch errors during conversion or final saving
        print(f"Error converting or writing solution file {output_dir / 'solution.yaml'}: {e}")
        # Clean up directory if final save fails
        try: shutil.rmtree(output_dir)
        except OSError: pass
        return False, "io_error"


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
        print(f"Error: Could not create dataset directory {dataset_path}: {e}")
        return

    # --- Determine existing and needed cases ---
    existing_successful_cases = 0
    highest_existing_index = -1
    try:
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name.startswith("case_"):
                if (item / "input.yaml").exists() and (item / "solution.yaml").exists():
                     existing_successful_cases += 1
                try:
                     index = int(item.name.split('_')[-1])
                     highest_existing_index = max(highest_existing_index, index)
                except (ValueError, IndexError):
                     print(f"Warning: Could not parse index from directory name: {item.name}")
    except Exception as e:
         print(f"Warning: Error analyzing existing cases in {dataset_path}: {e}. Starting generation index from 0.")
         highest_existing_index = -1

    needed_cases = num_target_cases - existing_successful_cases
    start_index = highest_existing_index + 1

    if needed_cases <= 0:
        print(f"Target of {num_target_cases} successful cases already met or exceeded in {dataset_path} ({existing_successful_cases} found). Skipping generation.")
        return

    print(f"Found {existing_successful_cases} existing successful cases. Highest index: {highest_existing_index}.")
    print(f"Generating {needed_cases} new successful solutions to reach target of {num_target_cases}...")

    # --- Get Generation Parameters ---
    cbs_map_shape = config.get("map_shape") # Expected [width, height]
    nb_obstacles = config.get("nb_obstacles")
    nb_agents = config.get("nb_agents")
    cbs_timeout = config.get("cbs_timeout_seconds", 60)
    if not cbs_map_shape or len(cbs_map_shape) != 2 or nb_obstacles is None or nb_agents is None:
         print("Error: Config missing 'map_shape' [width, height], 'nb_obstacles', or 'nb_agents' for generation.")
         return
    print(f"(Using CBS map shape {cbs_map_shape}, {nb_obstacles} obstacles, {nb_agents} agents, timeout {cbs_timeout}s)")


    # --- Generation Loop ---
    failure_counts = { "input_gen_failed": 0, "timeout": 0, "no_solution_found": 0, "cbs_error": 0, "io_error": 0, "no_agents": 0, "unknown": 0 }
    generated_this_run = 0
    current_case_index = start_index
    max_generation_attempts = needed_cases * 5 + 200

    pbar = tqdm(total=needed_cases, desc=f"Generating ({dataset_path.name})", unit="case")
    attempts_this_run = 0
    while generated_this_run < needed_cases and attempts_this_run < max_generation_attempts:
        attempts_this_run += 1
        case_path = dataset_path / f"case_{current_case_index}"

        # --- Generate Input ---
        input_data = gen_input(cbs_map_shape, nb_obstacles, nb_agents)

        # --- Generate Solution (data_gen handles cleanup on failure) ---
        success, reason = data_gen(input_data, case_path, cbs_timeout_seconds=cbs_timeout)

        if success:
            generated_this_run += 1
            pbar.update(1)
        else:
            reason_key = reason.split(":")[0]
            failure_counts[reason_key] = failure_counts.get(reason_key, 0) + 1
            # Optional: double check cleanup if data_gen failed but dir exists
            if case_path.exists() and case_path.is_dir():
                 # print(f"Warning: data_gen failed but directory {case_path} still exists. Attempting removal.") # Debug
                 try: shutil.rmtree(case_path)
                 except OSError as e: print(f"Error removing failed case dir {case_path}: {e}")

        current_case_index += 1
        pbar.set_postfix({"Success": generated_this_run, "Fails": attempts_this_run - generated_this_run})

    pbar.close()

    # --- Final Summary ---
    if attempts_this_run >= max_generation_attempts and generated_this_run < needed_cases:
         print(f"\nWarning: Reached maximum generation attempts ({max_generation_attempts}) but only generated {generated_this_run}/{needed_cases} new cases.")

    final_successful_cases = sum(1 for item in dataset_path.iterdir() if item.is_dir() and item.name.startswith("case_") and (item / "input.yaml").exists() and (item / "solution.yaml").exists())
    print(f"\n--- Generation Finished for: {dataset_path.name} ---")
    print(f"Total successful cases in directory: {final_successful_cases}")
    print(f"Generated {generated_this_run} new successful cases in this run.")

    total_failed = sum(failure_counts.values())
    if total_failed > 0:
        print(f"Failures during this run ({total_failed} total attempts failed):")
        sorted_failures = sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_failures:
            if count > 0:
                print(f"  - {reason}: {count}")


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