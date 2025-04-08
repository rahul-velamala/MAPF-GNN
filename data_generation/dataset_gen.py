# File: data_generation/dataset_gen.py
import sys
import os
import yaml
# import torch # Unused in this file?
import argparse
import numpy as np
from cbs.cbs import Environment, CBS # Use absolute import from project root
from tqdm import tqdm # For progress bars
import signal # For timeout handling
import errno # For checking specific errors
import shutil # For removing failed directories

# --- Timeout Handling ---
class TimeoutError(Exception):
    """Custom exception for timeouts."""
    pass

def handle_timeout(signum, frame):
    """Signal handler that raises our custom TimeoutError."""
    raise TimeoutError("CBS search timed out")
# --- End Timeout Handling ---


# VVVVVVVV  GEN_INPUT FUNCTION DEFINITION VVVVVVVV
def gen_input(dimensions: tuple[int, int], nb_obs: int, nb_agents: int) -> dict:
    """
    Generates a dictionary defining agents (random start/goal) and
    map (dimensions, random obstacles) for a CBS problem instance.
    """
    input_dict = {"agents": [], "map": {"dimensions": list(dimensions), "obstacles": []}} # Use list for dimensions for YAML compatibility

    # Initialize obstacles list here
    generated_obstacles = [] # Use a distinct name or ensure correct scope

    occupied = set() # Keep track of all occupied cells

    # Helper to check if a position is valid (within bounds and not occupied)
    def is_valid(pos, current_occupied):
        x, y = pos
        if not (0 <= x < dimensions[0] and 0 <= y < dimensions[1]): return False
        if tuple(pos) in current_occupied: return False
        return True

    # --- Generate Obstacles ---
    for _ in range(nb_obs):
        attempts = 0
        while attempts < 100: # Safety break
            obstacle_pos = [
                np.random.randint(0, dimensions[0]),
                np.random.randint(0, dimensions[1]),
            ]
            if is_valid(obstacle_pos, occupied):
                # Append to the initialized list
                generated_obstacles.append(tuple(obstacle_pos)) # Store as tuple
                occupied.add(tuple(obstacle_pos))
                break
            attempts += 1
        if attempts == 100:
             # Optionally print warning if obstacle placement fails often
             pass # print(f"Warning: Could not place obstacle after 100 attempts.")

    # Assign the generated list to the dict
    input_dict["map"]["obstacles"] = generated_obstacles # Add generated obstacles

    # --- Generate Agent Starts and Goals ---
    for agent_id in range(nb_agents):
        start_pos, goal_pos = None, None
        # Assign Start
        attempts = 0
        while attempts < 100:
            potential_start = [np.random.randint(0, dimensions[0]), np.random.randint(0, dimensions[1])]
            if is_valid(potential_start, occupied):
                start_pos = potential_start
                occupied.add(tuple(start_pos))
                break
            attempts += 1
        if start_pos is None: return None # Failed to place start

        # Assign Goal
        attempts = 0
        while attempts < 100:
            potential_goal = [np.random.randint(0, dimensions[0]), np.random.randint(0, dimensions[1])]
            if tuple(potential_goal) != tuple(start_pos) and is_valid(potential_goal, occupied):
                goal_pos = potential_goal
                occupied.add(tuple(goal_pos))
                break
            attempts += 1
        if goal_pos is None: return None # Failed to place goal

        input_dict["agents"].append(
            {"start": list(start_pos), "goal": list(goal_pos), "name": f"agent{agent_id}"}
        )

    return input_dict
# ^^^^^^^^ END OF gen_input FUNCTION DEFINITION ^^^^^^^^


# --- Modify data_gen to include timeout ---
def data_gen(input_dict, output_path, cbs_timeout_seconds=60): # Add timeout parameter (default 60s)
    os.makedirs(output_path, exist_ok=True)

    if input_dict is None:
        return False, "input_gen_failed" # Return failure type

    param = input_dict
    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]

    if not agents:
        parameters_path = os.path.join(output_path, "input.yaml")
        try:
            with open(parameters_path, "w") as parameters_file:
                 yaml.safe_dump(param, parameters_file)
            return False, "no_agents"
        except Exception as e:
             # print(f"Error writing input file for {output_path}: {e}") # Less verbose
             return False, "io_error"

    env = Environment(dimension, agents, obstacles)
    cbs = CBS(env, verbose=False)

    solution = None
    search_failed_reason = "unknown"
    # --- Setup signal alarm ---
    # Important: This only works on Unix-like systems (Linux, macOS)
    if hasattr(signal, 'SIGALRM'):
        original_handler = signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(cbs_timeout_seconds) # Set the alarm
    else:
        original_handler = None # Cannot use signals on this OS

    try:
        # --- Call CBS search ---
        solution = cbs.search()
        # --- Disable the alarm if search finished in time ---
        if hasattr(signal, 'SIGALRM'):
             signal.alarm(0)

        if not solution:
            search_failed_reason = "no_solution_found"
            return False, search_failed_reason

    except TimeoutError:
        # --- Timeout occurred ---
        search_failed_reason = "timeout"
        return False, search_failed_reason
    except Exception as e:
        # --- Other errors during search ---
        search_failed_reason = f"cbs_error:_{type(e).__name__}"
        return False, search_failed_reason
    finally:
        # --- Restore original signal handler ---
        if hasattr(signal, 'SIGALRM') and original_handler is not None:
            signal.signal(signal.SIGALRM, original_handler)
            signal.alarm(0) # Ensure alarm is off

    # --- If we got here, search was successful ---
    output = dict()
    output["schedule"] = solution
    cost = sum(len(path) - 1 for path in solution.values() if path)
    output["cost"] = cost

    solution_path = os.path.join(output_path, "solution.yaml")
    parameters_path = os.path.join(output_path, "input.yaml")

    try:
        with open(solution_path, "w") as solution_file:
            yaml.safe_dump(output, solution_file)
        with open(parameters_path, "w") as parameters_file:
            yaml.safe_dump(param, parameters_file)
        return True, "success" # Return success type
    except Exception as e:
        # print(f"Error writing output files for {output_path}: {e}") # Less verbose
        return False, "io_error"


# --- Modify create_solutions to pass timeout and handle failure types ---
def create_solutions(path, num_target_cases, config):
    os.makedirs(path, exist_ok=True)
    try:
        existing_dirs = {d for d in os.listdir(path) if d.startswith("case_") and os.path.isdir(os.path.join(path, d))}
        existing_case_indices = {int(d.split('_')[-1]) for d in existing_dirs}
        current_total_cases = len(existing_dirs)
        cases_ready = max(existing_case_indices) if existing_case_indices else 0
    except Exception as e: # Catch potential errors during listing/parsing
         print(f"Warning: Error analyzing existing cases in {path}: {e}. Starting count from 0.")
         current_total_cases = 0
         cases_ready = 0

    needed_cases = num_target_cases - current_total_cases
    cbs_timeout = config.get("cbs_timeout_seconds", 60) # Get timeout from config, default 60

    if needed_cases <= 0:
        print(f"Target of {num_target_cases} cases already met or exceeded in {path} ({current_total_cases} found). Skipping generation.")
        return

    print(f"Found {current_total_cases} existing cases. Highest index: {cases_ready}.")
    print(f"Generating {needed_cases} new solutions to reach target of {num_target_cases}...")
    print(f"(Using CBS timeout of {cbs_timeout} seconds per case)")

    # Keep track of different failure reasons
    failure_counts = {
        "input_gen_failed": 0,
        "timeout": 0,
        "no_solution_found": 0,
        "cbs_error": 0, # Catch-all for other CBS errors
        "io_error": 0,
        "no_agents": 0,
        "unknown": 0
    }

    current_case_index = cases_ready + 1
    generated_this_run = 0
    max_attempts = needed_cases * 3 + 100 # Allow buffer for failures

    pbar = tqdm(total=needed_cases, desc=f"Generating Solutions ({os.path.basename(path)})", unit="case")

    attempts_this_run = 0
    while generated_this_run < needed_cases and attempts_this_run < max_attempts:
        attempts_this_run += 1
        case_path = os.path.join(path, f"case_{current_case_index}")

        if os.path.exists(case_path):
            current_case_index += 1
            continue

        inpt = gen_input(
            config["map_shape"], config["nb_obstacles"], config["nb_agents"]
        )

        # data_gen now returns (bool_success, reason_string)
        success, reason = data_gen(inpt, case_path, cbs_timeout_seconds=cbs_timeout)

        if success:
            generated_this_run += 1
            pbar.update(1)
        else:
            # Increment specific failure counter
            reason_key = reason.split(":")[0] # Get base reason like 'cbs_error' from 'cbs_error:_TypeError'
            failure_counts[reason_key] = failure_counts.get(reason_key, 0) + 1
            # Remove failed case directory if it exists (might not exist if input gen failed)
            if os.path.exists(case_path):
                 try:
                     # Only remove if it's a directory to avoid removing unrelated files
                     if os.path.isdir(case_path):
                          shutil.rmtree(case_path)
                 except OSError as e:
                      # Handle potential race conditions or permission errors during removal
                      if e.errno != errno.ENOENT: # Ignore "No such file or directory"
                           print(f"Warning: Error removing failed case dir {case_path}: {e}")
                 except Exception as e:
                      print(f"Warning: Unexpected error removing failed case dir {case_path}: {e}")


        current_case_index += 1

    pbar.close()

    if attempts_this_run >= max_attempts and generated_this_run < needed_cases:
         print(f"\nWarning: Reached maximum generation attempts ({max_attempts}) but only generated {generated_this_run}/{needed_cases} new cases.")

    final_total_cases = len([d for d in os.listdir(path) if d.startswith("case_") and os.path.isdir(os.path.join(path, d))]) # More robust count
    print(f"\nFinished generating. Total cases in {path}: {final_total_cases}")
    print(f"Generated {generated_this_run} new cases successfully in this run.")
    # Print summary of failures
    total_failed = sum(failure_counts.values())
    if total_failed > 0:
        print(f"Failures during generation ({total_failed} total attempts failed):")
        for reason, count in failure_counts.items():
            if count > 0:
                print(f"  - {reason}: {count}")


if __name__ == "__main__":
    # Example Usage:
    parser = argparse.ArgumentParser(description="Generate CBS datasets.")
    parser.add_argument("--path", type=str, default="dataset/cbs_generated", help="Base directory to store generated cases.")
    parser.add_argument("--num_cases", type=int, default=100, help="Total number of cases desired in the directory.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents per case.")
    parser.add_argument("--width", type=int, default=10, help="Map width.")
    parser.add_argument("--height", type=int, default=10, help="Map height.")
    parser.add_argument("--obstacles", type=int, default=10, help="Number of obstacles per case.")
    parser.add_argument("--timeout", type=int, default=60, help="CBS search timeout in seconds.")

    args = parser.parse_args()

    config = {
        "device": "cpu", # Usually not needed for generation, but harmless
        "map_shape": [args.width, args.height],
        "root_dir": args.path, # Use path argument
        "nb_agents": args.agents,
        "nb_obstacles": args.obstacles,
        "cbs_timeout_seconds": args.timeout
    }
    print(f"--- Starting Dataset Generation ---")
    print(f"Target Path: {args.path}")
    print(f"Target Total Cases: {args.num_cases}")
    print(f"Config per case: Agents={args.agents}, Size={args.width}x{args.height}, Obstacles={args.obstacles}, Timeout={args.timeout}s")

    create_solutions(args.path, args.num_cases, config)

    print(f"--- Dataset Generation Finished ---")