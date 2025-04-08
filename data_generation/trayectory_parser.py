# File: data_generation/trayectory_parser.py
import os
import yaml
import numpy as np
# import matplotlib.pyplot as plt # Unused
import argparse
# from pprint import pprint # Unused
from tqdm import tqdm # Import tqdm

def get_longest_path(schedule):
    # ... (function remains the same) ...
    longest = 0
    if not schedule: # Handle empty schedule
        return 0
    for agent_path in schedule.values():
        if agent_path: # Check if agent has a path
             longest = max(longest, len(agent_path)) # Use number of states
    return longest

def parse_trayectories(schedule):
    # ... (function remains the same) ...
    if not schedule:
        return np.empty((0,0), dtype=np.int32), np.empty((0,2), dtype=np.int32)

    num_agents = len(schedule)
    longest_path_len = get_longest_path(schedule) # Number of states (t=0 to t=T)

    if longest_path_len == 0: # Handle cases where all paths are empty
         return np.empty((num_agents,0), dtype=np.int32), np.empty((num_agents,2), dtype=np.int32)

    num_actions = longest_path_len - 1
    if num_actions < 0: num_actions = 0 # Handle path length 1

    trayect = np.zeros((num_agents, num_actions), dtype=np.int32) # Action 0 (idle) is default
    startings = np.zeros((num_agents, 2), dtype=np.int32)

    action_map = {
        (0, 0): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3, (0, -1): 4,
    }

    agent_idx = 0
    for agent_name, path in schedule.items():
        if not path:
            startings[agent_idx][0] = -1
            startings[agent_idx][1] = -1
            agent_idx += 1
            continue

        startings[agent_idx][0] = path[0]["x"]
        startings[agent_idx][1] = path[0]["y"]

        for i in range(num_actions):
            if i + 1 < len(path):
                prev_x, prev_y = path[i]["x"], path[i]["y"]
                next_x, next_y = path[i+1]["x"], path[i+1]["y"]
                action_tuple = (next_x - prev_x, next_y - prev_y)
                trayect[agent_idx][i] = action_map.get(action_tuple, 0) # Use get with default 0
            else:
                trayect[agent_idx][i] = 0
        agent_idx += 1
    return trayect, startings


def parse_traject(path):
    """Parses solution.yaml to trajectory.npy for all valid cases in the directory."""
    try:
        cases = [d for d in os.listdir(path) if d.startswith("case_") and os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        print(f"Error: Directory not found - {path}. Skipping parsing.")
        return

    if not cases:
        print(f"No 'case_*' directories found in {path}. Skipping parsing.")
        return

    print(f"Parsing trajectories for {len(cases)} cases in {path}...")
    parsed_count = 0
    skipped_count = 0

    # --- Wrap the loop with tqdm ---
    for case_dir in tqdm(cases, desc=f"Parsing Trajectories ({os.path.basename(path)})", unit="case"):
        case_path = os.path.join(path, case_dir)
        solution_path = os.path.join(case_path, "solution.yaml")

        if not os.path.exists(solution_path):
            skipped_count += 1
            continue

        try:
            with open(solution_path) as states_file:
                schedule_data = yaml.load(states_file, Loader=yaml.FullLoader)

            if not schedule_data or "schedule" not in schedule_data:
                 skipped_count += 1
                 continue

            combined_schedule = schedule_data.get("schedule", {}) # Use get for safety

            # Parse the valid schedule
            t, s = parse_trayectories(combined_schedule)

            # Save the trajectory if parsing was successful and result is not empty
            if t.size > 0:
                traj_save_path = os.path.join(case_path, "trajectory.npy")
                np.save(traj_save_path, t)
                parsed_count += 1
                # --- Remove periodic print ---
                # if parsed_count % 25 == 0:
                #     print(f"Parsed Trajectory -- [{parsed_count}/{len(cases) - skipped_count} valid cases] (Processed {case_dir})")
            else:
                 if combined_schedule: # Only count as skipped if schedule wasn't empty
                    skipped_count += 1

        except yaml.YAMLError as e:
            # print(f"Skipping {case_dir}: Error loading solution.yaml: {e}") # Less verbose
            skipped_count += 1
        except Exception as e:
            # print(f"Skipping {case_dir}: Unexpected error during parsing: {e}") # Less verbose
            skipped_count += 1
    # tqdm automatically prints the final bar

    print(f"\nParsing finished for path: {path}")
    print(f"Successfully parsed: {parsed_count} cases.")
    print(f"Skipped: {skipped_count} cases (missing files or errors).")


if __name__ == "__main__":
    # ... (if __name__ block remains the same) ...
    path = "dataset/obs_test"
    print(f"Running trajectory parsing test on path: {path}")
    parse_traject(path)