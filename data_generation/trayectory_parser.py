# File: data_generation/trayectory_parser.py
# (Added missing State/Location import)

import os
import yaml
import numpy as np
import argparse
from tqdm import tqdm

# --- ADD IMPORTS FROM CBS ---
try:
    # Assumes cbs directory is sibling to data_generation or accessible in PYTHONPATH
    from cbs.cbs import State, Location # <<<--- ADDED IMPORT
except ImportError:
    # Fallback if directory structure is different (e.g., running script directly)
    try:
        from ..cbs.cbs import State, Location # <<<--- ADDED IMPORT
    except ImportError:
        raise ImportError("Could not import State or Location class from cbs.cbs. Check paths.")
# --- --------------------- ---

def get_longest_path(schedule):
    """Calculates the maximum length (number of states) of any agent's path."""
    longest = 0
    if not schedule: # Handle empty schedule dict
        return 0
    for agent_path in schedule.values():
        if agent_path and isinstance(agent_path, list): # Check if agent has a path and it's a list
             longest = max(longest, len(agent_path)) # len(path) gives number of states
    return longest

def parse_trayectories(schedule):
    """
    Parses a CBS schedule dictionary into numpy arrays for actions and start positions.
    Handles the internal CBS format (list of State objects per agent).

    Args:
        schedule (dict): Dictionary where keys are agent names and values are lists of
                         State objects [State(t, Location(x,y)), ...].

    Returns:
        tuple: (trayect, startings)
            - trayect (np.ndarray): Shape (num_agents, num_actions), actions for each agent over time.
            - startings (np.ndarray): Shape (num_agents, 2), starting [x, y] for each agent.
            Returns empty arrays if schedule is invalid or contains no actions.
    """
    if not schedule or not isinstance(schedule, dict):
        return np.empty((0,0), dtype=np.int32), np.empty((0,2), dtype=np.int32)

    num_agents = len(schedule)
    if num_agents == 0:
        return np.empty((0,0), dtype=np.int32), np.empty((0,2), dtype=np.int32)

    # Calculate longest path based on list length (number of State objects)
    longest_path_len = 0
    if not schedule:
        longest_path_len = 0
    else:
        for agent_path in schedule.values():
            if agent_path and isinstance(agent_path, list):
                 longest_path_len = max(longest_path_len, len(agent_path))

    if longest_path_len <= 1:
         # Handle no actions case, extract starting positions
         startings = np.zeros((num_agents, 2), dtype=np.int32)
         agent_idx = 0
         for agent_name, path in schedule.items():
             # Check if path is valid and non-empty and contains State objects
             if path and isinstance(path, list) and len(path) > 0 and isinstance(path[0], State): # Now State is defined
                 startings[agent_idx][0] = path[0].location.x # Access attribute
                 startings[agent_idx][1] = path[0].location.y # Access attribute
             else:
                 startings[agent_idx][0] = -1
                 startings[agent_idx][1] = -1
             agent_idx += 1
         return np.empty((num_agents, 0), dtype=np.int32), startings # 0 actions

    num_actions = longest_path_len - 1

    trayect = np.zeros((num_agents, num_actions), dtype=np.int32)
    startings = np.zeros((num_agents, 2), dtype=np.int32)

    # Action mapping from (delta_x, delta_y) based on CBS coords to GraphEnv actions:
    action_map_cbs_delta_to_env_action = {
        (0, 0): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3, (0, 1): 4,
    }

    agent_idx = 0
    for agent_name, path in schedule.items():
        # Check if path is valid and contains State objects
        if not path or not isinstance(path, list) or len(path) == 0 or not isinstance(path[0], State): # Now State is defined
            startings[agent_idx][0] = -1
            startings[agent_idx][1] = -1
            agent_idx += 1
            continue

        # Record starting position from the first State object's Location
        startings[agent_idx][0] = path[0].location.x # Access attribute
        startings[agent_idx][1] = path[0].location.y # Access attribute

        for i in range(num_actions):
            if i + 1 < len(path):
                try:
                    # Access location attributes of State objects
                    prev_loc = path[i].location
                    next_loc = path[i+1].location

                    delta_x = next_loc.x - prev_loc.x
                    delta_y = next_loc.y - prev_loc.y
                    action_tuple = (delta_x, delta_y)

                    trayect[agent_idx, i] = action_map_cbs_delta_to_env_action.get(action_tuple, 0) # Default to Idle
                except AttributeError:
                    # Handle cases where path elements might not be State/Location objects unexpectedly
                    print(f"Warning: Invalid object type in path for {agent_name} at step {i}. Using Idle action.")
                    trayect[agent_idx, i] = 0
            else:
                trayect[agent_idx, i] = 0 # Idle if path ended

        agent_idx += 1

    return trayect, startings


def parse_traject(path):
    """Parses solution.yaml to trajectory.npy for all valid cases in the directory."""
    # ... (Function body remains the same as previous version) ...
    try:
        cases = sorted([
            d for d in os.listdir(path)
            if d.startswith("case_") and os.path.isdir(os.path.join(path, d))
        ], key=lambda x: int(x.split('_')[-1]))
    except FileNotFoundError:
        print(f"Error: Directory not found - {path}. Skipping parsing.")
        return
    except Exception as e:
        print(f"Error listing cases in {path}: {e}")
        return

    if not cases:
        print(f"No 'case_*' directories found in {path}. Skipping parsing.")
        return

    print(f"\n--- Parsing Trajectories for Dataset: {os.path.basename(path)} ---")
    print(f"Found {len(cases)} potential cases in {path}.")
    parsed_count = 0
    skipped_count = 0
    skipped_reasons = {"no_solution": 0, "yaml_error": 0, "parse_error": 0, "empty_schedule": 0, "no_actions": 0}

    pbar = tqdm(cases, desc=f"Parsing {os.path.basename(path)}", unit="case")
    for case_dir in pbar:
        case_path = os.path.join(path, case_dir)
        solution_path = os.path.join(case_path, "solution.yaml")
        traj_save_path = os.path.join(case_path, "trajectory.npy")

        if os.path.exists(traj_save_path):
            continue

        if not os.path.exists(solution_path):
            skipped_count += 1
            skipped_reasons["no_solution"] += 1
            pbar.set_postfix({"Parsed": parsed_count, "Skip": skipped_count})
            continue

        try:
            with open(solution_path) as states_file:
                schedule_data = yaml.safe_load(states_file)

            if not isinstance(schedule_data, dict) or "schedule" not in schedule_data:
                 skipped_count += 1
                 skipped_reasons["yaml_error"] += 1
                 pbar.set_postfix({"Parsed": parsed_count, "Skip": skipped_count})
                 continue

            combined_schedule = schedule_data["schedule"]

            if not combined_schedule or not isinstance(combined_schedule, dict):
                 skipped_count += 1
                 skipped_reasons["empty_schedule"] += 1
                 pbar.set_postfix({"Parsed": parsed_count, "Skip": skipped_count})
                 continue

            t, s = parse_trayectories(combined_schedule)

            if t.size > 0:
                np.save(traj_save_path, t)
                parsed_count += 1
            else:
                 skipped_count += 1
                 skipped_reasons["no_actions"] += 1

        except yaml.YAMLError as e:
            skipped_count += 1
            skipped_reasons["yaml_error"] += 1
        except Exception as e:
            skipped_count += 1
            skipped_reasons["parse_error"] += 1
        finally:
             pbar.set_postfix({"Parsed": parsed_count, "Skip": skipped_count})

    pbar.close()
    print(f"\n--- Parsing Finished for: {os.path.basename(path)} ---")
    print(f"Successfully parsed and saved non-empty trajectories: {parsed_count} cases.")
    if skipped_count > 0:
        print(f"Skipped saving trajectories for: {skipped_count} cases.")
        print("Skip Reasons:", skipped_reasons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse CBS solutions into trajectory arrays.")
    parser.add_argument("path", help="Directory containing case subdirectories with solution.yaml files")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: Provided path '{args.path}' is not a valid directory.")
    else:
        print(f"--- Running trajectory parsing on path: {args.path} ---")
        parse_traject(args.path)
        print(f"--- Trajectory parsing finished for path: {args.path} ---")