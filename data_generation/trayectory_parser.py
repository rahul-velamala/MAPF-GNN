# File: data_generation/trayectory_parser.py
# (Handles parsing solution.yaml to actions, using pathlib)

import os
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
# Basic config, inherit level from main_data if run via that
logging.basicConfig(level=logging.INFO)

# --- ADD IMPORTS FROM CBS ---
# This might fail if cbs isn't directly importable, relies on correct execution context
try:
    from cbs.cbs import State, Location
except ImportError:
    try:
        from ..cbs.cbs import State, Location
    except ImportError:
        logger.warning("Could not import State/Location from cbs.cbs. Parsing might fail if internal format used.")
        # Define dummy classes if needed for basic parsing from YAML output format
        class Location: # Dummy if import fails
            def __init__(self, x=-1, y=-1): self.x=x; self.y=y
        class State: # Dummy if import fails
            def __init__(self, t=-1, loc=None): self.time=t; self.location=loc if loc else Location()
# --- --------------------- ---


def parse_cbs_solution_dict(schedule_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses a CBS schedule dictionary (YAML output format) into numpy arrays
    for actions and start positions compatible with GraphEnv.

    Args:
        schedule_dict (dict): Dictionary where keys are agent names and values are lists of
                              dicts [{'t': time, 'x': col, 'y': row}, ...].

    Returns:
        tuple: (trayect, startings)
            - trayect (np.ndarray): Shape (num_agents, num_actions), actions [0-4].
            - startings (np.ndarray): Shape (num_agents, 2), starting [row, col].
            Returns empty arrays if schedule is invalid or contains no actions.
    """
    if not schedule_dict or not isinstance(schedule_dict, dict):
        return np.empty((0,0), dtype=np.int64), np.empty((0,2), dtype=np.int64)

    num_agents = len(schedule_dict)
    if num_agents == 0:
        return np.empty((0,0), dtype=np.int64), np.empty((0,2), dtype=np.int64)

    # Calculate the maximum number of steps (T) based on the path length
    longest_path_len = 0
    for agent_path in schedule_dict.values():
        if agent_path and isinstance(agent_path, list):
            longest_path_len = max(longest_path_len, len(agent_path))

    # If longest path is 0 or 1 state, there are no actions
    if longest_path_len <= 1:
        startings = np.zeros((num_agents, 2), dtype=np.int64)
        agent_idx = 0
        for agent_name, path in schedule_dict.items():
            if path and isinstance(path, list) and len(path) > 0 and isinstance(path[0], dict):
                startings[agent_idx, 0] = path[0].get('y', -1) # Row = y
                startings[agent_idx, 1] = path[0].get('x', -1) # Col = x
            else:
                startings[agent_idx, :] = -1
            agent_idx += 1
        return np.empty((num_agents, 0), dtype=np.int64), startings # 0 actions

    num_actions = longest_path_len - 1

    # Initialize arrays with int64 for compatibility with torch.LongTensor
    trayect = np.zeros((num_agents, num_actions), dtype=np.int64)
    startings = np.zeros((num_agents, 2), dtype=np.int64) # Store [row, col]

    # Action mapping from (delta_row, delta_col) to GraphEnv actions:
    action_map_delta_rc_to_action = {
        (0, 0): 0,  # Idle
        (0, 1): 1,  # Right (dx=1)
        (-1, 0): 2, # Up    (dy=-1)
        (0, -1): 3, # Left  (dx=-1)
        (1, 0): 4,  # Down  (dy=1)
    }

    agent_idx = 0
    for agent_name, path in schedule_dict.items():
        # Check path validity
        if not path or not isinstance(path, list) or len(path) == 0 or not isinstance(path[0], dict):
            logger.warning(f"Invalid path data for agent {agent_name}. Setting start to -1.")
            startings[agent_idx, :] = -1
            # Fill actions with Idle (0) or leave as 0? Let's use Idle.
            trayect[agent_idx, :] = 0
            agent_idx += 1
            continue

        # Record starting position [row, col]
        startings[agent_idx, 0] = path[0].get('y', -1) # Row
        startings[agent_idx, 1] = path[0].get('x', -1) # Col

        # Extract actions
        prev_row, prev_col = startings[agent_idx, 0], startings[agent_idx, 1]
        for i in range(num_actions):
            if i + 1 < len(path): # If agent's path continues
                try:
                    next_state = path[i+1]
                    next_row = next_state.get('y', prev_row) # Default to prev if key missing
                    next_col = next_state.get('x', prev_col)

                    delta_row = next_row - prev_row
                    delta_col = next_col - prev_col
                    action_tuple = (delta_row, delta_col)

                    # Map delta to GraphEnv action index
                    trayect[agent_idx, i] = action_map_delta_rc_to_action.get(action_tuple, 0) # Default to Idle(0)

                    # Update prev for next iteration
                    prev_row, prev_col = next_row, next_col
                except (TypeError, KeyError) as e:
                    logger.warning(f"Error processing step {i+1} for agent {agent_name}: {e}. Using Idle.")
                    trayect[agent_idx, i] = 0
                    # Keep prev_row/col as they were
            else:
                # If agent's path ended, assume Idle action
                trayect[agent_idx, i] = 0

        agent_idx += 1

    return trayect, startings


def parse_traject(path: Path):
    """Parses solution.yaml to trajectory.npy for all valid cases in the directory."""
    if not path.is_dir():
        logger.error(f"Directory not found - {path}. Skipping parsing.")
        return

    try:
        # Use Path.glob to find case directories
        cases = sorted(
            [d for d in path.glob("case_*") if d.is_dir()],
            key=lambda x: int(x.name.split('_')[-1])
        )
    except Exception as e:
        logger.error(f"Error listing cases in {path}: {e}", exc_info=True)
        return

    if not cases:
        logger.warning(f"No 'case_*' directories found in {path}. Skipping parsing.")
        return

    logger.info(f"\n--- Parsing Trajectories for Dataset: {path.name} ---")
    logger.info(f"Found {len(cases)} potential cases in {path}.")
    parsed_count = 0
    skipped_count = 0
    skipped_reasons = {"no_solution": 0, "yaml_error": 0, "parse_error": 0, "empty_schedule": 0, "no_actions": 0}

    pbar = tqdm(cases, desc=f"Parsing {path.name}", unit="case")
    for case_path in pbar: # case_path is now a Path object
        solution_path = case_path / "solution.yaml"
        traj_save_path = case_path / "trajectory.npy" # Save as trajectory.npy

        # Skip if already parsed
        if traj_save_path.exists():
            logger.debug(f"Skipping {case_path.name}: trajectory.npy already exists.")
            # Optionally count existing ones: parsed_count += 1
            continue

        if not solution_path.exists():
            logger.debug(f"Skipping {case_path.name}: solution.yaml not found.")
            skipped_count += 1
            skipped_reasons["no_solution"] += 1
            pbar.set_postfix({"Parsed": parsed_count, "Skip": skipped_count})
            continue

        try:
            with open(solution_path, 'r') as states_file:
                # Use safe_load as we expect standard YAML structure now
                schedule_data = yaml.safe_load(states_file)

            if not isinstance(schedule_data, dict) or "schedule" not in schedule_data:
                logger.warning(f"Skipping {case_path.name}: Invalid structure or missing 'schedule' key in solution.yaml.")
                skipped_count += 1
                skipped_reasons["yaml_error"] += 1
                continue # Skip to next case

            cbs_schedule_dict = schedule_data.get("schedule", {})

            if not cbs_schedule_dict or not isinstance(cbs_schedule_dict, dict):
                logger.warning(f"Skipping {case_path.name}: 'schedule' section is empty or not a dictionary.")
                skipped_count += 1
                skipped_reasons["empty_schedule"] += 1
                continue # Skip to next case

            # Parse the dictionary into actions and start positions
            trajectory_actions, starting_pos = parse_cbs_solution_dict(cbs_schedule_dict)

            # Only save if there are actions
            if trajectory_actions.size > 0:
                np.save(traj_save_path, trajectory_actions)
                # Optionally save starting positions if needed elsewhere, though record.py reads input.yaml
                # np.save(case_path / "start_pos.npy", starting_pos)
                parsed_count += 1
            else:
                logger.debug(f"Skipping {case_path.name}: Parsed trajectory resulted in 0 actions.")
                skipped_count += 1
                skipped_reasons["no_actions"] += 1

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML for {case_path.name}: {e}", exc_info=True)
            skipped_count += 1
            skipped_reasons["yaml_error"] += 1
        except Exception as e:
            logger.error(f"Error parsing trajectory for {case_path.name}: {e}", exc_info=True)
            skipped_count += 1
            skipped_reasons["parse_error"] += 1
        finally:
            pbar.set_postfix({"Parsed": parsed_count, "Skip": skipped_count})

    pbar.close()
    logger.info(f"\n--- Parsing Finished for: {path.name} ---")
    logger.info(f"Successfully parsed and saved non-empty trajectories: {parsed_count} cases.")
    if skipped_count > 0:
        logger.info(f"Skipped saving trajectories for: {skipped_count} cases.")
        logger.info(f"Skip Reasons: {skipped_reasons}")


# --- Main execution block for running standalone ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse CBS solutions (solution.yaml) into trajectory action arrays (trajectory.npy).")
    parser.add_argument("path", help="Directory containing case subdirectories (e.g., dataset/train)")
    args = parser.parse_args()

    data_path = Path(args.path)

    if not data_path.is_dir():
        print(f"Error: Provided path '{args.path}' is not a valid directory.")
    else:
        print(f"--- Running trajectory parsing on path: {data_path.resolve()} ---")
        parse_traject(data_path)
        print(f"--- Trajectory parsing finished for path: {data_path.resolve()} ---")