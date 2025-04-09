# File: data_generation/record.py
# (Modified for consistency with updated GraphEnv and robust recording)

import os
import yaml
import numpy as np
from tqdm import tqdm # Import tqdm for progress bar

# Use relative import assuming called from main_data.py in the parent directory
try:
    from grid.env_graph_gridv1 import GraphEnv
except ImportError:
    # Fallback for direct execution or different project structure
    try:
        from ..grid.env_graph_gridv1 import GraphEnv
    except ImportError:
        raise ImportError("Could not import GraphEnv. Ensure grid/env_graph_gridv1.py exists and is accessible.")

def make_env(case_path, config):
    """
    Creates a GraphEnv environment instance based on the input.yaml
    found in the specified case directory.
    Args:
        case_path (str): Path to the specific case directory (e.g., 'dataset/train/case_1').
        config (dict): The main configuration dictionary, expected to contain
                       'num_agents', 'board_size', 'sensing_range', 'pad', 'max_time', etc.
    Returns:
        GraphEnv instance or None if creation fails.
    """
    input_yaml_path = os.path.join(case_path, "input.yaml")
    if not os.path.exists(input_yaml_path):
        # print(f"Warning: input.yaml not found in {case_path}. Skipping env creation.") # Less verbose
        return None

    try:
        with open(input_yaml_path, 'r') as input_params:
            params = yaml.safe_load(input_params) # Use safe_load for security
    except yaml.YAMLError as e:
        print(f"Error loading YAML from {input_yaml_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading {input_yaml_path}: {e}")
        return None

    if not isinstance(params, dict) or "agents" not in params or "map" not in params:
        print(f"Warning: Invalid or incomplete input.yaml structure in {case_path}.")
        return None
    if not isinstance(params["map"], dict) or "dimensions" not in params["map"]:
         print(f"Warning: Missing 'map'/'dimensions' in {input_yaml_path}.")
         return None
    if not isinstance(params["agents"], list):
         print(f"Warning: 'agents' key is not a list in {input_yaml_path}.")
         return None

    nb_agents_from_yaml = len(params["agents"])
    if nb_agents_from_yaml == 0:
        # print(f"Warning: No agents defined in {input_yaml_path}.") # Less verbose
        return None # Cannot create env with 0 agents easily

    # --- Consistency Checks ---
    config_num_agents = config.get("num_agents")
    if config_num_agents is None:
         print("Error: 'num_agents' missing in main config passed to make_env.")
         return None
    if nb_agents_from_yaml != config_num_agents:
         print(f"Warning: Agent count mismatch in {case_path}. YAML={nb_agents_from_yaml}, Config={config_num_agents}. Using value from config ({config_num_agents}).")
         # Note: GraphEnv will use config_num_agents, potentially leading to issues if YAML is source of truth

    dimensions_yaml = params["map"]["dimensions"] # CBS format [width, height]
    config_board_size = config.get("board_size") # Expected [rows, cols]
    if config_board_size is None or len(config_board_size) != 2:
         print("Error: 'board_size' [rows, cols] missing or invalid in main config.")
         return None

    # CBS [width, height] vs GraphEnv [rows, cols]
    if dimensions_yaml[0] != config_board_size[1] or dimensions_yaml[1] != config_board_size[0]:
        print(f"Warning: Map dimensions mismatch in {case_path}. YAML(w,h)={dimensions_yaml}, Config(r,c)={config_board_size}. Using Config dimensions.")
        # GraphEnv uses config_board_size

    # --- Prepare Env Args ---
    obstacles_yaml = params["map"].get("obstacles", []) # List of [x, y]
    # Convert CBS obstacles [x=col, y=row] to GraphEnv obstacles [row, col]
    # Ensure data type is integer
    obstacles_list = np.array([[item[1], item[0]] for item in obstacles_yaml if isinstance(item, (list, tuple)) and len(item)==2], dtype=np.int32).reshape(-1, 2) if obstacles_yaml else np.empty((0,2), dtype=np.int32)

    # Initialize numpy arrays for start/goal [row, col]
    starting_pos = np.zeros((config_num_agents, 2), dtype=np.int32)
    goals_env = np.zeros((config_num_agents, 2), dtype=np.int32)

    for i, agent_data in enumerate(params["agents"]):
         if i >= config_num_agents:
             print(f"Warning: {case_path} has more agents ({nb_agents_from_yaml}) in YAML than config ({config_num_agents}). Ignoring extra agents.")
             break
         try:
            if not isinstance(agent_data, dict) or "start" not in agent_data or "goal" not in agent_data:
                 print(f"Error: Invalid agent data format for agent {i} in {input_yaml_path}.")
                 return None
            # CBS start/goal are [x=col, y=row] -> convert to GraphEnv [row, col]
            start_cbs = agent_data["start"]
            goal_cbs = agent_data["goal"]
            if not isinstance(start_cbs, (list, tuple)) or len(start_cbs) != 2 or not isinstance(goal_cbs, (list, tuple)) or len(goal_cbs) != 2:
                 print(f"Error: Invalid start/goal format for agent {i} in {input_yaml_path}.")
                 return None

            starting_pos[i, :] = np.array([start_cbs[1], start_cbs[0]], dtype=np.int32) # [row, col]
            goals_env[i, :] = np.array([goal_cbs[1], goal_cbs[0]], dtype=np.int32)       # [row, col]
         except (KeyError, IndexError, TypeError, ValueError) as e:
             print(f"Error processing agent {i} data in {input_yaml_path}: {e}")
             return None # Invalid agent data

    # Ensure required config keys exist for GraphEnv initialization via config
    required_keys_for_env = ["sensing_range", "pad", "board_size", "num_agents", "max_time"] # Add others if GraphEnv needs more
    if not all(key in config for key in required_keys_for_env):
         missing_keys = [k for k in required_keys_for_env if k not in config]
         print(f"Error: Main config passed to make_env missing required key(s) for GraphEnv: {missing_keys}")
         return None

    try:
        # Instantiate GraphEnv, passing the main config and specific instance details
        env = GraphEnv(
            config=config, # Pass the whole config dict
            goal=goals_env, # Use the converted [row, col] goals
            # sensing_range and pad are now read from config inside GraphEnv
            starting_positions=starting_pos, # Use the converted [row, col] starts
            obstacles=obstacles_list, # Use the converted [row, col] obstacles
        )
        return env
    except Exception as e:
        print(f"Error initializing GraphEnv for {case_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def record_env(path, config):
    """
    Processes expert trajectories in a given dataset path, simulates them in the
    GraphEnv, and records 3-channel FOVs ('states.npy') and Adjacency Matrices ('gso.npy').

    Args:
        path (str): Path to the dataset directory (e.g., 'dataset/5_8_28/train').
        config (dict): The main configuration dictionary.
    """
    try:
        # Sort cases numerically based on the index after 'case_'
        cases = sorted([
            d for d in os.listdir(path)
            if d.startswith("case_") and os.path.isdir(os.path.join(path, d))
        ], key=lambda x: int(x.split('_')[-1]))
    except FileNotFoundError:
        print(f"Error: Dataset directory not found: {path}. Cannot record.")
        return
    except Exception as e:
        print(f"Error listing cases in {path}: {e}")
        return

    if not cases:
        print(f"No 'case_*' directories found in {path}. Nothing to record.")
        return

    print(f"\n--- Recording States/GSOs for Dataset: {os.path.basename(path)} ---")
    print(f"Found {len(cases)} potential cases in {path}.")

    recorded_count = 0
    skipped_count = 0
    skipped_reasons = {"env_creation": 0, "traj_load": 0, "traj_empty": 0, "agent_mismatch": 0, "sim_error": 0}

    # Use tqdm for progress bar over cases
    pbar = tqdm(cases, desc=f"Recording {os.path.basename(path)}", unit="case")
    for case_dir in pbar:
        case_path = os.path.join(path, case_dir)
        trajectory_path = os.path.join(case_path, "trajectory.npy")
        states_save_path = os.path.join(case_path, "states.npy")
        gso_save_path = os.path.join(case_path, "gso.npy")

        # Skip if already recorded
        if os.path.exists(states_save_path) and os.path.exists(gso_save_path):
            # recorded_count += 1 # Optionally count existing ones if needed
            # pbar.set_postfix({"Recorded": recorded_count, "Skipped": skipped_count})
            continue # Skip to next case

        env = None # Ensure env is reset for each case try-finally block
        try:
            # 1. Create environment for this specific case
            env = make_env(case_path, config)
            if env is None:
                skipped_count += 1
                skipped_reasons["env_creation"] += 1
                pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count})
                continue # Skip case if env creation failed

            # 2. Load trajectory actions [N, T]
            if not os.path.exists(trajectory_path):
                 skipped_count += 1
                 skipped_reasons["traj_load"] += 1
                 pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count})
                 continue

            trajectory_actions = np.load(trajectory_path) # Shape (N, T)
            num_agents_traj = trajectory_actions.shape[0]
            num_timesteps = trajectory_actions.shape[1] # T = number of actions

            # Basic validation of loaded trajectory
            if num_timesteps == 0:
                 skipped_count += 1
                 skipped_reasons["traj_empty"] += 1
                 pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count})
                 continue
            if num_agents_traj != env.nb_agents:
                 # This indicates a mismatch between the trajectory file and the env config/input.yaml
                 print(f"\nWarning: Agent number mismatch in {case_dir}. Trajectory={num_agents_traj}, Env={env.nb_agents}. Skipping.")
                 skipped_count += 1
                 skipped_reasons["agent_mismatch"] += 1
                 pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count})
                 continue

            # 3. Reset environment to the correct starting state
            # Seed can be omitted here if not needed for recording consistency itself
            initial_obs, _ = env.reset() # This sets env to start pos from input.yaml

            # 4. Initialize recording arrays
            # FOV shape: (N, C, H, W), GSO shape: (N, N) from initial_obs
            fov_shape = initial_obs['fov'].shape
            adj_shape = initial_obs['adj_matrix'].shape
            if len(fov_shape) != 4 or fov_shape[0] != env.nb_agents: # Basic sanity check
                 print(f"\nError: Unexpected initial FOV shape {fov_shape} for {case_dir}. Skipping.")
                 skipped_count += 1; skipped_reasons["sim_error"] += 1
                 pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count}); continue
            if len(adj_shape) != 2 or adj_shape[0] != env.nb_agents or adj_shape[1] != env.nb_agents:
                 print(f"\nError: Unexpected initial GSO shape {adj_shape} for {case_dir}. Skipping.")
                 skipped_count += 1; skipped_reasons["sim_error"] += 1
                 pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count}); continue

            # State recordings: (T+1, N, C, H, W) - includes initial state
            # GSO recordings: (T+1, N, N) - includes initial state GSO
            # Use float32 as expected by models
            recordings_fov = np.zeros((num_timesteps + 1,) + fov_shape, dtype=np.float32)
            recordings_gso = np.zeros((num_timesteps + 1,) + adj_shape, dtype=np.float32)

            # Store initial state (t=0)
            recordings_fov[0] = initial_obs['fov']
            recordings_gso[0] = initial_obs['adj_matrix']

            # 5. Simulate trajectory steps
            current_obs = initial_obs
            for i in range(num_timesteps): # Simulate T actions from t=0 to T-1
                actions_at_step_i = trajectory_actions[:, i] # Actions for step i

                # Step the environment using the expert action
                current_obs, _, terminated, truncated, _ = env.step(actions_at_step_i)

                # Store observation *after* action i (this is the state at t=i+1)
                recordings_fov[i + 1] = current_obs['fov']
                recordings_gso[i + 1] = current_obs['adj_matrix']

                # Optional: Check for early termination if the expert path finished before T steps
                # This shouldn't happen if T is derived correctly from the trajectory file, but as a safeguard:
                if terminated or truncated:
                     # If terminated early, the remaining steps in trajectory are likely invalid.
                     # We have recorded up to state i+1. We might need to pad or truncate.
                     # For simplicity, let's assume T matches the actual expert path length.
                     # If expert terminated at step k < T, trajectory[:, k:] might be padding.
                     # print(f"\nWarning: Env terminated/truncated early at step {i+1}/{num_timesteps} for {case_dir}.")
                     # recordings_fov = recordings_fov[:i+2] # Keep states up to the point of termination
                     # recordings_gso = recordings_gso[:i+2]
                     break # Stop simulation for this case

            # 6. Save recorded data
            np.save(states_save_path, recordings_fov)
            np.save(gso_save_path, recordings_gso)
            recorded_count += 1

        except FileNotFoundError as e:
             # Handles cases where trajectory exists but maybe input.yaml disappeared mid-run
             print(f"\nError: File not found during processing of {case_dir}: {e}")
             skipped_count += 1; skipped_reasons["env_creation"] += 1 # Count as env error
        except (ValueError, IndexError, RuntimeError, TypeError) as e:
            # Catch potential errors during simulation or saving
            print(f"\nError processing {case_dir} at step approx {i if 'i' in locals() else 0}: {e}")
            import traceback; traceback.print_exc() # Print full traceback for debugging
            skipped_count += 1; skipped_reasons["sim_error"] += 1
        except Exception as e:
            # Catch any other unexpected errors
            print(f"\nUnexpected error processing {case_dir}: {e}")
            import traceback; traceback.print_exc()
            skipped_count += 1; skipped_reasons["sim_error"] += 1
        finally:
            if env is not None:
                env.close() # Ensure rendering window is closed if opened
            pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count})

    pbar.close()
    print(f"\n--- Recording Finished for: {os.path.basename(path)} ---")
    print(f"Successfully recorded state/GSO for: {recorded_count} cases.")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} cases.")
        print("Skip Reasons:", skipped_reasons)
    # --- End record_env ---

# No __main__ block needed if run via main_data.py