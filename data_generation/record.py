# File: data_generation/record.py
# (Revised for consistency with updated GraphEnv, robust recording, pathlib)

import os
import yaml
import numpy as np
from tqdm import tqdm # Import tqdm for progress bar
from pathlib import Path
import logging
import traceback

logger = logging.getLogger(__name__)
# Basic config, inherit level from main_data if run via that
logging.basicConfig(level=logging.INFO)

# Use relative import assuming called from main_data.py in the parent directory
try:
    from grid.env_graph_gridv1 import GraphEnv
except ImportError:
    # Fallback for direct execution or different project structure
    try:
        from ..grid.env_graph_gridv1 import GraphEnv
    except ImportError:
        logger.error("Could not import GraphEnv.", exc_info=True)
        raise ImportError("Could not import GraphEnv. Ensure grid/env_graph_gridv1.py exists and is accessible.")

def make_env(case_path: Path, config: dict) -> GraphEnv | None:
    """
    Creates a GraphEnv environment instance based on the input.yaml
    found in the specified case directory.
    Args:
        case_path (Path): Path object for the specific case directory (e.g., 'dataset/train/case_1').
        config (dict): The main configuration dictionary, containing necessary
                       parameters for GraphEnv init (num_agents, board_size, sensing_range, pad, max_time).
    Returns:
        GraphEnv instance or None if creation fails.
    """
    input_yaml_path = case_path / "input.yaml"
    if not input_yaml_path.exists():
        logger.debug(f"input.yaml not found in {case_path}. Skipping env creation.")
        return None

    try:
        with open(input_yaml_path, 'r') as input_params:
            params = yaml.safe_load(input_params)
    except yaml.YAMLError as e:
        logger.error(f"Error loading YAML from {input_yaml_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error reading {input_yaml_path}: {e}", exc_info=True)
        return None

    # --- Validate YAML Structure ---
    if not isinstance(params, dict) or "agents" not in params or "map" not in params:
        logger.warning(f"Invalid or incomplete input.yaml structure in {case_path}.")
        return None
    if not isinstance(params["map"], dict) or "dimensions" not in params["map"]:
        logger.warning(f"Missing 'map'/'dimensions' in {input_yaml_path}.")
        return None
    if not isinstance(params["agents"], list):
        logger.warning(f"'agents' key is not a list in {input_yaml_path}.")
        return None

    nb_agents_from_yaml = len(params["agents"])
    if nb_agents_from_yaml == 0:
        logger.warning(f"No agents defined in {input_yaml_path}.")
        return None

    # --- Check Consistency with Main Config ---
    config_num_agents = config.get("num_agents")
    if config_num_agents is None:
        logger.error("'num_agents' missing in main config passed to make_env.")
        return None
    config_num_agents = int(config_num_agents)

    if nb_agents_from_yaml != config_num_agents:
        logger.warning(f"Agent count mismatch in {case_path.name}. YAML={nb_agents_from_yaml}, Config={config_num_agents}. Using value from config ({config_num_agents}).")
        # GraphEnv will use config_num_agents, ensure this is intended.

    dimensions_yaml = params["map"]["dimensions"] # CBS format [width, height]
    config_board_size = config.get("board_size") # Expected [rows, cols]
    if config_board_size is None or len(config_board_size) != 2:
        logger.error("'board_size' [rows, cols] missing or invalid in main config.")
        return None
    config_rows, config_cols = map(int, config_board_size)

    # Compare CBS [width, height] vs GraphEnv [rows, cols]
    if dimensions_yaml[0] != config_cols or dimensions_yaml[1] != config_rows:
        logger.warning(f"Map dimensions mismatch in {case_path.name}. YAML(w,h)={dimensions_yaml}, Config(r,c)=[{config_rows},{config_cols}]. Using Config dimensions.")

    # --- Prepare Env Args ---
    obstacles_yaml = params["map"].get("obstacles", []) # List of [x=col, y=row]
    # Convert CBS obstacles [x=col, y=row] to GraphEnv obstacles [row, col]
    obstacles_list = np.array([[item[1], item[0]] for item in obstacles_yaml if isinstance(item, (list, tuple)) and len(item)==2], dtype=np.int32).reshape(-1, 2) if obstacles_yaml else np.empty((0,2), dtype=np.int32)

    # Initialize numpy arrays for start/goal [row, col] using config_num_agents
    starting_pos = np.zeros((config_num_agents, 2), dtype=np.int32)
    goals_env = np.zeros((config_num_agents, 2), dtype=np.int32)

    # Read only up to config_num_agents from YAML
    for i, agent_data in enumerate(params["agents"]):
        if i >= config_num_agents: break # Stop if YAML has more agents than config
        try:
            if not isinstance(agent_data, dict) or "start" not in agent_data or "goal" not in agent_data:
                raise ValueError(f"Invalid agent data format for agent {i}")
            start_cbs = agent_data["start"] # [x=col, y=row]
            goal_cbs = agent_data["goal"]   # [x=col, y=row]
            if not isinstance(start_cbs, (list, tuple)) or len(start_cbs) != 2 or \
               not isinstance(goal_cbs, (list, tuple)) or len(goal_cbs) != 2:
                raise ValueError(f"Invalid start/goal format for agent {i}")

            starting_pos[i, :] = [int(start_cbs[1]), int(start_cbs[0])] # [row, col]
            goals_env[i, :] = [int(goal_cbs[1]), int(goal_cbs[0])]       # [row, col]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Error processing agent {i} data in {input_yaml_path}: {e}", exc_info=True)
            return None # Invalid agent data

    # --- Check for required GraphEnv config keys ---
    required_keys_for_env = ["sensing_range", "pad", "board_size", "num_agents", "max_time"]
    if not all(key in config for key in required_keys_for_env):
        missing_keys = [k for k in required_keys_for_env if k not in config]
        logger.error(f"Main config passed to make_env missing required key(s) for GraphEnv: {missing_keys}")
        return None

    try:
        # Instantiate GraphEnv, passing the main config and specific instance details
        env = GraphEnv(
            config=config, # Pass the whole config dict
            goal=goals_env, # Use the converted [row, col] goals
            starting_positions=starting_pos, # Use the converted [row, col] starts
            obstacles=obstacles_list, # Use the converted [row, col] obstacles
            # sensing_range, pad, etc. are read from config inside GraphEnv __init__
        )
        return env
    except Exception as e:
        logger.error(f"Error initializing GraphEnv for {case_path.name}: {e}", exc_info=True)
        return None


def record_env(path: Path, config: dict):
    """
    Processes expert trajectories in a given dataset path, simulates them in the
    GraphEnv, and records 3-channel FOVs ('states.npy') and Adjacency Matrices ('gso.npy').

    Args:
        path (Path): Path to the dataset directory (e.g., 'dataset/5_8_28/train').
        config (dict): The main configuration dictionary.
    """
    if not path.is_dir():
        logger.error(f"Dataset directory not found: {path}. Cannot record.")
        return

    try:
        # Sort cases numerically based on the index after 'case_'
        cases = sorted(
            [d for d in path.glob("case_*") if d.is_dir()],
            key=lambda x: int(x.name.split('_')[-1])
        )
    except Exception as e:
        logger.error(f"Error listing cases in {path}: {e}", exc_info=True)
        return

    if not cases:
        logger.warning(f"No 'case_*' directories found in {path}. Nothing to record.")
        return

    logger.info(f"\n--- Recording States/GSOs for Dataset: {path.name} ---")
    logger.info(f"Found {len(cases)} potential cases in {path}.")

    # --- Pre-calculate expected shapes from config ---
    try:
        N = int(config["num_agents"])
        C = 3 # Fixed channels
        pad_val = int(config["pad"])
        H = W = (pad_val * 2) - 1
        expected_fov_shape = (N, C, H, W)
        expected_gso_shape = (N, N)
    except (KeyError, ValueError, TypeError) as e:
         logger.error(f"Invalid config for determining expected shapes: {e}", exc_info=True)
         return

    recorded_count = 0
    skipped_count = 0
    sim_error_count = 0
    stats = {"env_creation": 0, "traj_load": 0, "traj_empty": 0, "agent_mismatch": 0, "sim_error": 0, "shape_error": 0}

    pbar = tqdm(cases, desc=f"Recording {path.name}", unit="case")
    for case_path in pbar: # case_path is a Path object
        trajectory_path = case_path / "trajectory.npy"
        states_save_path = case_path / "states.npy"
        gso_save_path = case_path / "gso.npy"

        # Skip if already recorded
        if states_save_path.exists() and gso_save_path.exists():
            # Optionally count existing: recorded_count += 1
            continue

        env = None # Ensure env is reset for each case
        try:
            # 1. Create environment for this specific case
            env = make_env(case_path, config)
            if env is None:
                skipped_count += 1; stats["env_creation"] += 1; continue

            # 2. Load trajectory actions [N, T]
            if not trajectory_path.exists():
                logger.debug(f"Skipping {case_path.name}: trajectory.npy not found.")
                skipped_count += 1; stats["traj_load"] += 1; continue

            trajectory_actions = np.load(trajectory_path) # Shape (N, T)
            if trajectory_actions.ndim != 2:
                 logger.warning(f"Skipping {case_path.name}: trajectory.npy has unexpected ndim {trajectory_actions.ndim} (expected 2).")
                 skipped_count += 1; stats["traj_load"] += 1; continue

            num_agents_traj = trajectory_actions.shape[0]
            num_timesteps = trajectory_actions.shape[1] # T = number of actions

            # Basic validation of loaded trajectory
            if num_timesteps == 0:
                logger.debug(f"Skipping {case_path.name}: trajectory.npy has 0 timesteps.")
                skipped_count += 1; stats["traj_empty"] += 1; continue
            if num_agents_traj != env.nb_agents:
                logger.warning(f"Skipping {case_path.name}: Agent mismatch trajectory ({num_agents_traj}) vs env ({env.nb_agents}).")
                skipped_count += 1; stats["agent_mismatch"] += 1; continue

            # 3. Reset environment to the correct starting state
            initial_obs, _ = env.reset(seed=np.random.randint(1e6)) # Use random seed per recording

            # 4. Initialize recording arrays & validate initial obs shape
            initial_fov = initial_obs.get('fov')
            initial_gso = initial_obs.get('adj_matrix')

            if initial_fov is None or initial_fov.shape != expected_fov_shape:
                logger.warning(f"Skipping {case_path.name}: Initial FOV shape error. Expected {expected_fov_shape}, Got {initial_fov.shape if initial_fov is not None else 'None'}.")
                skipped_count += 1; stats["shape_error"] += 1; continue
            if initial_gso is None or initial_gso.shape != expected_gso_shape:
                logger.warning(f"Skipping {case_path.name}: Initial GSO shape error. Expected {expected_gso_shape}, Got {initial_gso.shape if initial_gso is not None else 'None'}.")
                skipped_count += 1; stats["shape_error"] += 1; continue

            # Recordings have shape (T+1, ...) to include initial state
            recordings_fov = np.zeros((num_timesteps + 1,) + expected_fov_shape, dtype=np.float32)
            recordings_gso = np.zeros((num_timesteps + 1,) + expected_gso_shape, dtype=np.float32)

            # Store initial state (t=0)
            recordings_fov[0] = initial_fov
            recordings_gso[0] = initial_gso

            # 5. Simulate trajectory steps
            current_obs = initial_obs
            sim_successful = True
            for i in range(num_timesteps): # Simulate T actions from t=0 to T-1
                actions_at_step_i = trajectory_actions[:, i] # Actions for step i

                # Step the environment using the expert action
                current_obs, _, terminated, truncated, _ = env.step(actions_at_step_i)

                # Store observation *after* action i (this is the state at t=i+1)
                obs_fov = current_obs.get('fov')
                obs_gso = current_obs.get('adj_matrix')

                # Validate shapes during simulation
                if obs_fov is None or obs_fov.shape != expected_fov_shape or \
                   obs_gso is None or obs_gso.shape != expected_gso_shape:
                    logger.warning(f"Shape error during simulation step {i+1} for {case_path.name}. Skipping saving.")
                    sim_successful = False; stats["shape_error"] += 1; break

                recordings_fov[i + 1] = obs_fov
                recordings_gso[i + 1] = obs_gso

                # Optional: Check for early termination if the expert path finished before T steps
                if terminated or truncated:
                    logger.debug(f"Env terminated/truncated early at step {i+1}/{num_timesteps} for {case_path.name}. Recording stopped.")
                    # Trim arrays to the actual number of steps recorded (i+2 includes state after last action)
                    recordings_fov = recordings_fov[:i+2]
                    recordings_gso = recordings_gso[:i+2]
                    # Also need to adjust the corresponding trajectory if saving it
                    # trajectory_actions = trajectory_actions[:, :i+1] # Actions up to step i
                    break # Stop simulation for this case

            # 6. Save recorded data if simulation was successful
            if sim_successful:
                np.save(states_save_path, recordings_fov)
                np.save(gso_save_path, recordings_gso)
                recorded_count += 1
            else:
                skipped_count += 1 # Increment skip count if sim failed

        except FileNotFoundError as e:
            logger.warning(f"File not found during processing of {case_path.name}: {e}")
            skipped_count += 1; stats["traj_load"] += 1
        except Exception as e:
            logger.error(f"Error processing {case_path.name}: {e}", exc_info=True)
            skipped_count += 1; stats["sim_error"] += 1
        finally:
            if env is not None: env.close()
            pbar.set_postfix({"Rec": recorded_count, "Skip": skipped_count})

    pbar.close()
    logger.info(f"\n--- Recording Finished for: {path.name} ---")
    logger.info(f"Successfully recorded state/GSO for: {recorded_count} cases.")
    if skipped_count > 0:
        logger.info(f"Skipped recording for: {skipped_count} cases.")
        logger.info(f"Skip Reasons: {stats}")
# --- End record_env ---

# No __main__ block needed if run via main_data.py