# File: train.py
# (Revised Version with DAgger/Online Expert, Collision Shielding, Robustness Fixes)

import sys
import os
import argparse
import time
import traceback
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import random # For selecting cases for OE
import shutil # For potentially cleaning failed OE runs
import copy   # For deep copying env for expert simulation
import signal # For CBS timeout handling
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset # Import Dataset types
import logging

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- --------------- ---

# --- Assuming these imports work when running from project root ---
try:
    logger.info("Importing environment and data modules...")
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    from data_loader import GNNDataLoader, CreateDataset # Need CreateDataset info potentially
    from data_generation.record import make_env # Helper to create env from case path
    logger.info("Importing CBS modules...")
    from cbs.cbs import Environment as CBSEnvironment # Rename to avoid clash
    from cbs.cbs import CBS, State, Location # Import CBS components
    # Import the function that parses the CBS solution dictionary format
    from data_generation.trayectory_parser import parse_cbs_solution_dict as parse_traject_cbs 
    logger.info("Imports successful.")

    # --- Local Timeout Handling for CBS ---
    class TimeoutError(Exception): pass
    def handle_timeout(signum, frame): raise TimeoutError("CBS search timed out")
    can_use_alarm = hasattr(signal, 'SIGALRM') # Check if SIGALRM is available
    if not can_use_alarm: logger.warning("Signal alarms (SIGALRM) not available on this OS. CBS timeout may not be strictly enforced.")
    # --- ------------------------------- ---
except ImportError as e:
    logger.error(f"FATAL ERROR importing project modules: {e}", exc_info=True)
    logger.error("Please ensure:")
    logger.error("  1. You are running python from the main project directory (e.g., 'rahul-velamala-mapf-gnn').")
    logger.error("  2. All necessary __init__.py files exist in subdirectories.")
    logger.error("  3. The required libraries (torch, numpy, yaml, etc.) are installed.")
    sys.exit(1)
# --- ----------------------------------------------------------- ---

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Train GNN or Baseline MAPF models with optional DAgger.")
parser.add_argument(
    "--config", type=str, default="configs/config_gnn.yaml",
    help="Path to the YAML configuration file"
)
parser.add_argument(
    "--oe_disable", action="store_true", # Flag to disable Online Expert
    help="Disable the Online Expert (DAgger) data aggregation."
)
parser.add_argument(
    "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Set the logging level."
)
args = parser.parse_args()
# ========================

# --- Configure Logging Level ---
log_level_arg = getattr(logging, args.log_level.upper(), logging.INFO)
logging.basicConfig(level=log_level_arg, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(log_level_arg)
# --- ------------------------- ---

# --- Load Configuration ---
config_file_path = Path(args.config)
logger.info(f"Loading configuration from: {config_file_path}")
try:
    with open(config_file_path, "r") as config_path_obj: # Use path object
        config = yaml.safe_load(config_path_obj)
        if config is None: raise ValueError("Config file is empty or invalid.")
except Exception as e:
    logger.error(f"Could not load or parse config file '{config_file_path}': {e}", exc_info=True)
    sys.exit(1)
# --- ------------------ ---

# --- Setup based on Config ---
try:
    net_type = config.get("net_type", "gnn")
    exp_name = config.get("exp_name", "default_experiment")
    epochs = int(config.get("epochs", 50))
    max_steps_eval = int(config.get("max_steps", 60)) # Max steps for evaluation episodes
    max_steps_train_inference = int(config.get("max_steps_train_inference", max_steps_eval * 3)) # Timeout for OE inference
    eval_frequency = int(config.get("eval_frequency", 5))
    tests_episodes_eval = int(config.get("tests_episodes", 25)) # Num eval episodes per evaluation phase
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    num_agents_config = int(config.get("num_agents", 5))
    pad_config = int(config.get("pad", 3))

    # --- Online Expert (OE) Config ---
    use_online_expert = not args.oe_disable
    oe_config = config.get("online_expert", {}) # Get OE sub-dict
    oe_frequency = int(oe_config.get("frequency_epochs", 4))
    oe_num_cases_to_run = int(oe_config.get("num_cases_to_run", 500))
    oe_cbs_timeout = int(oe_config.get("cbs_timeout_seconds", 10))

except (ValueError, TypeError) as e:
     logger.error(f"Invalid numerical value in configuration: {e}", exc_info=True)
     sys.exit(1)
except KeyError as e:
     logger.error(f"Missing required key in configuration: {e}", exc_info=True)
     sys.exit(1)

# --- Device Setup ---
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {config['device']}")
# --- ----------- ---

# --- Results Directory ---
exp_name_cleaned = exp_name.replace('\\', '/') # Replace backslashes for consistency
results_dir = Path("results") / exp_name_cleaned
try:
    results_dir.mkdir(parents=True, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create results directory {results_dir}: {e}", exc_info=True)
    sys.exit(1)
logger.info(f"Results will be saved in: {results_dir.resolve()}")
# --- ----------------- ---

# --- Model Selection ---
NetworkClass = None
try:
    logger.info(f"Selecting model based on net_type='{net_type}'...")
    if net_type == "baseline":
        from models.framework_baseline import Network as NetworkClass
        logger.info("Using Model: Baseline Network")
    elif net_type == "gnn":
        msg_type = config.get("msg_type", "gcn").lower()
        # framework_gnn.py handles both 'gcn' and 'message' dynamically
        from models.framework_gnn import Network as NetworkClass
        logger.info(f"Using Model: GNN Network (msg_type='{msg_type}')")
        # Ensure config reflects intended type if framework_gnn handles both
        config['msg_type'] = msg_type # Ensure it's set for model init
    else:
        raise ValueError(f"Unknown 'net_type' in config: '{net_type}'")

    if NetworkClass is None: # Should not happen
        raise ImportError("NetworkClass was not assigned.")
except (ImportError, ValueError, KeyError) as e:
     logger.error(f"Failed to import or select model based on config: {e}", exc_info=True)
     sys.exit(1)
# --- --------------- ---

# --- Save Effective Config ---
config_save_path = results_dir / "config_used.yaml"
try:
    config_to_save = config.copy()
    if 'device' in config_to_save and isinstance(config_to_save['device'], torch.device):
        config_to_save['device'] = str(config_to_save["device"])

    with open(config_save_path, "w") as config_path_out:
        yaml.dump(config_to_save, config_path_out, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved effective config to {config_save_path}")
except Exception as e:
    logger.error(f"Could not save config to '{config_save_path}': {e}")
# --- ----------------------- ---

# === Helper Function: Run Inference with Collision Shielding ===
def run_inference_with_shielding(
    model: nn.Module,
    env: GraphEnv,
    max_steps_inference: int,
    device: torch.device,
    net_type: str
    ) -> tuple[dict | None, bool, bool]:
    """
    Runs the current model on the environment with collision shielding applied.
    Detects success, timeout (truncation), or deadlock.

    Args:
        model: The trained model (in eval mode).
        env: An initialized GraphEnv instance (will be reset).
        max_steps_inference (int): Maximum number of steps for this episode.
        device: Torch device ('cuda' or 'cpu').
        net_type (str): 'gnn' or 'baseline'.

    Returns:
        tuple: (history, is_success, is_timeout)
            history (dict | None): If successful run, {'states': list_fov, 'gsos': list_gso,
                                'model_actions': list_proposed, 'shielded_actions': list_actual}
                                Contains data *up to* the step before termination/truncation.
                                Returns None if reset or first step fails.
            is_success (bool): True if env reached terminated state before timeout.
            is_timeout (bool): True if env reached max_steps_inference or was truncated.
    """
    model.eval()
    try:
        obs, info = env.reset() # Reset env for this run
    except Exception as e:
         logger.error(f"Error resetting environment during inference run: {e}", exc_info=True)
         return None, False, True # Indicate failure/timeout

    terminated = False
    truncated = False
    history = {'states': [], 'gsos': [], 'model_actions': [], 'shielded_actions': []}
    idle_action = 0 # Assuming 0 is the idle action index in GraphEnv

    step_count = 0
    while not terminated and not truncated:
        # Check step limit BEFORE taking the step
        if step_count >= max_steps_inference:
            truncated = True
            break

        # Store current state (FOV, GSO) before taking action
        try:
            current_fov_np = obs["fov"] # Shape (N, C, H, W)
            current_gso_np = obs["adj_matrix"] # Shape (N, N)
            history['states'].append(current_fov_np.copy())
            history['gsos'].append(current_gso_np.copy())
        except KeyError as e:
             logger.error(f"Missing key {e} in observation dict at step {env.time}.", exc_info=True)
             return history, False, True # Treat as failure/timeout
        except Exception as e:
             logger.error(f"Error accessing observation at step {env.time}: {e}", exc_info=True)
             return history, False, True

        # Prepare observation for model
        try:
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device) # Add batch dim B=1
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device) # Add batch dim B=1
        except Exception as e:
             logger.error(f"Error converting observation to tensor at step {env.time}: {e}", exc_info=True)
             return history, False, True

        # Get action from model
        with torch.no_grad():
            try:
                if net_type == 'gnn':
                    # Ensure model receives both fov and gso
                    action_scores = model(fov_tensor, gso_tensor) # Expect (1, N, A)
                else: # baseline
                    action_scores = model(fov_tensor) # Baseline only needs FOV
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy() # Shape (N,)
            except Exception as e:
                 logger.error(f"Error during model forward pass at step {env.time}: {e}", exc_info=True)
                 return history, False, True

        history['model_actions'].append(proposed_actions.copy())

        # --- Apply Collision Shielding (Identical to create_gif.py / example.py) ---
        shielded_actions = proposed_actions.copy()
        current_pos_y = env.positionY.copy(); current_pos_x = env.positionX.copy()
        next_pos_y = current_pos_y.copy(); next_pos_x = current_pos_x.copy()
        active_mask = ~env.reached_goal

        for agent_id in np.where(active_mask)[0]:
             act = proposed_actions[agent_id]; dy, dx = env.action_map_dy_dx.get(act, (0,0))
             next_pos_y[agent_id] += dy; next_pos_x[agent_id] += dx
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
        next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)

        if env.obstacles.size > 0:
            active_indices = np.where(active_mask)[0]
            if len(active_indices) > 0:
                 proposed_coords_active=np.stack([next_pos_y[active_indices],next_pos_x[active_indices]],axis=1)
                 obs_coll_active_mask=np.any(np.all(proposed_coords_active[:,np.newaxis,:] == env.obstacles[np.newaxis,:,:],axis=2),axis=1)
                 colliding_agent_indices = active_indices[obs_coll_active_mask]
                 if colliding_agent_indices.size > 0:
                      shielded_actions[colliding_agent_indices] = idle_action
                      next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
                      next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]
                      active_mask[colliding_agent_indices] = False

        active_indices = np.where(active_mask)[0]
        if len(active_indices) > 1:
            next_coords_check=np.stack([next_pos_y[active_indices], next_pos_x[active_indices]],axis=1)
            current_coords_check=np.stack([current_pos_y[active_indices], current_pos_x[active_indices]],axis=1)
            unique_coords, unique_map, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
            vertex_collision_agents = active_indices[np.isin(unique_map, np.where(counts > 1)[0])]
            swapping_agents_list = []
            rel_idx = np.arange(len(active_indices))
            for i in rel_idx:
                 for j in range(i + 1, len(active_indices)):
                     if np.array_equal(next_coords_check[i], current_coords_check[j]) and np.array_equal(next_coords_check[j], current_coords_check[i]):
                         swapping_agents_list.extend([active_indices[i], active_indices[j]])
            swapping_collision_agents = np.unique(swapping_agents_list)
            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents])).astype(int)
            if agents_to_shield_idx.size > 0:
                 shielded_actions[agents_to_shield_idx] = idle_action
        # --- End Collision Shielding ---

        history['shielded_actions'].append(shielded_actions.copy())

        # Step the environment *with shielded actions*
        try:
            obs, reward, terminated, truncated_env, info = env.step(shielded_actions)
            # Environment can signal truncation (e.g. internal error), respect it
            truncated = truncated or truncated_env
            step_count = env.time # Update step count based on env time
        except Exception as e:
             logger.error(f"Error during env.step at step {env.time}: {e}", exc_info=True)
             return history, False, True # Treat as failure/timeout

    # --- After loop finishes ---
    # is_success: achieved goal AND did not hit step limit/timeout
    is_success = terminated and not truncated
    is_timeout = truncated # Timeout if step limit reached OR env signaled truncation

    return history, is_success, is_timeout


# === Helper Function: Call CBS Expert ===
def call_expert_from_state(env_state_info: dict, cbs_timeout_s: int) -> dict | None:
    """
    Creates a CBS problem from the current GraphEnv state and runs the expert.
    Args:
        env_state_info (dict): Contains current 'positions' [N,2](row,col),
                               'goals' [N,2](row,col), 'obstacles' [M,2](row,col),
                               'board_dims' [rows,cols].
        cbs_timeout_s (int): Timeout for the CBS search.
    Returns:
        Dict (CBS solution format {agent_name: [{'t':..,'x':..,'y':..},..]}) or None.
    """
    try:
        # --- Convert GraphEnv state to CBS input format ---
        agents_data = []
        current_positions_rc = env_state_info['positions'] # [row, col]
        goal_positions_rc = env_state_info['goals']       # [row, col]
        num_agents = len(current_positions_rc)

        for i in range(num_agents):
             # Convert GraphEnv [row, col] to CBS [x=col, y=row]
             start_xy = [int(current_positions_rc[i, 1]), int(current_positions_rc[i, 0])]
             goal_xy = [int(goal_positions_rc[i, 1]), int(goal_positions_rc[i, 0])]
             agents_data.append({"start": start_xy, "goal": goal_xy, "name": f"agent{i}"})

        # Convert GraphEnv [rows, cols] to CBS [width=cols, height=rows]
        board_dims_wh = [int(env_state_info['board_dims'][1]), int(env_state_info['board_dims'][0])]

        # Convert GraphEnv obstacles [row, col] to CBS obstacles [x=col, y=row]
        obstacles_xy_list = [[int(obs[1]), int(obs[0])] for obs in env_state_info['obstacles']] if env_state_info['obstacles'].size > 0 else []

        map_data = {"dimensions": board_dims_wh, "obstacles": obstacles_xy_list}
        # --- ----------------------------------------- ---

        # --- Initialize CBS ---
        cbs_env = CBSEnvironment(map_data["dimensions"], agents_data, map_data["obstacles"])
        cbs_solver = CBS(cbs_env, verbose=False) # Keep verbose false for expert calls

    except Exception as e:
        logger.error(f"OE Expert Error: Failed to initialize CBS environment/solver: {e}", exc_info=True)
        return None

    # --- Run CBS with Timeout ---
    solution = None
    original_handler = None
    if can_use_alarm:
        try:
            original_handler = signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(cbs_timeout_s)
        except ValueError as e: # Handle error if running in a thread where signals are restricted
             logger.warning(f"Could not set SIGALRM handler (maybe running in thread?): {e}. CBS timeout may not work.")
             original_handler = None # Ensure it's None

    try:
        solution = cbs_solver.search() # Returns dict in output format or {}
        if can_use_alarm and original_handler: signal.alarm(0) # Disable alarm

        if not solution: # CBS returns {} on failure
             # logger.debug("OE Expert Info: CBS found no solution for the given state.")
             return None

    except TimeoutError:
        # logger.debug(f"OE Expert Info: CBS expert timed out after {cbs_timeout_s}s.")
        return None # Timeout occurred
    except Exception as e:
        logger.error(f"OE Expert Error: CBS search failed unexpectedly: {e}", exc_info=True)
        return None # Other error during search
    finally:
        # Restore original signal handler only if it was set
        if can_use_alarm and original_handler is not None:
            try:
                signal.signal(signal.SIGALRM, original_handler)
                signal.alarm(0) # Ensure alarm is off
            except ValueError: pass # Ignore error if handler cannot be restored

    # --- Return CBS solution dictionary (already in output format) ---
    return solution


# === Helper Function: Aggregate Expert Data via Simulation ===
def aggregate_expert_data(
    expert_solution_cbs: dict,
    start_env_instance: GraphEnv,
    config: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Simulates the expert path from the deadlock start state in a *copy* of the
    environment and records the sequence of (FOV, GSO, Expert_Action).

    Args:
        expert_solution_cbs (dict): Raw output from call_expert_from_state (CBS YAML format).
        start_env_instance (GraphEnv): Environment instance *at the deadlock state*.
        config (dict): Main configuration (needed for GraphEnv parameters like pad).

    Returns:
        Tuple: (numpy_states, numpy_gsos, numpy_expert_actions) or None if failed.
               Shapes: [T_agg, N, C, H, W], [T_agg, N, N], [T_agg, N]
               where T_agg is the number of steps in the expert path simulation.
    """
    if not expert_solution_cbs or 'schedule' not in expert_solution_cbs or not expert_solution_cbs['schedule']:
        logger.debug("OE Aggregate: Invalid or empty expert solution provided.")
        return None

    # 1. Parse expert CBS solution into GraphEnv actions array [N, T_expert]
    try:
        # parse_traject_cbs expects CBS schedule format, returns GraphEnv actions
        # It needs the CBS format {agent: [{'t':..,'x':..,'y':..},..]}
        expert_actions_np, _ = parse_traject_cbs(expert_solution_cbs['schedule']) # Shape (N, T_expert)
        if expert_actions_np.size == 0:
             logger.debug("OE Aggregate Info: Expert path from CBS has 0 actions.")
             return None
        num_expert_actions = expert_actions_np.shape[1]
    except Exception as e:
        logger.error(f"OE Aggregate Error: Failed to parse expert CBS solution: {e}", exc_info=True)
        return None

    # 2. Simulate expert actions in a DEEP COPY of the environment from the deadlock state
    env_sim = None
    try:
        # Create a completely independent copy of the environment at the deadlock state
        # Ensure rendering is off for the simulation copy
        original_render_mode = start_env_instance.render_mode
        start_env_instance.render_mode = None # Temporarily disable rendering for copy
        env_sim = copy.deepcopy(start_env_instance)
        start_env_instance.render_mode = original_render_mode # Restore original mode
        # CRITICAL: Reset the time counter and max_time for the simulation copy
        env_sim.time = 0
        env_sim.max_time = num_expert_actions + 10 # Give some buffer

        # logger.debug(f"OE Aggregate: Simulating expert path (len {num_expert_actions}) in copied Env (max_time={env_sim.max_time}).")
    except Exception as e:
         logger.error(f"OE Aggregate Error: Failed to deepcopy environment: {e}", exc_info=True)
         if env_sim: env_sim.close()
         return None

    # Lists to store data from simulation
    aggregated_states_list = [] # Holds FOV arrays (N, C, H, W)
    aggregated_gsos_list = []   # Holds GSO arrays (N, N)
    aggregated_actions_list = []# Holds action arrays (N,)

    try:
        # Get the state *at the deadlock* (time t=0 for this simulation)
        # Need to get observation from the copied env, ensure board is updated first
        env_sim.updateBoard()
        current_obs_sim = env_sim.getObservations()
        aggregated_states_list.append(current_obs_sim['fov'])
        aggregated_gsos_list.append(current_obs_sim['adj_matrix'])

        # Simulate each expert action step
        terminated_sim = False
        truncated_sim = False
        for t in range(num_expert_actions):
            if terminated_sim or truncated_sim: break # Stop if env terminates early

            actions_t = expert_actions_np[:, t] # Expert actions for this step
            aggregated_actions_list.append(actions_t) # Store action t

            # Step the simulation environment
            current_obs_sim, _, terminated_sim, truncated_sim, _ = env_sim.step(actions_t)

            # Store the state resulting from action t (state at sim time t+1)
            aggregated_states_list.append(current_obs_sim['fov'])
            aggregated_gsos_list.append(current_obs_sim['adj_matrix'])

        # --- Post-Simulation Processing ---
        num_sim_steps = len(aggregated_actions_list) # Number of actions successfully simulated
        if num_sim_steps == 0:
             logger.debug("OE Aggregate Info: No actions were simulated.")
             return None

        # We need pairs of (state_t, action_t, gso_t).
        # We have T_sim actions and T_sim+1 states/gsos recorded.
        # Use states[0...T_sim-1] and gsos[0...T_sim-1] with actions[0...T_sim-1].
        # Convert lists to numpy arrays
        final_states = np.stack(aggregated_states_list[:num_sim_steps]).astype(np.float32)
        final_gsos = np.stack(aggregated_gsos_list[:num_sim_steps]).astype(np.float32)
        final_actions = np.stack(aggregated_actions_list).astype(np.int64)

        # Final check for consistent lengths
        if not (final_states.shape[0] == final_gsos.shape[0] == final_actions.shape[0] == num_sim_steps):
             logger.error(f"OE Aggregate Error: Mismatch shapes after stacking. SimSteps:{num_sim_steps}, S:{final_states.shape}, G:{final_gsos.shape}, A:{final_actions.shape}")
             return None

        # logger.debug(f"OE Aggregate: Successfully aggregated {num_sim_steps} state-action pairs.")
        return final_states, final_gsos, final_actions

    except Exception as e:
         logger.error(f"OE Aggregate Error: Failed during expert path simulation or stacking: {e}", exc_info=True)
         return None
    finally:
         if env_sim is not None:
              env_sim.close() # Close the rendering window of the copied env if opened

# === Custom Dataset for Aggregated Expert Data ===
class AggregatedDataset(Dataset):
    """A Dataset to hold aggregated expert data collected during training."""
    def __init__(self, aggregated_list_of_dicts: list[dict], config_for_shape: dict):
        """
        Args:
            aggregated_list_of_dicts (list): A list where each element is a dictionary
                                             {'states': np.array, 'gsos': np.array, 'actions': np.array}
                                             from one expert rollout.
            config_for_shape (dict): Config needed to determine expected shapes if list is empty.
        """
        self.config = config_for_shape
        if not aggregated_list_of_dicts:
             logger.warning("AggregatedDataset: Initialized with empty data list.")
             self._initialize_empty()
        else:
             # Concatenate data from all dictionaries in the list
             try:
                 # Filter out empty arrays before concatenating
                 all_s = [d['states'] for d in aggregated_list_of_dicts if d['states'].size > 0]
                 all_g = [d['gsos'] for d in aggregated_list_of_dicts if d['gsos'].size > 0]
                 all_a = [d['actions'] for d in aggregated_list_of_dicts if d['actions'].size > 0]

                 if not all_s or not all_g or not all_a: # Check if any list became empty after filtering
                      logger.warning("AggregatedDataset: No valid data found after filtering empty arrays.")
                      self._initialize_empty()
                      return

                 self.all_states = np.concatenate(all_s, axis=0).astype(np.float32)
                 self.all_gsos = np.concatenate(all_g, axis=0).astype(np.float32)
                 self.all_actions = np.concatenate(all_a, axis=0).astype(np.int64)

                 logger.debug(f"AggregatedDataset: Combined data shapes: S={self.all_states.shape}, G={self.all_gsos.shape}, A={self.all_actions.shape}")
             except ValueError as e: # Handle potential concat errors (e.g., shape mismatch)
                  logger.error(f"ERROR in AggregatedDataset: Could not concatenate data: {e}", exc_info=True)
                  self._initialize_empty() # Initialize empty to prevent crashes

        self.count = len(self.all_states)
        if self.count > 0 and (self.count != len(self.all_gsos) or self.count != len(self.all_actions)):
             logger.error(f"ERROR in AggregatedDataset: Mismatch in sample count after concatenation! S:{len(self.all_states)}, G:{len(self.all_gsos)}, A:{len(self.all_actions)}")
             # Correct counts to minimum consistent length
             min_count = min(len(self.all_states), len(self.all_gsos), len(self.all_actions))
             self.all_states = self.all_states[:min_count]
             self.all_gsos = self.all_gsos[:min_count]
             self.all_actions = self.all_actions[:min_count]
             self.count = min_count
             logger.warning(f"Corrected aggregated dataset count to {self.count}")

    def _initialize_empty(self):
        """Initializes empty numpy arrays with correct shapes and dtypes."""
        N = int(self.config.get("num_agents", 0))
        C = 3 # Default channels from GraphEnv
        pad_val = int(self.config.get("pad", 3))
        H = W = (pad_val * 2) - 1
        self.all_states = np.empty((0, N, C, H, W), dtype=np.float32)
        self.all_gsos = np.empty((0, N, N), dtype=np.float32)
        self.all_actions = np.empty((0, N), dtype=np.int64)
        self.count = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(f"Index {index} out of bounds for AggregatedDataset with size {self.count}")

        state = torch.from_numpy(self.all_states[index]).float()
        gso = torch.from_numpy(self.all_gsos[index]).float()
        action = torch.from_numpy(self.all_actions[index]).long() # Actions must be Long

        # Return order must match DataLoader expectation in training loop: state, action, gso
        return state, action, gso

# === Main Training Script ===
if __name__ == "__main__":

    logger.info("\n----- Effective Configuration -----")
    pprint(config, indent=2)
    logger.info(f"Using device: {config['device']}")
    logger.info(f"Online Expert (DAgger): {'Enabled' if use_online_expert else 'Disabled'}")
    if use_online_expert:
         logger.info(f"  OE Frequency: Every {oe_frequency} epochs")
         logger.info(f"  OE Cases to run: {oe_num_cases_to_run}")
         logger.info(f"  OE CBS Timeout: {oe_cbs_timeout}s")
         logger.info(f"  OE Max Inference Steps: {max_steps_train_inference}")
    logger.info("---------------------------------\n")

    # --- Data Loading ---
    data_loader_manager = None
    train_loader = None
    valid_loader = None
    original_train_dataset = None
    try:
        data_loader_manager = GNNDataLoader(config)
        train_loader = data_loader_manager.train_loader
        valid_loader = data_loader_manager.valid_loader

        if not train_loader or not hasattr(train_loader, 'dataset') or len(train_loader.dataset) == 0:
             logger.error("Training data loader is unexpectedly empty or invalid after initialization.")
             sys.exit(1)

        # Keep a reference to the original dataset for OE DAgger
        original_train_dataset = train_loader.dataset
        logger.info(f"Initial training samples (timesteps): {len(original_train_dataset)}")
        if valid_loader: logger.info(f"Validation samples (timesteps): {len(valid_loader.dataset)}")

    except Exception as e:
        logger.error(f"Failed to initialize/load data: {e}", exc_info=True)
        sys.exit(1)
    # --- ------------ ---

    # --- Model, Optimizer, Criterion ---
    model = None
    optimizer = None
    criterion = None
    try:
        model = NetworkClass(config) # Instantiate the selected NetworkClass
        model.to(config["device"])
        logger.info(f"\nModel '{type(model).__name__}' initialized on {config['device']}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {num_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Use reduction='mean' by default - average loss over batch elements
        criterion = nn.CrossEntropyLoss()

    except Exception as e:
        logger.error(f"Failed init model/optimizer/criterion: {e}", exc_info=True)
        sys.exit(1)
    # --- --------------------------- ---

    # --- Training Loop Setup ---
    all_epoch_metrics = [] # List to store metrics dict per epoch
    best_eval_success_rate = -1.0 # Track best validation success rate
    aggregated_expert_data_list = [] # Holds dicts {'states':S,'gsos':G,'actions':A} from *all* OE runs
    current_train_dataset = original_train_dataset # Start with the original data

    # --- Sanity Check: Ensure original dataset has required attributes for OE ---
    if use_online_expert:
         # Need 'cases' (list of Path objects) and 'root_dir' (Path object)
         if not hasattr(original_train_dataset, 'cases') or not isinstance(getattr(original_train_dataset, 'cases', None), list) or \
            not hasattr(original_train_dataset, 'root_dir') or not isinstance(getattr(original_train_dataset, 'root_dir', None), Path):
             logger.error("Original training dataset object (required for OE) is missing 'cases' list or 'root_dir' Path attribute.")
             logger.error("Online Expert cannot function without access to the original case paths.")
             use_online_expert = False # Disable OE if dataset structure is wrong
             logger.warning("Disabled Online Expert due to incompatible dataset structure.")

    logger.info(f"\n--- Starting Training for {epochs} epochs ---")
    training_start_time = time.time()

    # --- Main Epoch Loop ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*15} Epoch {epoch+1}/{epochs} {'='*15}")

        # === Online Expert (DAgger) Data Aggregation Phase ===
        run_oe_this_epoch = use_online_expert and ((epoch + 1) % oe_frequency == 0) and epoch > 0 # Avoid running on epoch 0

        if run_oe_this_epoch:
            logger.info(f"--- Running Online Expert Data Aggregation (Epoch {epoch+1}) ---")
            oe_start_time = time.time()
            num_deadlocks_found = 0
            num_expert_calls = 0
            num_expert_success = 0
            newly_aggregated_samples_count = 0
            new_data_this_epoch = [] # Collect data only from this specific OE run

            # Get list of case Paths and root directory from the original dataset
            original_case_paths = getattr(original_train_dataset, 'cases', []) # Should be list of Path objects

            if not original_case_paths:
                logger.error("OE Error: Cannot run OE because original case list is unavailable or empty.")
            else:
                num_original_cases = len(original_case_paths)
                num_to_sample = min(oe_num_cases_to_run, num_original_cases)
                # Sample indices from the list of original cases
                indices_to_run = random.sample(range(num_original_cases), num_to_sample)
                logger.info(f"Selected {len(indices_to_run)}/{num_original_cases} cases for OE inference run.")

                oe_pbar = tqdm(indices_to_run, desc="OE Inference", unit="case", leave=False)
                for case_idx_in_orig_list in oe_pbar:
                    case_path = original_case_paths[case_idx_in_orig_list] # Get Path object
                    env_oe = None # Ensure env_oe is defined for finally block

                    try:
                        # Create environment for this case using helper function
                        # Pass the main config, make_env extracts necessary parts
                        env_oe = make_env(case_path, config)
                        if env_oe is None:
                            # logger.debug(f"OE Warning: Could not create env for case {case_path.name}. Skipping.")
                            continue

                        # Run current model with shielding to find deadlocks
                        # Make sure env has render_mode=None for OE runs
                        original_render_mode_oe = env_oe.render_mode
                        env_oe.render_mode = None
                        history, is_success, is_timeout = run_inference_with_shielding(
                            model, env_oe, max_steps_train_inference, config["device"], net_type
                        )
                        env_oe.render_mode = original_render_mode_oe # Restore mode

                        # Deadlock = timed out AND not successful
                        is_deadlock = is_timeout and not is_success

                        if is_deadlock:
                            num_deadlocks_found += 1
                            # The 'env_oe' instance is now AT the deadlock state
                            # Extract necessary info for CBS expert
                            deadlock_state_info = {
                                "positions": env_oe.get_current_positions(), # [N,2](row,col)
                                "goals": env_oe.goal.copy(),             # [N,2](row,col)
                                "obstacles": env_oe.obstacles.copy(),    # [M,2](row,col)
                                "board_dims": env_oe.config['board_size'] # [rows, cols]
                            }

                            # Call the expert solver
                            num_expert_calls += 1
                            expert_solution_dict = call_expert_from_state(deadlock_state_info, oe_cbs_timeout)

                            if expert_solution_dict:
                                num_expert_success += 1
                                # If expert found a path, simulate it to get (s,a,g) pairs
                                aggregated_data = aggregate_expert_data(expert_solution_dict, env_oe, config)

                                if aggregated_data:
                                    states_agg, gsos_agg, actions_agg = aggregated_data
                                    # Append dict of numpy arrays for temporary storage this epoch
                                    new_data_this_epoch.append({
                                        "states": states_agg, "gsos": gsos_agg, "actions": actions_agg
                                    })
                                    num_new = len(states_agg)
                                    newly_aggregated_samples_count += num_new
                                    oe_pbar.set_postfix({"Deadlocks": num_deadlocks_found, "ExpertOK": num_expert_success, "NewSamples": newly_aggregated_samples_count})
                            # else: Expert failed or timed out

                    except Exception as e:
                         logger.error(f"OE Error: Unhandled exception processing case {case_path.name}: {e}", exc_info=True);
                    finally:
                         if env_oe: env_oe.close() # Close env figure if opened

            # --- Update main aggregated list and potentially recreate DataLoader ---
            if newly_aggregated_samples_count > 0:
                logger.info(f"OE Aggregation Summary: Found {num_deadlocks_found} deadlocks. Expert succeeded {num_expert_success}/{num_expert_calls} times.")
                logger.info(f"Aggregated {newly_aggregated_samples_count} new state-action pairs this epoch.")

                # Add data collected this epoch to the main list tracking all aggregated data
                aggregated_expert_data_list.extend(new_data_this_epoch)
                total_aggregated_samples = sum(len(d['states']) for d in aggregated_expert_data_list)
                logger.info(f"Total aggregated expert samples accumulated: {total_aggregated_samples}")

                # Create a Dataset from ALL aggregated data collected so far
                # Pass config for shape inference if list becomes empty
                aggregated_dataset_all = AggregatedDataset(aggregated_expert_data_list, config)

                # Combine original dataset and ALL aggregated data for training
                if len(aggregated_dataset_all) > 0:
                    current_train_dataset = ConcatDataset([original_train_dataset, aggregated_dataset_all])
                    logger.info(f"Combined dataset size for training this epoch: {len(current_train_dataset)} samples.")
                else:
                     logger.warning("Aggregated dataset was empty, using only original data.")
                     current_train_dataset = original_train_dataset

                # Create a NEW DataLoader with the combined dataset for the training phase below
                try:
                    # Make sure persistent_workers matches initial loader settings
                    persistent = (config["num_workers"] > 0)
                    train_loader = DataLoader(
                        current_train_dataset,
                        batch_size=config["batch_size"], shuffle=True, # Shuffle combined data
                        num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
                        persistent_workers=persistent
                    )
                    logger.info("Created new DataLoader with combined original + aggregated data.")
                except Exception as e:
                    logger.error(f"ERROR creating combined DataLoader: {e}. Falling back to previous DataLoader.", exc_info=True)
                    # Fallback: reuse the previous train_loader (might miss latest data)
                    if train_loader is None: # Safety check if first OE run fails here
                         train_loader = data_loader_manager.train_loader # Very first loader
                         current_train_dataset = train_loader.dataset # Update ref
                    # If train_loader existed before, reuse it
            else:
                 logger.info("OE Aggregation: No new samples aggregated this epoch.")
                 # If no new data, continue using the existing train_loader (which might be combined from previous epochs)
                 if train_loader is None: # Should not happen after first epoch
                      logger.error("train_loader is None after OE phase with no new data. Cannot train.")
                      sys.exit(1)
                 current_train_dataset = train_loader.dataset # Update reference
                 logger.info(f"Continuing training with existing dataset size: {len(current_train_dataset)}")


            oe_duration = time.time() - oe_start_time
            logger.info(f"--- Online Expert Data Aggregation Finished ({oe_duration:.2f}s) ---")
        # === End Online Expert Block ===


        # ##### Training Phase #########
        model.train() # Set model to training mode
        epoch_train_loss = 0.0
        batches_processed = 0
        if train_loader is None: # Safety check
            logger.critical("FATAL ERROR: train_loader is None before training phase. Exiting.")
            sys.exit(1)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

        for i, batch_data in enumerate(train_pbar):
            # Expected return order from DataLoader: state, action, gso
            try:
                # Non-blocking only useful with CUDA and num_workers > 0
                non_blocking = config["device"].type == 'cuda' and config["num_workers"] > 0
                states_batch = batch_data[0].to(config["device"], non_blocking=non_blocking) # (B, N, C, H, W)
                target_actions_batch = batch_data[1].to(config["device"], non_blocking=non_blocking) # (B, N), LongTensor
                gso_batch = batch_data[2].to(config["device"], non_blocking=non_blocking) # (B, N, N)
            except IndexError:
                 logger.error(f"Incorrect data format received from DataLoader at batch {i}. Expected 3 items.", exc_info=True)
                 continue # Skip this batch
            except Exception as e:
                 logger.error(f"Error unpacking/moving batch {i} to device: {e}", exc_info=True)
                 continue # Skip this batch

            optimizer.zero_grad()

            # --- Forward Pass ---
            try:
                # Model expects (B, N, C, H, W) and (B, N, N)
                if net_type == 'gnn':
                     output_logits = model(states_batch, gso_batch) # Output: (B, N, A)
                else: # baseline
                     output_logits = model(states_batch) # Baseline doesn't use GSO
            except Exception as e:
                logger.error(f"Error during forward pass batch {i}: {e}", exc_info=True)
                logger.error(f"State shape: {states_batch.shape}, GSO shape: {gso_batch.shape}")
                continue # Skip batch

            # --- Loss Calculation ---
            try:
                # Reshape for CrossEntropyLoss: Output (B*N, A), Target (B*N)
                # B, N, A = output_logits.shape -> gives error if B=1 and squeeze happens
                batch_b, batch_n, num_actions_model = output_logits.shape
                if batch_n != num_agents_config:
                    logger.warning(f"Batch N ({batch_n}) != config N ({num_agents_config})")
                output_reshaped = output_logits.reshape(-1, num_actions_model) # (B*N, A)
                target_reshaped = target_actions_batch.reshape(-1) # (B*N)

                # Check model output action dimension
                expected_actions = 5 # Assuming 5 actions
                if num_actions_model != expected_actions:
                     logger.error(f"Model output action dimension mismatch! Expected {expected_actions}, got {num_actions_model}. Logits shape: {output_logits.shape}")
                     continue # Skip batch if action dimension is wrong

                # Ensure target actions are within the valid range [0, num_actions-1]
                if torch.any(target_reshaped < 0) or torch.any(target_reshaped >= expected_actions):
                    logger.error(f"Invalid target action found in batch {i}. Values outside [0, {expected_actions-1}]. Min: {target_reshaped.min()}, Max: {target_reshaped.max()}. Target shape: {target_actions_batch.shape}")
                    continue # Skip batch with invalid targets

                # Ensure shapes match for loss calculation
                if output_reshaped.shape[0] != target_reshaped.shape[0]:
                     logger.error(f"Shape mismatch for loss calculation batch {i}. Output samples: {output_reshaped.shape[0]}, Target samples: {target_reshaped.shape[0]}.")
                     continue

                batch_loss = criterion(output_reshaped, target_reshaped)

                # Check for NaN/Inf loss
                if not torch.isfinite(batch_loss):
                    logger.warning(f"Loss is NaN or Inf at batch {i}. Skipping backward pass.")
                    # Consider logging inputs/outputs that caused NaN loss if it persists
                    continue

            except Exception as e:
                 logger.error(f"Error during loss calculation batch {i}: {e}", exc_info=True)
                 continue # Skip batch

            # --- Backward Pass & Optimizer Step ---
            try:
                batch_loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimizer step batch {i}: {e}", exc_info=True)
                continue # Skip optimization step if backward fails

            epoch_train_loss += batch_loss.item()
            batches_processed += 1
            if batches_processed > 0:
                train_pbar.set_postfix({"AvgLoss": epoch_train_loss / batches_processed})
        # --- End Training Batch Loop ---
        train_pbar.close()

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_train_loss / batches_processed if batches_processed > 0 else 0.0
        epoch_duration_train = time.time() - epoch_start_time # Track training time separetely
        logger.info(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f} | Train Duration: {epoch_duration_train:.2f}s")

        # Store epoch metrics
        current_epoch_data = {
            "Epoch": epoch + 1,
            "Average Training Loss": avg_epoch_loss,
            "Evaluation Episode Success Rate": np.nan, # Placeholder
            "Evaluation Avg Steps (Success)": np.nan, # Placeholder
            "Evaluation Episodes Tested": 0,
            "Evaluation Episodes Succeeded": 0,
            "Training Samples Used": len(current_train_dataset) # Samples used in this epoch
        }

        # ######### Evaluation Phase #########
        # Evaluate periodically and at the final epoch
        run_eval = ((epoch + 1) % eval_frequency == 0) or ((epoch + 1) == epochs)
        if run_eval and tests_episodes_eval > 0:
            eval_start_time = time.time()
            logger.info(f"\n--- Running Evaluation after Epoch {epoch+1} ---")
            model.eval() # Set model to evaluation mode
            eval_success_count = 0
            eval_steps_success = [] # Store steps only for successful episodes

            # --- Use evaluation specific parameters if defined in config ---
            eval_board_dims = config.get("eval_board_size", config.get("board_size", [16, 16]))
            eval_obstacles_count = config.get("eval_obstacles", config.get("obstacles", 6))
            eval_agents_count = config.get("eval_num_agents", config.get("num_agents", 4))
            # Ensure eval agent count matches model's num_agents if model is fixed size
            if hasattr(model, 'num_agents') and eval_agents_count != model.num_agents:
                 logger.warning(f"Eval agent count ({eval_agents_count}) differs from model ({model.num_agents}). Using model's count for eval env.")
                 eval_agents_count = model.num_agents
            elif eval_agents_count != num_agents_config: # Fallback check against global config N
                 logger.warning(f"Eval agent count ({eval_agents_count}) differs from global config N ({num_agents_config}). Using global config N for eval env.")
                 eval_agents_count = num_agents_config

            eval_sensing_range = config.get("eval_sensing_range", config.get("sensing_range", 4))
            eval_pad = config.get("eval_pad", config.get("pad", 3)) # Use same pad as model trained on
            eval_max_steps_run = int(config.get("eval_max_steps", max_steps_eval)) # Use specific eval steps limit

            eval_pbar = tqdm(range(tests_episodes_eval), desc=f"Epoch {epoch+1} Evaluation", leave=False, unit="ep")
            for episode_idx in eval_pbar:
                env_eval = None # Ensure env is defined for finally block
                try:
                    # Generate a new random scenario for each evaluation episode
                    eval_obstacles_ep = create_obstacles(eval_board_dims, eval_obstacles_count)
                    eval_start_pos_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep)
                    eval_goals_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep, current_starts=eval_start_pos_ep)

                    # Create eval env instance using eval parameters
                    eval_config_instance = config.copy() # Start with base config
                    eval_config_instance.update({ # Override with eval params
                         "board_size": eval_board_dims,
                         "num_agents": eval_agents_count,
                         "sensing_range": eval_sensing_range,
                         "pad": eval_pad,
                         "max_time": eval_max_steps_run, # Use eval step limit
                         "render_mode": None # Ensure no rendering during batch evaluation
                    })

                    env_eval = GraphEnv(config=eval_config_instance,
                                        goal=eval_goals_ep, obstacles=eval_obstacles_ep,
                                        starting_positions=eval_start_pos_ep)

                    # Run inference *with shielding* to get performance
                    # run_inference_with_shielding resets the env internally
                    _, is_success, _ = run_inference_with_shielding(
                        model, env_eval, eval_max_steps_run, config["device"], net_type
                    )

                    if is_success:
                        eval_success_count += 1
                        eval_steps_success.append(env_eval.time) # Record steps taken for success

                    eval_pbar.set_postfix({"Success": f"{eval_success_count}/{episode_idx+1}"})

                except Exception as e:
                    logger.error(f"Error during evaluation episode {episode_idx+1}: {e}", exc_info=True)
                    # Continue to next episode
                finally:
                     if env_eval: env_eval.close() # Close env if opened

            # --- End Eval Loop ---
            eval_pbar.close()
            eval_success_rate = eval_success_count / tests_episodes_eval if tests_episodes_eval > 0 else 0.0
            avg_steps_succ = np.mean(eval_steps_success) if eval_steps_success else np.nan
            eval_duration = time.time() - eval_start_time

            logger.info(f"Evaluation Complete: SR={eval_success_rate:.4f} ({eval_success_count}/{tests_episodes_eval}), AvgSteps(Succ)={avg_steps_succ:.2f} | Duration={eval_duration:.2f}s")

            # Update epoch metrics
            current_epoch_data.update({
                "Evaluation Episode Success Rate": eval_success_rate,
                "Evaluation Avg Steps (Success)": avg_steps_succ,
                "Evaluation Episodes Tested": tests_episodes_eval,
                "Evaluation Episodes Succeeded": eval_success_count,
            })

            # --- Save Best Model based on Eval Success Rate ---
            if eval_success_rate >= best_eval_success_rate:
                 logger.info(f"*** New best evaluation success rate ({eval_success_rate:.4f}), saving model... ***")
                 best_eval_success_rate = eval_success_rate
                 best_model_path = results_dir / "model_best.pt"
                 try:
                      torch.save(model.state_dict(), best_model_path)
                 except Exception as e:
                      logger.error(f"Failed to save best model: {e}", exc_info=True)
            # --- --------------- ---
        # --- End Evaluation Phase ---

        # Append metrics for this epoch
        all_epoch_metrics.append(current_epoch_data)

        # --- Save Metrics Periodically ---
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            try:
                 metrics_df_partial = pd.DataFrame(all_epoch_metrics)
                 excel_path_partial = results_dir / "training_metrics_partial.xlsx"
                 metrics_df_partial.to_excel(excel_path_partial, index=False, engine='openpyxl')
            except ImportError:
                 logger.warning("Module 'openpyxl' not installed. Cannot save partial metrics to Excel. Install with 'pip install openpyxl'.")
            except Exception as e:
                 logger.warning(f"Failed to save partial metrics to Excel: {e}", exc_info=True)
        # --- ------------------------- ---

        epoch_total_duration = time.time() - epoch_start_time
        logger.info(f"--- Epoch {epoch+1} Finished (Total Duration: {epoch_total_duration:.2f}s) ---")

    # --- End Epoch Loop ---

    total_training_time = time.time() - training_start_time
    logger.info(f"\n--- Training Finished ({total_training_time:.2f}s total) ---")

    # --- Saving Final Results ---
    metrics_df = pd.DataFrame(all_epoch_metrics)
    excel_path = results_dir / "training_metrics.xlsx"
    csv_path = results_dir / "training_metrics.csv"
    logger.info("\nSaving final metrics...")
    try:
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info(f"Saved final epoch metrics to Excel: {excel_path}")
    except ImportError:
        logger.warning("Module 'openpyxl' not found. Saving metrics to CSV instead.")
        try: metrics_df.to_csv(csv_path, index=False); logger.info(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv: logger.error(f"Failed to save metrics to CSV: {e_csv}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to save metrics to Excel: {e}. Attempting CSV save.", exc_info=True)
        try: metrics_df.to_csv(csv_path, index=False); logger.info(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv: logger.error(f"Failed to save metrics to CSV: {e_csv}", exc_info=True)

    # Save final model state
    final_model_path = results_dir / "model_final.pt"
    try:
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model state to {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}", exc_info=True)

    # --- Plotting ---
    logger.info("\nGenerating training plots...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True) # Share x-axis (epochs)
        fig.suptitle(f"Training Metrics: {exp_name_cleaned}", fontsize=14)

        epochs_axis = metrics_df["Epoch"]

        # Plot Training Loss
        ax = axes[0]
        ax.plot(epochs_axis, metrics_df["Average Training Loss"], marker='.', linestyle='-', color='tab:blue')
        ax.set_title("Average Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=':')

        # Plot Evaluation Metrics (only if eval was run)
        eval_ran = metrics_df["Evaluation Episode Success Rate"].notna().any()
        eval_df = metrics_df.dropna(subset=["Evaluation Episode Success Rate"]) if eval_ran else pd.DataFrame()

        # Plot Success Rate
        ax = axes[1]
        if eval_ran and not eval_df.empty:
            ax.plot(eval_df["Epoch"], eval_df["Evaluation Episode Success Rate"], marker='o', linestyle='-', color='tab:green')
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
        ax.set_title("Evaluation Success Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Success Rate")
        ax.grid(True, linestyle=':')

        # Plot Avg Steps for Successful Runs
        ax = axes[2]
        if eval_ran and not eval_df.empty:
            valid_steps_df = eval_df.dropna(subset=["Evaluation Avg Steps (Success)"])
            if not valid_steps_df.empty:
                ax.plot(valid_steps_df["Epoch"], valid_steps_df["Evaluation Avg Steps (Success)"], marker='s', linestyle='-', color='tab:red')
                ax.set_ylabel("Average Steps")
            else:
                ax.text(0.5, 0.5, 'No successful\neval runs', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
                ax.set_ylabel("Average Steps (N/A)")
        else:
            ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
            ax.set_ylabel("Average Steps (N/A)")
        ax.set_title("Avg Steps (Successful Eval Episodes)")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=':')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plot_path = results_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved plots to: {plot_path}")
        plt.close(fig) # Close the plot figure
    except Exception as e:
        logger.warning(f"Failed generate plots: {e}", exc_info=True)
    # --- --------- ---

    logger.info("--- Script Finished ---")
