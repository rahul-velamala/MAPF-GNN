# File: train.py
# (Modified Version with DAgger/Online Expert, Collision Shielding in Eval/OE, Fixes)

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

# --- Add project root to sys.path if needed (adjust based on execution context) ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
# print(f"PROJECT_ROOT: {PROJECT_ROOT}") # Debug print
# print(f"sys.path: {sys.path}") # Debug print
# --- ------------------------------------------------------------------------- ---

# --- Assuming these imports work when running from project root ---
try:
    print("Importing environment and data modules...")
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    from data_loader import GNNDataLoader, CreateDataset # Need CreateDataset info potentially
    from data_generation.record import make_env # Helper to create env from case path
    print("Importing CBS modules...")
    # --- Import CBS for Online Expert ---
    from cbs.cbs import Environment as CBSEnvironment # Rename to avoid clash
    from cbs.cbs import CBS
    from data_generation.trayectory_parser import parse_trayectories as parse_traject_cbs # Use specific name
    print("Imports successful.")

    # --- Local Timeout Handling for CBS ---
    class TimeoutError(Exception): pass
    def handle_timeout(signum, frame): raise TimeoutError("CBS search timed out")
    # --- ------------------------------- ---
except ImportError as e:
    print(f"\nFATAL ERROR importing project modules: {e}")
    print("Please ensure:")
    print("  1. You are running python from the main project directory (e.g., 'rahul-velamala-mapf-gnn').")
    print("  2. All necessary __init__.py files exist in subdirectories.")
    print("  3. The required libraries (torch, numpy, yaml, etc.) are installed.")
    # traceback.print_exc() # Uncomment for more details
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
args = parser.parse_args()
# ========================

# --- Load Configuration ---
config_file_path = args.config
print(f"\nLoading configuration from: {config_file_path}")
try:
    with open(config_file_path, "r") as config_path:
        # Use safe_load for security
        config = yaml.safe_load(config_path)
        if config is None: # Handle empty config file
             raise ValueError("Config file is empty or invalid.")
except Exception as e:
    print(f"ERROR: Could not load or parse config file '{config_file_path}': {e}")
    sys.exit(1)
# --- ------------------ ---

# --- Setup based on Config ---
try:
    net_type = config.get("net_type", "gnn")
    exp_name = config.get("exp_name", "default_experiment")
    epochs = int(config.get("epochs", 50))
    # Max steps for evaluation episodes run during training
    max_steps_eval = int(config.get("max_steps", 60))
    # Longer timeout for inference runs during Online Expert (to detect actual deadlocks)
    max_steps_train_inference = int(config.get("max_steps_train_inference", max_steps_eval * 3))
    eval_frequency = int(config.get("eval_frequency", 5))
    tests_episodes_eval = int(config.get("tests_episodes", 25)) # Num eval episodes per evaluation phase
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    num_agents_config = int(config.get("num_agents", 5)) # Ensure integer
    # Assume pad is correctly set in config (used by model init and env)
    pad_config = int(config.get("pad", 3))

    # --- Online Expert (OE) Config ---
    use_online_expert = not args.oe_disable
    oe_config = config.get("online_expert", {}) # Get OE sub-dict, default to empty dict
    oe_frequency = int(oe_config.get("frequency_epochs", 4)) # How often to run OE
    oe_num_cases_to_run = int(oe_config.get("num_cases_to_run", 500)) # Cases to check per OE run
    oe_cbs_timeout = int(oe_config.get("cbs_timeout_seconds", 10)) # Timeout for expert solver

except (ValueError, TypeError) as e:
     print(f"ERROR: Invalid numerical value in configuration: {e}")
     sys.exit(1)
except KeyError as e:
     print(f"ERROR: Missing required key in configuration: {e}")
     sys.exit(1)

# --- Device Setup ---
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --- ----------- ---

# --- Results Directory ---
# Replace potential Windows backslashes for consistency
exp_name_cleaned = exp_name.replace('\\', '/')
# Now Path() can be used:
results_dir = Path("results") / exp_name_cleaned # <<<--- This line caused the error
try:
    results_dir.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"ERROR: Could not create results directory {results_dir}: {e}")
    sys.exit(1)
# --- ----------------- ---

# --- Model Selection ---
NetworkClass = None
try:
    if net_type == "baseline":
        from models.framework_baseline import Network as NetworkClass
        print("Using Model: Baseline Network")
    elif net_type == "gnn":
        msg_type = config.get("msg_type", "gcn").lower()
        if msg_type == 'message':
             from models.framework_gnn_message import Network as NetworkClass
             print("Using Model: GNN Message Passing Network")
        elif msg_type == 'gcn':
             from models.framework_gnn import Network as NetworkClass
             print("Using Model: GNN (GCN) Network")
        else:
             raise ValueError(f"Unsupported 'msg_type' in config: {msg_type}")
    else:
        raise ValueError(f"Unknown 'net_type' in config: '{net_type}'")
    if NetworkClass is None: # Should not happen if logic above is correct
        raise ImportError("NetworkClass was not assigned.")
except (ImportError, ValueError, KeyError) as e:
     print(f"ERROR: Failed to import or select model based on config ('net_type': {net_type}, 'msg_type': {config.get('msg_type')}): {e}")
     sys.exit(1)
# --- --------------- ---

# --- Save Effective Config ---
config_save_path = results_dir / "config_used.yaml"
try:
    config_to_save = config.copy() # Save a copy
    # Convert device object to string for saving
    if 'device' in config_to_save and not isinstance(config_to_save['device'], str):
        config_to_save['device'] = str(config_to_save["device"])
    # Optionally convert Path objects to strings if needed for YAML
    # config_to_save['train']['root_dir'] = str(config_to_save['train']['root_dir'])
    # config_to_save['valid']['root_dir'] = str(config_to_save['valid']['root_dir'])

    with open(config_save_path, "w") as config_path_out:
        yaml.dump(config_to_save, config_path_out, default_flow_style=False, sort_keys=False)
    print(f"Saved effective config to {config_save_path}")
except Exception as e:
    print(f"ERROR: Could not save config to '{config_save_path}': {e}")
    # Continue training, but config is not saved
# --- ----------------------- ---


# === Helper Function: Run Inference with Collision Shielding ===
def run_inference_with_shielding(   model, env, max_steps_inference, device, net_type):
    """
    Runs the current model on the environment with collision shielding applied.
    Detects success, timeout (truncation), or deadlock.

    Args:
        model: The trained model (in eval mode).
        env: An initialized GraphEnv instance.
        max_steps_inference (int): Maximum number of steps for this episode.
        device: Torch device ('cuda' or 'cpu').
        net_type (str): 'gnn' or 'baseline'.

    Returns:
        tuple: (history, is_success, is_timeout)
            history (dict): {'states': list_of_fov_np, 'gsos': list_of_gso_np,
                              'model_actions': list_of_np, 'shielded_actions': list_of_np}
                            States/GSOs are recorded *before* the corresponding action.
            is_success (bool): True if env reached terminated state.
            is_timeout (bool): True if env reached max_steps_inference.
                                (is_deadlock is implicitly not is_success and is_timeout)
    """
    model.eval() # Ensure model is in evaluation mode
    try:
        obs, info = env.reset() # Reset env for this run
    except Exception as e:
         print(f"\nError resetting environment during inference run: {e}")
         return {}, False, True # Indicate failure/timeout

    terminated = False
    truncated = False
    history = {'states': [], 'gsos': [], 'model_actions': [], 'shielded_actions': []}
    idle_action = 0 # Assuming 0 is the idle action index in GraphEnv

    step_count = 0
    while not terminated and not truncated and step_count < max_steps_inference:
        # Store current state (FOV, GSO) before taking action
        try:
            current_fov_np = obs["fov"] # Shape (N, C, H, W)
            current_gso_np = obs["adj_matrix"] # Shape (N, N)
            history['states'].append(current_fov_np.copy())
            history['gsos'].append(current_gso_np.copy())
        except KeyError as e:
             print(f"Error: Missing key {e} in observation dict at step {env.time}.")
             return history, False, True # Treat as failure/timeout

        # Prepare observation for model
        try:
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device) # Add batch dim B=1
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device) # Add batch dim B=1
        except Exception as e:
             print(f"Error converting observation to tensor: {e}")
             return history, False, True

        # Get action from model
        with torch.no_grad():
            try:
                if net_type == 'gnn':
                    action_scores = model(fov_tensor, gso_tensor) # Expect (1, N, A)
                else: # baseline
                    action_scores = model(fov_tensor)
                # Get actions with highest score for each agent
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy() # Shape (N,)
            except Exception as e:
                 print(f"Error during model forward pass: {e}")
                 return history, False, True

        history['model_actions'].append(proposed_actions.copy())

        # --- Apply Collision Shielding (Sec V.G / Alg 1 style) ---
        shielded_actions = proposed_actions.copy()
        current_pos_y = env.positionY.copy() # Shape (N,)
        current_pos_x = env.positionX.copy() # Shape (N,)
        next_pos_y = current_pos_y.copy()
        next_pos_x = current_pos_x.copy()
        needs_shielding = np.zeros(env.nb_agents, dtype=bool) # Tracks agents whose actions were changed
        active_mask = ~env.reached_goal # Agents not yet at goal

        # 1. Calculate proposed next positions for active agents
        for agent_id in np.where(active_mask)[0]:
             act = proposed_actions[agent_id]
             # Ensure action is valid, default to Idle (0) if not
             dy, dx = env.action_map_dy_dx.get(act, (0,0))
             next_pos_y[agent_id] += dy
             next_pos_x[agent_id] += dx

        # 2. Clamp proposed positions to boundaries (only active agents)
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
        next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)

        # 3. Check Obstacle Collisions (only active agents)
        if env.obstacles.size > 0:
            active_indices = np.where(active_mask)[0]
            # Consider only agents still active after potential boundary clip
            if len(active_indices) > 0:
                 proposed_coords_active = np.stack([next_pos_y[active_indices], next_pos_x[active_indices]], axis=1)
                 # Efficient check using broadcasting
                 obs_coll_active_mask = np.any(np.all(proposed_coords_active[:, np.newaxis, :] == env.obstacles[np.newaxis, :, :], axis=2), axis=1)
                 # Map back to original agent indices
                 colliding_agent_indices = active_indices[obs_coll_active_mask]

                 if colliding_agent_indices.size > 0:
                      shielded_actions[colliding_agent_indices] = idle_action # Shield: set to idle
                      needs_shielding[colliding_agent_indices] = True
                      # Revert positions for these agents for agent-agent check
                      next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
                      next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]
                      active_mask[colliding_agent_indices] = False # Mark as inactive for agent-agent check

        # 4. Check Agent-Agent Collisions (Vertex & Swapping)
        # Consider only agents that are still active (not at goal, didn't hit obstacle)
        active_indices = np.where(active_mask)[0]
        agents_to_shield_idx = np.array([], dtype=int) # Initialize empty

        if len(active_indices) > 1: # Need at least two agents to collide
            # Get coordinates relative to this check
            next_coords_check = np.stack([next_pos_y[active_indices], next_pos_x[active_indices]], axis=1)
            current_coords_check = np.stack([current_pos_y[active_indices], current_pos_x[active_indices]], axis=1)

            # Vertex collisions
            unique_coords, unique_map_indices, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
            colliding_cell_indices = np.where(counts > 1)[0]
            vertex_collision_mask_rel = np.isin(unique_map_indices, colliding_cell_indices)
            # Ensure dtype=int
            vertex_collision_agents = active_indices[vertex_collision_mask_rel].astype(int)
            
            # Edge collisions (swapping)
            swapping_collision_agents_list = []
            relative_indices = np.arange(len(active_indices))
            for i_rel in relative_indices:
                 for j_rel in range(i_rel + 1, len(active_indices)):
                     if np.array_equal(next_coords_check[i_rel], current_coords_check[j_rel]) and \
                        np.array_equal(next_coords_check[j_rel], current_coords_check[i_rel]):
                         swapping_collision_agents_list.extend([active_indices[i_rel], active_indices[j_rel]])
            # Ensure dtype=int
            swapping_collision_agents = np.unique(swapping_collision_agents_list).astype(int)

            # Combine collision indices, ensuring integer type
            # !!! FIX: Ensure concatenation results in integer array !!!
            if vertex_collision_agents.size > 0 or swapping_collision_agents.size > 0:
                 agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents])).astype(int)
            # else: agents_to_shield_idx remains empty integer array

        # Apply shielding for agent-agent collisions
        # Check size BEFORE indexing
        if agents_to_shield_idx.size > 0:
            shielded_actions[agents_to_shield_idx] = idle_action # Indexing now safe
            needs_shielding[agents_to_shield_idx] = True
        # --- End Collision Shielding ---

        history['shielded_actions'].append(shielded_actions.copy())

        # Step the environment *with shielded actions*
        try:
            obs, reward, terminated, truncated, info = env.step(shielded_actions)
            # Check explicit timeout based on max_steps_inference
            truncated = truncated or (env.time >= max_steps_inference)
            step_count = env.time # Update step count based on env time
        except Exception as e:
             print(f"\nError during env.step in inference run (Time: {env.time}): {e}")
             # traceback.print_exc() # Uncomment for debugging
             return history, False, True # Treat as failure/timeout

    # --- After loop finishes ---
    is_success = terminated and not truncated # Success only if terminated *before* truncation/timeout
    is_timeout = truncated or (step_count >= max_steps_inference) # Timeout if truncated or reached step limit

    return history, is_success, is_timeout


# === Helper Function: Call CBS Expert ===
def call_expert_from_state(env_state_info, cbs_timeout_s):
    """
    Creates a CBS problem from the current GraphEnv state and runs the expert.
    Args:
        env_state_info (dict): Contains current 'positions' [N,2](row,col),
                               'goals' [N,2](row,col), 'obstacles' [M,2](row,col),
                               'board_dims' [rows,cols].
        cbs_timeout_s (int): Timeout for the CBS search.
    Returns:
        Dict (expert solution {agent_name: [{'t':..,'x':..,'y':..},..]}) or None.
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
             agents_data.append({
                 "start": start_xy,
                 "goal": goal_xy,
                 "name": f"agent{i}"
             })

        # Convert GraphEnv [rows, cols] to CBS [width=cols, height=rows]
        board_dims_wh = [int(env_state_info['board_dims'][1]), int(env_state_info['board_dims'][0])]

        # Convert GraphEnv obstacles [row, col] to CBS obstacles [x=col, y=row]
        obstacles_xy = [[int(obs[1]), int(obs[0])] for obs in env_state_info['obstacles']] if env_state_info['obstacles'].size > 0 else []

        map_data = {
            "dimensions": board_dims_wh,
            "obstacles": obstacles_xy
        }
        # --- ----------------------------------------- ---

        # --- Initialize CBS ---
        cbs_env = CBSEnvironment(map_data["dimensions"], agents_data, map_data["obstacles"])
        cbs_solver = CBS(cbs_env, verbose=False) # Keep verbose false

    except Exception as e:
        print(f"OE Expert Error: Failed to initialize CBS environment/solver: {e}")
        return None

    # --- Run CBS with Timeout ---
    solution = None
    can_use_alarm = hasattr(signal, 'SIGALRM')
    original_handler = None

    if can_use_alarm:
        original_handler = signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(cbs_timeout_s)
    else: # Cannot use alarm (e.g., Windows)
        # Consider adding a time check within CBS search loop if possible, or accept no timeout
        if cbs_timeout_s < 600: # Only warn if timeout is reasonably short
            print("OE Expert Warning: Signal alarms not available on this OS. CBS timeout may not be enforced.")

    try:
        solution = cbs_solver.search()
        if can_use_alarm: signal.alarm(0) # Disable alarm if search finished

        if not solution:
             # print("OE Expert Info: CBS found no solution for the given state.") # Less verbose
             return None # No solution is a valid outcome for CBS

    except TimeoutError:
        # print(f"OE Expert Warning: CBS expert timed out after {cbs_timeout_s}s.") # Less verbose
        return None # Timeout occurred
    except Exception as e:
        print(f"OE Expert Error: CBS search failed unexpectedly: {e}")
        # traceback.print_exc() # Uncomment for debugging CBS errors
        return None # Other error during search
    finally:
        # Restore original signal handler only if it was set
        if can_use_alarm and original_handler is not None:
            signal.signal(signal.SIGALRM, original_handler)
            signal.alarm(0) # Ensure alarm is off

    # --- Return CBS solution dictionary ---
    return solution


# === Helper Function: Aggregate Expert Data via Simulation ===
def aggregate_expert_data(expert_solution_cbs, start_env_instance, config):
    """
    Simulates the expert path from the deadlock start state in a *copy* of the
    environment and records the sequence of (FOV, GSO, Expert_Action).

    Args:
        expert_solution_cbs (dict): Raw output from call_expert_from_state (CBS format).
        start_env_instance (GraphEnv): Environment instance *at the deadlock state*.
        config (dict): Main configuration (needed for GraphEnv parameters like pad).

    Returns:
        Tuple: (numpy_states, numpy_gsos, numpy_expert_actions) or None if failed.
               Shapes: [T_agg, N, C, H, W], [T_agg, N, N], [T_agg, N]
               where T_agg is the number of steps in the expert path simulation.
    """
    if not expert_solution_cbs: return None

    # 1. Parse expert CBS solution into GraphEnv actions array [N, T_expert]
    try:
        # parse_traject_cbs expects CBS schedule format, returns GraphEnv actions
        expert_actions_np, _ = parse_traject_cbs(expert_solution_cbs) # Shape (N, T_expert)
        num_expert_actions = expert_actions_np.shape[1]
        if num_expert_actions == 0:
             # print("OE Aggregate Info: Expert path from CBS has 0 actions.")
             return None
    except Exception as e:
        print(f"OE Aggregate Error: Failed to parse expert CBS solution: {e}"); return None

    # 2. Simulate expert actions in a DEEP COPY of the environment from the deadlock state
    try:
        # Create a completely independent copy of the environment at the deadlock state
        env_sim = copy.deepcopy(start_env_instance)
        # CRITICAL FIX: Reset the time counter for the simulation copy
        env_sim.time = 0
        # Ensure simulation env uses appropriate max_time if needed (can be longer than original episode)
        # env_sim.max_time = num_expert_actions + 10 # Or some other reasonable limit

        # print(f"[OE Agg] Simulating expert path (len {num_expert_actions}) in copied Env (max_time={env_sim.max_time})...") # Debug
    except Exception as e:
         print(f"OE Aggregate Error: Failed to deepcopy environment: {e}"); return None

    # Lists to store data from simulation
    aggregated_states = [] # Holds FOV arrays (N, C, H, W)
    aggregated_gsos = []   # Holds GSO arrays (N, N)
    aggregated_actions = []# Holds action arrays (N,)

    try:
        # Get the state *at the deadlock* (time t=0 for this simulation)
        # Need to get observation from the copied env
        current_obs_sim = env_sim.getObservations() # Use the copied env
        aggregated_states.append(current_obs_sim['fov'])
        aggregated_gsos.append(current_obs_sim['adj_matrix'])

        # Simulate each expert action step
        terminated_sim = False
        truncated_sim = False
        for t in range(num_expert_actions):
            actions_t = expert_actions_np[:, t] # Expert actions for this step
            aggregated_actions.append(actions_t) # Store action t

            # Step the simulation environment
            current_obs_sim, _, terminated_sim, truncated_sim, _ = env_sim.step(actions_t)

            # Store the state resulting from action t (state at sim time t+1)
            aggregated_states.append(current_obs_sim['fov'])
            aggregated_gsos.append(current_obs_sim['adj_matrix'])

            # Check if simulation ended prematurely (expert reached goal or env truncated)
            if terminated_sim:
                # print(f"[OE Agg] Expert path reached goal at step {t+1}/{num_expert_actions}.") # Info
                break # Stop simulation
            if truncated_sim:
                 # This might happen if expert path is longer than env_sim.max_time
                 print(f"OE Aggregate Warning: Simulation env truncated during expert path sim at step {t+1}/{num_expert_actions} (Env Time: {env_sim.time}). Max time {env_sim.max_time}.")
                 break # Stop simulation, aggregate partial path

        # --- Post-Simulation Processing ---
        num_sim_steps = len(aggregated_actions) # Number of actions successfully simulated
        if num_sim_steps == 0:
             # print("OE Aggregate Info: No actions were simulated (e.g., expert path empty or immediate termination).")
             return None

        # We need pairs of (state_t, action_t, gso_t).
        # We have T_sim actions and T_sim+1 states/gsos.
        # Use states[0...T_sim-1] and gsos[0...T_sim-1] with actions[0...T_sim-1].
        final_states = np.stack(aggregated_states[:num_sim_steps]) # Shape [T_sim, N, C, H, W]
        final_gsos = np.stack(aggregated_gsos[:num_sim_steps])     # Shape [T_sim, N, N]
        final_actions = np.stack(aggregated_actions)           # Shape [T_sim, N]

        # Final check for consistent lengths
        if not (final_states.shape[0] == final_gsos.shape[0] == final_actions.shape[0] == num_sim_steps):
             print(f"OE Aggregate Error: Mismatch shapes after stacking. SimSteps:{num_sim_steps}, S:{final_states.shape}, G:{final_gsos.shape}, A:{final_actions.shape}"); return None

        # print(f"[OE Agg] Successfully aggregated {num_sim_steps} state-action pairs.") # Debug
        return final_states, final_gsos, final_actions

    except Exception as e:
         print(f"OE Aggregate Error: Failed during expert path simulation or stacking: {e}");
         # traceback.print_exc() # Uncomment for debug
         return None
    finally:
         if 'env_sim' in locals() and env_sim is not None:
              env_sim.close() # Close the rendering window of the copied env if opened

# === Custom Dataset for Aggregated Expert Data ===
class AggregatedDataset(Dataset):
    """A Dataset to hold aggregated expert data collected during training."""
    def __init__(self, aggregated_list_of_dicts):
        """
        Args:
            aggregated_list_of_dicts (list): A list where each element is a dictionary
                                             {'states': np.array, 'gsos': np.array, 'actions': np.array}
                                             from one expert rollout.
        """
        if not aggregated_list_of_dicts:
             print("AggregatedDataset: Initialized with empty data.")
             # Initialize with empty arrays of expected dtype and ndim, 0 samples
             self.all_states = np.empty((0, num_agents_config, 3, pad_config*2-1, pad_config*2-1), dtype=np.float32)
             self.all_gsos = np.empty((0, num_agents_config, num_agents_config), dtype=np.float32)
             self.all_actions = np.empty((0, num_agents_config), dtype=np.int64) # Actions should be Long
        else:
             # Concatenate data from all dictionaries in the list
             try:
                 self.all_states = np.concatenate([d['states'] for d in aggregated_list_of_dicts if d['states'].size > 0], axis=0)
                 self.all_gsos = np.concatenate([d['gsos'] for d in aggregated_list_of_dicts if d['gsos'].size > 0], axis=0)
                 self.all_actions = np.concatenate([d['actions'] for d in aggregated_list_of_dicts if d['actions'].size > 0], axis=0)

                 # Ensure actions are int64 for LongTensor conversion
                 if self.all_actions.dtype != np.int64:
                      self.all_actions = self.all_actions.astype(np.int64)

                 print(f"AggregatedDataset: Combined data shapes: S={self.all_states.shape}, G={self.all_gsos.shape}, A={self.all_actions.shape}")
             except ValueError as e: # Handle potential concat errors (e.g., shape mismatch)
                  print(f"ERROR in AggregatedDataset: Could not concatenate data: {e}")
                  # Initialize empty to prevent crashes
                  self.__init__([]) # Re-initialize empty

        self.count = len(self.all_states)
        if self.count > 0 and (self.count != len(self.all_gsos) or self.count != len(self.all_actions)):
             print(f"ERROR in AggregatedDataset: Mismatch in sample count after concatenation! S:{len(self.all_states)}, G:{len(self.all_gsos)}, A:{len(self.all_actions)}")
             # Handle inconsistency, e.g., re-initialize empty or take the minimum count
             min_count = min(len(self.all_states), len(self.all_gsos), len(self.all_actions))
             self.all_states = self.all_states[:min_count]
             self.all_gsos = self.all_gsos[:min_count]
             self.all_actions = self.all_actions[:min_count]
             self.count = min_count
             print(f"Corrected count to {self.count}")


    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(f"Index {index} out of bounds for AggregatedDataset with size {self.count}")

        # Extract data and convert to tensors
        state = torch.from_numpy(self.all_states[index]).float()
        gso = torch.from_numpy(self.all_gsos[index]).float()
        action = torch.from_numpy(self.all_actions[index]).long() # Actions must be Long

        # Return order must match DataLoader expectation in training loop: state, action, gso
        return state, action, gso

# === Main Training Script ===
if __name__ == "__main__":

    print("\n----- Effective Configuration -----")
    # Use pprint for better dict formatting
    pprint(config, indent=2)
    print(f"Using device: {config['device']}")
    print(f"Online Expert (DAgger): {'Enabled' if use_online_expert else 'Disabled'}")
    if use_online_expert:
         print(f"  OE Frequency: Every {oe_frequency} epochs")
         print(f"  OE Cases to run: {oe_num_cases_to_run}")
         print(f"  OE CBS Timeout: {oe_cbs_timeout}s")
         print(f"  OE Max Inference Steps: {max_steps_train_inference}")
    print("---------------------------------\n")

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
             # Error should have been raised by GNNDataLoader if dataset was empty
             print("ERROR: Training data loader is unexpectedly empty or invalid after initialization.")
             sys.exit(1)

        # Keep a reference to the original dataset for OE DAgger
        original_train_dataset = train_loader.dataset
        print(f"Initial training samples (timesteps): {len(original_train_dataset)}")
        if valid_loader: print(f"Validation samples (timesteps): {len(valid_loader.dataset)}")

    except Exception as e:
        print(f"ERROR: Failed to initialize/load data: {e}");
        traceback.print_exc(); sys.exit(1)
    # --- ------------ ---

    # --- Model, Optimizer, Criterion ---
    model = None
    optimizer = None
    criterion = None
    try:
        # Instantiate the selected NetworkClass
        model = NetworkClass(config)
        model.to(config["device"])
        print(f"\nModel '{type(model).__name__}' initialized on {config['device']}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss() # Standard loss for classification (predicting action)

    except Exception as e:
        print(f"ERROR: Failed init model/optimizer/criterion: {e}");
        traceback.print_exc(); sys.exit(1)
    # --- --------------------------- ---

    # --- Training Loop Setup ---
    all_epoch_metrics = [] # List to store metrics dict per epoch
    best_eval_success_rate = -1.0 # Track best validation success rate
    aggregated_expert_data_list = [] # Holds dicts {'states':S,'gsos':G,'actions':A} from *all* OE runs
    current_train_dataset = original_train_dataset # Start with the original data

    # --- Sanity Check: Ensure original dataset has required attributes for OE ---
    if use_online_expert:
         if not hasattr(original_train_dataset, 'cases') or not hasattr(original_train_dataset, 'root_dir'):
             print("ERROR: Original training dataset object (required for OE) is missing 'cases' or 'root_dir' attribute.")
             print("Online Expert cannot function without access to the original case paths.")
             use_online_expert = False # Disable OE if dataset structure is wrong
             print("Disabled Online Expert.")

    print(f"\n--- Starting Training for {epochs} epochs ---")
    training_start_time = time.time()

    # --- Main Epoch Loop ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*10} Epoch {epoch+1}/{epochs} {'='*10}")

        # === Online Expert (DAgger) Data Aggregation Phase ===
        # Run OE at specified frequency (e.g., epoch 4, 8, 12...)
        run_oe_this_epoch = use_online_expert and ((epoch + 1) % oe_frequency == 0)

        if run_oe_this_epoch:
            print(f"--- Running Online Expert Data Aggregation (Epoch {epoch+1}) ---")
            oe_start_time = time.time()
            num_deadlocks_found = 0
            num_expert_calls = 0
            num_expert_success = 0
            newly_aggregated_samples_count = 0
            new_data_this_epoch = [] # Collect new data only for this specific OE run

            # Get list of case names and root directory from the original dataset
            # Assumes original_train_dataset is an instance of CreateDataset
            original_case_names = getattr(original_train_dataset, 'cases', [])
            original_root_dir = getattr(original_train_dataset, 'root_dir', None)

            if not original_case_names or not original_root_dir:
                print("OE Error: Cannot run OE because original case list or root dir is unavailable.")
            else:
                num_original_cases = len(original_case_names)
                # Sample indices from the list of original cases
                indices_to_run = random.sample(range(num_original_cases), min(oe_num_cases_to_run, num_original_cases))
                print(f"Selected {len(indices_to_run)} cases for OE inference run.")

                oe_pbar = tqdm(indices_to_run, desc="OE Inference", unit="case", leave=False)
                for case_idx_in_orig_list in oe_pbar:
                    case_name = original_case_names[case_idx_in_orig_list]
                    case_path = original_root_dir / case_name # Use pathlib concatenation
                    env_oe = None # Ensure env_oe is defined for finally block

                    try:
                        # Create environment for this case using helper function
                        env_oe = make_env(case_path, config)
                        if env_oe is None:
                            # print(f"OE Warning: Could not create env for case {case_name}. Skipping.") # Less verbose
                            continue

                        # Run current model with shielding to find deadlocks
                        history, is_success, is_timeout = run_inference_with_shielding(
                            model, env_oe, max_steps_train_inference, config["device"], net_type
                        )

                        # Deadlock = timed out AND not successful
                        is_deadlock = is_timeout and not is_success

                        if is_deadlock:
                            num_deadlocks_found += 1
                            # The 'env_oe' instance is now AT the deadlock state
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
                                # If expert found a path, simulate it to get (s,a,g) pairs
                                aggregated_data = aggregate_expert_data(expert_solution_dict, env_oe, config)

                                if aggregated_data:
                                    num_expert_success += 1
                                    states_agg, gsos_agg, actions_agg = aggregated_data
                                    # Append dict of numpy arrays for temporary storage this epoch
                                    new_data_this_epoch.append({
                                        "states": states_agg, "gsos": gsos_agg, "actions": actions_agg
                                    })
                                    newly_aggregated_samples_count += len(states_agg) # Count samples (timesteps)
                                    oe_pbar.set_postfix({"Deadlocks": num_deadlocks_found, "ExpertOK": num_expert_success, "NewSamples": newly_aggregated_samples_count})
                            # else: Expert failed or timed out - no new data for this deadlock

                    except Exception as e:
                         print(f"\nOE Error: Unhandled exception processing case {case_name}: {e}");
                         traceback.print_exc()
                    finally:
                         if env_oe: env_oe.close() # Close env figure if opened

            # --- Update main aggregated list and potentially recreate DataLoader ---
            if newly_aggregated_samples_count > 0:
                print(f"\nOE Aggregation Summary: Found {num_deadlocks_found} deadlocks. Expert succeeded {num_expert_success}/{num_expert_calls} times.")
                print(f"Aggregated {newly_aggregated_samples_count} new state-action pairs this epoch.")

                # Add data collected this epoch to the main list tracking all aggregated data
                aggregated_expert_data_list.extend(new_data_this_epoch)
                total_aggregated_samples = sum(len(d['states']) for d in aggregated_expert_data_list)
                print(f"Total aggregated expert samples accumulated: {total_aggregated_samples}")

                # Create a Dataset from ALL aggregated data collected so far
                aggregated_dataset_all = AggregatedDataset(aggregated_expert_data_list)

                # Combine original dataset and ALL aggregated data for training
                current_train_dataset = ConcatDataset([original_train_dataset, aggregated_dataset_all])
                print(f"Combined dataset size for training this epoch: {len(current_train_dataset)} samples.")

                # Create a NEW DataLoader with the combined dataset for the training phase below
                try:
                    train_loader = DataLoader(
                        current_train_dataset,
                        batch_size=config["batch_size"], shuffle=True, # Shuffle combined data
                        num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
                    )
                    print("Created new DataLoader with combined original + aggregated data.")
                except Exception as e:
                    print(f"ERROR creating combined DataLoader: {e}. Falling back to previous DataLoader.")
                    # Fallback: reuse the previous train_loader (might miss latest data)
                    if train_loader is None: # Safety check if first OE run fails here
                         train_loader = data_loader_manager.train_loader # Very first loader
            else:
                 print("OE Aggregation: No new samples aggregated this epoch.")
                 # If no new data, continue using the existing train_loader (which might be combined from previous epochs)
                 # Ensure train_loader exists
                 if train_loader is None:
                      print("ERROR: train_loader is None after OE phase with no new data. Cannot train.")
                      sys.exit(1)
                 current_train_dataset = train_loader.dataset # Update reference
                 print(f"Continuing training with existing dataset size: {len(current_train_dataset)}")


            oe_duration = time.time() - oe_start_time
            print(f"--- Online Expert Data Aggregation Finished ({oe_duration:.2f}s) ---")
        # === End Online Expert Block ===


        # ##### Training Phase #########
        model.train() # Set model to training mode
        epoch_train_loss = 0.0
        batches_processed = 0
        if train_loader is None:
            print("FATAL ERROR: train_loader is None before training phase. Exiting.")
            sys.exit(1)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

        for i, batch_data in enumerate(train_pbar):
            # Expected return order from DataLoader: state, action, gso
            try:
                states_batch = batch_data[0].to(config["device"], non_blocking=True) # (B, N, C, H, W)
                target_actions_batch = batch_data[1].to(config["device"], non_blocking=True) # (B, N), LongTensor
                gso_batch = batch_data[2].to(config["device"], non_blocking=True) # (B, N, N)
            except IndexError:
                 print(f"\nError: Incorrect data format received from DataLoader at batch {i}. Expected 3 items.")
                 continue # Skip this batch
            except Exception as e:
                 print(f"\nError unpacking/moving batch {i} to device: {e}")
                 continue # Skip this batch

            optimizer.zero_grad()

            # --- Forward Pass ---
            try:
                # Model expects (B, N, C, H, W) and (B, N, N)
                output_logits = model(states_batch, gso_batch) # Output: (B, N, A)
            except Exception as e:
                print(f"\nError during forward pass batch {i}: {e}")
                # Optionally print shapes for debugging:
                # print(f"State shape: {states_batch.shape}, GSO shape: {gso_batch.shape}")
                # traceback.print_exc() # Uncomment for full traceback
                continue # Skip batch

            # --- Loss Calculation ---
            try:
                # Reshape for CrossEntropyLoss: Output (B*N, A), Target (B*N)
                num_actions_model = output_logits.shape[-1]
                output_reshaped = output_logits.reshape(-1, num_actions_model) # (B*N, A)
                target_reshaped = target_actions_batch.reshape(-1) # (B*N)

                # Check if model output has the expected number of actions (5)
                if num_actions_model != 5:
                     print(f"\nShape Error: Model output has {num_actions_model} actions, expected 5. Batch {i}. Logits shape: {output_logits.shape}")
                     continue # Skip batch if action dimension is wrong

                # Ensure target actions are within the valid range [0, 4]
                if torch.any(target_reshaped < 0) or torch.any(target_reshaped >= 5):
                    print(f"\nValue Error: Invalid target action found in batch {i}. Values outside [0, 4]. Min: {target_reshaped.min()}, Max: {target_reshaped.max()}. Target shape: {target_actions_batch.shape}")
                    continue # Skip batch with invalid targets

                # Ensure shapes match for loss calculation
                if output_reshaped.shape[0] != target_reshaped.shape[0]:
                     print(f"\nShape Error: Mismatch for loss calculation batch {i}. Output samples: {output_reshaped.shape[0]}, Target samples: {target_reshaped.shape[0]}.")
                     continue

                batch_loss = criterion(output_reshaped, target_reshaped)

                # Check for NaN/Inf loss
                if not torch.isfinite(batch_loss):
                    print(f"\nWarning: Loss is NaN or Inf at batch {i}. Skipping backward pass.")
                    # Consider logging inputs/outputs that caused NaN loss if it persists
                    continue

            except IndexError as e:
                 # Catches potential errors if reshaping fails or target access is wrong
                 print(f"\nIndexError during loss calculation batch {i}: {e}.")
                 print(f"Output shape: {output_logits.shape}, Target shape: {target_actions_batch.shape}")
                 continue
            except Exception as e:
                 print(f"\nError during loss calculation batch {i}: {e}")
                 continue # Skip batch

            # --- Backward Pass & Optimizer Step ---
            try:
                batch_loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except Exception as e:
                print(f"\nError during backward pass or optimizer step batch {i}: {e}")
                # Continue training even if one batch fails optimization
                continue

            epoch_train_loss += batch_loss.item()
            batches_processed += 1
            if batches_processed > 0:
                train_pbar.set_postfix({"AvgLoss": epoch_train_loss / batches_processed})
        # --- End Training Batch Loop ---

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_train_loss / batches_processed if batches_processed > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f} | Duration: {epoch_duration:.2f}s")

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
            print(f"\n--- Running Evaluation after Epoch {epoch+1} ---")
            eval_start_time = time.time()
            model.eval() # Set model to evaluation mode
            eval_success_count = 0
            eval_steps_success = [] # Store steps only for successful episodes

            # --- Use evaluation specific parameters if defined in config ---
            eval_board_dims = config.get("eval_board_size", config.get("board_size", [16, 16]))
            eval_obstacles_count = config.get("eval_obstacles", config.get("obstacles", 6))
            eval_agents_count = config.get("eval_num_agents", config.get("num_agents", 4))
            # Ensure eval agent count matches model's num_agents if model is fixed size
            if eval_agents_count != model.num_agents:
                 print(f"Warning: Eval agent count ({eval_agents_count}) differs from model agent count ({model.num_agents}). Using model's count for eval env.")
                 eval_agents_count = model.num_agents
            eval_sensing_range = config.get("eval_sensing_range", config.get("sensing_range", 4))
            eval_pad = config.get("eval_pad", config.get("pad", 3)) # Use same pad as model trained on
            # Use max_steps_eval for evaluation episode length
            eval_max_steps_run = max_steps_eval # Renamed for clarity

            eval_pbar = tqdm(range(tests_episodes_eval), desc=f"Epoch {epoch+1} Evaluation", leave=False, unit="ep")
            for episode in eval_pbar:
                env_eval = None # Ensure env is defined for finally block
                try:
                    # Generate a new random scenario for each evaluation episode
                    eval_obstacles_ep = create_obstacles(eval_board_dims, eval_obstacles_count)
                    eval_start_pos_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep)
                    # Ensure goals avoid obstacles AND starts
                    eval_temp_occ = np.vstack([eval_obstacles_ep, eval_start_pos_ep]) if eval_obstacles_ep.size > 0 else eval_start_pos_ep
                    eval_goals_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep, current_starts=eval_start_pos_ep)

                    # Create eval env instance using eval parameters
                    eval_config_instance = config.copy() # Start with base config
                    eval_config_instance.update({ # Override with eval params
                         "board_size": eval_board_dims,
                         "num_agents": eval_agents_count,
                         "sensing_range": eval_sensing_range,
                         "pad": eval_pad,
                         "max_time": eval_max_steps_run # Use eval step limit
                    })

                    env_eval = GraphEnv(config=eval_config_instance,
                                        goal=eval_goals_ep, obstacles=eval_obstacles_ep,
                                        starting_positions=eval_start_pos_ep)

                    # Run inference *with shielding* to get performance
                    _, is_success, _ = run_inference_with_shielding(
                        model, env_eval, eval_max_steps_run, config["device"], net_type
                    )

                    if is_success:
                        eval_success_count += 1
                        eval_steps_success.append(env_eval.time) # Record steps taken for success

                    eval_pbar.set_postfix({"Success": f"{eval_success_count}/{episode+1}"})

                except Exception as e:
                    print(f"\nError during evaluation episode {episode+1}: {e}")
                    # traceback.print_exc() # Uncomment for debug
                    # Continue to next episode
                finally:
                     if env_eval: env_eval.close() # Close env figure if opened

            # --- End Eval Loop ---
            eval_success_rate = eval_success_count / tests_episodes_eval if tests_episodes_eval > 0 else 0.0
            avg_steps_succ = np.mean(eval_steps_success) if eval_steps_success else np.nan
            eval_duration = time.time() - eval_start_time

            print(f"Evaluation Complete: SR={eval_success_rate:.4f} ({eval_success_count}/{tests_episodes_eval}), AvgSteps(Succ)={avg_steps_succ:.2f} | Duration={eval_duration:.2f}s")

            # Update epoch metrics
            current_epoch_data.update({
                "Evaluation Episode Success Rate": eval_success_rate,
                "Evaluation Avg Steps (Success)": avg_steps_succ,
                "Evaluation Episodes Tested": tests_episodes_eval,
                "Evaluation Episodes Succeeded": eval_success_count,
            })

            # --- Save Best Model ---
            if eval_success_rate >= best_eval_success_rate:
                 print(f"*** New best evaluation success rate ({eval_success_rate:.4f}), saving model... ***")
                 best_eval_success_rate = eval_success_rate
                 best_model_path = results_dir / f"model_best.pt"
                 try:
                      torch.save(model.state_dict(), best_model_path)
                 except Exception as e:
                      print(f"Warning: Failed to save best model: {e}")
            # --- --------------- ---
        # --- End Evaluation Phase ---

        # Append metrics for this epoch
        all_epoch_metrics.append(current_epoch_data)

        # --- Save Metrics Periodically ---
        # Save partial metrics every N epochs and at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            try:
                 metrics_df_partial = pd.DataFrame(all_epoch_metrics)
                 excel_path_partial = results_dir / "training_metrics_partial.xlsx"
                 # Use openpyxl engine
                 metrics_df_partial.to_excel(excel_path_partial, index=False, engine='openpyxl')
            except ImportError:
                 print("Warning: 'openpyxl' not installed. Cannot save partial metrics to Excel.")
            except Exception as e:
                 print(f"Warning: Failed to save partial metrics to Excel: {e}")
        # --- ------------------------- ---

    # --- End Epoch Loop ---

    total_training_time = time.time() - training_start_time
    print(f"\n--- Training Finished ({total_training_time:.2f}s total) ---")

    # --- Saving Final Results ---
    metrics_df = pd.DataFrame(all_epoch_metrics)
    excel_path = results_dir / "training_metrics.xlsx"
    csv_path = results_dir / "training_metrics.csv"
    print("\nSaving final metrics...")
    try:
        # Ensure openpyxl is installed: pip install openpyxl
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Saved final epoch metrics to Excel: {excel_path}")
    except ImportError:
        print("Warning: 'openpyxl' not found. Cannot save metrics to Excel. Install with 'pip install openpyxl'.")
        try: # Attempt CSV saving as fallback
             metrics_df.to_csv(csv_path, index=False)
             print(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv: print(f"Warning: Failed to save metrics to CSV: {e_csv}")
    except Exception as e:
        print(f"Warning: Failed to save metrics to Excel: {e}.")
        try: # Attempt CSV saving as fallback
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv: print(f"Warning: Failed to save metrics to CSV: {e_csv}")

    # Save final model state
    final_model_path = results_dir / "model_final.pt"
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model state to {final_model_path}")
    except Exception as e:
        print(f"Warning: Failed to save final model: {e}")

    # --- Plotting ---
    print("\nGenerating training plots...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Metrics: {exp_name_cleaned}", fontsize=14)

        # Plot Training Loss
        ax = axes[0]
        ax.plot(metrics_df["Epoch"], metrics_df["Average Training Loss"], marker='.', linestyle='-', color='tab:blue')
        ax.set_title("Average Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=':')

        # Plot Evaluation Metrics (only if eval was run)
        eval_ran = metrics_df["Evaluation Episode Success Rate"].notna().any()
        if eval_ran:
            eval_df = metrics_df.dropna(subset=["Evaluation Episode Success Rate"]) # Filter epochs where eval ran

            # Plot Success Rate
            ax = axes[1]
            ax.plot(eval_df["Epoch"], eval_df["Evaluation Episode Success Rate"], marker='o', linestyle='-', color='tab:green')
            ax.set_title("Evaluation Success Rate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=':')

            # Plot Avg Steps for Successful Runs
            ax = axes[2]
            # Only plot if there are successful runs with valid step counts
            valid_steps_df = eval_df.dropna(subset=["Evaluation Avg Steps (Success)"])
            if not valid_steps_df.empty:
                ax.plot(valid_steps_df["Epoch"], valid_steps_df["Evaluation Avg Steps (Success)"], marker='s', linestyle='-', color='tab:red')
                ax.set_ylabel("Average Steps")
            else:
                # Handle case where no episodes were ever successful
                ax.text(0.5, 0.5, 'No successful\neval runs', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
                ax.set_ylabel("Average Steps (N/A)")

            ax.set_title("Avg Steps (Successful Eval Episodes)")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=':')
        else:
             # Handle case where evaluation was never run
             for i in [1, 2]:
                 ax = axes[i]
                 ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
                 ax.set_title(f"Evaluation Metric {i}")
                 ax.set_xlabel("Epoch"); ax.grid(True, linestyle=':')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plot_path = results_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plots to: {plot_path}")
        plt.close(fig) # Close the plot figure
    except Exception as e:
        print(f"Warning: Failed generate plots: {e}")
    # --- --------- ---

    print("\n--- Script Finished ---")