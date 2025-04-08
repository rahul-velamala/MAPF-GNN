# File: train.py
# (Modified Version with DAgger/Online Expert and Fixes)

import sys
import os
import argparse
import time
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import random # For selecting cases for OE
import shutil # For potentially cleaning failed OE runs
import copy   # <<<--- Corrected: Added missing import ---<<<
import signal # For CBS timeout

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset # Import Dataset types

# --- Assuming these imports work when running from project root ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles # Import make_env helper
    from data_loader import GNNDataLoader, CreateDataset # Need CreateDataset for OE data
    from data_generation.record import make_env
    # --- Import CBS for Online Expert ---
    from cbs.cbs import Environment as CBSEnvironment # Rename to avoid clash
    from cbs.cbs import CBS # Removed State/Location imports as not needed directly here
    from data_generation.trayectory_parser import parse_trayectories as parse_traject # Import helper

    # --- Local Timeout Handling ---
    class TimeoutError(Exception): pass
    def handle_timeout(signum, frame): raise TimeoutError("CBS search timed out")
    # --- ----------------------- ---
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running python from the 'rahuldevlab-cloud-project' directory.")
    sys.exit(1)
# --- ----------------------------------------------------------- ---

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Train GNN or Baseline MAPF models.")
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
print(f"Loading configuration from: {config_file_path}")
try:
    with open(config_file_path, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)
except Exception as e:
    print(f"ERROR: Could not load or parse config file '{config_file_path}': {e}")
    sys.exit(1)
# --- ------------------ ---

# --- Setup based on Config ---
net_type = config.get("net_type", "gnn")
exp_name = config.get("exp_name", "default_experiment")
tests_episodes = config.get("tests_episodes", 25)
epochs = config.get("epochs", 50)
max_steps_eval = config.get("max_steps", 60)
max_steps_train_inference = config.get("max_steps_train_inference", max_steps_eval * 3)
print(f"Using max steps for training inference (OE deadlock check): {max_steps_train_inference}")

eval_frequency = config.get("eval_frequency", 5)
# --- Convert LR and WD to float ---
try:
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
except ValueError:
    print("ERROR: Could not convert learning_rate or weight_decay from config to float.")
    sys.exit(1)
# --- -------------------------- ---
num_agents_config = config.get("num_agents", 5)

# --- Online Expert (OE) Config ---
use_online_expert = not args.oe_disable
oe_config = config.get("online_expert", {})
oe_frequency = oe_config.get("frequency_epochs", 4)
oe_num_cases = oe_config.get("num_cases_to_run", 500)
oe_cbs_timeout = oe_config.get("cbs_timeout_seconds", 10)
# --- ------------------------- ---

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_name = exp_name.replace('\\', '/')
results_dir = os.path.join("results", exp_name)

# --- Model Selection ---
try:
    if net_type == "baseline":
        from models.framework_baseline import Network
        print("Using Baseline Network")
    elif net_type == "gnn":
        # Assuming framework_gnn uses GCNLayer by default if msg_type isn't 'message'
        msg_type = config.get("msg_type", "gcn")
        if msg_type == 'message':
             from models.framework_gnn_message import Network
             print("Using GNN Message Passing Network")
        else:
             from models.framework_gnn import Network
             print("Using GNN (GCN) Network")
    else:
        raise ValueError(f"Unknown net_type in config: '{net_type}'")
except Exception as e:
     print(f"ERROR: Failed to import or validate model '{net_type}': {e}")
     sys.exit(1)
# --- --------------- ---

# --- Results Directory and Config Saving ---
os.makedirs(results_dir, exist_ok=True)
config_save_path = os.path.join(results_dir, "config_used.yaml")
try:
    config_to_save = config.copy() # Save a copy
    # Convert device object to string for saving
    if 'device' in config_to_save and not isinstance(config_to_save['device'], str):
        config_to_save['device'] = str(config_to_save["device"])
    # Also convert numericals back to standard format if needed for readability
    config_to_save['learning_rate'] = learning_rate
    config_to_save['weight_decay'] = weight_decay
    with open(config_save_path, "w") as config_path_out:
        yaml.dump(config_to_save, config_path_out, default_flow_style=False, sort_keys=False)
    print(f"Saved effective config to {config_save_path}")
except Exception as e:
    print(f"ERROR: Could not save config to '{config_save_path}': {e}")
    sys.exit(1)
# --- ----------------------------------- ---

# === Helper Function for Online Expert: Run Inference ===
def run_inference_for_oe(model, env, max_steps_inference, device, net_type):
    """
    Runs the current model on the environment until goal, collision, or timeout.
    Returns trajectory history (states, gsos, actions) and success status.
    Includes Collision Shielding.
    """
    model.eval()
    obs, _ = env.reset()
    terminated = False
    truncated = False
    history = {'states': [], 'gsos': [], 'model_actions': [], 'shielded_actions': []}
    idle_action = 0 # Assuming 0 is the idle action index

    while not terminated and not truncated:
        # Store current state before action
        current_fov = torch.tensor(obs["fov"]).float().to(device)
        current_gso = torch.tensor(obs["adj_matrix"]).float().to(device)
        history['states'].append(current_fov.cpu().numpy()) # Store numpy arrays
        history['gsos'].append(current_gso.cpu().numpy())

        # Get action from model
        with torch.no_grad():
            fov_batch = current_fov.unsqueeze(0)
            gso_batch = current_gso.unsqueeze(0)
            if net_type == 'gnn':
                action_scores = model(fov_batch, gso_batch)
            else: # baseline
                action_scores = model(fov_batch)
            proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
        history['model_actions'].append(proposed_actions.copy())

        # --- Apply Collision Shielding (Sec V.G / Alg 1) ---
        shielded_actions = proposed_actions.copy()
        current_pos_y = env.positionY.copy()
        current_pos_x = env.positionX.copy()
        next_pos_y = current_pos_y.copy()
        next_pos_x = current_pos_x.copy()
        needs_shielding = np.zeros(env.nb_agents, dtype=bool)
        active_mask = ~env.reached_goal

        # 1. Calculate proposed next positions for active agents
        for agent_id in np.where(active_mask)[0]:
             act = proposed_actions[agent_id]
             # Ensure action is valid
             if act not in env.action_map_dy_dx: act = idle_action
             dy, dx = env.action_map_dy_dx.get(act, (0,0))
             next_pos_y[agent_id] += dy
             next_pos_x[agent_id] += dx
        # Clamp to boundaries
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
        next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)

        # 2. Check Obstacle Collisions
        if env.obstacles.size > 0:
            next_coords_active = np.stack([next_pos_y[active_mask], next_pos_x[active_mask]], axis=1)
            # Use broadcasting for efficient check against obstacles array
            obs_coll_active_mask = np.any(np.all(next_coords_active[:, np.newaxis, :] == env.obstacles[np.newaxis, :, :], axis=2), axis=1)
            colliding_agent_indices = np.where(active_mask)[0][obs_coll_active_mask]
            shielded_actions[colliding_agent_indices] = idle_action # Shield: set to idle
            needs_shielding[colliding_agent_indices] = True
            # Revert position in temp arrays for subsequent checks
            next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
            next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]

        # 3. Check Agent-Agent Collisions (Vertex & Swapping) among active, non-obstacle-colliding agents
        check_agent_coll_mask = active_mask & (~needs_shielding)
        check_indices = np.where(check_agent_coll_mask)[0]

        agents_to_shield_idx = np.array([], dtype=int) # Initialize as empty int array
        if len(check_indices) > 1:
            relative_indices = np.arange(len(check_indices))
            next_coords_check = np.stack([next_pos_y[check_indices], next_pos_x[check_indices]], axis=1)
            current_coords_check = np.stack([current_pos_y[check_indices], current_pos_x[check_indices]], axis=1)

            # Vertex collisions
            unique_coords, unique_map_indices, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
            colliding_cell_indices = np.where(counts > 1)[0]
            vertex_collision_mask_rel = np.isin(unique_map_indices, colliding_cell_indices)
            vertex_collision_agents = check_indices[vertex_collision_mask_rel]

            # Edge collisions (swapping)
            swapping_collision_agents_list = []
            # Iterate using relative indices and their length
            for i in relative_indices:
                 for j in range(i + 1, len(relative_indices)): # Use length of relative_indices
                     agent_i_idx = check_indices[i] # Get original index
                     agent_j_idx = check_indices[j] # Get original index
                     # Check swap using relative indices i, j
                     if np.array_equal(next_coords_check[i], current_coords_check[j]) and \
                        np.array_equal(next_coords_check[j], current_coords_check[i]):
                         swapping_collision_agents_list.extend([agent_i_idx, agent_j_idx])
            swapping_collision_agents = np.unique(swapping_collision_agents_list)

            # Combine collision indices, handling potential empty arrays
            if vertex_collision_agents.size == 0 and swapping_collision_agents.size == 0:
                agents_to_shield_idx = np.array([], dtype=int)
            elif vertex_collision_agents.size == 0:
                agents_to_shield_idx = swapping_collision_agents
            elif swapping_collision_agents.size == 0:
                agents_to_shield_idx = vertex_collision_agents
            else:
                 agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents]))

        # --- Apply Shielding based on Agent-Agent Collisions ---
        if agents_to_shield_idx.size > 0:
            shielded_actions[agents_to_shield_idx] = idle_action
            needs_shielding[agents_to_shield_idx] = True # Mark shielded

        history['shielded_actions'].append(shielded_actions.copy())
        # --- End Collision Shielding ---

        # Step the environment *with shielded actions*
        try:
            obs, reward, terminated, truncated, info = env.step(shielded_actions)
            truncated = truncated or (env.time >= max_steps_inference) # Add explicit timeout check
        except Exception as e:
             print(f"\nError during env.step in OE inference (Time: {env.time}): {e}")
             return history, False, True # Failed, assume deadlock

    # After loop finishes
    is_success = terminated and not truncated
    is_deadlock = not is_success # Deadlock if not successful (timeout or stuck)

    return history, is_success, is_deadlock


# === Helper Function for Online Expert: Call CBS Expert ===
def call_expert_from_state(env_state, cbs_timeout_s):
    """
    Creates a CBS problem from the current env state and runs the expert.
    Args:
        env_state (dict): Contains 'positions' [N,2](row,col), 'goals' [N,2](row,col),
                          'obstacles' [M,2](row,col), 'board_dims' [rows,cols].
        cbs_timeout_s (int): Timeout for the CBS search.
    Returns:
        Dict (expert solution {agent_name: [{'t':..,'x':..,'y':..},..]}) or None.
    """
    agents_data = []
    current_positions = env_state['positions'] # [row, col]
    goal_positions = env_state['goals']       # [row, col]
    for i in range(len(current_positions)):
         agents_data.append({
             "start": [current_positions[i, 1], current_positions[i, 0]], # CBS uses [x=col, y=row]
             "goal": [goal_positions[i, 1], goal_positions[i, 0]],       # CBS uses [x=col, y=row]
             "name": f"agent{i}"
         })

    map_data = {
        "dimensions": [env_state['board_dims'][1], env_state['board_dims'][0]], # CBS uses [width=cols, height=rows]
        "obstacles": [ [obs[1], obs[0]] for obs in env_state['obstacles']] if env_state['obstacles'].size > 0 else [] # CBS uses list of [x=col, y=row]
    }

    try:
         cbs_env = CBSEnvironment(map_data["dimensions"], agents_data, map_data["obstacles"])
         cbs_solver = CBS(cbs_env, verbose=False)
    except Exception as e:
        print(f"OE Expert Error: Failed to initialize CBS environment: {e}")
        return None

    solution = None
    if hasattr(signal, 'SIGALRM'): # Check if SIGALRM is available (Unix-like)
        original_handler = signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(cbs_timeout_s)
    else: original_handler = None

    try:
        solution = cbs_solver.search()
        if hasattr(signal, 'SIGALRM'): signal.alarm(0) # Disable alarm if search finished
        if not solution:
             # print("OE Expert Info: CBS found no solution.") # Less verbose
             return None
    except TimeoutError:
        # print(f"OE Expert Warning: CBS expert timed out.") # Less verbose
        return None
    except Exception as e:
        print(f"OE Expert Error: CBS search failed: {e}")
        return None
    finally:
        # Restore original signal handler only if it was set
        if hasattr(signal, 'SIGALRM') and original_handler is not None:
            signal.signal(signal.SIGALRM, original_handler)
            signal.alarm(0) # Ensure alarm is off in finally block

    return solution


# === Helper Function for Online Expert: Aggregate Data ===
def aggregate_expert_data(expert_solution, start_env, config):
    """
    Simulates the expert path from the deadlock start state and records
    the sequence of (FOV, GSO, Expert_Action).
    Args:
        expert_solution (dict): Output from call_expert_from_state.
        start_env (GraphEnv): Environment instance *at the deadlock state*.
        config (dict): Main configuration.
    Returns:
        Tuple: (numpy_states, numpy_gsos, numpy_expert_actions) or None if failed.
               Shapes: [T_agg, N, C, H, W], [T_agg, N, N], [T_agg, N]
    """
    if not expert_solution: return None
    num_agents = start_env.nb_agents

    # 1. Parse expert solution into actions array [N, T_expert]
    try:
        expert_actions_np, _ = parse_traject(expert_solution)
        num_expert_steps = expert_actions_np.shape[1]
        if num_expert_steps == 0: return None
        # print(f"\n[OE Agg] Expert path length: {num_expert_steps} steps.") # Optional debug
    except Exception as e:
        print(f"OE Aggregate Error: Failed to parse expert solution: {e}"); return None

    # 2. Simulate expert actions in the environment from the deadlock start state
    env = copy.deepcopy(start_env)

    # <<< --- FIX: Reset the time counter for the simulation --- >>>
    env.time = 0
    # <<< --- ------------------------------------------------ >>>

    # print(f"[OE Agg] Simulating in Env with max_time: {env.max_time}") # Optional debug

    aggregated_states = []
    aggregated_gsos = []
    aggregated_actions = []

    # Record initial state (the deadlock state) BEFORE first action
    # Re-get obs as deepcopy might not perfectly replicate internal state if used by getObservations
    obs = env.getObservations()
    aggregated_states.append(obs["fov"])
    aggregated_gsos.append(obs["adj_matrix"])

    for t in range(num_expert_steps):
        actions_t = expert_actions_np[:, t]
        aggregated_actions.append(actions_t) # Store action t

        try:
            # Step with expert action
            obs, _, terminated, truncated, _ = env.step(actions_t, emb=None)

            # Record state resulting from action t (state at t+1)
            aggregated_states.append(obs["fov"])
            aggregated_gsos.append(obs["adj_matrix"])

            # Optional Debug Print
            # if terminated or truncated:
            #    print(f"[OE Agg] Step {t+1}/{num_expert_steps} (Env Time: {env.time}): Term={terminated}, Trunc={truncated}")

            if terminated: break # Stop simulation if goal is reached by expert path

            # If truncated based on env.max_time, it's now a *real* issue (path too long)
            if truncated:
                 print(f"OE Aggregate Warning: Env truncated during expert path sim at step {t+1}/{num_expert_steps} (Env Time: {env.time}). Max time is {env.max_time}. Expert path might be too long.")
                 # Decide whether to break or continue aggregating the partial path.
                 # Breaking here means we only get partial data if expert path > env.max_time
                 break
        except Exception as e:
             print(f"OE Aggregate Error: Failed during expert path sim step {t}: {e}"); return None

    # Convert lists of numpy arrays to single large numpy arrays
    try:
        num_pairs = len(aggregated_actions)
        # Check if any actions were actually aggregated
        if num_pairs == 0:
             # print("OE Aggregate Info: No actions aggregated (e.g., expert path was empty or simulation failed immediately).") # Less verbose
             return None

        # Use states[0]..states[num_pairs-1] and actions[0]..actions[num_pairs-1]
        # Total states recorded = num_pairs + 1 (includes initial state)
        # Total GSOs recorded = num_pairs + 1
        final_states = np.stack(aggregated_states[:num_pairs]) # Shape [T_agg, N, C, H, W]
        final_gsos = np.stack(aggregated_gsos[:num_pairs])     # Shape [T_agg, N, N]
        final_actions = np.stack(aggregated_actions)           # Shape [T_agg, N]

        if not (final_states.shape[0] == final_gsos.shape[0] == final_actions.shape[0] == num_pairs):
             print(f"OE Aggregate Error: Mismatch shapes after stacking. Pairs:{num_pairs}, S:{final_states.shape}, G:{final_gsos.shape}, A:{final_actions.shape}"); return None

        return final_states, final_gsos, final_actions
    except Exception as e:
         print(f"OE Aggregate Error: Failed to stack aggregated data: {e}"); return None


# === Main Training Script ===
if __name__ == "__main__":

    print("\n----- Effective Configuration -----")
    pprint(config)
    print(f"Using device: {config['device']}")
    print(f"Online Expert (DAgger): {'Enabled' if use_online_expert else 'Disabled'}")
    if use_online_expert:
         print(f"  OE Frequency: Every {oe_frequency} epochs")
         print(f"  OE Cases per Run: {oe_num_cases}")
         print(f"  OE CBS Timeout: {oe_cbs_timeout}s")
    print("---------------------------------\n")

    # --- Data Loading ---
    try:
        data_loader_manager = GNNDataLoader(config)
        train_loader = data_loader_manager.train_loader
        valid_loader = data_loader_manager.valid_loader

        if not train_loader or len(train_loader.dataset) == 0:
             print("ERROR: Training data loader empty."); sys.exit(1)
        original_train_dataset = train_loader.dataset
        print(f"Initial training samples (timesteps): {len(original_train_dataset)}")
        if valid_loader: print(f"Validation samples (timesteps): {len(valid_loader.dataset)}")
        else: print("No validation data loader created.")
    except Exception as e:
        print(f"ERROR: Failed to initialize/load data: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    # --- ------------ ---

    # --- Model, Optimizer, Criterion ---
    try:
        model = Network(config)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        model.to(config["device"])
        print(f"Model '{type(model).__name__}' initialized on {config['device']}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
    except Exception as e:
        print(f"ERROR: Failed init model/optimizer/criterion: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    # --- --------------------------- ---

    # --- Training Loop Setup ---
    all_epoch_metrics = []
    best_eval_success_rate = -1.0
    aggregated_expert_data_list = [] # Holds dicts {'states':S,'gsos':G,'actions':A}
    # --- ----------------- ---

    print(f"\n--- Starting Training for {epochs} epochs ---")
    training_start_time = time.time()

    # --- Main Epoch Loop ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # === Online Expert (DAgger) Data Aggregation ===
        run_oe_this_epoch = use_online_expert and ((epoch + 1) % oe_frequency == 0)

        if run_oe_this_epoch:
            print(f"--- Running Online Expert Data Aggregation (Epoch {epoch+1}) ---")
            oe_start_time = time.time()
            num_deadlocks_found = 0
            num_expert_calls = 0
            num_expert_success = 0
            newly_aggregated_samples_count = 0
            new_data_this_epoch = [] # Collect new data for this epoch only

            # Ensure original_train_dataset has 'cases' attribute and 'dir_path'
            if not hasattr(original_train_dataset, 'cases') or not hasattr(original_train_dataset, 'dir_path'):
                 print("OE Error: Original dataset object missing 'cases' or 'dir_path' attribute. Cannot select cases.")
            else:
                num_original_cases = len(original_train_dataset.cases)
                if num_original_cases == 0:
                    print("OE Warning: Original dataset has no cases listed. Cannot run OE.")
                else:
                    indices_to_run = random.sample(range(num_original_cases), min(oe_num_cases, num_original_cases))
                    print(f"Selected {len(indices_to_run)} cases for OE inference.")

                    oe_pbar = tqdm(indices_to_run, desc="OE Inference", unit="case", leave=False)
                    for case_idx_in_orig_dataset in oe_pbar:
                        case_name = original_train_dataset.cases[case_idx_in_orig_dataset]
                        case_path = os.path.join(original_train_dataset.dir_path, case_name)
                        env_oe = None # Ensure env_oe is defined for finally block
                        try:
                            env_oe = make_env(case_path, config) # Use helper from record.py
                            if env_oe is None: continue

                            history, is_success, is_deadlock = run_inference_for_oe(
                                model, env_oe, max_steps_train_inference, config["device"], net_type
                            )

                            if is_deadlock:
                                num_deadlocks_found += 1
                                # Use env_oe which is now AT the deadlock state after run_inference loop finished
                                deadlock_state_info = {
                                    "positions": env_oe.get_current_positions(), # [N,2](row,col)
                                    "goals": env_oe.goal.copy(),             # [N,2](row,col)
                                    "obstacles": env_oe.obstacles.copy(),    # [M,2](row,col)
                                    "board_dims": env_oe.config['board_size'] # [rows, cols]
                                }
                                num_expert_calls += 1
                                expert_solution_dict = call_expert_from_state(deadlock_state_info, oe_cbs_timeout)

                                if expert_solution_dict:
                                    # Pass env_oe (at deadlock state) to aggregation function
                                    aggregated_data = aggregate_expert_data(expert_solution_dict, env_oe, config)
                                    if aggregated_data:
                                        num_expert_success += 1
                                        states_agg, gsos_agg, actions_agg = aggregated_data
                                        # Append dict of numpy arrays for temporary storage
                                        new_data_this_epoch.append({
                                            "states": states_agg, "gsos": gsos_agg, "actions": actions_agg
                                        })
                                        newly_aggregated_samples_count += len(states_agg) # Count samples (timesteps)
                                        oe_pbar.set_postfix({"Deadlocks": num_deadlocks_found, "ExpertOK": num_expert_success, "NewSamples": newly_aggregated_samples_count})

                        except Exception as e:
                             print(f"\nOE Error: Unhandled exception processing {case_name}: {e}");
                             import traceback; traceback.print_exc()
                        finally:
                             if env_oe: env_oe.close() # Close env figure if opened

            # --- Update main aggregated list and recreate DataLoader ---
            if newly_aggregated_samples_count > 0:
                print(f"\nOE Aggregation Summary: Found {num_deadlocks_found} deadlocks, Expert succeeded {num_expert_success}/{num_expert_calls} times.")
                print(f"Aggregated {newly_aggregated_samples_count} new (state, action) pairs this epoch.")

                # Add data collected this epoch to the main list
                aggregated_expert_data_list.extend(new_data_this_epoch)
                total_aggregated_samples = sum(len(d['states']) for d in aggregated_expert_data_list)
                print(f"Total aggregated expert samples now: {total_aggregated_samples}")

                # Create a Dataset from ALL aggregated data so far
                class AggregatedDataset(Dataset):
                    def __init__(self, aggregated_list):
                        if not aggregated_list: # Handle empty list
                             # Match shapes from original dataset if possible, else define explicitly
                             # Assuming original_train_dataset.states etc. exist and are populated
                             state_shape_postfix = getattr(original_train_dataset, 'states', np.empty((0,0,0,0,0))).shape[1:]
                             gso_shape_postfix = getattr(original_train_dataset, 'gsos', np.empty((0,0,0))).shape[1:]
                             action_shape_postfix = getattr(original_train_dataset, 'trajectories', np.empty((0,0))).shape[1:]

                             self.all_states = np.empty((0,) + state_shape_postfix, dtype=np.float32)
                             self.all_gsos = np.empty((0,) + gso_shape_postfix, dtype=np.float32)
                             self.all_actions = np.empty((0,) + action_shape_postfix, dtype=np.int64)
                        else:
                             self.all_states = np.concatenate([d['states'] for d in aggregated_list], axis=0)
                             self.all_gsos = np.concatenate([d['gsos'] for d in aggregated_list], axis=0)
                             self.all_actions = np.concatenate([d['actions'] for d in aggregated_list], axis=0)
                        self.count = len(self.all_states)
                        if self.count > 0: print(f"AggregatedDataset: Total Shapes S:{self.all_states.shape}, G:{self.all_gsos.shape}, A:{self.all_actions.shape}")

                    def __len__(self): return self.count
                    def __getitem__(self, index):
                        state = torch.from_numpy(self.all_states[index]).float()
                        gso = torch.from_numpy(self.all_gsos[index]).float()
                        action = torch.from_numpy(self.all_actions[index]).long()
                        # Return order must match DataLoader expectation in training loop
                        return state, action, gso # Order: State, Action, GSO

                aggregated_dataset = AggregatedDataset(aggregated_expert_data_list)

                # Combine original and aggregated datasets for this epoch's training
                combined_dataset = ConcatDataset([original_train_dataset, aggregated_dataset])
                print(f"Combined dataset size for this epoch: {len(combined_dataset)} samples.")

                # Create new DataLoader for combined data
                train_loader = DataLoader(
                    combined_dataset, batch_size=config["batch_size"], shuffle=True,
                    num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
                )
            else:
                 print("OE Aggregation: No new samples aggregated this epoch.")
                 # If no new data, reuse the existing train_loader (which might contain past aggregations)
                 # Ensure train_loader is defined from previous epoch or initial load
                 if 'train_loader' not in locals() or train_loader is None: # Check if train_loader is valid
                      train_loader = data_loader_manager.train_loader # Fallback to initial

            oe_duration = time.time() - oe_start_time
            print(f"--- Online Expert Data Aggregation Finished ({oe_duration:.2f}s) ---")
        # === End Online Expert Block ===

        # ##### Training Phase #########
        if train_loader is None: # Safety check if fallback failed
            print("ERROR: train_loader is None before training phase. Exiting.")
            sys.exit(1)

        model.train()
        epoch_train_loss = 0.0
        batches_processed = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

        for i, batch_data in enumerate(train_pbar):
            try:
                # DataLoader order: state, action, gso (based on CreateDataset/AggregatedDataset)
                states = batch_data[0].to(config["device"], non_blocking=True)
                target_actions = batch_data[1].to(config["device"], non_blocking=True)
                gso = batch_data[2].to(config["device"], non_blocking=True)
            except Exception as e:
                 print(f"\nError unpack/move batch {i}: {e}")
                 continue # Skip batch

            optimizer.zero_grad()
            try: # Forward Pass
                if net_type == 'gnn': output_logits = model(states, gso) # [B, N, Actions]
                else: output_logits = model(states)
            except Exception as e:
                print(f"\nError forward pass batch {i}: {e}")
                continue # Skip batch

            try: # Loss Calc
                # Ensure output has the correct number of actions (5)
                if output_logits.shape[-1] != 5:
                     print(f"\nShape mismatch: Model output has {output_logits.shape[-1]} actions, expected 5. Batch {i}"); continue

                output_reshaped = output_logits.reshape(-1, 5) # [B*N, Actions]
                target_reshaped = target_actions.reshape(-1).long() # [B*N]

                # Ensure target actions are within the valid range [0, 4]
                if torch.any(target_reshaped < 0) or torch.any(target_reshaped > 4):
                    print(f"\nInvalid target action found in batch {i}. Values outside [0, 4]. Min: {target_reshaped.min()}, Max: {target_reshaped.max()}"); continue

                if output_reshaped.shape[0] != target_reshaped.shape[0]:
                     print(f"\nShape mismatch loss batch {i}: Output {output_reshaped.shape[0]}, Target {target_reshaped.shape[0]}"); continue

                batch_loss = criterion(output_reshaped, target_reshaped)
                if not torch.isfinite(batch_loss): print(f"\nWarning: Loss NaN/Inf batch {i}"); continue

            except IndexError as e:
                 print(f"\nIndexError during loss calculation batch {i}: {e}. Target shape: {target_reshaped.shape}, Target values (sample): {target_reshaped[:10]}, Max: {target_reshaped.max()}, Min: {target_reshaped.min()}")
                 continue # Skip this batch
            except Exception as e:
                 print(f"\nError loss calc batch {i}: {e}")
                 continue # Skip batch

            try: # Backward Pass & Opt Step
                batch_loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"\nError backward/opt batch {i}: {e}")
                continue # Don't stop training, just skip opt step for this batch

            epoch_train_loss += batch_loss.item()
            batches_processed += 1
            if batches_processed > 0: train_pbar.set_postfix({"AvgLoss": epoch_train_loss / batches_processed})
        # --- End Training Batch Loop ---

        avg_epoch_loss = epoch_train_loss / batches_processed if batches_processed > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f} | Duration: {epoch_duration:.2f}s")

        current_epoch_data = {
            "Epoch": epoch + 1, "Average Training Loss": avg_epoch_loss,
            "Evaluation Episode Success Rate": np.nan, "Evaluation Avg Flow Time (Success)": np.nan,
            "Evaluation Episodes Tested": 0, "Evaluation Episodes Succeeded": 0,
            "Training Samples Used": len(train_loader.dataset) if train_loader and train_loader.dataset else 0
        }

        # ######### Evaluation Phase #########
        run_eval = ((epoch + 1) % eval_frequency == 0) or ((epoch + 1) == epochs)
        if run_eval and tests_episodes > 0:
            print(f"\n--- Running Evaluation after Epoch {epoch+1} ---")
            eval_start_time = time.time()
            model.eval()
            eval_success_count = 0
            eval_flow_times_success = []
            # Use evaluation specific parameters if defined, otherwise fallback to general config
            board_dims_eval = config.get("eval_board_size", config.get("board_size", [16, 16]))
            obstacles_count_eval = config.get("eval_obstacles", config.get("obstacles", 6))
            agents_count_eval = config.get("eval_num_agents", config.get("num_agents", 4))
            sensing_range_eval = config.get("eval_sensing_range", config.get("sensing_range", 4))
            pad_eval = config.get("eval_pad", config.get("pad", 3))
            max_steps_eval_run = config.get("eval_max_steps", config.get("max_steps", 60))


            eval_pbar = tqdm(range(tests_episodes), desc=f"Epoch {epoch+1} Evaluation", leave=False, unit="ep")
            for episode in eval_pbar:
                env_eval = None # Ensure env is defined for finally block
                try:
                    # Generate a new random scenario for each evaluation episode
                    obstacles_eval_ep = create_obstacles(board_dims_eval, obstacles_count_eval)
                    start_pos_eval_ep = create_goals(board_dims_eval, agents_count_eval, obstacles_eval_ep)
                    temp_obs_goals = np.vstack([obstacles_eval_ep, start_pos_eval_ep]) if obstacles_eval_ep.size > 0 else start_pos_eval_ep
                    goals_eval_ep = create_goals(board_dims_eval, agents_count_eval, temp_obs_goals)

                    env_eval = GraphEnv(config, goal=goals_eval_ep, obstacles=obstacles_eval_ep,
                                        starting_positions=start_pos_eval_ep,
                                        sensing_range=sensing_range_eval, pad=pad_eval)

                    # Use inference function (already has shielding)
                    _, is_success, _ = run_inference_for_oe(
                        model, env_eval, max_steps_eval_run, config["device"], net_type
                    )
                    if is_success:
                        eval_success_count += 1
                        eval_flow_times_success.append(env_eval.time)
                    eval_pbar.set_postfix({"Success": f"{eval_success_count}/{episode+1}"})
                except Exception as e:
                    print(f"\nError eval episode {episode}: {e}")
                    import traceback; traceback.print_exc()
                    continue # Skip to next episode on error
                finally:
                     if env_eval: env_eval.close() # Ensure env is closed even if error occurs

            # --- End Eval Loop ---
            eval_success_rate = eval_success_count / tests_episodes if tests_episodes > 0 else 0.0
            avg_flow_time = np.mean(eval_flow_times_success) if eval_flow_times_success else np.nan
            eval_duration = time.time() - eval_start_time
            current_epoch_data.update({
                "Evaluation Episode Success Rate": eval_success_rate,
                "Evaluation Avg Flow Time (Success)": avg_flow_time,
                "Evaluation Episodes Tested": tests_episodes,
                "Evaluation Episodes Succeeded": eval_success_count,
            })
            print(f"Eval Complete: SR={eval_success_rate:.4f}, AvgSteps={avg_flow_time:.2f} | Dur={eval_duration:.2f}s")

            # Save best model based on success rate
            if eval_success_rate >= best_eval_success_rate:
                 print(f"New best eval SR ({eval_success_rate:.4f}), saving model...")
                 best_eval_success_rate = eval_success_rate
                 best_model_path = os.path.join(results_dir, f"model_best.pt")
                 try: torch.save(model.state_dict(), best_model_path)
                 except Exception as e: print(f"Warn: Failed save best model: {e}")

        all_epoch_metrics.append(current_epoch_data)
        # Save metrics periodically and at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            try:
                 metrics_df_partial = pd.DataFrame(all_epoch_metrics)
                 excel_path_partial = os.path.join(results_dir, "training_metrics_partial.xlsx")
                 metrics_df_partial.to_excel(excel_path_partial, index=False, engine='openpyxl')
            except ImportError: pass # openpyxl not installed
            except Exception as e: print(f"Warning: Failed to save partial metrics: {e}")
    # --- End Epoch Loop ---

    total_training_time = time.time() - training_start_time
    print(f"\n--- Training Finished ({total_training_time:.2f}s total) ---")

    # --- Saving Final Results ---
    metrics_df = pd.DataFrame(all_epoch_metrics)
    excel_path = os.path.join(results_dir, "training_metrics.xlsx")
    csv_path = os.path.join(results_dir, "training_metrics.csv")
    try:
        # Ensure openpyxl is installed: pip install openpyxl
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Saved final epoch metrics to Excel: {excel_path}")
    except ImportError:
        print("Warning: 'openpyxl' not found. Cannot save metrics to Excel. Install with 'pip install openpyxl'. Attempting CSV.")
        try:
             metrics_df.to_csv(csv_path, index=False)
             print(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv: print(f"Warning: Failed to save metrics to CSV: {e_csv}")
    except Exception as e:
        print(f"Warning: Failed to save metrics to Excel: {e}. Attempting CSV.")
        try:
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv: print(f"Warning: Failed to save metrics to CSV: {e_csv}")

    # Save final model
    final_model_path = os.path.join(results_dir, "model_final.pt")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
    except Exception as e:
        print(f"Warning: Failed to save final model: {e}")

    # --- Plotting ---
    print("\n--- Generating Plots ---")
    try:
        plt.figure(figsize=(18, 5))
        # Plot Training Loss
        plt.subplot(1, 3, 1); plt.plot(metrics_df["Epoch"], metrics_df["Average Training Loss"], marker='.'); plt.title("Avg Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.tight_layout(pad=2.0)

        # Plot Evaluation Metrics (only if eval was run)
        eval_df = metrics_df.dropna(subset=["Evaluation Episode Success Rate"]) # Filter epochs where eval ran
        if not eval_df.empty:
            # Plot Success Rate
            plt.subplot(1, 3, 2); plt.plot(eval_df["Epoch"], eval_df["Evaluation Episode Success Rate"], marker='o'); plt.title("Eval Success Rate"); plt.xlabel("Epoch"); plt.ylabel("Success Rate"); plt.ylim(-0.05, 1.05); plt.grid(True); plt.tight_layout(pad=2.0)

            # Plot Flow Time (Steps) for successful runs
            plt.subplot(1, 3, 3)
            valid_flow_df = eval_df.dropna(subset=["Evaluation Avg Flow Time (Success)"]) # Filter out epochs with 0 success
            if not valid_flow_df.empty:
                plt.plot(valid_flow_df["Epoch"], valid_flow_df["Evaluation Avg Flow Time (Success)"], marker='o')
            else:
                # Handle case where no episodes were ever successful
                plt.text(0.5, 0.5, 'No successful eval runs', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Eval Flow Time (Avg Success)"); plt.xlabel("Epoch"); plt.ylabel("Steps"); plt.grid(True); plt.tight_layout(pad=2.0)
        else:
             # Handle case where evaluation was never run
             for i in [2, 3]:
                 plt.subplot(1, 3, i); plt.text(0.5, 0.5, 'No eval data', ha='center', va='center', transform=plt.gca().transAxes); plt.title(f"Eval Metric {i-1}"); plt.xlabel("Epoch"); plt.grid(True)

        plot_path = os.path.join(results_dir, "training_plots.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plots: {plot_path}")
        plt.close() # Close the plot figure
    except Exception as e:
        print(f"Warning: Failed generate plots: {e}")
    # --- --------- ---

    print("\n--- Script Finished ---")