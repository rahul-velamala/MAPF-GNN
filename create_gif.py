# File: create_gif.py
# (Modified to save scenario and path to YAML)

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import imageio  # Library for creating GIFs
from tqdm import tqdm # Add progress bar for simulation
from pathlib import Path # Use Pathlib
import logging # Use standard logging
from collections import defaultdict # To store paths

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---

# --- Import environment and model components ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
except ImportError:
    logger.error("Error: Could not import environment classes."); sys.exit(1)
# --- --------------------------------------- ---

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate GIF and Scenario YAML for MAPF.")
parser.add_argument("--config", type=str, default="configs/config_gnn.yaml", help="Path to the training configuration file.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt) file.")
parser.add_argument("--output_gif", type=str, default="mapf_visualization.gif", help="Filename for the output GIF.")
parser.add_argument("--output_yaml", type=str, default="mapf_scenario_path.yaml", help="Filename for the output YAML with scenario and path.") # New Argument
parser.add_argument("--seed", type=int, default=None, help="Random seed for scenario generation.")
parser.add_argument("--gif_duration", type=float, default=0.2, help="Duration (seconds) per GIF frame.")
parser.add_argument("--max_steps", type=int, default=None, help="Override max simulation steps from config.")
args = parser.parse_args()
# --- --------------- ---

# --- Set Seed ---
if args.seed is not None:
    logger.info(f"Using random seed: {args.seed}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
# --- -------- ---

# --- Load Configuration ---
config_path = Path(args.config)
if not config_path.is_file(): logger.error(f"Config file not found: {config_path}"); sys.exit(1)
try:
    with open(config_path, "r") as f: config = yaml.safe_load(f)
    if config is None: raise ValueError("Config file empty/invalid.")
except Exception as e: logger.error(f"Error loading config {config_path}: {e}"); sys.exit(1)
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {config['device']}")
# --- ------------------ ---

# --- Dynamically Import Model Class ---
NetworkClass = None
try:
    net_type = config.get("net_type", "gnn"); logger.info(f"Loading model type: {net_type}")
    if net_type == "baseline": from models.framework_baseline import Network as NetworkClass
    elif net_type == "gnn":
        from models.framework_gnn import Network as NetworkClass
        msg_type = config.get("msg_type", "gcn").lower(); config['msg_type'] = msg_type
        logger.info(f"Imported GNN Network (config msg_type: {msg_type}).")
    else: raise ValueError(f"Unknown net_type: {net_type}")
    if NetworkClass is None: raise ImportError("Network class not assigned.")
    logger.info(f"Imported NetworkClass: {NetworkClass.__name__}")
except Exception as e: logger.error(f"Error importing model class: {e}", exc_info=True); sys.exit(1)
# --- ------------------------------ ---

# --- Load Model ---
model = NetworkClass(config); model.to(config["device"])
model_load_path = Path(args.model_path)
logger.info(f"Loading model from: {model_load_path}")
if not model_load_path.is_file(): logger.error(f"Model file not found: {model_load_path}."); sys.exit(1)
try:
    model.load_state_dict(torch.load(model_load_path, map_location=config["device"]))
    model.eval(); logger.info("Model loaded successfully.")
except Exception as e: logger.error(f"Error loading model state_dict: {e}", exc_info=True); sys.exit(1)
# --- ---------- ---

# --- Setup Environment for a Single Episode ---
logger.info("Setting up environment...")
env = None
# Store initial scenario details for YAML output
scenario_details_for_yaml = {}
try:
    board_dims_tuple = tuple(config.get("eval_board_size", config.get("board_size", [20, 20])))
    num_obstacles_gen = config.get("eval_obstacles", config.get("obstacles", 40))
    num_agents_env = config.get("eval_num_agents", config.get("num_agents", 10))
    if hasattr(model, 'num_agents') and num_agents_env != model.num_agents:
         logger.warning(f"Config N ({num_agents_env}) != model N ({model.num_agents}). Using model's: {model.num_agents}.")
         num_agents_env = model.num_agents

    obstacles_arr = create_obstacles(board_dims_tuple, num_obstacles_gen)
    start_pos_arr = create_goals(board_dims_tuple, num_agents_env, obstacles=obstacles_arr)
    goals_arr = create_goals(board_dims_tuple, num_agents_env, obstacles=np.vstack([obstacles_arr, start_pos_arr]) if obstacles_arr.size > 0 else start_pos_arr)

    # Store for YAML (convert to lists of simple ints for YAML readability)
    # Note: GraphEnv uses (row, col), YAML often uses (x, y) where x=col, y=row
    scenario_details_for_yaml['map_dimensions_wh'] = [int(board_dims_tuple[1]), int(board_dims_tuple[0])] # width, height
    scenario_details_for_yaml['obstacles_xy'] = [[int(o[1]), int(o[0])] for o in obstacles_arr.tolist()]
    scenario_details_for_yaml['agents'] = []
    for i in range(num_agents_env):
        scenario_details_for_yaml['agents'].append({
            'name': f'agent{i}',
            'start_xy': [int(start_pos_arr[i, 1]), int(start_pos_arr[i, 0])], # col, row
            'goal_xy': [int(goals_arr[i, 1]), int(goals_arr[i, 0])]      # col, row
        })

    env_config_instance = config.copy()
    env_config_instance["num_agents"] = num_agents_env
    env_config_instance["board_size"] = list(board_dims_tuple) # rows, cols
    env_config_instance["pad"] = config.get("eval_pad", config.get("pad", 6))
    env_config_instance["sensing_range"] = config.get("eval_sensing_range", config.get("sensing_range", 5))
    max_steps_sim = args.max_steps if args.max_steps is not None else config.get("eval_max_steps", config.get("max_steps", config.get("max_time", 150)))
    env_config_instance["max_time"] = max_steps_sim
    env_config_instance["render_mode"] = "rgb_array"

    env = GraphEnv(config=env_config_instance, goal=goals_arr, obstacles=obstacles_arr, starting_positions=start_pos_arr)
    obs, info = env.reset(seed=args.seed)
    logger.info(f"Simulating: Board={env.board_rows}x{env.board_cols}, Agents={env.nb_agents}, Pad={env.pad}, MaxSteps={max_steps_sim}")
except Exception as e: logger.error(f"Error setting up env: {e}", exc_info=True); sys.exit(1)
# --- ------------------------------------ ---

# --- Simulation, Frame Capture, and Path Recording ---
frames = []
# Initialize path storage: {agent_name: [{'t': time, 'x': col, 'y': row}, ...]}
agent_paths_yaml = {f'agent{i}': [] for i in range(env.nb_agents)}

logger.info(f"Starting simulation for max {max_steps_sim} steps...")
terminated = False; truncated = False
sim_pbar = tqdm(range(max_steps_sim), desc="Simulating Episode", unit="step")

# Record initial positions (t=0)
current_positions_rc = env.get_current_positions() # Returns (N, 2) [row, col]
for i in range(env.nb_agents):
    agent_paths_yaml[f'agent{i}'].append({
        't': 0,
        'x': int(current_positions_rc[i, 1]), # col
        'y': int(current_positions_rc[i, 0])  # row
    })

for step in sim_pbar:
    if terminated or truncated: break

    # 1. Render current state (BEFORE step)
    try:
        frame = env.render(mode='rgb_array')
        if frame is not None: frames.append(frame)
        else: logger.warning(f"env.render returned None at step {env.time}.")
    except Exception as e: logger.error(f"Error env.render @ step {env.time}: {e}", exc_info=True); sys.exit(1)

    # 2. Prepare observation
    try:
        fov_tensor = torch.from_numpy(obs["fov"]).float().unsqueeze(0).to(config["device"])
        gso_tensor = torch.from_numpy(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])
    except Exception as e: logger.error(f"Error processing obs @ step {env.time}: {e}", exc_info=True); sys.exit(1)

    # 3. Get action from model
    with torch.no_grad():
        try:
            if net_type == 'gnn': action_scores = model(fov_tensor, gso_tensor)
            else: action_scores = model(fov_tensor)
            proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
        except Exception as e: logger.error(f"Error model forward @ step {env.time}: {e}", exc_info=True); sys.exit(1)

    # 4. Step environment (handles collisions internally)
    try:
        obs, reward, terminated, truncated_env, info = env.step(proposed_actions)
        truncated = truncated or truncated_env or (env.time >= max_steps_sim)
    except Exception as e: logger.error(f"Error env.step @ time {env.time}: {e}", exc_info=True); sys.exit(1)

    # 5. Record new positions for YAML (AFTER step)
    current_positions_rc = env.get_current_positions() # [row, col]
    for i in range(env.nb_agents):
        agent_paths_yaml[f'agent{i}'].append({
            't': env.time, # Current time after step
            'x': int(current_positions_rc[i, 1]), # col
            'y': int(current_positions_rc[i, 0])  # row
        })

    sim_pbar.set_postfix({"Term": terminated, "Trunc": truncated, "AtGoal": info.get('agents_at_goal',[]).sum(), "Step": env.time})
# --- End Simulation Loop ---
sim_pbar.close()

# Capture final frame if needed
if (terminated or truncated) and (not frames or env.time > frames[-1].shape[0] if frames else True): # Avoid duplicate final frame
    logger.info(f"Sim ended. Term={terminated}, Trunc={truncated}, FinalStep={env.time}")
    try:
        final_frame = env.render(mode='rgb_array')
        if final_frame is not None: frames.append(final_frame); logger.info("Appended final frame.")
        else: logger.warning("Could not render final frame.")
    except Exception as e: logger.warning(f"Error rendering final frame: {e}", exc_info=True)

if env: env.close(); logger.info("Environment closed.")

# --- Save Frames as GIF ---
output_gif_path = Path(args.output_gif).resolve()
if frames:
    logger.info(f"\nSaving {len(frames)} frames to {output_gif_path}...")
    try:
        imageio.mimsave(output_gif_path, frames, duration=int(args.gif_duration * 1000), loop=0)
        logger.info(f"GIF saved successfully to {output_gif_path}")
    except Exception as e: logger.error(f"Error saving GIF: {e}", exc_info=True)
else: logger.warning("\nNo frames captured. Cannot create GIF.")

# --- Save Scenario and Path to YAML ---
output_yaml_path = Path(args.output_yaml).resolve()
yaml_data_to_save = {
    'scenario': scenario_details_for_yaml,
    'executed_schedule': agent_paths_yaml,
    'status': {'success': bool(terminated and not truncated), 'steps_taken': env.time if env else max_steps_sim}
}
logger.info(f"\nSaving scenario and path data to {output_yaml_path}...")
try:
    with open(output_yaml_path, 'w') as f_yaml:
        yaml.dump(yaml_data_to_save, f_yaml, default_flow_style=None, sort_keys=False, indent=2)
    logger.info(f"Scenario and path YAML saved successfully to {output_yaml_path}")
except Exception as e:
    logger.error(f"Error saving scenario YAML: {e}", exc_info=True)

logger.info("\nScript finished.")