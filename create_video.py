# File: create_video.py
# (Modified with Collision Shielding)

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import imageio  # Still use imageio
from tqdm import tqdm # Add progress bar for simulation

# --- Add necessary paths ---
sys.path.append("configs"); sys.path.append("models"); sys.path.append(".")

# --- Import environment and model components ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
except ImportError:
    print("Error: Could not import environment classes."); sys.exit(1)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate video visualization for MAPF using a trained model.")
parser.add_argument("--config", type=str, default="configs/config_gnn.yaml", help="Path to the configuration file.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt) file.")
parser.add_argument("--output_file", type=str, default="mapf_visualization.mp4", help="Filename for the output video (e.g., .mp4).")
parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible scenario.")
parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
parser.add_argument("--max_steps", type=int, default=None, help="Override max simulation steps from config.")


args = parser.parse_args()

# --- Set Seed ---
if args.seed is not None:
    print(f"Using random seed: {args.seed}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

# --- Load Configuration ---
try:
    with open(args.config, "r") as config_path: config = yaml.load(config_path, Loader=yaml.FullLoader)
except Exception as e: print(f"Error loading config {args.config}: {e}"); sys.exit(1)

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config['device']}")
net_type = config["net_type"]

# --- Dynamically Import Model Class ---
try:
    if net_type == "baseline": from models.framework_baseline import Network
    elif net_type == "gnn": from models.framework_gnn import Network
    else: raise ValueError(f"Unknown net_type in config: {net_type}")
    print(f"Using network type: {net_type}")
except Exception as e: print(f"Error importing model class: {e}"); sys.exit(1)

# --- Load Model ---
model = Network(config); model.to(config["device"])
print(f"Loading model from: {args.model_path}")
if not os.path.exists(args.model_path): raise FileNotFoundError(f"Model file not found at: {args.model_path}.")
try:
    model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
    model.eval(); print("Model loaded successfully.")
except Exception as e: print(f"Error loading model state_dict: {e}"); sys.exit(1)

# --- Setup Environment ---
print("Setting up environment for one episode...")
try:
    board_dims = config.get("board_size", [28, 28])
    if isinstance(board_dims, int): board_dims = [board_dims, board_dims]
    num_obstacles = config.get("obstacles", 8)
    obstacles = create_obstacles(board_dims, num_obstacles)
    num_agents = config.get("num_agents", 5)
    start_pos = create_goals(board_dims, num_agents, obstacles)
    temp_obs_goals = np.vstack([obstacles, start_pos]) if obstacles.size > 0 else start_pos
    goals = create_goals(board_dims, num_agents, temp_obs_goals)

    env = GraphEnv(config, goal=goals, obstacles=obstacles, starting_positions=start_pos,
                   sensing_range=config.get("sensing_range", 4), pad=config.get("pad", 3))

    if args.seed is not None: obs, info = env.reset(seed=args.seed)
    else: obs, info = env.reset()
    print("Environment reset.")
except Exception as e: print(f"Error setting up environment: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

# --- Simulation and Video Frame Writing ---
max_steps_sim = args.max_steps if args.max_steps is not None else config.get("max_steps", 60)
print(f"Starting simulation for max {max_steps_sim} steps...")

# Initialize video writer
video_writer = None
frame_count = 0
try:
    video_writer = imageio.get_writer(args.output_file, fps=args.fps, format='FFMPEG', codec='libx264')
    print(f"Initialized video writer for {args.output_file} at {args.fps} FPS.")
except Exception as e:
    print(f"\nError initializing video writer: {e}. Ensure ffmpeg is installed (`pip install imageio[ffmpeg]`)."); env.close(); sys.exit(1)

terminated = False
truncated = False
idle_action = 0

try: # Wrap simulation in try...finally to ensure writer is closed
    sim_pbar = tqdm(range(max_steps_sim), desc="Simulating Episode", unit="step")
    for step in sim_pbar:
        if terminated or truncated: break

        # 1. Render current state
        frame = None
        try:
            frame = env.render(mode='rgb_array')
            if frame is None: continue # Skip if render fails
        except Exception as e: print(f"\nError during env.render: {e}. Stopping video generation."); break

        # Append frame to video writer
        try: video_writer.append_data(frame); frame_count += 1
        except Exception as e: print(f"\nError appending frame to video writer: {e}"); break

        # 2. Prepare observation for model
        try:
            fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
            gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])
        except KeyError as e: print(f"Error: Missing key {e} in observation dict."); break

        # 3. Get action from model
        with torch.no_grad():
            if net_type == 'gnn': action_scores = model(fov, gso)
            else: action_scores = model(fov)
            proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()

        # 4. Apply Collision Shielding (Identical logic as create_gif.py)
        shielded_actions = proposed_actions.copy()
        current_pos_y = env.positionY.copy(); current_pos_x = env.positionX.copy()
        next_pos_y = current_pos_y.copy(); next_pos_x = current_pos_x.copy()
        needs_shielding = np.zeros(env.nb_agents, dtype=bool)
        active_mask = ~env.reached_goal
        # Calc proposed positions
        for agent_id in np.where(active_mask)[0]:
             act = proposed_actions[agent_id]; dy, dx = env.action_map_dy_dx.get(act, (0,0))
             next_pos_y[agent_id] += dy; next_pos_x[agent_id] += dx
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
        next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)
        # Obstacle Collisions
        if env.obstacles.size > 0:
            next_coords_active = np.stack([next_pos_y[active_mask], next_pos_x[active_mask]], axis=1)
            obs_coll_active_mask = np.any(np.all(next_coords_active[:, np.newaxis, :] == env.obstacles[np.newaxis, :, :], axis=2), axis=1)
            colliding_agent_indices = np.where(active_mask)[0][obs_coll_active_mask]
            shielded_actions[colliding_agent_indices] = idle_action
            needs_shielding[colliding_agent_indices] = True
            next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
            next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]
        # Agent-Agent Collisions
        check_agent_coll_mask = active_mask & (~needs_shielding)
        check_indices = np.where(check_agent_coll_mask)[0]
        if len(check_indices) > 1:
            next_coords_check = np.stack([next_pos_y[check_indices], next_pos_x[check_indices]], axis=1)
            current_coords_check = np.stack([current_pos_y[check_indices], current_pos_x[check_indices]], axis=1)
            unique_coords, unique_indices, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
            colliding_cell_indices = np.where(counts > 1)[0]
            vertex_collision_mask_rel = np.isin(unique_indices, colliding_cell_indices)
            vertex_collision_agents = check_indices[vertex_collision_mask_rel]
            swapping_collision_agents_list = []
            for i in range(len(check_indices)):
                 for j in range(i + 1, len(check_indices)):
                     agent_i_idx = check_indices[i]; agent_j_idx = check_indices[j]
                     if np.array_equal(next_coords_check[i], current_coords_check[j]) and np.array_equal(next_coords_check[j], current_coords_check[i]):
                         swapping_collision_agents_list.extend([agent_i_idx, agent_j_idx])
            swapping_collision_agents = np.unique(swapping_collision_agents_list)
            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents]))
            shielded_actions[agents_to_shield_idx] = idle_action
        # --- End Collision Shielding ---

        # 5. Step environment with shielded actions
        try:
            obs, reward, terminated, truncated, info = env.step(shielded_actions)
            truncated = truncated or (step >= max_steps_sim - 1)
        except Exception as e: print(f"\nError during env.step: {e}"); break

        sim_pbar.set_postfix({"Term": terminated, "Trunc": truncated, "AtGoal": info['agents_at_goal'].sum()})

    # --- End Simulation Loop ---
    sim_pbar.close()

    # Capture final frame
    if terminated or truncated:
        try:
            final_frame = env.render(mode='rgb_array')
            if final_frame is not None: video_writer.append_data(final_frame); frame_count += 1
        except Exception: pass

finally: # Ensure resources are closed
    if video_writer is not None:
        try: video_writer.close(); print(f"\nVideo writer closed. {frame_count} frames saved to {args.output_file}")
        except Exception as e: print(f"Error closing video writer: {e}")
    env.close()

if frame_count > 0: print("Video saved successfully.")
else: print("\nNo frames were captured or saved.")
print("Script finished.")