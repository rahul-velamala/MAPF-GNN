# File: create_video.py
# (Modified with Collision Shielding and Robust Setup)

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import imageio # Still use imageio for video writing via ffmpeg
from tqdm import tqdm # Add progress bar for simulation
from pathlib import Path # Use Pathlib

# --- Add necessary paths if running from root ---
# (Assuming standard project structure)
# --- -------------------------------------- ---

# --- Import environment and model components ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Network class dynamically imported later
except ImportError:
    print("Error: Could not import environment classes.")
    print("Ensure 'grid/env_graph_gridv1.py' exists and you're running from the project root.")
    sys.exit(1)
# --- --------------------------------------- ---

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate video visualization for MAPF using a trained model with Collision Shielding.")
parser.add_argument("--config", type=str, default="configs/config_gnn.yaml", help="Path to the configuration file used for training.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt) file.")
parser.add_argument("--output_file", type=str, default="mapf_visualization.mp4", help="Filename for the output video (e.g., .mp4).")
parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible scenario generation.")
parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
parser.add_argument("--max_steps", type=int, default=None, help="Override max simulation steps from config.")
args = parser.parse_args()
# --- --------------- ---

# --- Set Seed ---
if args.seed is not None:
    print(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
# --- -------- ---

# --- Load Configuration ---
config_path = Path(args.config)
if not config_path.is_file():
    print(f"ERROR: Config file not found at {config_path}")
    sys.exit(1)
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None: raise ValueError("Config file is empty or invalid.")
except Exception as e:
    print(f"Error loading config {config_path}: {e}")
    sys.exit(1)

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config['device']}")
# --- ------------------ ---

# --- Dynamically Import Model Class ---
NetworkClass = None
try:
    net_type = config.get("net_type", "gnn") # Default to gnn
    if net_type == "baseline":
        from models.framework_baseline import Network as NetworkClass
    elif net_type == "gnn":
        msg_type = config.get("msg_type", "gcn").lower()
        if msg_type == 'message':
             from models.framework_gnn_message import Network as NetworkClass
        else: # Default GNN is GCN
             from models.framework_gnn import Network as NetworkClass
    else:
        raise ValueError(f"Unknown net_type in config: {net_type}")
    print(f"Using network type: {net_type}" + (f" ({msg_type})" if net_type == "gnn" else ""))
except Exception as e:
     print(f"Error importing model class: {e}")
     sys.exit(1)
# --- ------------------------------ ---

# --- Load Model ---
model = NetworkClass(config)
model.to(config["device"])
model_load_path = Path(args.model_path)
print(f"Loading model from: {model_load_path}")
if not model_load_path.is_file():
    raise FileNotFoundError(f"Model file not found at: {model_load_path}.")
try:
    model.load_state_dict(torch.load(model_load_path, map_location=config["device"]))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    sys.exit(1)
# --- ---------- ---

# --- Setup Environment ---
print("Setting up environment for one episode...")
env = None
try:
    # Use parameters from config to generate a scenario
    board_dims = config.get("board_size", [28, 28])
    if isinstance(board_dims, int): board_dims = [board_dims, board_dims]
    if not isinstance(board_dims, (list, tuple)) or len(board_dims) != 2:
         raise ValueError("Invalid 'board_size' in config.")

    num_obstacles = config.get("obstacles", 8)
    obstacles = create_obstacles(board_dims, num_obstacles)

    num_agents = config.get("num_agents", 5)
    if hasattr(model, 'num_agents') and num_agents != model.num_agents:
         print(f"Warning: Config num_agents ({num_agents}) differs from model num_agents ({model.num_agents}). Using model's count.")
         num_agents = model.num_agents

    start_pos = create_goals(board_dims, num_agents, obstacles=obstacles)
    temp_obstacles_for_goals = np.vstack([obstacles, start_pos]) if obstacles.size > 0 else start_pos
    goals = create_goals(board_dims, num_agents, obstacles=obstacles, current_starts=start_pos)

    env_config_instance = config.copy()
    env_config_instance["num_agents"] = num_agents
    env_config_instance["board_size"] = board_dims
    env_config_instance["pad"] = config.get("pad", 3)
    max_steps_sim = args.max_steps if args.max_steps is not None else config.get("max_steps", config.get("max_time", 120))
    env_config_instance["max_time"] = max_steps_sim

    env = GraphEnv(config=env_config_instance,
                   goal=goals,
                   obstacles=obstacles,
                   starting_positions=start_pos)

    obs, info = env.reset(seed=args.seed)
    print("Environment reset.")

except Exception as e:
     print(f"Error setting up environment: {e}")
     import traceback; traceback.print_exc()
     if env: env.close()
     sys.exit(1)
# --- ----------------- ---

# --- Simulation and Video Frame Writing ---
# Use max_steps_sim determined earlier
print(f"Starting simulation for max {max_steps_sim} steps...")
output_video_path = Path(args.output_file)

# Initialize video writer
video_writer = None
frame_count = 0
try:
    # Specify codec, pixel format may be needed depending on player compatibility
    # Common pixel format for broad compatibility: 'yuv420p'
    # Quality/bitrate can also be specified via ffmpeg_params
    video_writer = imageio.get_writer(output_video_path, fps=args.fps, format='FFMPEG',
                                       codec='libx264', quality=8, pixelformat='yuv420p')
    print(f"Initialized video writer for {output_video_path} at {args.fps} FPS.")
except ImportError:
    print("\nError: `imageio` library not found. Cannot save video.")
    print("Install using: pip install imageio")
    if env: env.close();
    sys.exit(1)
except Exception as e:
    print(f"\nError initializing video writer: {e}.")
    print("Ensure ffmpeg is installed and accessible to imageio.")
    print("Try: pip install imageio[ffmpeg]")
    if env: env.close();
    sys.exit(1)


terminated = False
truncated = False
idle_action = 0
simulation_error = False

try: # Wrap simulation in try...finally to ensure writer is closed
    sim_pbar = tqdm(range(max_steps_sim), desc="Simulating Episode", unit="step")
    for step in sim_pbar:
        if terminated or truncated: break

        # 1. Render current state
        frame = None
        try:
            frame = env.render(mode='rgb_array')
            if frame is None:
                print(f"Warning: env.render returned None at step {env.time}. Skipping frame.")
                continue # Skip if render fails
        except Exception as e:
            print(f"\nError during env.render at step {env.time}: {e}. Stopping video generation.")
            simulation_error = True; break

        # Append frame to video writer
        try:
            video_writer.append_data(frame)
            frame_count += 1
        except Exception as e:
            print(f"\nError appending frame {frame_count} to video writer: {e}. Stopping.")
            simulation_error = True; break

        # 2. Prepare observation for model
        try:
            current_fov_np = obs["fov"]
            current_gso_np = obs["adj_matrix"]
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(config["device"])
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(config["device"])
        except KeyError as e:
            print(f"Error: Missing key {e} in observation dict at step {env.time}. Keys: {obs.keys()}")
            simulation_error = True; break
        except Exception as e:
             print(f"Error processing observation: {e}")
             simulation_error = True; break

        # 3. Get action from model
        with torch.no_grad():
            try:
                if net_type == 'gnn':
                    action_scores = model(fov_tensor, gso_tensor)
                else: # baseline
                    action_scores = model(fov_tensor)
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
            except Exception as e:
                 print(f"Error during model forward pass: {e}")
                 simulation_error = True; break

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
            active_indices = np.where(active_mask)[0]
            if len(active_indices) > 0:
                 proposed_coords_active=np.stack([next_pos_y[active_indices],next_pos_x[active_indices]],axis=1)
                 obs_coll_active_mask=np.any(np.all(proposed_coords_active[:,np.newaxis,:] == env.obstacles[np.newaxis,:,:],axis=2),axis=1)
                 colliding_agent_indices = active_indices[obs_coll_active_mask]
                 if colliding_agent_indices.size > 0:
                      shielded_actions[colliding_agent_indices] = idle_action
                      needs_shielding[colliding_agent_indices] = True
                      next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
                      next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]
                      active_mask[colliding_agent_indices] = False
        # Agent-Agent Collisions
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
            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents]))
            if agents_to_shield_idx.size > 0:
                 shielded_actions[agents_to_shield_idx] = idle_action
        # --- End Collision Shielding ---

        # 5. Step environment with shielded actions
        try:
            obs, reward, terminated, truncated, info = env.step(shielded_actions)
            truncated = truncated or (step >= max_steps_sim - 1)
        except Exception as e:
            print(f"\nError during env.step: {e}")
            simulation_error = True; break

        sim_pbar.set_postfix({"Term": terminated, "Trunc": truncated, "AtGoal": info['agents_at_goal'].sum(), "Step": env.time})

    # --- End Simulation Loop ---
    sim_pbar.close()

    # Capture final frame if simulation didn't end in error
    if not simulation_error and (terminated or truncated):
        try:
            final_frame = env.render(mode='rgb_array')
            if final_frame is not None:
                 video_writer.append_data(final_frame)
                 frame_count += 1
        except Exception as e:
             print(f"Warning: Could not render/append final frame: {e}")

finally: # Ensure resources are closed
    if video_writer is not None:
        try:
            video_writer.close()
            print(f"\nVideo writer closed. {frame_count} frames saved to {output_video_path.resolve()}")
        except Exception as e:
            print(f"Error closing video writer: {e}")
    if env is not None:
        env.close()

if frame_count > 0 and not simulation_error:
    print("Video saved successfully.")
elif simulation_error:
    print("\nVideo generation stopped due to simulation error. Output might be incomplete or corrupt.")
else:
    print("\nNo frames were captured or saved.")
print("Script finished.")