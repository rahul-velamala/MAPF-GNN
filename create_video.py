# File: create_video.py
import sys
import os
import yaml
import argparse
import numpy as np
import torch
import imageio  # Still use imageio, but differently

# --- Add necessary paths ---
sys.path.append("configs")
sys.path.append("models")
sys.path.append(".")

# --- Import environment and model components ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
except ImportError:
    print("Error: Could not import environment classes.")
    print("Ensure 'grid/env_graph_gridv1.py' exists and the correct path is in sys.path")
    sys.exit(1)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate video visualization for MAPF using a trained model.")
parser.add_argument("--config", type=str, default="configs/config_gnn.yaml",
                    help="Path to the configuration file.")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the saved model (.pt) file.")
# <<< CHANGED: Output argument defaults to .mp4 >>>
parser.add_argument("--output_file", type=str, default="mapf_visualization.mp4",
                    help="Filename for the output video (e.g., .mp4).")
parser.add_argument("--seed", type=int, default=None,
                    help="Optional random seed for reproducible scenario generation.")
# <<< CHANGED: Argument for video speed (FPS) >>>
parser.add_argument("--fps", type=int, default=10,
                    help="Frames per second for the output video.")

args = parser.parse_args()

# --- Set Seed ---
if args.seed is not None:
    print(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# --- Load Configuration ---
try:
    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)
except FileNotFoundError:
    print(f"Error: Config file not found at {args.config}")
    sys.exit(1)

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config['device']}")

net_type = config["net_type"]
msg_type = config.get("msg_type", None)

# --- Dynamically Import Model Class ---
try:
    if net_type == "gnn":
        if msg_type == "message":
            from models.framework_gnn_message import Network
        else:
            from models.framework_gnn import Network
    elif net_type == "baseline":
        from models.framework_baseline import Network
    else:
        raise ValueError(f"Unknown net_type in config: {net_type}")
except ImportError as e:
     print(f"Error importing model class: {e}")
     sys.exit(1)
except ValueError as e:
    print(e)
    sys.exit(1)

# --- Load Model ---
model = Network(config)
model.to(config["device"])
print(f"Loading model from: {args.model_path}")
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found at: {args.model_path}.")
try:
    model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    sys.exit(1)

# --- Setup Environment ---
print("Setting up environment...")
board_dims = config.get("board_size", [28, 28])
if isinstance(board_dims, int): board_dims = [board_dims, board_dims]
num_obstacles = config.get("obstacles", 8)
obstacles = create_obstacles(board_dims, num_obstacles)
num_agents = config.get("num_agents", 5)
goals = create_goals(board_dims, num_agents, obstacles)
env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=config.get("sensing_range", 4))

if args.seed is not None:
    obs, info = env.reset(seed=args.seed)
else:
    obs, info = env.reset()
print("Environment reset.")

# --- Simulation and Video Frame Writing ---
max_steps_sim = config.get("max_steps", 60)
print(f"Starting simulation for max {max_steps_sim} steps...")

# <<< CHANGED: Initialize video writer >>>
try:
    # Get the writer object
    video_writer = imageio.get_writer(args.output_file, fps=args.fps, format='FFMPEG', codec='libx264') # Specify format and codec
    print(f"Initialized video writer for {args.output_file} at {args.fps} FPS.")
except Exception as e:
    print(f"\nError initializing video writer: {e}")
    print("Ensure ffmpeg is installed and accessible by imageio (`pip install imageio[ffmpeg]`).")
    env.close()
    sys.exit(1)


terminated = False
truncated = False
frame_count = 0

try: # Wrap simulation in try...finally to ensure writer is closed
    while not terminated and not truncated:
        # 1. Render current state
        frame = None # Initialize frame to None
        try:
            frame = env.render(mode='rgb_array')
            if frame is None:
                 print(f"Warning: env.render(mode='rgb_array') returned None at step {env.time}. Skipping frame.")
                 continue # Skip to next step if frame is None
        except Exception as e:
            print(f"\nError during env.render(mode='rgb_array'): {e}")
            print("Stopping video generation.")
            break # Exit the loop cleanly

        # <<< CHANGED: Append frame directly to video writer >>>
        try:
            video_writer.append_data(frame)
            frame_count += 1
        except Exception as e:
            print(f"\nError appending frame to video writer: {e}")
            # Decide whether to continue or stop
            break


        # 2. Prepare observation for model
        if not isinstance(obs, dict):
             print(f"Error: Observation is not a dictionary at step {env.time}. Got type: {type(obs)}")
             break
        if "fov" not in obs or "adj_matrix" not in obs:
            print(f"Error: Missing 'fov' or 'adj_matrix' in observation keys at step {env.time}. Keys: {obs.keys()}")
            break
        fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
        gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])

        # 3. Get action from model
        with torch.no_grad():
            if net_type == "gnn":
                action_scores = model(fov, gso)
            elif net_type == "baseline":
                action_scores = model(fov)
            else:
                 raise ValueError(f"Unknown net_type during inference: {net_type}")
            action = action_scores.cpu().squeeze(0).numpy()
            action = np.argmax(action, axis=1)

        # 4. Step the environment
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except TypeError as e:
            print(f"Error during env.step: {e}")
            break
        except Exception as e:
             print(f"Unexpected error during env.step: {e}")
             break

        # 5. Check termination conditions (implicit in loop condition now)
        if terminated:
            print(f"\nAll agents reached their goal in {env.time} steps.")
            # Render and append final frame if possible
            try:
                final_frame = env.render(mode='rgb_array')
                if final_frame is not None:
                    video_writer.append_data(final_frame)
                    frame_count += 1
            except Exception: pass # Ignore error on final render

        if truncated:
            print(f"\nMax steps ({max_steps_sim}) reached.")
            # Render and append final frame if possible
            try:
                final_frame = env.render(mode='rgb_array')
                if final_frame is not None:
                    video_writer.append_data(final_frame)
                    frame_count += 1
            except Exception: pass # Ignore error on final render

finally: # <<< CHANGED: Ensure resources are closed >>>
    # Close the video writer
    if 'video_writer' in locals() and video_writer is not None:
        try:
            video_writer.close()
            print(f"\nVideo writer closed. {frame_count} frames saved to {args.output_file}")
        except Exception as e:
            print(f"Error closing video writer: {e}")

    # Close the environment
    env.close()

if frame_count > 0:
    print("Video saved successfully.")
else:
    print("\nNo frames were captured or saved to video.")

print("Script finished.")