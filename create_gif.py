# File: create_gif.py
import sys
import os
import yaml
import argparse
import numpy as np
import torch
import imageio  # Library for creating GIFs

# --- Add necessary paths (same as example.py) ---
# Adjust these if your project structure is different
sys.path.append("configs")
sys.path.append("models")
sys.path.append(".") # Add current directory in case grid module is there

# --- Import environment and model components ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
except ImportError:
    print("Error: Could not import environment classes.")
    print("Ensure 'grid/env_graph_gridv1.py' exists and the current directory or relevant path is in sys.path")
    sys.exit(1)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate GIF visualization for MAPF using a trained model.")
parser.add_argument("--config", type=str, default="configs/config_gnn.yaml",
                    help="Path to the configuration file.")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the saved model (.pt) file.")
parser.add_argument("--output_gif", type=str, default="mapf_visualization.gif",
                    help="Filename for the output GIF.")
parser.add_argument("--seed", type=int, default=None,
                    help="Optional random seed for reproducible scenario generation.")
parser.add_argument("--gif_duration", type=float, default=0.2,
                    help="Duration (in seconds) for each frame in the GIF.")

args = parser.parse_args()

# --- Set Seed (Optional) ---
if args.seed is not None:
    print(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Add seeding for the environment itself if it supports it
    # env.reset(seed=args.seed) # Call this after env is created

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
        if msg_type == "message": # Assuming 'message' implies gcn msg type? Adjust if needed
            from models.framework_gnn_message import Network
        else: # Assuming default GNN if msg_type is None or different
            from models.framework_gnn import Network
    elif net_type == "baseline":
        from models.framework_baseline import Network
    else:
        raise ValueError(f"Unknown net_type in config: {net_type}")
except ImportError as e:
     print(f"Error importing model class: {e}")
     print("Ensure the model files exist in the 'models' directory and paths are correct.")
     sys.exit(1)
except ValueError as e:
    print(e)
    sys.exit(1)

# --- Load Model ---
model = Network(config)
model.to(config["device"])

print(f"Loading model from: {args.model_path}")
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found at: {args.model_path}. Please check the path.")

try:
    model.load_state_dict(torch.load(args.model_path, map_location=config["device"]))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    sys.exit(1)

# --- Setup Environment for a Single Episode ---
print("Setting up environment for one episode...")
# Ensure board_size is used correctly (expects list/tuple)
board_dims = config.get("board_size", [28, 28]) # Default if missing
if isinstance(board_dims, int): # Handle old config format
    board_dims = [board_dims, board_dims]

num_obstacles = config.get("obstacles", 8) # Get number of obstacles
obstacles = create_obstacles(board_dims, num_obstacles)

num_agents = config.get("num_agents", 5)
# Ensure goals are created avoiding obstacles
goals = create_goals(board_dims, num_agents, obstacles)

# Pass goals, obstacles directly. sensing_range is optional.
env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=config.get("sensing_range", 4))

# Seed the environment if a seed was provided
if args.seed is not None:
    obs, info = env.reset(seed=args.seed) # <<< CHANGED: Unpack reset tuple and pass seed
else:
    obs, info = env.reset() # <<< CHANGED: Unpack reset tuple

# emb = env.getEmbedding() # getEmbedding might be deprecated if embedding is part of obs
print("Environment reset.")

# --- Simulation and Frame Capture ---
frames = []
max_steps_sim = config.get("max_steps", 60) # Get max_steps from config
print(f"Starting simulation for max {max_steps_sim} steps...")

terminated = False
truncated = False

# Loop until terminated (all goals reached) or truncated (max steps)
while not terminated and not truncated:
    # 1. Render current state and store frame
    try:
        # Ensure render mode matches what your env provides
        frame = env.render(mode='rgb_array')
        if frame is None:
             print(f"Warning: env.render(mode='rgb_array') returned None at step {env.time}. Skipping frame.")
             print("Check your GraphEnv's render implementation.")
        else:
            frames.append(frame)
    except Exception as e:
        print(f"\nError during env.render(mode='rgb_array'): {e}")
        print("Ensure your GraphEnv class supports render(mode='rgb_array') and returns a NumPy array.")
        print("Stopping GIF generation.")
        # Clean up environment before exiting
        env.close()
        sys.exit(1)

    # 2. Prepare observation for model
    # Make sure 'fov' and 'adj_matrix' are keys in the observation dict returned by env
    if not isinstance(obs, dict):
         print(f"Error: Observation is not a dictionary at step {env.time}. Got type: {type(obs)}")
         env.close()
         sys.exit(1)
    if "fov" not in obs or "adj_matrix" not in obs:
        print(f"Error: Missing 'fov' or 'adj_matrix' in observation keys at step {env.time}. Keys: {obs.keys()}")
        env.close()
        sys.exit(1)

    fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
    # Assuming 'adj_matrix' is the key for the graph structure input
    gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])


    # 3. Get action from model
    with torch.no_grad():
        if net_type == "gnn":
            action_scores = model(fov, gso)
        elif net_type == "baseline":
            action_scores = model(fov)
        else:
             # This case should ideally be caught earlier, but added for safety
             raise ValueError(f"Unknown net_type during inference: {net_type}")

        action = action_scores.cpu().squeeze(0).numpy()
        action = np.argmax(action, axis=1) # Get action index per agent

    # 4. Step the environment
    # Assuming step does not require embedding if model doesn't provide it?
    # If step requires embedding, get it from obs: emb = obs['embeddings']
    try:
        # <<< CHANGED: Unpack step tuple correctly
        obs, reward, terminated, truncated, info = env.step(action)
    except TypeError as e:
        print(f"Error during env.step: {e}")
        print("Check if env.step() expects an embedding argument that wasn't provided.")
        # Example if embedding is needed:
        # current_embedding = torch.tensor(obs['embeddings']).float().to(config['device']) # Or however it's represented
        # obs, reward, terminated, truncated, info = env.step(action, current_embedding)
        env.close()
        sys.exit(1)


    # 5. Check for termination conditions (already done by unpacking step result)
    if terminated:
        print(f"\nAll agents reached their goal in {env.time} steps.")
        # Capture the final frame
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Warning: Could not render final 'terminated' frame: {e}")
        # Loop condition 'terminated' will be true, so loop exits

    if truncated:
        print(f"\nMax steps ({max_steps_sim} reached or env.time >= max_time).")
         # Capture the final frame
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                 frames.append(frame)
        except Exception as e:
            print(f"Warning: Could not render final 'truncated' frame: {e}")
        # Loop condition 'truncated' will be true, so loop exits

# --- Close Environment ---
env.close() # Important to close plot windows etc.

# --- Save Frames as GIF ---
if frames:
    print(f"\nSaving {len(frames)} frames to {args.output_gif}...")
    try:
        # Use 'duration' directly for imageio v2.9+
        # For older versions, you might need 'fps = 1 / args.gif_duration'
        imageio.mimsave(args.output_gif, frames, duration=args.gif_duration, loop=0) # loop=0 means infinite loop
        print("GIF saved successfully.")
    except Exception as e:
        print(f"\nError saving GIF: {e}")
        print("Ensure imageio is installed correctly (`pip install imageio imageio[ffmpeg]`).")
else:
    print("\nNo frames were captured. Cannot create GIF.")

print("Script finished.")