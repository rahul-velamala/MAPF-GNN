# File: create_gif.py
# (Modified with REDUNDANT Collision Shielding Removed)

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

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---


# --- Add necessary paths if running from root ---
# (Assuming standard project structure - adjust if needed)
# script_dir = Path(__file__).parent.resolve()
# project_root = script_dir.parent
# sys.path.append(str(project_root))
# --- -------------------------------------- ---


# --- Import environment and model components ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Network class dynamically imported later
except ImportError:
    logger.error("Error: Could not import environment classes.")
    logger.error("Ensure 'grid/env_graph_gridv1.py' exists and necessary paths are set.")
    sys.exit(1)
# --- --------------------------------------- ---


# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate GIF visualization for MAPF using a trained model.")
parser.add_argument("--config", type=str, default="configs/config_gnn.yaml", help="Path to the configuration file used for training.") # May need config_train_mat.yaml
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt) file.")
parser.add_argument("--output_gif", type=str, default="mapf_visualization.gif", help="Filename for the output GIF.")
parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible scenario generation.")
parser.add_argument("--gif_duration", type=float, default=0.2, help="Duration (in seconds) for each frame in the GIF.")
parser.add_argument("--max_steps", type=int, default=None, help="Override max simulation steps from config.")
args = parser.parse_args()
# --- --------------- ---


# --- Set Seed ---
if args.seed is not None:
    logger.info(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
# --- -------- ---


# --- Load Configuration ---
config_path = Path(args.config)
if not config_path.is_file():
    logger.error(f"ERROR: Config file not found at {config_path}")
    sys.exit(1)
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None: raise ValueError("Config file is empty or invalid.")
except Exception as e:
    logger.error(f"Error loading config {config_path}: {e}")
    sys.exit(1)

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {config['device']}")
# --- ------------------ ---


# --- Dynamically Import Model Class ---
NetworkClass = None
try:
    net_type = config.get("net_type", "gnn") # Default to gnn
    logger.info(f"Attempting to load model type: {net_type}")
    if net_type == "baseline":
        from models.framework_baseline import Network as NetworkClass
        logger.info("Imported Baseline Network.")
    elif net_type == "gnn":
        # GNN model might have variations, check config further if needed
        msg_type = config.get("msg_type", "gcn").lower()
        # Assuming framework_gnn handles different msg_types internally based on config
        from models.framework_gnn import Network as NetworkClass
        logger.info(f"Imported GNN Network (config msg_type: {msg_type}).")
        # Older structure check (remove if framework_gnn handles types):
        # if msg_type == 'message':
        #      from models.framework_gnn_message import Network as NetworkClass
        # else: # Default GNN is GCN/ADC handled by framework_gnn
        #      from models.framework_gnn import Network as NetworkClass
    else:
        raise ValueError(f"Unknown net_type in config: {net_type}")
    if NetworkClass is None: raise ImportError("Network class was not assigned.")
    logger.info(f"Successfully imported NetworkClass: {NetworkClass.__name__}")

except Exception as e:
     logger.error(f"Error importing model class for net_type '{net_type}': {e}", exc_info=True)
     sys.exit(1)
# --- ------------------------------ ---


# --- Load Model ---
model = NetworkClass(config)
model.to(config["device"])
model_load_path = Path(args.model_path)
logger.info(f"Loading model from: {model_load_path}")
if not model_load_path.is_file():
    logger.error(f"Model file not found at: {model_load_path}.")
    sys.exit(1)
try:
    model.load_state_dict(torch.load(model_load_path, map_location=config["device"]))
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model state_dict: {e}", exc_info=True)
    sys.exit(1)
# --- ---------- ---


# --- Setup Environment for a Single Episode ---
logger.info("Setting up environment for one episode...")
env = None
try:
    # Use parameters from config to generate a scenario
    # Use eval_board_size if defined, otherwise board_size
    board_dims = config.get("eval_board_size", config.get("board_size", [20, 20]))
    if isinstance(board_dims, int): board_dims = [board_dims, board_dims]
    if not isinstance(board_dims, (list, tuple)) or len(board_dims) != 2:
         raise ValueError("Invalid 'board_size' or 'eval_board_size' in config.")

    # Use eval_obstacles if defined, otherwise obstacles
    num_obstacles = config.get("eval_obstacles", config.get("obstacles", 40)) # Match defaults
    obstacles = create_obstacles(tuple(board_dims), num_obstacles) # Use tuple for size

    # Use eval_num_agents if defined, otherwise num_agents
    num_agents = config.get("eval_num_agents", config.get("num_agents", 10)) # Match defaults
    # Ensure num_agents matches model if model size is fixed (check attribute existence)
    if hasattr(model, 'num_agents') and num_agents != model.num_agents:
         logger.warning(f"Config num_agents ({num_agents}) differs from model num_agents ({model.num_agents}). Using model's count: {model.num_agents}.")
         num_agents = model.num_agents

    # Generate valid starts and goals avoiding obstacles and each other
    start_pos = create_goals(tuple(board_dims), num_agents, obstacles=obstacles)
    # Avoid placing goals on obstacles OR start positions
    temp_obstacles_for_goals = np.vstack([obstacles, start_pos]) if obstacles.size > 0 else start_pos
    goals = create_goals(tuple(board_dims), num_agents, obstacles=temp_obstacles_for_goals) # Pass combined obstacles

    # Pass necessary config params to Env
    env_config_instance = config.copy()
    # Ensure env uses correct agent count, board size, pad etc.
    env_config_instance["num_agents"] = num_agents
    env_config_instance["board_size"] = board_dims
    # Use eval_pad if defined, otherwise pad (CRITICAL for FOV match)
    env_config_instance["pad"] = config.get("eval_pad", config.get("pad", 6)) # Match defaults
    # Use eval_sensing_range if defined, otherwise sensing_range
    env_config_instance["sensing_range"] = config.get("eval_sensing_range", config.get("sensing_range", 5)) # Match defaults

    # Set max_time for env based on args or config (prefer eval_max_steps)
    max_steps_sim = args.max_steps if args.max_steps is not None else config.get("eval_max_steps", config.get("max_steps", config.get("max_time", 150))) # Match defaults
    env_config_instance["max_time"] = max_steps_sim

    env_config_instance["render_mode"] = "rgb_array" # Force rgb_array for gif creation

    env = GraphEnv(config=env_config_instance,
                   goal=goals,
                   obstacles=obstacles,
                   starting_positions=start_pos)

    # Reset with seed if provided
    obs, info = env.reset(seed=args.seed)
    logger.info("Environment reset and ready.")
    logger.info(f"Simulating with Board: {env.board_rows}x{env.board_cols}, Agents: {env.nb_agents}, Pad: {env.pad}, MaxSteps: {max_steps_sim}")

except Exception as e:
     logger.error(f"Error setting up environment: {e}", exc_info=True)
     if env: env.close()
     sys.exit(1)
# --- ------------------------------------ ---


# --- Simulation and Frame Capture ---
frames = []
logger.info(f"Starting simulation for max {max_steps_sim} steps...")

terminated = False
truncated = False

sim_pbar = tqdm(range(max_steps_sim), desc="Simulating Episode", unit="step")

for step in sim_pbar:
    if terminated or truncated:
        break

    # 1. Render current state and store frame (BEFORE step)
    try:
        # Render using the environment's forced 'rgb_array' mode
        frame = env.render(mode='rgb_array')
        if frame is None:
             logger.warning(f"env.render returned None at step {env.time}. Skipping frame.")
        else:
            frames.append(frame)
    except Exception as e:
        logger.error(f"\nError during env.render at step {env.time}: {e}. Stopping GIF generation.", exc_info=True)
        if env: env.close()
        sys.exit(1)

    # 2. Prepare observation for model
    try:
        current_fov_np = obs["fov"]
        current_gso_np = obs["adj_matrix"]
        # Add batch dimension (B=1)
        fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(config["device"])
        gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(config["device"])
    except KeyError as e:
        logger.error(f"Error: Missing key {e} in observation dict at step {env.time}. Keys: {list(obs.keys())}")
        if env: env.close()
        sys.exit(1)
    except Exception as e:
         logger.error(f"Error processing observation at step {env.time}: {e}", exc_info=True)
         if env: env.close()
         sys.exit(1)

    # 3. Get action from model
    with torch.no_grad():
        try:
            if net_type == 'gnn':
                action_scores = model(fov_tensor, gso_tensor) # Shape (B, N, A)
            else: # baseline
                action_scores = model(fov_tensor) # Shape (B, N, A)
            # Get actions for the single batch item -> Shape (N,)
            proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
        except Exception as e:
             logger.error(f"Error during model forward pass at step {env.time}: {e}", exc_info=True)
             if env: env.close()
             sys.exit(1)

    # 4. --- NO EXTERNAL COLLISION SHIELDING HERE ---
    # The environment's step function will handle collisions.

    # 5. Step the environment *with raw model actions*
    try:
        # Use proposed_actions directly
        obs, reward, terminated, truncated_env, info = env.step(proposed_actions)
        # Update truncation flag also based on env time vs max_steps_sim
        truncated = truncated or truncated_env or (env.time >= max_steps_sim)
    except Exception as e:
        logger.error(f"\nError during env.step at time {env.time}: {e}", exc_info=True)
        if env: env.close()
        sys.exit(1)

    # Update progress bar description
    sim_pbar.set_postfix({
        "Term": terminated,
        "Trunc": truncated,
        "AtGoal": info.get('agents_at_goal', np.array([])).sum(), # Use .get for safety
        "Step": env.time
    })

# --- End Simulation Loop ---
sim_pbar.close()

# Capture the final frame if episode finished or truncated
if terminated or truncated:
    logger.info(f"Simulation ended. Terminated={terminated}, Truncated={truncated}, Final Step={env.time}")
    try:
        final_frame = env.render(mode='rgb_array')
        if final_frame is not None:
             frames.append(final_frame)
             logger.info("Appended final frame.")
        else:
             logger.warning("Could not render final frame (returned None).")
    except Exception as e:
        logger.warning(f"Error rendering final frame: {e}", exc_info=True)

if env:
    env.close()
    logger.info("Environment closed.")

# --- Save Frames as GIF ---
output_gif_path = Path(args.output_gif).resolve() # Get absolute path
if frames:
    logger.info(f"\nSaving {len(frames)} frames to {output_gif_path}...")
    try:
        # imageio v3 syntax: duration is per frame in milliseconds
        imageio.mimsave(output_gif_path, frames, duration=int(args.gif_duration * 1000), loop=0)
        # # imageio v2 syntax: duration is per frame in seconds
        # imageio.mimsave(output_gif_path, frames, duration=args.gif_duration, loop=0)
        logger.info(f"GIF saved successfully to {output_gif_path}")
    except ImportError:
         logger.error("\nError: `imageio` library not found. Cannot save GIF.")
         logger.error("Install using: pip install imageio")
    except Exception as e:
        logger.error(f"\nError saving GIF: {e}", exc_info=True)
        logger.error("Ensure imageio is installed (`pip install imageio`).")
        logger.error("You might need ffmpeg for some formats (`pip install imageio[ffmpeg]`).")
else:
    logger.warning("\nNo frames were captured. Cannot create GIF.")

logger.info("\nScript finished.")