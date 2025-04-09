# File: example.py
# (Modified for Robust Model Loading, Collision Shielding during Eval)

import sys
import os
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path # Use Pathlib

# --- Assuming these imports work when running from project root ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Network class is dynamically imported based on config later
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running python from the main project directory.")
    sys.exit(1)
# --- ----------------------------- ---

def run_evaluation_episode(model, env, max_steps_eval, device, net_type):
    """
    Runs a single evaluation episode with the model and collision shielding.
    (Logic adapted from run_inference_with_shielding in train.py)

    Returns:
        tuple: (is_success, steps_taken)
    """
    model.eval() # Ensure model is in evaluation mode
    try:
        obs, info = env.reset() # Reset env for this episode
    except Exception as e:
         print(f"\nError resetting environment for evaluation episode: {e}")
         return False, max_steps_eval # Count as failure with max steps

    terminated = False
    truncated = False
    idle_action = 0 # Assuming 0 is the idle action index in GraphEnv
    steps_taken = 0

    while not terminated and not truncated and steps_taken < max_steps_eval:
        # Prepare observation for model
        try:
            current_fov_np = obs["fov"]
            current_gso_np = obs["adj_matrix"]
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device)
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device)
        except Exception as e:
             print(f"\nError processing observation at step {env.time}: {e}")
             return False, max_steps_eval # Treat as failure

        # Get action from model
        with torch.no_grad():
            try:
                if net_type == 'gnn':
                    action_scores = model(fov_tensor, gso_tensor)
                else: # baseline
                    action_scores = model(fov_tensor)
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
            except Exception as e:
                 print(f"\nError during model forward pass at step {env.time}: {e}")
                 return False, max_steps_eval # Treat as failure

        # --- Apply Collision Shielding ---
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
             dy, dx = env.action_map_dy_dx.get(act, (0,0)) # Default to idle
             next_pos_y[agent_id] += dy
             next_pos_x[agent_id] += dx

        # 2. Clamp proposed positions to boundaries
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
        next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)

        # 3. Check Obstacle Collisions
        if env.obstacles.size > 0:
            active_indices = np.where(active_mask)[0]
            if len(active_indices) > 0:
                proposed_coords_active = np.stack([next_pos_y[active_indices], next_pos_x[active_indices]], axis=1)
                obs_coll_active_mask = np.any(np.all(proposed_coords_active[:, np.newaxis, :] == env.obstacles[np.newaxis, :, :], axis=2), axis=1)
                colliding_agent_indices = active_indices[obs_coll_active_mask]
                if colliding_agent_indices.size > 0:
                    shielded_actions[colliding_agent_indices] = idle_action
                    needs_shielding[colliding_agent_indices] = True
                    next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
                    next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]
                    active_mask[colliding_agent_indices] = False

        # 4. Check Agent-Agent Collisions
        active_indices = np.where(active_mask)[0]
        if len(active_indices) > 1:
            next_coords_check = np.stack([next_pos_y[active_indices], next_pos_x[active_indices]], axis=1)
            current_coords_check = np.stack([current_pos_y[active_indices], current_pos_x[active_indices]], axis=1)

            unique_coords, unique_map_indices, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
            colliding_cell_indices = np.where(counts > 1)[0]
            vertex_collision_mask_rel = np.isin(unique_map_indices, colliding_cell_indices)
            vertex_collision_agents = active_indices[vertex_collision_mask_rel]

            swapping_collision_agents_list = []
            relative_indices = np.arange(len(active_indices))
            for i_rel in relative_indices:
                 for j_rel in range(i_rel + 1, len(active_indices)):
                     if np.array_equal(next_coords_check[i_rel], current_coords_check[j_rel]) and \
                        np.array_equal(next_coords_check[j_rel], current_coords_check[i_rel]):
                         swapping_collision_agents_list.extend([active_indices[i_rel], active_indices[j_rel]])
            swapping_collision_agents = np.unique(swapping_collision_agents_list)

            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents]))

            if agents_to_shield_idx.size > 0:
                shielded_actions[agents_to_shield_idx] = idle_action
                # needs_shielding[agents_to_shield_idx] = True # Mark shielded if needed later
        # --- End Collision Shielding ---

        # Step environment with shielded actions
        try:
            obs, reward, terminated, truncated, info = env.step(shielded_actions)
            steps_taken = env.time # Get current env time
            # Explicitly check truncation based on max_steps_eval
            truncated = truncated or (steps_taken >= max_steps_eval)
        except Exception as e:
            print(f"\nError during env.step at step {env.time}: {e}")
            return False, max_steps_eval # Treat as failure

        # Optional: Render during evaluation
        # env.render()

    # --- After loop ---
    is_success = terminated and not truncated
    final_steps = env.time # Use the environment's final time

    return is_success, final_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MAPF model.")
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml", help="Path to the YAML configuration file used for training.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a specific model.pt file (overrides config's exp_name).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes to run.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps per episode from config.")
    parser.add_argument("--seed", type=int, default=None, help="Set random seed for numpy and torch for reproducibility.")
    args = parser.parse_args()

    # --- Set Seed ---
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    # --- -------- ---

    # --- Load Config ---
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
    # --- ----------- ---

    # --- Identify Model Type and Import ---
    NetworkClass = None
    try:
        net_type = config.get("net_type", "gnn") # Default to gnn if missing
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
    except (ImportError, ValueError, KeyError) as e:
        print(f"Error importing model based on config: {e}")
        sys.exit(1)
    # --- ----------------------------- ---

    # --- Load Model ---
    model = NetworkClass(config)
    model.to(config["device"])
    model.eval() # Set to evaluation mode

    model_load_path = None
    if args.model_path:
         model_load_path_arg = Path(args.model_path)
         if model_load_path_arg.is_file():
              model_load_path = model_load_path_arg
         else:
              print(f"Warning: --model_path '{args.model_path}' not found. Trying default locations.")

    if model_load_path is None:
        # Default to loading from the experiment directory specified in the config
        exp_name = config.get("exp_name", "default_experiment").replace('\\', '/')
        results_dir = Path("results") / exp_name
        model_best_path = results_dir / "model_best.pt"
        model_final_path = results_dir / "model_final.pt"

        if model_best_path.is_file():
            model_load_path = model_best_path
        elif model_final_path.is_file():
            print("Warning: model_best.pt not found. Loading model_final.pt instead.")
            model_load_path = model_final_path
        else:
            # Try older potential location structure
            old_model_path = Path("trained_models") / exp_name / "model.pt"
            if old_model_path.is_file():
                 model_load_path = old_model_path
            else:
                 print(f"ERROR: Model file not found. Searched:")
                 print(f"  1. --model_path argument (if provided)")
                 print(f"  2. {model_best_path}")
                 print(f"  3. {model_final_path}")
                 print(f"  4. {old_model_path} (legacy)")
                 print(f"Please check config's 'exp_name' or provide a valid --model_path.")
                 sys.exit(1)

    print(f"Loading model from: {model_load_path}")
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=config["device"]))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict from {model_load_path}: {e}")
        sys.exit(1)
    # --- ---------- ---

    # --- Evaluation Loop ---
    num_test_episodes = args.episodes
    # Determine max steps per episode for evaluation runs
    if args.max_steps is not None:
         max_steps_per_episode = args.max_steps
         print(f"Using max steps override: {max_steps_per_episode}")
    else:
         # Use evaluation-specific max_steps if available, else general max_steps/max_time
         max_steps_per_episode = config.get("eval_max_steps", config.get("max_steps", config.get("max_time", 60)))
         print(f"Using max steps from config: {max_steps_per_episode}")


    all_ep_success_flags = []
    all_ep_steps_taken = []

    print(f"\n--- Running {num_test_episodes} Test Episodes ---")
    eval_pbar = tqdm(range(num_test_episodes), desc="Evaluating", unit="episode")

    for episode in eval_pbar:
        env = None # Ensure env is defined for finally
        try:
            # --- Create Environment Instance for this episode ---
            # Use eval-specific parameters from config if they exist
            eval_board_dims = config.get("eval_board_size", config.get("board_size", [16, 16]))
            eval_obstacles_count = config.get("eval_obstacles", config.get("obstacles", 6))
            eval_agents_count = config.get("eval_num_agents", config.get("num_agents", 4))
            # Ensure env agent count matches loaded model
            if eval_agents_count != model.num_agents:
                 print(f"Warning: Eval agent count ({eval_agents_count}) in config differs from model agent count ({model.num_agents}). Using model's count ({model.num_agents}) for env.")
                 eval_agents_count = model.num_agents
            eval_sensing_range = config.get("eval_sensing_range", config.get("sensing_range", 4))
            eval_pad = config.get("eval_pad", config.get("pad", 3)) # Should match model's FOV

            # Generate random scenario
            eval_obstacles_ep = create_obstacles(eval_board_dims, eval_obstacles_count)
            eval_start_pos_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep)
            eval_goals_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep, current_starts=eval_start_pos_ep)

            # Create a config dict specifically for this env instance
            env_config_instance = config.copy()
            env_config_instance.update({
                 "board_size": eval_board_dims,
                 "num_agents": eval_agents_count,
                 "sensing_range": eval_sensing_range,
                 "pad": eval_pad,
                 "max_time": max_steps_per_episode # Env's internal max_time matches eval steps
            })

            env = GraphEnv(config=env_config_instance,
                           goal=eval_goals_ep, obstacles=eval_obstacles_ep,
                           starting_positions=eval_start_pos_ep)

            # --- Run the episode using the helper function ---
            is_success, steps = run_evaluation_episode(
                model, env, max_steps_per_episode, config["device"], net_type
            )

            all_ep_success_flags.append(is_success)
            all_ep_steps_taken.append(steps) # Store actual steps taken
            eval_pbar.set_postfix({"LastResult": "Success" if is_success else "Failure", "Steps": steps})

        except Exception as e:
             print(f"\nError creating/running environment for episode {episode+1}: {e}")
             traceback.print_exc()
             all_ep_success_flags.append(False) # Count as failure
             all_ep_steps_taken.append(max_steps_per_episode)
             # Continue to next episode
        finally:
             if env: env.close() # Close environment figure if open
    # --- End Evaluation Loop ---
    eval_pbar.close()

    # --- Aggregate and Print Results ---
    success_rate = np.mean(all_ep_success_flags) if all_ep_success_flags else 0.0
    steps_array = np.array(all_ep_steps_taken)
    successful_mask = np.array(all_ep_success_flags)

    if np.any(successful_mask):
        avg_steps_success = np.mean(steps_array[successful_mask])
        std_steps_success = np.std(steps_array[successful_mask])
    else:
        avg_steps_success = np.nan
        std_steps_success = np.nan

    num_successful = sum(all_ep_success_flags)
    num_failed = num_test_episodes - num_successful

    print(f"\n\n--- Overall Test Results ({num_test_episodes} episodes) ---")
    print(f"Success Rate: {success_rate:.4f} ({num_successful}/{num_test_episodes})")
    print(f"Failure Rate: {1.0 - success_rate:.4f} ({num_failed}/{num_test_episodes})")
    if not np.isnan(avg_steps_success):
        print(f"Average Steps (Successful Episodes): {avg_steps_success:.2f} (StdDev: {std_steps_success:.2f})")
    else:
        print("Average Steps (Successful Episodes): N/A (No successful episodes)")
    # Optionally, print average steps for failed episodes
    if num_failed > 0:
         avg_steps_fail = np.mean(steps_array[~successful_mask])
         print(f"Average Steps (Failed Episodes): {avg_steps_fail:.2f}")
    print("-" * 40)