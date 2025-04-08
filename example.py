# File: example.py
# (Modified with Collision Shielding)

import sys
import os
import yaml
import argparse
import numpy as np
import torch

# --- Assuming these imports work ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Import Network class based on config net_type
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)
# --- ----------------------------- ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml")
    # Add argument to load a specific model checkpoint if needed
    parser.add_argument("--model_path", type=str, default=None, help="Path to specific model.pt file (overrides config exp_name)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes to run") # Changed from tests_episodes
    args = parser.parse_args()

    # --- Load Config ---
    try:
        with open(args.config, "r") as config_path:
            config = yaml.load(config_path, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error loading config {args.config}: {e}")
        sys.exit(1)

    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config['device']}")

    exp_name = config["exp_name"].replace('\\', '/')
    num_test_episodes = args.episodes
    net_type = config["net_type"]

    # --- Dynamically Import Model ---
    try:
        if net_type == "baseline":
            from models.framework_baseline import Network # Ensure this is also updated for 3 channels if used
        elif net_type == "gnn":
            from models.framework_gnn import Network # Assumes this is updated
        else:
            raise ValueError(f"Unknown net_type: {net_type}")
        print(f"Using network type: {net_type}")
    except Exception as e:
        print(f"Error importing model: {e}")
        sys.exit(1)
    # --- ------------------------ ---

    # --- Load Model ---
    model = Network(config)
    model.to(config["device"])

    if args.model_path:
         model_load_path = args.model_path
    else:
         # Default to loading 'model_best.pt' from the experiment directory specified in the config
         model_load_path = os.path.join("results", exp_name, "model_best.pt")
         if not os.path.exists(model_load_path):
             # Fallback to model_final.pt if best doesn't exist
             model_load_path = os.path.join("results", exp_name, "model_final.pt")

    print(f"Loading model from: {model_load_path}")
    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model file not found at: {model_load_path}. Check config's exp_name or provide --model_path.")

    try:
        model.load_state_dict(torch.load(model_load_path, map_location=config["device"]))
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)
    # --- ---------- ---

    # --- Evaluation Loop ---
    all_ep_success_flags = []
    all_ep_steps_taken = []
    max_steps_per_episode = config.get("max_steps", 60) # Max steps for evaluation episodes

    print(f"\n--- Running {num_test_episodes} Test Episodes ---")
    for episode in range(num_test_episodes):
        print(f"\n-- Episode {episode+1}/{num_test_episodes} --")
        # --- Create Environment Instance ---
        try:
            board_dims_eval = config.get("board_size", [16, 16])
            obstacles_count_eval = config.get("obstacles", 6)
            agents_count_eval = config.get("num_agents", 4)
            sensing_range_eval = config.get("sensing_range", 4)
            pad_eval = config.get("pad", 3)

            obstacles_eval = create_obstacles(board_dims_eval, obstacles_count_eval)
            start_pos_eval = create_goals(board_dims_eval, agents_count_eval, obstacles_eval)
            temp_obs_goals = np.vstack([obstacles_eval, start_pos_eval]) if obstacles_eval.size > 0 else start_pos_eval
            goals_eval = create_goals(board_dims_eval, agents_count_eval, temp_obs_goals)

            env = GraphEnv(config, goal=goals_eval, obstacles=obstacles_eval,
                           starting_positions=start_pos_eval,
                           sensing_range=sensing_range_eval, pad=pad_eval)

            obs, info = env.reset()
        except Exception as e:
             print(f"Error creating environment for episode {episode+1}: {e}")
             all_ep_success_flags.append(False)
             all_ep_steps_taken.append(max_steps_per_episode)
             continue # Skip to next episode

        terminated = False
        truncated = False
        idle_action = 0 # Assuming 0 is idle

        # --- Simulation Loop with Collision Shielding ---
        for step in range(max_steps_per_episode + 1): # Allow one extra check
            if terminated or truncated: break # Exit if already done

            # Prepare model inputs
            fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
            gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])

            # Get action from model
            with torch.no_grad():
                if net_type == 'gnn':
                    action_scores = model(fov, gso)
                else: # baseline
                    action_scores = model(fov)
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()

            # --- Apply Collision Shielding ---
            shielded_actions = proposed_actions.copy()
            current_pos_y = env.positionY.copy()
            current_pos_x = env.positionX.copy()
            next_pos_y = current_pos_y.copy()
            next_pos_x = current_pos_x.copy()
            needs_shielding = np.zeros(env.nb_agents, dtype=bool)
            active_mask = ~env.reached_goal

            # 1. Calc proposed positions for active agents
            for agent_id in np.where(active_mask)[0]:
                 act = proposed_actions[agent_id]
                 dy, dx = env.action_map_dy_dx.get(act, (0,0))
                 next_pos_y[agent_id] += dy
                 next_pos_x[agent_id] += dx
            next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
            next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)

            # 2. Check Obstacle Collisions
            if env.obstacles.size > 0:
                next_coords_active = np.stack([next_pos_y[active_mask], next_pos_x[active_mask]], axis=1)
                obs_coll_active_mask = np.any(np.all(next_coords_active[:, np.newaxis, :] == env.obstacles[np.newaxis, :, :], axis=2), axis=1)
                colliding_agent_indices = np.where(active_mask)[0][obs_coll_active_mask]
                shielded_actions[colliding_agent_indices] = idle_action
                needs_shielding[colliding_agent_indices] = True
                next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
                next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]

            # 3. Check Agent-Agent Collisions
            check_agent_coll_mask = active_mask & (~needs_shielding)
            check_indices = np.where(check_agent_coll_mask)[0]
            if len(check_indices) > 1:
                next_coords_check = np.stack([next_pos_y[check_indices], next_pos_x[check_indices]], axis=1)
                current_coords_check = np.stack([current_pos_y[check_indices], current_pos_x[check_indices]], axis=1)
                # Vertex
                unique_coords, unique_indices, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
                colliding_cell_indices = np.where(counts > 1)[0]
                vertex_collision_mask_rel = np.isin(unique_indices, colliding_cell_indices)
                vertex_collision_agents = check_indices[vertex_collision_mask_rel]
                # Edge
                swapping_collision_agents_list = []
                for i in range(len(check_indices)):
                     for j in range(i + 1, len(check_indices)):
                         agent_i_idx = check_indices[i]; agent_j_idx = check_indices[j]
                         if np.array_equal(next_coords_check[i], current_coords_check[j]) and np.array_equal(next_coords_check[j], current_coords_check[i]):
                             swapping_collision_agents_list.extend([agent_i_idx, agent_j_idx])
                swapping_collision_agents = np.unique(swapping_collision_agents_list)
                # Shield
                agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents]))
                shielded_actions[agents_to_shield_idx] = idle_action
                needs_shielding[agents_to_shield_idx] = True
            # --- End Collision Shielding ---

            # Step environment with shielded actions
            obs, reward, terminated, truncated, info = env.step(shielded_actions)
            truncated = truncated or (step >= max_steps_per_episode) # Explicit truncation check

            # Optional: Render
            # env.render(mode="human", printNeigh=False)

        # --- End of Simulation Loop ---
        success = terminated and not truncated
        steps = env.time
        all_ep_success_flags.append(success)
        all_ep_steps_taken.append(steps if success else max_steps_per_episode) # Use actual steps if success

        print(f"Episode {episode+1} Result: {'Success' if success else 'Failure (Timeout)'} in {steps} steps.")
        env.close() # Close environment figure if open
    # --- End Evaluation Loop ---

    # --- Aggregate and Print Results ---
    success_rate = np.mean(all_ep_success_flags)
    steps_array = np.array(all_ep_steps_taken)
    avg_steps_success = np.mean(steps_array[np.array(all_ep_success_flags)]) if np.any(all_ep_success_flags) else np.nan

    print(f"\n--- Overall Test Results ({num_test_episodes} episodes) ---")
    print(f"Success Rate (Episodes): {success_rate:.4f} ({sum(all_ep_success_flags)}/{num_test_episodes})")
    if not np.isnan(avg_steps_success):
        print(f"Average Steps (Successful Episodes): {avg_steps_success:.2f}")
    else:
        print("Average Steps (Successful Episodes): N/A (No successful episodes)")
    print("--- ------------------------------------ ---")