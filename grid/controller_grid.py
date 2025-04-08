# File: grid/controller_grid.py
# (Basic adaptation for testing with updated env)

import sys
import os
import numpy as np
import gymnasium as gym
import torch # Assuming GCNLayer uses torch

# Add project root to path if necessary
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Assuming GCNLayer might be used directly (though less common outside model frameworks)
    # If GCNLayer is part of your model, import the model instead.
    # from models.networks.gnn import GCNLayer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

import time

if __name__ == "__main__":
    print("--- Running Controller Grid Test ---")
    agents = 4
    bs = 16
    board_dims = [bs, bs]
    num_obstacles = 10
    sensing = 4
    pad_size = 3

    # --- Create a dummy config for the environment ---
    test_config = {
        "num_agents": agents,
        "board_size": board_dims,
        "max_time": 100, # Set a max time for the test episode
        "obstacles": num_obstacles, # Num obstacles to generate
        "sensing_range": sensing,
        "pad": pad_size,
        "min_time": 1, # Example value
        "device": "cpu" # Example value
    }

    # --- Generate a scenario ---
    obstacles = create_obstacles(board_dims, num_obstacles)
    starts = create_goals(board_dims, agents, obstacles)
    temp_obstacles_goals = np.vstack([obstacles, starts]) if obstacles.size > 0 else starts
    goals = create_goals(board_dims, agents, temp_obstacles_goals)

    # --- Initialize Environment ---
    try:
        env = GraphEnv(
            config=test_config,
            goal=goals,
            obstacles=obstacles,
            starting_positions=starts
        )
        env.render_mode = "human" # Render to screen for testing
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit(1)

    # --- Basic Simulation Loop ---
    num_test_steps = 50
    try:
        obs, info = env.reset(seed=123)
        start_time = time.time()

        for i in range(num_test_steps):
            # --- Get Actions (Random for this test) ---
            actions = env.action_space.sample() # Get random valid actions

            # --- GCN/Embedding Update (Example - adapt if needed) ---
            # If you were testing GCN directly, you'd do it here
            # gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0) # Add batch dim
            # node_features = ... # Get or create node features
            # embeddings_out = gcn_layer(node_features, gso)
            # For this simple test, we don't update embeddings

            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(actions)

            # Render
            env.render()

            if terminated or truncated:
                status = "Terminated (Success)" if terminated else "Truncated (Timeout)"
                print(f"\nEpisode finished at step {env.time}. Status: {status}")
                break

        end_time = time.time()
        print(f"\nSimulation of {env.time} steps finished.")
        print(f"Total time: {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close() # Close the environment window

    print("--- Controller Grid Test Finished ---")