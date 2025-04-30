# File: grid/controller_grid.py
# (Basic adaptation for testing with updated env_graph_gridv1)

import sys
import os
import numpy as np
import gymnasium as gym
import torch # Assuming GCNLayer might be used directly (though less common outside model frameworks)
import time
import logging

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # If GCNLayer is part of your model, import the model instead.
    # from models.networks.gnn import GCNLayer
except ImportError as e:
    logger.error(f"Error importing modules: {e}", exc_info=True)
    sys.exit(1)


if __name__ == "__main__":
    logger.info("--- Running Controller Grid Test ---")
    agents = 4
    bs = 16
    board_dims = [bs, bs]
    num_obstacles_to_gen = 10
    sensing_range_val = 4
    pad_val = 3 # For 5x5 FOV
    max_steps_test = 100

    # --- Create a config dictionary for the environment ---
    test_config = {
        "num_agents": agents,
        "board_size": board_dims,
        "max_time": max_steps_test, # Set a max time for the test episode
        "obstacles": num_obstacles_to_gen, # Num obstacles to generate
        "sensing_range": sensing_range_val,
        "pad": pad_val,
        "render_mode": "human", # Set render mode here
        "device": "cpu", # Add device if needed by other parts of config use
        "min_time": 1 # Example value if needed elsewhere
    }

    # --- Generate a scenario ---
    try:
        obstacles = create_obstacles(board_dims, num_obstacles_to_gen)
        starts = create_goals(board_dims, agents, obstacles)
        # Ensure goals avoid obstacles AND starts
        goals = create_goals(board_dims, agents, obstacles=obstacles, current_starts=starts)
    except Exception as e:
        logger.error(f"Error generating scenario: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize Environment ---
    env = None
    try:
        env = GraphEnv(
            config=test_config,
            goal=goals,
            obstacles=obstacles,
            starting_positions=starts
        )
    except Exception as e:
        logger.error(f"Error creating environment: {e}", exc_info=True)
        if env: env.close()
        sys.exit(1)

    # --- Basic Simulation Loop ---
    try:
        obs, info = env.reset(seed=123)
        logger.info(f"Environment Reset. Initial Positions: {info['positions'].tolist()}")
        if env.render_mode == "human": env.render() # Initial render
        time.sleep(1)
        start_time = time.time()
        total_reward = 0

        for i in range(max_steps_test):
            # --- Get Actions (Random for this test) ---
            actions = env.action_space.sample() # Get random valid actions

            # --- Embedding Update (Placeholder) ---
            # In a real scenario, embeddings might come from a model or be static
            # emb = model.get_embeddings(...)
            # env._updateEmbedding(emb) # If needed by env internally (current v1 doesn't use it in step)

            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward

            # Render (called within step if mode='human')

            print(f"\rStep: {info['time']}, Term: {terminated}, Trunc: {truncated}, Reward: {reward:.3f}, Reached: {np.sum(info['agents_at_goal'])}", end="")

            if terminated or truncated:
                status = "Terminated (Success)" if terminated else "Truncated (Timeout)"
                print(f"\nEpisode finished at step {info['time']}. Status: {status}, Total Reward: {total_reward:.3f}")
                break

        end_time = time.time()
        print(f"\nSimulation of {info['time']} steps finished.")
        print(f"Total time: {end_time - start_time:.2f}s")

    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
    finally:
        if env: env.close() # Close the environment window

    logger.info("--- Controller Grid Test Finished ---")

