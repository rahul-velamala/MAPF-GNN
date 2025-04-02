# File: example.py
import sys
import os # Import os for path joining

# MODIFIED: Removed absolute paths and used relative paths
# sys.path.append(r"configs") # Original
# sys.path.append(r"models") # Original
# These might not even be necessary if example.py is run from the project root
# If they are needed, use forward slashes or let python handle it:
# sys.path.append("configs")
# sys.path.append("models")
# Best practice: Ensure your project structure allows imports without sys.path manipulation
# e.g., run python -m retamalvictor-mapf-gnn.example from the directory *above* retamalvictor-mapf-gnn

sys.path.append("configs")
sys.path.append("models")
import yaml
import argparse
import numpy as np

import torch
from torch import nn
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MODIFIED: Default path uses forward slash (already correct)
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = config["exp_name"] # This might contain backslashes from the YAML, ensure YAML is fixed too!
    tests_episodes = config["tests_episodes"]
    net_type = config["net_type"]
    msg_type = config.get("msg_type", None) # Use .get for optional keys

    # Ensure exp_name uses forward slashes if it comes from config
    exp_name = exp_name.replace('\\', '/')

    if net_type == "gnn":
        if msg_type == "message":
            from models.framework_gnn_message import Network
        else:
            from models.framework_gnn import Network
    elif net_type == "baseline": # Use elif for clarity
        from models.framework_baseline import Network
    else:
        raise ValueError(f"Unknown net_type: {net_type}")


    success_rate = np.zeros((tests_episodes, 1))
    flow_time = np.zeros((tests_episodes, 1))
    all_goals = 0

    model = Network(config)
    model.to(config["device"])

    # MODIFIED: Construct path using os.path.join or forward slashes
    # model_path = rf"{exp_name}\model.pt" # Original Windows path
    model_path = os.path.join(exp_name, "model.pt") # Safer, platform-independent
    # Alternatively:
    # model_path = f"{exp_name}/model.pt" # Linux-style f-string

    print(f"Loading model from: {model_path}") # Add print for debugging
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Check config's exp_name and path separators.")

    model.load_state_dict(torch.load(model_path, map_location=config["device"])) # Added map_location
    model.eval()
    for episode in range(tests_episodes):
        obstacles = create_obstacles(config["board_size"], config["obstacles"])
        goals = create_goals(config["board_size"], config["num_agents"], obstacles)

        env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=config.get("sensing_range", 4)) # Added get for sensing range
        emb = env.getEmbedding()
        obs = env.reset()
        for i in range(config["max_steps"] + 10):
            fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
            gso = (
                torch.tensor(obs["adj_matrix"])
                .float()
                .unsqueeze(0)
                .to(config["device"])
            )
            with torch.no_grad():
                if net_type == "gnn":
                    action = model(fov, gso)
                elif net_type == "baseline":
                    action = model(fov)
                else:
                     raise ValueError(f"Unknown net_type during inference: {net_type}")

                action = action.cpu().squeeze(0).numpy()
            action = np.argmax(action, axis=1)

            obs, reward, done, info = env.step(action, emb)
            env.render(None)
            if done:
                print(f"Episode {episode+1}: All agents reached their goal in {env.time} steps.\n")
                all_goals += 1
                break
            if i == config["max_steps"] + 9:
                print(f"Episode {episode+1}: Max steps reached")

        metrics = env.computeMetrics()

        success_rate[episode] = metrics[0]
        flow_time[episode] = metrics[1]

    print(f"\n--- Results ({tests_episodes} episodes) ---")
    print("All goals reached count: ", all_goals)
    print("All goals reached mean: ", all_goals / tests_episodes)
    print("Success rate (per agent) mean: ", np.mean(success_rate))
    print("Success rate (per agent) std: ", np.std(success_rate))
    print("Flow time (finished episodes) max: ", np.max(flow_time[flow_time < (config['num_agents'] * config['max_time'])])) # Max of successful
    print("Flow time (finished episodes) mean: ", np.mean(flow_time[flow_time < (config['num_agents'] * config['max_time'])])) # Mean of successful
    print("Flow time (finished episodes) std: ", np.std(flow_time[flow_time < (config['num_agents'] * config['max_time'])])) # Std of successful