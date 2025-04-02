# File: train.py
import sys
import os
import argparse
import time
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd  # <--- MODIFICATION: Import pandas

import torch
from torch import nn
from torch import optim

# --- Assuming these imports work when running from project root ---
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from data_loader import GNNDataLoader
# ---                                                           ---

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Train GNN or Baseline MAPF models.")
parser.add_argument(
    "--config",
    type=str,
    default="configs/config_gnn.yaml",
    help="Path to the YAML configuration file (e.g., configs/config_gnn.yaml)"
)
args = parser.parse_args()
# ========================

# --- Load Configuration ---
config_file_path = args.config
print(f"Loading configuration from: {config_file_path}")
try:
    with open(config_file_path, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at '{config_file_path}'")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"ERROR: Could not parse configuration file '{config_file_path}': {e}")
    sys.exit(1)
# --- ------------------ ---

# --- Setup based on Config ---
net_type = config.get("net_type", "gnn")
exp_name = config.get("exp_name", "default_experiment")
tests_episodes = config.get("tests_episodes", 25)
epochs = config.get("epochs", 50)
max_steps = config.get("max_steps", 32)
eval_frequency = config.get("eval_frequency", 5) # <--- MODIFICATION: Get eval frequency from config (optional)

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_name = exp_name.replace('\\', '/')
results_dir = os.path.join("results", exp_name)

# --- Model Selection ---
try:
    if net_type == "baseline":
        from models.framework_baseline import Network
    elif net_type == "gnn":
        if config.get("msg_type") == 'message':
            from models.framework_gnn_message import Network
        else:
            from models.framework_gnn import Network
    else:
        raise ValueError(f"Unknown net_type in config: '{net_type}'")
except ImportError as e:
     print(f"ERROR: Failed to import model '{net_type}'. Check model files and dependencies: {e}")
     sys.exit(1)
# --- --------------- ---

# --- Results Directory and Config Saving ---
if not os.path.exists(results_dir):
    print(f"Creating results directory: {results_dir}")
    try:
        os.makedirs(results_dir)
    except OSError as e:
        print(f"ERROR: Could not create results directory '{results_dir}': {e}")
        sys.exit(1)

config_save_path = os.path.join(results_dir, "config.yaml")
try:
    with open(config_save_path, "w") as config_path_out:
        yaml.dump(config, config_path_out)
    print(f"Saved effective config to {config_save_path}")
except IOError as e:
    print(f"ERROR: Could not save config to '{config_save_path}': {e}")
    sys.exit(1)
# --- ----------------------------------- ---


if __name__ == "__main__":

    print("\n----- Effective Configuration -----")
    pprint(config)
    print(f"Using device: {config['device']}")
    print("---------------------------------\n")

    # --- Data Loading ---
    try:
        data_loader = GNNDataLoader(config)
        if not data_loader.train_loader or len(data_loader.train_loader.dataset) == 0:
             print("ERROR: Training data loader is empty.")
             sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize or load data: {e}")
        print("Relevant config sections:")
        print("  train:", config.get('train'))
        print("  valid:", config.get('valid'))
        print("  batch_size:", config.get('batch_size'))
        print("  num_workers:", config.get('num_workers'))
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # --- ------------ ---

    # --- Model, Optimizer, Criterion ---
    model = Network(config)
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 3e-4), weight_decay=config.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    model.to(config["device"])
    # --- --------------------------- ---

    # <--- MODIFICATION: Initialize list to store epoch data --->
    all_epoch_metrics = []

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # ##### Training #########
        model.train()
        train_loss = 0
        train_pbar = tqdm(data_loader.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for i, batch_data in enumerate(train_pbar):
            try:
                states, trajectories, gso = batch_data
            except ValueError as e:
                print(f"\nError unpacking batch {i}. Expected 3 items, got {len(batch_data)}. Error: {e}")
                continue

            optimizer.zero_grad()
            states = states.to(config["device"], non_blocking=True)
            trajectories = trajectories.to(config["device"], non_blocking=True)
            gso = gso.to(config["device"], non_blocking=True)

            try:
                if net_type == 'gnn':
                    output = model(states, gso)
                elif net_type == 'baseline':
                    output = model(states)
                else:
                    raise ValueError("Invalid net_type")
            except Exception as e:
                print(f"\nError during model forward pass (batch {i}): {e}")
                raise e

            if output.shape[:-1] != trajectories.shape or output.shape[-1] != 5:
                 print(f"\nShape mismatch error (batch {i}): Output={output.shape}, Target={trajectories.shape}")
                 continue

            output_reshaped = output.view(-1, 5)
            trajectories_reshaped = trajectories.view(-1)

            try:
                 batch_total_loss = criterion(output_reshaped, trajectories_reshaped.long())
            except Exception as e:
                 print(f"\nError during loss calculation (batch {i}): {e}")
                 continue

            if isinstance(batch_total_loss, torch.Tensor):
                batch_total_loss.backward()
                optimizer.step()
                train_loss += batch_total_loss.item()
                train_pbar.set_postfix({"Loss": train_loss / (i + 1)})
            else:
                print(f"Warning: Skipping backward pass for batch {i} due to non-tensor loss.")
        # --- End Training Loop for Epoch ---

        avg_epoch_loss = train_loss / len(data_loader.train_loader) if len(data_loader.train_loader) > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f} | Duration: {epoch_duration:.2f}s")

        # <--- MODIFICATION: Initialize epoch metrics dictionary --->
        current_epoch_data = {
            "Epoch": epoch + 1,
            "Average Training Loss": avg_epoch_loss,
            "Evaluation Success Rate": np.nan, # Default to NaN if no eval this epoch
            "Evaluation Flow Time": np.nan    # Default to NaN
        }

        ######### Evaluation (Run periodically) #########
        # Check if evaluation should run this epoch
        run_eval = (epoch + 1) % eval_frequency == 0 or (epoch + 1) == epochs

        if run_eval and tests_episodes > 0:
            eval_start_time = time.time()
            print(f"\n--- Running Evaluation after Epoch {epoch+1} ---")
            model.eval()
            eval_success_rate = []
            eval_flow_time = []
            eval_all_goals_count = 0
            eval_pbar = tqdm(range(tests_episodes), desc=f"Epoch {epoch+1} Evaluation", leave=False)

            board_dims = config.get("board_size", [16, 16])
            obstacles_count = config.get("obstacles", 6)
            agents_count = config.get("num_agents", 4)
            sensing_range = config.get("sensing_range", 4)

            for episode in eval_pbar:
                try:
                    obstacles = create_obstacles(board_dims, obstacles_count)
                    goals = create_goals(board_dims, agents_count, obstacles)
                    env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=sensing_range)
                    emb = env.getEmbedding()
                    obs = env.reset()
                    episode_done = False
                    for step in range(max_steps):
                        fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
                        gso = (
                            torch.tensor(obs["adj_matrix"])
                            .float()
                            .unsqueeze(0)
                            .to(config["device"])
                        )
                        with torch.no_grad():
                            if net_type == 'gnn':
                                action_probs = model(fov, gso)
                            elif net_type == 'baseline':
                                action_probs = model(fov)
                            else:
                               raise ValueError("Invalid net_type during eval")
                        action = action_probs.cpu().argmax(dim=-1).squeeze(0).numpy()
                        obs, reward, done, info = env.step(action, emb)
                        if done:
                            episode_done = True
                            break

                    metrics = env.computeMetrics()

                    if isinstance(metrics, (list, tuple)) and len(metrics) >= 2:
                         current_success_rate = metrics[0]
                         if np.isnan(current_success_rate):
                              eval_success_rate.append(0.0) # Append 0 if NaN
                         else:
                              eval_success_rate.append(current_success_rate)
                         eval_flow_time.append(metrics[1]) # Append flow time (can be inf)
                    else:
                         print(f"\nWarning: Eval metrics format issue: {metrics}. Skipping ep {episode}.")
                         eval_success_rate.append(0.0)
                         eval_flow_time.append(float('inf'))

                    if episode_done:
                        eval_all_goals_count += 1

                    current_avg_sr_display = np.mean(eval_success_rate) if eval_success_rate else 0.0
                    eval_pbar.set_postfix({"AvgSuccRate": f"{current_avg_sr_display:.3f}", "AllGoals": f"{eval_all_goals_count}/{episode+1}"})

                except Exception as e:
                    print(f"\nError during evaluation episode {episode}: {e}")
                    eval_success_rate.append(0.0)
                    eval_flow_time.append(float('inf'))
                    continue
            # --- End Eval Episode Loop ---

            # Calculate final evaluation metrics for the epoch
            avg_success_rate = np.nanmean(eval_success_rate) if eval_success_rate else 0.0
            max_possible_flowtime = agents_count * max_steps
            valid_flow_times = [ft for ft in eval_flow_time if ft is not None and ft <= max_possible_flowtime]
            avg_flow_time = np.mean(valid_flow_times) if valid_flow_times else float('inf')
            eval_duration = time.time() - eval_start_time

            # <--- MODIFICATION: Store eval results in the epoch dictionary --->
            current_epoch_data["Evaluation Success Rate"] = avg_success_rate
            # Replace inf with NaN for better Excel compatibility if desired, or keep inf
            current_epoch_data["Evaluation Flow Time"] = avg_flow_time if avg_flow_time != float('inf') else np.nan

            print(f"Evaluation Success rate (avg agent): {avg_success_rate:.4f}")
            print(f"Evaluation Episodes All Goals Reached: {eval_all_goals_count}/{tests_episodes}")
            print(f"Evaluation Flow time (avg valid): {avg_flow_time:.2f}")
            print(f"Evaluation Duration: {eval_duration:.2f}s")

            # Save model checkpoint
            checkpoint_path = os.path.join(results_dir, f"model_epoch_{epoch+1}.pt")
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            except IOError as e:
                 print(f"Warning: Failed to save checkpoint '{checkpoint_path}': {e}")
        ######### End of Evaluation Block #########

        # <--- MODIFICATION: Append the epoch's data to the main list --->
        all_epoch_metrics.append(current_epoch_data)

    # --- End of All Epochs ---
    print("\n--- Training Finished ---")

    # --- Saving Final Results ---

    # <--- MODIFICATION: Create DataFrame and save to Excel --->
    metrics_df = pd.DataFrame(all_epoch_metrics)
    excel_path = os.path.join(results_dir, "training_metrics.xlsx")
    try:
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Saved epoch metrics to Excel file: {excel_path}")
    except Exception as e:
        print(f"Warning: Failed to save metrics to Excel file '{excel_path}': {e}")
    # <-------------------------------------------------------->

    # Save loss and evaluation metrics also as numpy arrays (optional, redundant with Excel)
    loss_array = metrics_df["Average Training Loss"].to_numpy()
    eval_epoch_indices = metrics_df.loc[metrics_df["Evaluation Success Rate"].notna(), "Epoch"].to_numpy()
    success_rate_eval_array = metrics_df.loc[metrics_df["Evaluation Success Rate"].notna(), "Evaluation Success Rate"].to_numpy()
    # Handle potential NaNs if kept from avg_flow_time
    flow_time_eval_array = metrics_df.loc[metrics_df["Evaluation Flow Time"].notna(), "Evaluation Flow Time"].to_numpy()


    try:
        np.save(os.path.join(results_dir, "loss.npy"), loss_array)
        # Save evaluation metrics only if they exist
        if len(eval_epoch_indices) > 0:
            np.savez(os.path.join(results_dir, "evaluation_metrics.npz"),
                     epochs=eval_epoch_indices,
                     success_rate=success_rate_eval_array,
                     flow_time=flow_time_eval_array) # flow_time might contain NaN if no valid times found
        print(f"Saved metrics (loss.npy, evaluation_metrics.npz) to {results_dir}")
    except IOError as e:
        print(f"Warning: Failed to save .npy metrics files: {e}")

    # Save final model
    final_model_path = os.path.join(results_dir, "model.pt")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
    except IOError as e:
        print(f"Warning: Failed to save final model '{final_model_path}': {e}")
    # --- -------------------- ---

    # --- Plotting (using data from DataFrame) ---
    try:
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(metrics_df["Epoch"], metrics_df["Average Training Loss"])
        plt.title("Average Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # Filter DataFrame for valid evaluation data
        eval_df = metrics_df.dropna(subset=["Evaluation Success Rate", "Evaluation Flow Time"]) # Drop rows where eval wasn't run

        if not eval_df.empty:
            # Success Rate plot
            plt.subplot(1, 3, 2)
            plt.plot(eval_df["Epoch"], eval_df["Evaluation Success Rate"], marker='o')
            plt.title("Evaluation Success Rate (Avg Agent)")
            plt.xlabel("Epoch")
            plt.ylabel("Success Rate")
            plt.grid(True)
            plt.ylim(0, 1.05)

            # Flow Time plot
            plt.subplot(1, 3, 3)
            # Filter out potential NaNs if avg_flow_time resulted in NaN
            valid_flow_time_df = eval_df.dropna(subset=["Evaluation Flow Time"])
            if not valid_flow_time_df.empty:
                 plt.plot(valid_flow_time_df["Epoch"], valid_flow_time_df["Evaluation Flow Time"], marker='o')
            else:
                 print("Warning: No valid (non-NaN) flow time data to plot.")

            plt.title("Evaluation Flow Time (Avg Valid)")
            plt.xlabel("Epoch")
            plt.ylabel("Flow Time")
            plt.grid(True)
        else:
            print("Warning: No evaluation data found to plot.")


        plt.tight_layout()
        plot_path = os.path.join(results_dir, "training_plots.png")
        plt.savefig(plot_path)
        print(f"Saved training plots to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to generate or save plots: {e}")
        import traceback
        traceback.print_exc()
    # --- ----------------- ---