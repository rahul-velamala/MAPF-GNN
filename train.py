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
from torch.utils.data import DataLoader # Added DataLoader for type hinting

# --- Assuming these imports work when running from project root ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    from data_loader import GNNDataLoader
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running python from the 'rahul-velamala-mapf-gnn' directory.")
    sys.exit(1)
# ---                                                           ---

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Train GNN or Baseline MAPF models.")
parser.add_argument(
    "--config",
    type=str,
    default="configs/config_gnn.yaml", # Use forward slash
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
net_type = config.get("net_type", "gnn") # Default to gnn if missing
exp_name = config.get("exp_name", "default_experiment")
tests_episodes = config.get("tests_episodes", 25) # Episodes for evaluation
epochs = config.get("epochs", 50)
max_steps = config.get("max_steps", 32) # Max steps per *evaluation* episode
eval_frequency = config.get("eval_frequency", 5) # <--- MODIFICATION: How often to evaluate (epochs)
learning_rate = config.get("learning_rate", 3e-4)
weight_decay = config.get("weight_decay", 1e-4)

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Ensure exp_name uses forward slashes and create results dir
exp_name = exp_name.replace('\\', '/')
results_dir = os.path.join("results", exp_name) # Put results in a 'results' subfolder

# --- Model Selection ---
try:
    if net_type == "baseline":
        from models.framework_baseline import Network
        print("Using Baseline Network")
    elif net_type == "gnn":
        msg_type = config.get("msg_type", "gcn") # Default to 'gcn' if 'msg_type' is missing
        if msg_type == 'message':
            from models.framework_gnn_message import Network
            print("Using GNN Message Passing Network")
        else: # Default to standard GCN
            from models.framework_gnn import Network
            print("Using GNN (GCN) Network")
    else:
        raise ValueError(f"Unknown net_type in config: '{net_type}'")
except ImportError as e:
     print(f"ERROR: Failed to import model '{net_type}'. Check model files and dependencies: {e}")
     sys.exit(1)
except ValueError as e:
     print(f"ERROR: {e}")
     sys.exit(1)
# --- --------------- ---

# --- Results Directory and Config Saving ---
if not os.path.exists(results_dir):
    print(f"Creating results directory: {results_dir}")
    try:
        os.makedirs(results_dir, exist_ok=True) # Use exist_ok=True
    except OSError as e:
        print(f"ERROR: Could not create results directory '{results_dir}': {e}")
        sys.exit(1)

config_save_path = os.path.join(results_dir, "config_used.yaml") # Save effective config
try:
    # Add device to config before saving if not already present from loading
    if 'device' not in config: config['device'] = str(config["device"]) # Save device as string
    elif not isinstance(config['device'], str): config['device'] = str(config['device'])

    with open(config_save_path, "w") as config_path_out:
        yaml.dump(config, config_path_out, default_flow_style=False) # Better format
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
        train_loader: DataLoader = data_loader.train_loader
        valid_loader: DataLoader = data_loader.valid_loader # Can be None

        if not train_loader or len(train_loader.dataset) == 0:
             print("ERROR: Training data loader is empty or could not be created.")
             sys.exit(1)
        print(f"Training samples (timesteps): {len(train_loader.dataset)}")
        if valid_loader:
             print(f"Validation samples (timesteps): {len(valid_loader.dataset)}")
        else:
             print("No validation data loader created.")

    except Exception as e:
        print(f"ERROR: Failed to initialize or load data: {e}")
        print("\nRelevant config sections:")
        print(f"  train: {config.get('train')}")
        print(f"  valid: {config.get('valid')}")
        print(f"  batch_size: {config.get('batch_size')}")
        print(f"  num_workers: {config.get('num_workers')}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # --- ------------ ---

    # --- Model, Optimizer, Criterion ---
    try:
        model = Network(config)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Using CrossEntropyLoss for classification (action prediction)
        # It expects raw logits from the model and target class indices (long)
        criterion = nn.CrossEntropyLoss()
        model.to(config["device"])
        print(f"Model '{type(model).__name__}' initialized and moved to {config['device']}")
        # Optional: Print model summary or number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    except Exception as e:
        print(f"ERROR: Failed to initialize model, optimizer, or criterion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # --- --------------------------- ---

    # <--- MODIFICATION: Initialize list to store epoch data --->
    all_epoch_metrics = []
    best_eval_success_rate = -1.0 # Track best eval performance

    print(f"\n--- Starting Training for {epochs} epochs ---")
    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # ##### Training #########
        model.train() # Set model to training mode
        epoch_train_loss = 0.0
        batches_processed = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

        for i, batch_data in enumerate(train_pbar):
            try:
                # Ensure data is on the correct device
                states = batch_data[0].to(config["device"], non_blocking=True)
                trajectories = batch_data[1].to(config["device"], non_blocking=True) # Target actions
                gso = batch_data[2].to(config["device"], non_blocking=True)
            except (ValueError, IndexError) as e:
                print(f"\nError unpacking or moving batch {i} to device. Expected 3 items, got {len(batch_data)}. Error: {e}")
                continue # Skip this batch

            optimizer.zero_grad() # Zero gradients before forward pass

            # --- Forward Pass ---
            try:
                if net_type == 'gnn':
                    # GNN models expect states [B, N, C, H, W] and gso [B, N, N]
                    output_logits = model(states, gso) # Output shape: [B, N, NumActions]
                elif net_type == 'baseline':
                    # Baseline expects states [B, N, C, H, W]
                    output_logits = model(states) # Output shape: [B, N, NumActions]
                else:
                    # This case should have been caught earlier, but belts and suspenders
                    print(f"FATAL: Invalid net_type '{net_type}' during training loop.")
                    sys.exit(1)

                # --- Loss Calculation ---
                # output_logits shape: [B, N, NumActions=5]
                # trajectories shape: [B, N] (LongTensor of target action indices)

                # Check shapes before loss calculation
                expected_output_shape = (states.shape[0], config["num_agents"], 5)
                expected_target_shape = (states.shape[0], config["num_agents"])

                if output_logits.shape != expected_output_shape:
                     print(f"\nShape mismatch error (batch {i}): Output Logits={output_logits.shape}, Expected={expected_output_shape}")
                     continue # Skip batch if output shape is wrong
                if trajectories.shape != expected_target_shape:
                     print(f"\nShape mismatch error (batch {i}): Target Trajectories={trajectories.shape}, Expected={expected_target_shape}")
                     continue # Skip batch if target shape is wrong

                # Reshape for CrossEntropyLoss:
                # Logits: [B, N, C=5] -> [B*N, C=5]
                # Target: [B, N] -> [B*N]
                output_reshaped = output_logits.reshape(-1, 5) # (B*N, 5)
                # Ensure target is long and has the right shape
                trajectories_reshaped = trajectories.reshape(-1).long() # (B*N,)

                batch_loss = criterion(output_reshaped, trajectories_reshaped)

                # Check for NaN/Inf loss
                if not torch.isfinite(batch_loss):
                    print(f"\nWarning: NaN or Inf loss detected in batch {i}. Skipping backward pass.")
                    # Optionally: Investigate inputs/outputs causing NaN
                    # print("Output logits sample:", output_reshaped[:5])
                    # print("Target sample:", trajectories_reshaped[:5])
                    continue

            except Exception as e:
                print(f"\nError during model forward pass or loss calculation (batch {i}): {e}")
                print("Input shapes: states={}, gso={}, trajectories={}".format(states.shape, gso.shape, trajectories.shape))
                import traceback
                traceback.print_exc()
                continue # Skip this batch


            # --- Backward Pass & Optimization ---
            try:
                batch_loss.backward()
                # Optional: Gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += batch_loss.item()
                batches_processed += 1
                # Update progress bar
                train_pbar.set_postfix({"AvgLoss": epoch_train_loss / batches_processed})

            except Exception as e:
                 print(f"\nError during backward pass or optimizer step (batch {i}): {e}")
                 continue # Skip optimization if backward fails

        # --- End Training Loop for Epoch ---

        avg_epoch_loss = epoch_train_loss / batches_processed if batches_processed > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f} | Duration: {epoch_duration:.2f}s")

        # <--- MODIFICATION: Initialize epoch metrics dictionary --->
        current_epoch_data = {
            "Epoch": epoch + 1,
            "Average Training Loss": avg_epoch_loss,
            "Evaluation Episode Success Rate": np.nan, # Default to NaN if no eval this epoch
            "Evaluation Avg Flow Time (Success)": np.nan,    # Default to NaN
            "Evaluation Episodes Tested": 0,
            "Evaluation Episodes Succeeded": 0,
        }

        ######### Evaluation (Run periodically) #########
        # Check if evaluation should run this epoch
        run_eval = ((epoch + 1) % eval_frequency == 0) or ((epoch + 1) == epochs)

        if run_eval and tests_episodes > 0:
            eval_start_time = time.time()
            print(f"\n--- Running Evaluation after Epoch {epoch+1} ---")
            model.eval() # Set model to evaluation mode
            eval_episode_success_count = 0
            eval_flow_times_completed = [] # Store flow times ONLY for successful episodes
            eval_pbar = tqdm(range(tests_episodes), desc=f"Epoch {epoch+1} Evaluation", leave=False, unit="ep")

            # Get environment parameters from config
            board_dims = config.get("board_size", [16, 16]) # Default if missing
            obstacles_count = config.get("obstacles", 6)
            agents_count = config.get("num_agents", 4)
            sensing_range = config.get("sensing_range", 4) # Default if missing

            for episode in eval_pbar:
                try:
                    # --- Create Environment Instance ---
                    # Ensure goal generation doesn't overlap with obstacles/starts implicitly
                    obstacles = create_obstacles(board_dims, obstacles_count)
                    # Note: create_goals should ideally ensure goals are reachable and distinct
                    goals = create_goals(board_dims, agents_count, obstacles)
                    # The environment should handle random start positions if not provided
                    env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=sensing_range)
                    # emb = env.getEmbedding() # Embedding might not be needed if model doesn't use it explicitly
                    obs = env.reset()
                    episode_successfully_completed = False
                    # --- Run Simulation Episode ---
                    for step in range(max_steps):
                        # Prepare model inputs
                        # Ensure FOV and GSO are float tensors and have batch dim [1, N, ...]
                        fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
                        gso = (
                            torch.tensor(obs["adj_matrix"])
                            .float()
                            .unsqueeze(0)
                            .to(config["device"])
                        )

                        with torch.no_grad(): # Disable gradient calculation for evaluation
                            if net_type == 'gnn':
                                action_logits = model(fov, gso) # [1, N, NumActions]
                            elif net_type == 'baseline':
                                action_logits = model(fov) # [1, N, NumActions]
                            else:
                               # Should not happen
                               raise ValueError("Invalid net_type during eval")

                        # Get actions (select action with highest logit/probability)
                        # Argmax over the last dimension (actions) -> [1, N]
                        action = action_logits.argmax(dim=-1).squeeze(0).cpu().numpy() # Shape [N,]

                        # Step the environment
                        obs, reward, done, info = env.step(action, env.getEmbedding()) # Pass current embedding if needed

                        if done: # Check if the environment signaled completion (all agents at goal)
                            episode_successfully_completed = True
                            break # End episode early if done

                    # --- End of Simulation Episode ---

                    if episode_successfully_completed:
                        eval_episode_success_count += 1
                        eval_flow_times_completed.append(env.time) # Record time taken for this successful episode
                    # else: episode failed (timeout or other condition)

                    # Update progress bar postfix
                    current_success_rate_display = eval_episode_success_count / (episode + 1)
                    eval_pbar.set_postfix({
                        "Success": f"{eval_episode_success_count}/{episode+1}",
                        "SR": f"{current_success_rate_display:.3f}"
                    })

                except Exception as e:
                    print(f"\nError during evaluation episode {episode}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to the next episode
                    continue
            # --- End Eval Episode Loop ---

            # Calculate final evaluation metrics for the epoch
            eval_episode_success_rate = eval_episode_success_count / tests_episodes if tests_episodes > 0 else 0.0
            avg_flow_time_success = np.mean(eval_flow_times_completed) if eval_flow_times_completed else np.nan # Avg time for SUCCESSFUL episodes

            eval_duration = time.time() - eval_start_time

            # <--- MODIFICATION: Store eval results in the epoch dictionary --->
            current_epoch_data["Evaluation Episode Success Rate"] = eval_episode_success_rate
            current_epoch_data["Evaluation Avg Flow Time (Success)"] = avg_flow_time_success
            current_epoch_data["Evaluation Episodes Tested"] = tests_episodes
            current_epoch_data["Evaluation Episodes Succeeded"] = eval_episode_success_count

            print(f"Evaluation Complete:")
            print(f"  Episodes Succeeded: {eval_episode_success_count}/{tests_episodes}")
            print(f"  Episode Success Rate: {eval_episode_success_rate:.4f}")
            if not np.isnan(avg_flow_time_success):
                 print(f"  Avg Flow Time (Successful Eps): {avg_flow_time_success:.2f} steps")
            else:
                 print(f"  Avg Flow Time (Successful Eps): N/A (No episodes succeeded)")
            print(f"  Evaluation Duration: {eval_duration:.2f}s")

            # Save model checkpoint IF this epoch is the best so far based on success rate
            if eval_episode_success_rate >= best_eval_success_rate:
                 print(f"New best evaluation success rate ({eval_episode_success_rate:.4f}), saving model...")
                 best_eval_success_rate = eval_episode_success_rate
                 best_model_path = os.path.join(results_dir, f"model_best.pt")
                 try:
                     torch.save(model.state_dict(), best_model_path)
                     print(f"Saved best model checkpoint: {best_model_path}")
                 except IOError as e:
                      print(f"Warning: Failed to save best model checkpoint '{best_model_path}': {e}")

            # Optional: Save checkpoint every eval period regardless of performance
            # checkpoint_path = os.path.join(results_dir, f"model_epoch_{epoch+1}.pt")
            # try:
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print(f"Saved checkpoint: {checkpoint_path}")
            # except IOError as e:
            #      print(f"Warning: Failed to save checkpoint '{checkpoint_path}': {e}")

        ######### End of Evaluation Block #########
        else:
            # Print message if skipping evaluation this epoch
            if tests_episodes <= 0:
                print("Evaluation skipped (tests_episodes is 0).")
            elif not run_eval:
                 print(f"Evaluation skipped this epoch (runs every {eval_frequency} epochs).")


        # <--- MODIFICATION: Append the epoch's data to the main list --->
        all_epoch_metrics.append(current_epoch_data)

        # --- Optional: Save metrics dataframe periodically (e.g., every 10 epochs) ---
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            try:
                temp_df = pd.DataFrame(all_epoch_metrics)
                temp_excel_path = os.path.join(results_dir, "training_metrics_partial.xlsx")
                temp_df.to_excel(temp_excel_path, index=False, engine='openpyxl')
            except Exception: # Ignore errors during partial save
                pass


    # --- End of All Epochs ---
    total_training_time = time.time() - training_start_time
    print(f"\n--- Training Finished ({total_training_time:.2f}s total) ---")

    # --- Saving Final Results ---

    # <--- MODIFICATION: Create DataFrame and save to Excel --->
    metrics_df = pd.DataFrame(all_epoch_metrics)
    excel_path = os.path.join(results_dir, "training_metrics.xlsx")
    try:
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Saved final epoch metrics to Excel file: {excel_path}")
    except Exception as e:
        print(f"Warning: Failed to save metrics to Excel file '{excel_path}': {e}")
        print("Attempting to save as CSV instead...")
        try:
            csv_path = os.path.join(results_dir, "training_metrics.csv")
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved final epoch metrics to CSV file: {csv_path}")
        except Exception as e_csv:
            print(f"Warning: Failed to save metrics to CSV file '{csv_path}': {e_csv}")
    # <-------------------------------------------------------->

    # Save final model (last epoch's state)
    final_model_path = os.path.join(results_dir, "model_final.pt")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model (last epoch) to {final_model_path}")
    except IOError as e:
        print(f"Warning: Failed to save final model '{final_model_path}': {e}")
    # --- -------------------- ---

    # --- Plotting (using data from DataFrame) ---
    print("\n--- Generating Plots ---")
    try:
        plt.figure(figsize=(18, 5)) # Wider figure for 3 plots

        # --- Loss Plot ---
        plt.subplot(1, 3, 1)
        plt.plot(metrics_df["Epoch"], metrics_df["Average Training Loss"], marker='.')
        plt.title("Average Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout(pad=2.0) # Add padding between plots

        # --- Evaluation Plots (only if evaluation was run) ---
        # Filter DataFrame for rows where evaluation was actually performed
        eval_df = metrics_df.dropna(subset=["Evaluation Episode Success Rate"])

        if not eval_df.empty:
            # --- Episode Success Rate Plot ---
            plt.subplot(1, 3, 2)
            plt.plot(eval_df["Epoch"], eval_df["Evaluation Episode Success Rate"], marker='o', linestyle='-')
            plt.title("Evaluation Episode Success Rate")
            plt.xlabel("Epoch")
            plt.ylabel("Success Rate (Episodes)")
            plt.ylim(-0.05, 1.05) # Set Y axis from 0 to 1
            plt.grid(True)
            plt.tight_layout(pad=2.0)

            # --- Average Flow Time (Successful Episodes) Plot ---
            plt.subplot(1, 3, 3)
            # Filter out potential NaNs if avg_flow_time resulted in NaN (no successful runs)
            valid_flow_time_df = eval_df.dropna(subset=["Evaluation Avg Flow Time (Success)"])
            if not valid_flow_time_df.empty:
                 plt.plot(valid_flow_time_df["Epoch"], valid_flow_time_df["Evaluation Avg Flow Time (Success)"], marker='o', linestyle='-')
                 plt.ylabel("Avg Flow Time (Steps)")
            else:
                 # If no valid flow times, plot nothing or a horizontal line indicating failure
                 # Plotting nothing is clearer
                 plt.text(0.5, 0.5, 'No successful episodes', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                 plt.ylabel("Avg Flow Time (Steps) - N/A")


            plt.title("Evaluation Flow Time (Avg Successful Eps)")
            plt.xlabel("Epoch")
            # plt.ylabel("Flow Time (Steps)") # Set above conditionally
            plt.grid(True)
            plt.tight_layout(pad=2.0)
        else:
            print("Warning: No evaluation data found to plot evaluation metrics.")
            # Optionally add placeholder text to the empty subplots
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, 'No evaluation data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title("Evaluation Episode Success Rate")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.subplot(1, 3, 3)
            plt.text(0.5, 0.5, 'No evaluation data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title("Evaluation Flow Time")
            plt.xlabel("Epoch")
            plt.grid(True)


        # plt.tight_layout() # Call once after all subplots are added
        plot_path = os.path.join(results_dir, "training_plots.png")
        plt.savefig(plot_path, dpi=150) # Increase DPI for clarity if needed
        print(f"Saved training plots to {plot_path}")
        plt.close() # Close the figure to free memory
    except Exception as e:
        print(f"Warning: Failed to generate or save plots: {e}")
        import traceback
        traceback.print_exc()
    # --- ----------------- ---

    print("\n--- Script Finished ---")