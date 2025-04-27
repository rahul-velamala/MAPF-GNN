# File: train.py
# (Revised Version - Uses MatDataLoader, OE Disabled, Logging Imported, REDUNDANT Shielding Removed from Inference)

import sys
import os
import argparse
import time
import traceback
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import random # Still needed for evaluation seeding if used
import shutil # For potentially cleaning failed OE runs (though OE is disabled)
import copy   # For deep copying env for expert simulation (OE disabled)
import signal # For CBS timeout handling (OE disabled)
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader # Keep base DataLoader import

import logging # <<< --- IMPORT ADDED --- <<<

# --- Setup Logging ---
logger = logging.getLogger(__name__)
# Configure logger if not already configured by a higher-level script
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- --------------- ---

# --- Project Imports ---
try:
    logger.info("Importing environment and data modules...")
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Import the specific DataLoader manager we want to use
    from data_loader import MatDataLoader as GNNDataLoader # <<< Uses MatDataLoader now
    logger.info("Imports successful.")
except ImportError as e:
    logger.error(f"FATAL ERROR importing project modules: {e}", exc_info=True)
    logger.error("Please ensure:")
    logger.error("  1. You are running python from the main project directory.")
    logger.error("  2. Necessary files (mat_dataset.py, data_loader.py, env_graph_gridv1.py) exist.")
    sys.exit(1)
# --- ----------------- ---

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Train GNN or Baseline MAPF models using .mat dataset.")
parser.add_argument(
    "--config", type=str, default="configs/config_train_mat.yaml", # Point to new default
    help="Path to the YAML configuration file for .mat data training."
)
parser.add_argument(
    "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Set the logging level."
)
# NOTE: --oe_disable flag is now ignored as OE is hard-disabled below
args = parser.parse_args()
# ========================

# --- Configure Logging Level ---
log_level_arg = getattr(logging, args.log_level.upper(), logging.INFO)
# Reconfigure logger with the specified level
logging.basicConfig(level=log_level_arg, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s', force=True)
logger = logging.getLogger(__name__) # Get logger again after reconfiguring
# --- ------------------------- ---

# --- Load Configuration ---
config_file_path = Path(args.config)
logger.info(f"Loading configuration from: {config_file_path}")
try:
    with open(config_file_path, "r") as config_path_obj:
        config = yaml.safe_load(config_path_obj)
        if config is None: raise ValueError("Config file is empty or invalid.")
except Exception as e:
    logger.error(f"Could not load or parse config file '{config_file_path}': {e}", exc_info=True)
    sys.exit(1)
# --- ------------------ ---

# --- Setup based on Config ---
try:
    net_type = config.get("net_type", "gnn")
    exp_name = config.get("exp_name", "default_mat_experiment")
    epochs = int(config.get("epochs", 50))
    max_steps_eval = int(config.get("max_steps", 60)) # Max steps for evaluation episodes
    eval_frequency = int(config.get("eval_frequency", 5))
    tests_episodes_eval = int(config.get("tests_episodes", 25)) # Num eval episodes
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    num_agents_config = int(config.get("num_agents")) # Critical: Must match .mat data
    pad_config = int(config.get("pad"))             # Critical: Must match .mat data

    # --- Online Expert (OE) Config ---
    # !!! Disabling OE when using MatDataset as it requires different setup !!!
    use_online_expert = False # Explicitly disable
    logger.warning("Online Expert (DAgger) is DISABLED for training with MatDataset.")
    # --- End OE Config ---

except (ValueError, TypeError, KeyError) as e:
     logger.error(f"Missing or invalid required key in configuration: {e}", exc_info=True)
     sys.exit(1)

# --- Device Setup ---
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {config['device']}")
# --- ----------- ---

# --- Results Directory ---
exp_name_cleaned = exp_name.replace('\\', '/')
results_dir = Path("results") / exp_name_cleaned
try:
    results_dir.mkdir(parents=True, exist_ok=True)
except OSError as e: logger.error(f"Could not create results dir {results_dir}: {e}", exc_info=True); sys.exit(1)
logger.info(f"Results will be saved in: {results_dir.resolve()}")
# --- ----------------- ---

# --- Model Selection ---
NetworkClass = None
try:
    logger.info(f"Selecting model based on net_type='{net_type}'...")
    if net_type == "baseline":
        from models.framework_baseline import Network as NetworkClass
        logger.info("Using Model: Baseline Network")
    elif net_type == "gnn":
        msg_type = config.get("msg_type", "gcn").lower()
        from models.framework_gnn import Network as NetworkClass
        logger.info(f"Using Model: GNN Network (msg_type='{msg_type}')")
        config['msg_type'] = msg_type # Ensure it's set for model init
    else: raise ValueError(f"Unknown 'net_type' in config: '{net_type}'")
    if NetworkClass is None: raise ImportError("NetworkClass was not assigned.")
    # Import utils_weights here, as it's used later
    from models.networks.utils_weights import weights_init
except (ImportError, ValueError, KeyError) as e:
     logger.error(f"Failed import/select model: {e}", exc_info=True); sys.exit(1)
# --- --------------- ---

# --- Save Effective Config ---
config_save_path = results_dir / "config_used.yaml"
try:
    config_to_save = config.copy()
    if 'device' in config_to_save and isinstance(config_to_save['device'], torch.device):
        config_to_save['device'] = str(config_to_save["device"])
    if 'online_expert' in config_to_save: del config_to_save['online_expert']
    with open(config_save_path, "w") as f_out: yaml.dump(config_to_save, f_out, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved effective config to {config_save_path}")
except Exception as e: logger.error(f"Could not save config: {e}")
# --- ----------------------- ---


# === Helper Function: Run Inference (No External Shielding) ===
# (Relies on Env's internal collision handling)
# NOTE: Renamed function slightly for clarity, but kept original name call below for compatibility.
# Original name "run_inference_with_shielding" is kept in main loop call.
def run_inference_without_external_shielding(
    model: nn.Module, env: GraphEnv, max_steps_inference: int, device: torch.device, net_type: str
    ) -> tuple[dict | None, bool, bool]:
    model.eval()
    try:
        obs, info = env.reset()
    except Exception as e: logger.error(f"Error reset inference: {e}", exc_info=True); return None, False, True

    terminated = False; truncated = False
    # History can still be useful for debugging if needed
    history = {'states': [], 'gsos': [], 'model_actions': []} # Simplified history
    step_count = 0
    total_collisions_env = 0 # Track collisions reported by env info

    while not terminated and not truncated:
        if step_count >= max_steps_inference:
            truncated = True
            logger.debug(f"Eval Episode Timeout at step {step_count}")
            break

        try: # Prepare observation for model
            current_fov_np = obs["fov"]
            current_gso_np = obs["adj_matrix"]
            # history['states'].append(current_fov_np.copy()) # Optional: uncomment if needed for debug
            # history['gsos'].append(current_gso_np.copy()) # Optional: uncomment if needed for debug

            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device)
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device)
        except KeyError as e:
            logger.error(f"Error accessing observation key {e} at env time {env.time}", exc_info=True)
            return history, False, True
        except Exception as e:
            logger.error(f"Error converting observation to tensor at env time {env.time}: {e}", exc_info=True)
            return history, False, True

        with torch.no_grad(): # Get model's proposed action
            try:
                if net_type == 'gnn':
                    action_scores = model(fov_tensor, gso_tensor)
                else: # baseline
                    action_scores = model(fov_tensor)
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
                # history['model_actions'].append(proposed_actions.copy()) # Optional

            except Exception as e:
                logger.error(f"Error during model forward pass at env time {env.time}: {e}", exc_info=True)
                return history, False, True

        # --- NO EXTERNAL COLLISION SHIELDING PERFORMED HERE ---
        # The environment's step function will handle collisions based on proposed_actions.

        try: # Step environment with the model's proposed actions
            obs, reward, terminated, truncated_env, info = env.step(proposed_actions)
            truncated = truncated or truncated_env # Combine truncation flags
            step_count = env.time # Update step count based on environment time

            # Optional: Log collisions detected by the environment
            if "collisions_this_step" in info and np.any(info["collisions_this_step"]):
                num_collided = np.sum(info["collisions_this_step"])
                total_collisions_env += num_collided
                # logger.debug(f"Step {env.time}: Env detected {num_collided} collision(s) for agents {np.where(info['collisions_this_step'])[0]}")

        except Exception as e:
            logger.error(f"Error during env.step at time {env.time}: {e}", exc_info=True)
            return history, False, True # Treat env error as failure
    # --- After loop ---
    is_success = terminated and not truncated # Success only if terminated *without* being truncated
    is_timeout = truncated

    logger.debug(f"Eval Episode End: Success={is_success}, Timeout={is_timeout}, Steps={env.time}, EnvReportedCollisions={total_collisions_env}")

    # Return history (can be empty if unused), success status, timeout status
    return history, is_success, is_timeout
# === END of run_inference_without_external_shielding ===

# Alias for backward compatibility in the main loop call below
run_inference_with_shielding = run_inference_without_external_shielding


# === Main Training Script ===
if __name__ == "__main__":

    logger.info("\n----- Effective Configuration -----")
    print_config = config.copy()
    if 'device' in print_config and isinstance(print_config['device'], torch.device): print_config['device'] = str(print_config['device'])
    pprint(print_config, indent=2)
    device = config["device"] # Use the actual torch.device object
    logger.info(f"Using device: {device}")
    logger.info(f"Online Expert (DAgger): {'Enabled' if use_online_expert else 'Disabled'}")
    logger.info("---------------------------------\n")

    # --- Data Loading (Using MatDataLoader) ---
    data_loader_manager = None; train_loader = None; valid_loader = None
    try:
        data_loader_manager = GNNDataLoader(config) # GNNDataLoader is alias for MatDataLoader
        train_loader = data_loader_manager.train_loader
        valid_loader = data_loader_manager.valid_loader
        if not train_loader: logger.critical("MAT Training data loader failed to initialize."); sys.exit(1)
        if len(train_loader.dataset) == 0: logger.critical("MAT Training data loader is empty."); sys.exit(1)
        logger.info(f"Training samples (timesteps): {len(train_loader.dataset)}")
        if valid_loader: logger.info(f"Validation samples (timesteps): {len(valid_loader.dataset)}")
        else: logger.info("No validation loader configured.")
    except Exception as e: logger.error(f"Failed to initialize/load MAT data: {e}", exc_info=True); sys.exit(1)
    # --- ------------ ---

    # --- Model, Optimizer, Criterion ---
    model = None; optimizer = None; criterion = None
    try:
        model = NetworkClass(config)
        model.to(device)
        model.apply(weights_init) # Apply weight initialization
        logger.info(f"\nModel '{type(model).__name__}' initialized on {device}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {num_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
    except Exception as e: logger.error(f"Failed init model/optimizer/criterion: {e}", exc_info=True); sys.exit(1)
    # --- --------------------------- ---

    # --- Training Loop Setup ---
    all_epoch_metrics = []
    best_eval_success_rate = -1.0

    logger.info(f"\n--- Starting Training for {epochs} epochs ---")
    training_start_time = time.time()

    # --- Main Epoch Loop ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*15} Epoch {epoch+1}/{epochs} {'='*15}")

        # === Online Expert -> DISABLED ===
        run_oe_this_epoch = False
        if run_oe_this_epoch: pass
        # === End OE ===

        # ##### Training Phase #########
        model.train()
        epoch_train_loss = 0.0; batches_processed = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")
        for i, batch_data in enumerate(train_pbar):
            try: # Unpack and move data
                non_blocking = device.type == 'cuda' and config.get('num_workers', 0) > 0
                fov_batch = batch_data[0].to(device, non_blocking=non_blocking)
                target_actions_batch = batch_data[1].to(device, non_blocking=non_blocking)
                gso_batch = batch_data[2].to(device, non_blocking=non_blocking)
            except Exception as e: logger.error(f"Error moving batch {i}: {e}"); continue
            optimizer.zero_grad()
            try: # Forward pass
                if net_type == 'gnn': output_logits = model(fov_batch, gso_batch)
                else: output_logits = model(fov_batch)
            except Exception as e: logger.error(f"Error forward pass {i}: {e}"); continue
            try: # Loss calculation
                batch_b, batch_n, num_actions_model = output_logits.shape
                output_reshaped = output_logits.reshape(-1, num_actions_model)
                target_reshaped = target_actions_batch.reshape(-1)
                if num_actions_model != 5: logger.error(f"Action dim mismatch {num_actions_model}"); continue
                if torch.any(target_reshaped < 0) or torch.any(target_reshaped >= 5): logger.error(f"Invalid target index {target_reshaped.min()}-{target_reshaped.max()}"); continue
                if output_reshaped.shape[0] != target_reshaped.shape[0]: logger.error(f"Loss shape mismatch {i}"); continue
                batch_loss = criterion(output_reshaped, target_reshaped)
                if not torch.isfinite(batch_loss): logger.warning(f"Loss NaN/Inf {i}. Skipping."); continue
            except Exception as e: logger.error(f"Error loss calc {i}: {e}"); continue
            try: # Backward pass & step
                batch_loss.backward(); optimizer.step()
            except Exception as e: logger.error(f"Error backward/step {i}: {e}"); continue
            epoch_train_loss += batch_loss.item(); batches_processed += 1
            if batches_processed > 0: train_pbar.set_postfix({"AvgLoss": epoch_train_loss / batches_processed})
        train_pbar.close() # End Training Batch Loop

        avg_epoch_loss = epoch_train_loss / batches_processed if batches_processed > 0 else 0.0
        epoch_duration_train = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} Avg Training Loss: {avg_epoch_loss:.4f} | Train Duration: {epoch_duration_train:.2f}s")

        # Store epoch metrics (placeholders for eval)
        current_epoch_data = {
            "Epoch": epoch + 1, "Average Training Loss": avg_epoch_loss,
            "Evaluation Episode Success Rate": np.nan, "Evaluation Avg Steps (Success)": np.nan,
            "Evaluation Episodes Tested": 0, "Evaluation Episodes Succeeded": 0,
            "Training Samples Used": len(train_loader.dataset)
        }

        # ######### Evaluation Phase #########
        run_eval = ((epoch + 1) % eval_frequency == 0) or ((epoch + 1) == epochs)
        # Always run evaluation if validation loader exists, for simplicity with static data
        # if run_eval and valid_loader is not None:
        if valid_loader is not None and run_eval: # Check valid_loader first
            eval_start_time = time.time()
            logger.info(f"\n--- Running Evaluation after Epoch {epoch+1} ---")
            model.eval()
            eval_success_count = 0; eval_steps_success = []

            # Get eval parameters from config, fallback to training params
            eval_board_dims = config.get("eval_board_size", config["board_size"])
            eval_obstacles_count = config.get("eval_obstacles", config.get("nb_obstacles", 10)) # nb_obstacles might not exist
            eval_agents_count = num_agents_config
            eval_sensing_range = config.get("eval_sensing_range", config["sensing_range"])
            eval_pad = pad_config
            eval_max_steps_run = int(config.get("eval_max_steps", max_steps_eval))

            eval_pbar = tqdm(range(tests_episodes_eval), desc=f"Epoch {epoch+1} Evaluation", leave=False, unit="ep")
            for episode_idx in eval_pbar:
                env_eval = None
                try: # Create random scenario
                    # Use a consistent seed if needed for debugging, otherwise random
                    eval_seed = None # Or epoch*1000 + episode_idx for semi-reproducible eval
                    if eval_seed is not None: # Only seed if specified
                        np.random.seed(eval_seed) # Seed numpy for scenario generation

                    eval_obstacles_ep = create_obstacles(eval_board_dims, eval_obstacles_count)
                    eval_start_pos_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep)
                    # Ensure start!=goal within the loop
                    attempts_goal = 0
                    while attempts_goal < 100:
                        eval_goals_ep = create_goals(eval_board_dims, eval_agents_count, obstacles=eval_obstacles_ep, current_starts=eval_start_pos_ep)
                        if not np.any(np.all(eval_start_pos_ep == eval_goals_ep, axis=1)):
                            break # Found goals different from starts
                        attempts_goal += 1
                    if attempts_goal >= 100:
                        logger.warning(f"Could not generate non-overlapping start/goals for eval ep {episode_idx+1}, skipping episode.")
                        continue

                    # Create env instance
                    eval_config_instance = config.copy()
                    eval_config_instance.update({ "board_size": eval_board_dims, "num_agents": eval_agents_count,
                                                  "sensing_range": eval_sensing_range, "pad": eval_pad,
                                                  "max_time": eval_max_steps_run, "render_mode": None })
                    env_eval = GraphEnv(config=eval_config_instance, goal=eval_goals_ep, obstacles=eval_obstacles_ep, starting_positions=eval_start_pos_ep)

                    # Run inference (using the modified function via the alias)
                    # Pass the actual device object
                    _, is_success, is_timeout = run_inference_with_shielding(model, env_eval, eval_max_steps_run, config["device"], net_type)

                    if is_success:
                        eval_success_count += 1
                        eval_steps_success.append(env_eval.time)
                    # Optional: log failures/timeouts
                    # elif is_timeout:
                    #     logger.debug(f"Eval ep {episode_idx+1} timed out at step {env_eval.time}")
                    # else:
                    #     logger.debug(f"Eval ep {episode_idx+1} failed (not success, not timeout) at step {env_eval.time}")

                    eval_pbar.set_postfix({"Success": f"{eval_success_count}/{episode_idx+1}"})

                except Exception as e: logger.error(f"Error during evaluation episode {episode_idx+1}: {e}", exc_info=True)
                finally:
                     if env_eval: env_eval.close()
            eval_pbar.close() # End Eval Episode Loop

            eval_success_rate = eval_success_count / tests_episodes_eval if tests_episodes_eval > 0 else 0.0
            avg_steps_succ = np.mean(eval_steps_success) if eval_steps_success else np.nan
            eval_duration = time.time() - eval_start_time
            logger.info(f"Evaluation Complete: SR={eval_success_rate:.4f}, AvgSteps(Succ)={avg_steps_succ:.2f} | Duration={eval_duration:.2f}s")

            current_epoch_data.update({ "Evaluation Episode Success Rate": eval_success_rate, "Evaluation Avg Steps (Success)": avg_steps_succ,
                                       "Evaluation Episodes Tested": tests_episodes_eval, "Evaluation Episodes Succeeded": eval_success_count })

            # Save Best Model
            if eval_success_rate >= best_eval_success_rate:
                 logger.info(f"*** New best eval SR ({eval_success_rate:.4f}), saving model... ***")
                 best_eval_success_rate = eval_success_rate; best_model_path = results_dir / "model_best.pt"
                 try: torch.save(model.state_dict(), best_model_path)
                 except Exception as e: logger.error(f"Failed save best model: {e}")
        # elif run_eval and valid_loader is None:
        #      logger.info("Skipping evaluation phase: No validation data loader available.")
        elif not run_eval and valid_loader is not None:
             logger.debug(f"Skipping evaluation phase: Not an evaluation epoch (freq={eval_frequency}).")
        elif valid_loader is None:
             logger.info("Skipping evaluation phase: No validation data loader configured.")
        # --- End Evaluation Phase ---

        all_epoch_metrics.append(current_epoch_data)
        # Save Metrics Periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            try:
                 metrics_df_partial = pd.DataFrame(all_epoch_metrics)
                 excel_path_partial = results_dir / "training_metrics_partial.xlsx"
                 excel_path_partial.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                 metrics_df_partial.to_excel(excel_path_partial, index=False, engine='openpyxl')
            except ImportError: logger.warning("Module 'openpyxl' missing. Cannot save partial metrics to Excel.")
            except Exception as e: logger.warning(f"Failed save partial metrics: {e}")

        epoch_total_duration = time.time() - epoch_start_time
        logger.info(f"--- Epoch {epoch+1} Finished (Total Duration: {epoch_total_duration:.2f}s) ---")
    # --- End Epoch Loop ---

    total_training_time = time.time() - training_start_time
    logger.info(f"\n--- Training Finished ({total_training_time:.2f}s total) ---")

    # --- Saving Final Results ---
    metrics_df = pd.DataFrame(all_epoch_metrics)
    excel_path = results_dir / "training_metrics.xlsx"
    csv_path = results_dir / "training_metrics.csv"
    logger.info("\nSaving final metrics...")
    try:
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info(f"Saved final epoch metrics to Excel: {excel_path}")
    except ImportError:
        logger.warning("Module 'openpyxl' not found. Attempting to save metrics to CSV instead.")
        try:
            metrics_df.to_csv(csv_path, index=False)
            logger.info(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv:
            logger.error(f"Failed to save metrics to CSV: {e_csv}", exc_info=True)
    except Exception as e_excel:
        logger.error(f"Failed to save metrics to Excel: {e_excel}. Attempting CSV save.", exc_info=True)
        try:
            metrics_df.to_csv(csv_path, index=False)
            logger.info(f"Saved final epoch metrics to CSV: {csv_path}")
        except Exception as e_csv:
            logger.error(f"Failed to save metrics to CSV after Excel failure: {e_csv}", exc_info=True)

    # Save final model
    final_model_path = results_dir / "model_final.pt"
    try: torch.save(model.state_dict(), final_model_path); logger.info(f"Saved final model state to {final_model_path}")
    except Exception as e: logger.error(f"Failed save final model: {e}")

    # Plotting
    logger.info("\nGenerating training plots...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True); fig.suptitle(f"Training Metrics: {exp_name_cleaned}", fontsize=14)
        epochs_axis = metrics_df["Epoch"]
        # Loss
        ax = axes[0]; ax.plot(epochs_axis, metrics_df["Average Training Loss"], marker='.', linestyle='-', color='tab:blue'); ax.set_title("Average Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, linestyle=':')
        # SR
        eval_ran = metrics_df["Evaluation Episode Success Rate"].notna().any(); eval_df = metrics_df.dropna(subset=["Evaluation Episode Success Rate"]) if eval_ran else pd.DataFrame()
        ax = axes[1]
        if eval_ran and not eval_df.empty: ax.plot(eval_df["Epoch"], eval_df["Evaluation Episode Success Rate"], marker='o', linestyle='-', color='tab:green'); ax.set_ylim(-0.05, 1.05)
        else: ax.text(0.5, 0.5, 'No Eval Data', ha='center', va='center', transform=ax.transAxes, color='grey')
        ax.set_title("Evaluation Success Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("Success Rate"); ax.grid(True, linestyle=':')
        # Steps
        ax = axes[2]
        if eval_ran and not eval_df.empty:
            valid_steps_df = eval_df.dropna(subset=["Evaluation Avg Steps (Success)"])
            if not valid_steps_df.empty: ax.plot(valid_steps_df["Epoch"], valid_steps_df["Evaluation Avg Steps (Success)"], marker='s', linestyle='-', color='tab:red'); ax.set_ylabel("Average Steps")
            else: ax.text(0.5, 0.5, 'No Succ Runs', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_ylabel("Average Steps (N/A)")
        else: ax.text(0.5, 0.5, 'No Eval Data', ha='center', va='center', transform=ax.transAxes, color='grey'); ax.set_ylabel("Average Steps (N/A)")
        ax.set_title("Avg Steps (Successful Eval)"); ax.set_xlabel("Epoch"); ax.grid(True, linestyle=':')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plot_path = results_dir / "training_plots.png"; plt.savefig(plot_path, dpi=150); logger.info(f"Saved plots to: {plot_path}"); plt.close(fig)
    except Exception as e: logger.warning(f"Failed generate plots: {e}", exc_info=True)
    # --- End Saving ---

    logger.info("--- Script Finished ---")