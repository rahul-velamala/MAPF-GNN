# File: compare_models.py (Includes Size/Speed Comparison & Diagonal Fix)

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
import traceback
import random
import time # For timing inference

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- --------------- ---

# --- Project Imports ---
# Ensure this script is run from the root directory (rahul-velamala-mapf-gnn-adc)
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Model classes will be imported dynamically
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Please run this script from the project's root directory.")
    sys.exit(1)
# --- --------------- ---


def load_model_and_config(model_dir: Path) -> tuple[torch.nn.Module | None, dict | None, Path | None]:
    """Loads model, its config_used.yaml, and path to weights file."""
    logger.info(f"Loading model and config from: {model_dir}")
    model_load_path_found = None # Store the path to the loaded weights file
    if not model_dir.is_dir():
        logger.error(f"Model directory not found: {model_dir}")
        return None, None, None

    config_path = model_dir / "config_used.yaml"
    if not config_path.is_file():
        logger.error(f"config_used.yaml not found in {model_dir}")
        return None, None, None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None: raise ValueError("Config file is empty or invalid.")
        config["device"] = torch.device(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return None, None, None

    # --- Dynamically Import Model Class ---
    NetworkClass = None
    net_type = config.get("net_type", "gnn")
    msg_type = ""
    try:
        if net_type == "baseline":
            from models.framework_baseline import Network as NetworkClass
        elif net_type == "gnn":
            msg_type = config.get("msg_type", "gcn").lower()
            # framework_gnn handles gcn, message, adc types based on msg_type in config
            from models.framework_gnn import Network as NetworkClass
            config['msg_type'] = msg_type # Ensure it's set for model init
        else:
            raise ValueError(f"Unknown net_type in config: {net_type}")
        logger.info(f"  Using network type: {net_type}" + (f" ({msg_type})" if net_type == "gnn" else ""))
    except (ImportError, ValueError, KeyError) as e:
        logger.error(f"Error importing model class for {model_dir}: {e}")
        return None, config, None

    # --- Load Model Weights ---
    model = NetworkClass(config)
    model.to(config["device"])
    model.eval() # Set to evaluation mode

    model_best_path = model_dir / "model_best.pt"
    model_final_path = model_dir / "model_final.pt"

    if model_best_path.is_file():
        model_load_path_found = model_best_path
    elif model_final_path.is_file():
        logger.warning(f"model_best.pt not found in {model_dir}. Loading model_final.pt instead.")
        model_load_path_found = model_final_path
    else:
        logger.error(f"No model weights (model_best.pt or model_final.pt) found in {model_dir}")
        return None, config, None # Return config even if model fails

    logger.info(f"  Loading model weights from: {model_load_path_found}")
    try:
        model.load_state_dict(torch.load(model_load_path_found, map_location=config["device"]))
        logger.info("  Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model state_dict from {model_load_path_found}: {e}")
        return None, config, model_load_path_found # Return config even if model fails

    # Add net_type to config dict for convenience in evaluation function
    config['_net_type'] = net_type

    return model, config, model_load_path_found


def run_evaluation_episode(model: torch.nn.Module, env: GraphEnv, config: dict, max_steps_eval: int) -> tuple[bool, int]:
    """
    Runs a single evaluation episode with the model and collision shielding.
    (Adapted from example.py/train.py)

    Returns:
        tuple: (is_success, steps_taken)
    """
    model.eval()
    device = config["device"]
    net_type = config["_net_type"]
    num_agents_model = int(config["num_agents"])
    idle_action = 0 # Assuming 0 is the idle action index

    try:
        # Env should already be reset with the specific scenario
        obs = env.getObservations() # Get initial observation after reset
        terminated = np.all(env.reached_goal) # Check if already done
        truncated = env.time >= max_steps_eval
    except Exception as e:
         logger.error(f"\nError getting initial state for evaluation episode: {e}")
         return False, max_steps_eval # Count as failure with max steps

    steps_taken = env.time # Start from env's current time (should be 0 after reset)

    while not terminated and not truncated:
        # Check step limit BEFORE taking the step
        if steps_taken >= max_steps_eval:
            truncated = True
            break

        # Prepare observation for model
        try:
            current_fov_np = obs["fov"]
            current_gso_np = obs["adj_matrix"]
            # Ensure correct shape [B, N, ...] for model (B=1)
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device)
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device)
        except Exception as e:
             logger.error(f"\nError processing observation at step {steps_taken}: {e}")
             return False, max_steps_eval # Treat as failure

        # Get action from model
        with torch.no_grad():
            try:
                if net_type == 'gnn':
                    action_scores = model(fov_tensor, gso_tensor) # Expects (1, N, A)
                else: # baseline
                    action_scores = model(fov_tensor)
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy() # Shape (N,)
            except Exception as e:
                 logger.error(f"\nError during model forward pass at step {steps_taken}: {e}")
                 return False, max_steps_eval # Treat as failure

        # --- Apply Collision Shielding ---
        shielded_actions = proposed_actions.copy()
        current_pos_y = env.positionY.copy(); current_pos_x = env.positionX.copy()
        next_pos_y = current_pos_y.copy(); next_pos_x = current_pos_x.copy()
        active_mask = ~env.reached_goal

        for agent_id in np.where(active_mask)[0]:
             act = proposed_actions[agent_id]; dy, dx = env.action_map_dy_dx.get(act, (0,0))
             next_pos_y[agent_id] += dy; next_pos_x[agent_id] += dx
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1)
        next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)

        if env.obstacles.size > 0:
            active_indices = np.where(active_mask)[0]
            if len(active_indices) > 0:
                 proposed_coords_active=np.stack([next_pos_y[active_indices],next_pos_x[active_indices]],axis=1)
                 obs_coll_active_mask=np.any(np.all(proposed_coords_active[:,np.newaxis,:] == env.obstacles[np.newaxis,:,:],axis=2),axis=1)
                 colliding_agent_indices = active_indices[obs_coll_active_mask]
                 if colliding_agent_indices.size > 0:
                      shielded_actions[colliding_agent_indices] = idle_action
                      next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]
                      next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]
                      active_mask[colliding_agent_indices] = False

        active_indices = np.where(active_mask)[0]
        if len(active_indices) > 1:
            next_coords_check=np.stack([next_pos_y[active_indices], next_pos_x[active_indices]],axis=1)
            current_coords_check=np.stack([current_pos_y[active_indices], current_pos_x[active_indices]],axis=1)
            unique_coords, unique_map, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True)
            vertex_collision_agents = active_indices[np.isin(unique_map, np.where(counts > 1)[0])]
            swapping_agents_list = []
            rel_idx = np.arange(len(active_indices))
            for i in rel_idx:
                 for j in range(i + 1, len(active_indices)):
                     if np.array_equal(next_coords_check[i], current_coords_check[j]) and np.array_equal(next_coords_check[j], current_coords_check[i]):
                         swapping_agents_list.extend([active_indices[i], active_indices[j]])
            swapping_collision_agents = np.unique(swapping_agents_list)
            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents])).astype(int)
            if agents_to_shield_idx.size > 0:
                 shielded_actions[agents_to_shield_idx] = idle_action
        # --- End Collision Shielding ---

        # Step environment with shielded actions
        try:
            obs, reward, terminated, truncated_env, info = env.step(shielded_actions)
            # Environment can signal truncation (e.g. internal error), respect it
            truncated = truncated_env or (info['time'] >= max_steps_eval)
            steps_taken = info['time'] # Update step count based on env time
        except Exception as e:
            logger.error(f"\nError during env.step at step {steps_taken}: {e}")
            return False, max_steps_eval # Treat as failure

    # --- After loop finishes ---
    is_success = terminated and not truncated
    final_steps = env.time # Use the environment's final time

    return is_success, final_steps


def get_model_size(model: torch.nn.Module, model_path: Path) -> dict:
    """Calculates trainable parameters and file size."""
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    file_size_bytes = model_path.stat().st_size if model_path.exists() else 0
    file_size_mb = file_size_bytes / (1024 * 1024)
    return {
        "trainable_params": num_trainable_params,
        "file_size_mb": file_size_mb
    }


def measure_inference_speed(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    batch_size: int,
    num_iterations: int,
    warmup_iterations: int = 10
) -> dict:
    """Measures inference time and throughput."""
    model.eval()
    net_type = config["_net_type"]
    num_agents = int(config["num_agents"])
    pad = int(config["pad"])
    fov_size = (pad * 2) - 1
    fov_channels = 3 # Assuming 3 channels: Obstacles/Agents, Goal, Self

    # --- Generate Dummy Input Data ---
    logger.info(f"  Generating dummy data: Batch={batch_size}, N={num_agents}, C={fov_channels}, H=W={fov_size}")
    dummy_fov = torch.randn(batch_size, num_agents, fov_channels, fov_size, fov_size, device=device, dtype=torch.float32)
    dummy_gso = None
    if net_type == 'gnn':
        dummy_gso = torch.rand(batch_size, num_agents, num_agents, device=device, dtype=torch.float32)
        # Simple GSO: random, could be more structured (e.g., based on distance) if needed
        dummy_gso = (dummy_gso + dummy_gso.transpose(1, 2)) / 2 # Make symmetric
        # Fix for fill_diagonal_ error:
        diag_view = torch.diagonal(dummy_gso, offset=0, dim1=-2, dim2=-1) # Get view of diagonals
        diag_view.zero_() # Zero them out in-place

    logger.info(f"  Running {warmup_iterations} warm-up iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            if net_type == 'gnn':
                _ = model(dummy_fov, dummy_gso)
            else:
                _ = model(dummy_fov)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    logger.info(f"  Running {num_iterations} timed iterations...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            if net_type == 'gnn':
                _ = model(dummy_fov, dummy_gso)
            else:
                _ = model(dummy_fov)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_batch_ms = (total_time / num_iterations) * 1000
    # Throughput in terms of individual *timesteps* (batch items) processed per second
    total_samples_processed = batch_size * num_iterations
    throughput_samples_per_sec = total_samples_processed / total_time

    return {
        "avg_batch_time_ms": avg_time_per_batch_ms,
        "throughput_samples_sec": throughput_samples_per_sec,
        "total_time_sec": total_time,
        "batch_size": batch_size,
        "iterations": num_iterations
    }


# --- Main Comparison Logic ---
def main(args):
    # Set Seed
    if args.seed is not None: logger.info(f"Using fixed random seed: {args.seed}"); np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed);
    else: logger.info("Using random seed.")

    # --- Determine Device for Speed Test ---
    if args.speed_device:
        speed_test_device = torch.device(args.speed_device)
    else:
        speed_test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Speed tests will run on: {speed_test_device}")

    # --- Load Models ---
    model1, config1, path1 = load_model_and_config(args.model1_dir)
    model2, config2, path2 = load_model_and_config(args.model2_dir)

    if model1 is None or config1 is None or path1 is None or \
       model2 is None or config2 is None or path2 is None:
        logger.error("Failed to load models/configs/paths. Exiting.")
        sys.exit(1)

    # --- Compare Size ---
    logger.info("\n--- Model Size Comparison ---")
    size1 = get_model_size(model1, path1)
    size2 = get_model_size(model2, path2)

    print(f"Model: {args.model1_dir.name}")
    print(f"  Trainable Parameters: {size1['trainable_params']:,}")
    print(f"  File Size: {size1['file_size_mb']:.2f} MB")
    print(f"Model: {args.model2_dir.name}")
    print(f"  Trainable Parameters: {size2['trainable_params']:,}")
    print(f"  File Size: {size2['file_size_mb']:.2f} MB")

    # --- Compare Speed ---
    logger.info("\n--- Model Speed Comparison ---")
    speed1 = measure_inference_speed(model1, config1, speed_test_device, args.speed_batch_size, args.speed_iterations)
    speed2 = measure_inference_speed(model2, config2, speed_test_device, args.speed_batch_size, args.speed_iterations)

    print(f"Device: {speed_test_device} | Iterations: {args.speed_iterations} | Batch Size: {args.speed_batch_size}")
    print(f"Model: {args.model1_dir.name}")
    print(f"  Avg. Batch Time: {speed1['avg_batch_time_ms']:.3f} ms")
    print(f"  Throughput: {speed1['throughput_samples_sec']:,.2f} samples/sec")
    print(f"Model: {args.model2_dir.name}")
    print(f"  Avg. Batch Time: {speed2['avg_batch_time_ms']:.3f} ms")
    print(f"  Throughput: {speed2['throughput_samples_sec']:,.2f} samples/sec")

    # --- (Optional) Store Size/Speed Results ---
    perf_results = {
        "model": [args.model1_dir.name, args.model2_dir.name],
        "trainable_params": [size1['trainable_params'], size2['trainable_params']],
        "file_size_mb": [size1['file_size_mb'], size2['file_size_mb']],
        "avg_batch_time_ms": [speed1['avg_batch_time_ms'], speed2['avg_batch_time_ms']],
        "throughput_samples_sec": [speed1['throughput_samples_sec'], speed2['throughput_samples_sec']],
        "speed_test_device": [str(speed_test_device)] * 2,
        "speed_batch_size": [args.speed_batch_size] * 2,
    }
    perf_df = pd.DataFrame(perf_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perf_csv_path = args.output_dir / "size_speed_comparison.csv"
    perf_df.to_csv(perf_csv_path, index=False)
    logger.info(f"\nSize and speed comparison saved to {perf_csv_path}")

    # --- Performance Evaluation (Success Rate / Steps) ---
    # This part remains the same as the previous version...
    logger.info("\n--- Model Performance Comparison (Success Rate & Steps) ---")
    # Sanity Check: Agent Count
    num_agents_model1 = int(config1["num_agents"])
    num_agents_model2 = int(config2["num_agents"])
    if num_agents_model1 != num_agents_model2:
        logger.error(f"Models have different agent counts ({num_agents_model1} vs {num_agents_model2}). Cannot perform SR/Steps comparison.")
        sys.exit(1)
    num_eval_agents = num_agents_model1 # Use this consistent agent count
    logger.info(f"SR/Steps comparison using {num_eval_agents} agents.")

    # Parse Evaluation Scenarios
    eval_settings = []
    if not args.eval_dims and not args.eval_obstacles: eval_settings.append({"dims": (16, 16), "obstacles": 8})
    else:
        dims_list = [(int(d.split(',')[0]), int(d.split(',')[1])) for d in args.eval_dims] if args.eval_dims else [(16,16)]
        obs_list = [int(o) for o in args.eval_obstacles] if args.eval_obstacles else [8]
        for dim in dims_list:
            for obs_count in obs_list: eval_settings.append({"dims": dim, "obstacles": obs_count})
    logger.info(f"Evaluation Settings: {eval_settings}")

    all_results = []
    pbar_settings = tqdm(eval_settings, desc="Scenario Settings")
    for setting in pbar_settings:
        eval_dims = setting["dims"]; eval_rows, eval_cols = eval_dims
        eval_obstacles_count = setting["obstacles"]
        setting_name = f"Map{eval_rows}x{eval_cols}_Obs{eval_obstacles_count}"
        pbar_settings.set_postfix({"Current": setting_name})
        results_model1 = {"success": [], "steps": []}; results_model2 = {"success": [], "steps": []}
        pbar_episodes = tqdm(range(args.episodes), desc=f"  Episodes ({setting_name})", leave=False)
        for episode in pbar_episodes:
            episode_seed = args.seed + episode if args.seed is not None else None; env_eval = None
            try:
                # 1. Generate Scenario
                if episode_seed is not None: np.random.seed(episode_seed)
                eval_obstacles_ep = create_obstacles(eval_dims, eval_obstacles_count)
                eval_start_pos_ep = create_goals(eval_dims, num_eval_agents, obstacles=eval_obstacles_ep)
                eval_goals_ep = create_goals(eval_dims, num_eval_agents, obstacles=eval_obstacles_ep, current_starts=eval_start_pos_ep)

                # 2. Evaluate Model 1
                env_config_m1 = config1.copy()
                env_config_m1.update({"board_size":[eval_rows,eval_cols],"num_agents":num_eval_agents,"max_time":args.max_steps,"render_mode":None})
                env_eval = GraphEnv(config=env_config_m1, goal=eval_goals_ep, obstacles=eval_obstacles_ep, starting_positions=eval_start_pos_ep)
                _, _ = env_eval.reset(seed=episode_seed) # Reset with seed
                success1, steps1 = run_evaluation_episode(model1, env_eval, config1, args.max_steps)
                results_model1["success"].append(success1);
                if success1: results_model1["steps"].append(steps1)
                env_eval.close()

                # 3. Evaluate Model 2 (same scenario)
                env_config_m2 = config2.copy()
                env_config_m2.update({"board_size":[eval_rows,eval_cols],"num_agents":num_eval_agents,"max_time":args.max_steps,"render_mode":None})
                env_eval = GraphEnv(config=env_config_m2, goal=eval_goals_ep, obstacles=eval_obstacles_ep, starting_positions=eval_start_pos_ep)
                _, _ = env_eval.reset(seed=episode_seed) # Reset with same seed
                success2, steps2 = run_evaluation_episode(model2, env_eval, config2, args.max_steps)
                results_model2["success"].append(success2);
                if success2: results_model2["steps"].append(steps2)
                env_eval.close()

            except Exception as e:
                 logger.error(f"Error eval ep {episode+1} setting {setting_name}: {e}", exc_info=True)
                 results_model1["success"].append(False); results_model2["success"].append(False)
            finally:
                 if env_eval is not None: env_eval.close()

        # Aggregate Setting Results
        sr1 = np.mean(results_model1["success"]) if results_model1["success"] else 0.0; avg_steps1 = np.mean(results_model1["steps"]) if results_model1["steps"] else np.nan
        sr2 = np.mean(results_model2["success"]) if results_model2["success"] else 0.0; avg_steps2 = np.mean(results_model2["steps"]) if results_model2["steps"] else np.nan
        all_results.append({"Setting": setting_name,"Map Size": f"{eval_rows}x{eval_cols}","Obstacles": eval_obstacles_count,"Agents": num_eval_agents,
                            f"{args.model1_dir.name}_SR": sr1, f"{args.model1_dir.name}_AvgSteps": avg_steps1,
                            f"{args.model2_dir.name}_SR": sr2, f"{args.model2_dir.name}_AvgSteps": avg_steps2})
        logger.info(f"  {setting_name}: M1 SR={sr1:.3f}, Steps={avg_steps1:.2f} | M2 SR={sr2:.3f}, Steps={avg_steps2:.2f}")

    # Save Performance Results & Plots
    if not all_results: logger.warning("No performance evaluation results generated."); sys.exit(0)
    results_df = pd.DataFrame(all_results)
    csv_path = args.output_dir / "performance_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Performance comparison results saved to {csv_path}")

    # Plotting (same as before)
    model1_name = args.model1_dir.name; model2_name = args.model2_dir.name; settings_labels = results_df["Setting"].tolist()
    x = np.arange(len(settings_labels)); width = 0.35
    # SR Plot
    fig_sr, ax_sr = plt.subplots(figsize=(max(10, len(settings_labels)*1.5), 6))
    rects1_sr = ax_sr.bar(x - width/2, results_df[f"{model1_name}_SR"], width, label=model1_name, color='skyblue')
    rects2_sr = ax_sr.bar(x + width/2, results_df[f"{model2_name}_SR"], width, label=model2_name, color='lightcoral')
    ax_sr.set_ylabel('Success Rate'); ax_sr.set_title('Model Comparison: Success Rate'); ax_sr.set_xticks(x); ax_sr.set_xticklabels(settings_labels, rotation=45, ha="right")
    ax_sr.legend(); ax_sr.set_ylim(0, 1.1); ax_sr.bar_label(rects1_sr, padding=3, fmt='%.2f', fontsize=8); ax_sr.bar_label(rects2_sr, padding=3, fmt='%.2f', fontsize=8)
    fig_sr.tight_layout(); sr_plot_path = args.output_dir / "performance_comparison_success_rate.png"; fig_sr.savefig(sr_plot_path)
    logger.info(f"Success Rate plot saved to {sr_plot_path}"); plt.close(fig_sr)
    # Steps Plot
    fig_steps, ax_steps = plt.subplots(figsize=(max(10, len(settings_labels)*1.5), 6))
    steps1 = results_df[f"{model1_name}_AvgSteps"].fillna(0); steps2 = results_df[f"{model2_name}_AvgSteps"].fillna(0)
    rects1_steps = ax_steps.bar(x - width/2, steps1, width, label=model1_name, color='skyblue')
    rects2_steps = ax_steps.bar(x + width/2, steps2, width, label=model2_name, color='lightcoral')
    ax_steps.set_ylabel('Average Steps (Successful Episodes)'); ax_steps.set_title('Model Comparison: Average Steps'); ax_steps.set_xticks(x); ax_steps.set_xticklabels(settings_labels, rotation=45, ha="right")
    ax_steps.legend(); ax_steps.set_ylim(0, args.max_steps * 1.1); ax_steps.bar_label(rects1_steps, padding=3, fmt='%.1f', fontsize=8); ax_steps.bar_label(rects2_steps, padding=3, fmt='%.1f', fontsize=8)
    fig_steps.tight_layout(); steps_plot_path = args.output_dir / "performance_comparison_avg_steps.png"; fig_steps.savefig(steps_plot_path)
    logger.info(f"Average Steps plot saved to {steps_plot_path}"); plt.close(fig_steps)

    # Final Conclusion
    logger.info("\n--- Final Summary ---")
    print("\nSize & Speed:")
    print(perf_df.to_string(index=False)) # Print size/speed table
    avg_sr1 = results_df[f"{model1_name}_SR"].mean(); avg_sr2 = results_df[f"{model2_name}_SR"].mean()
    avg_steps1_overall = results_df[f"{model1_name}_AvgSteps"].mean(); avg_steps2_overall = results_df[f"{model2_name}_AvgSteps"].mean()
    logger.info(f"\nOverall Avg Performance (SR/Steps):")
    logger.info(f"{model1_name}: Avg SR = {avg_sr1:.3f}, Avg Steps (Succ) = {avg_steps1_overall:.2f}")
    logger.info(f"{model2_name}: Avg SR = {avg_sr2:.3f}, Avg Steps (Succ) = {avg_steps2_overall:.2f}")

    # Combine conclusions based on size, speed, and performance
    print("\nComparative Conclusion:")
    # Size
    if size1['trainable_params'] < size2['trainable_params'] * 0.95: # Model 1 significantly smaller
        print(f"- Size: {model1_name} is significantly smaller ({size1['trainable_params']:,} vs {size2['trainable_params']:,} params).")
    elif size2['trainable_params'] < size1['trainable_params'] * 0.95: # Model 2 significantly smaller
        print(f"- Size: {model2_name} is significantly smaller ({size2['trainable_params']:,} vs {size1['trainable_params']:,} params).")
    else:
        print("- Size: Models have comparable parameter counts.")
    # Speed
    if speed1['avg_batch_time_ms'] < speed2['avg_batch_time_ms'] * 0.95: # Model 1 significantly faster
        print(f"- Speed: {model1_name} is significantly faster ({speed1['avg_batch_time_ms']:.2f}ms vs {speed2['avg_batch_time_ms']:.2f}ms avg batch time).")
    elif speed2['avg_batch_time_ms'] < speed1['avg_batch_time_ms'] * 0.95: # Model 2 significantly faster
        print(f"- Speed: {model2_name} is significantly faster ({speed2['avg_batch_time_ms']:.2f}ms vs {speed1['avg_batch_time_ms']:.2f}ms avg batch time).")
    else:
        print("- Speed: Models have comparable inference speeds.")
    # Performance (SR/Steps)
    if avg_sr1 > avg_sr2 + 0.02: # Model 1 better SR
        print(f"- Performance: {model1_name} achieved a higher average Success Rate ({avg_sr1:.3f} vs {avg_sr2:.3f}).")
        if avg_steps1_overall < avg_steps2_overall: print(f"  It was also faster on average ({avg_steps1_overall:.2f} vs {avg_steps2_overall:.2f} steps).")
    elif avg_sr2 > avg_sr1 + 0.02: # Model 2 better SR
        print(f"- Performance: {model2_name} achieved a higher average Success Rate ({avg_sr2:.3f} vs {avg_sr1:.3f}).")
        if avg_steps2_overall < avg_steps1_overall: print(f"  It was also faster on average ({avg_steps2_overall:.2f} vs {avg_steps1_overall:.2f} steps).")
    else: # Similar SR
        print(f"- Performance: Models have similar average Success Rates (SR1≈{avg_sr1:.3f}, SR2≈{avg_sr2:.3f}).")
        if avg_steps1_overall < avg_steps2_overall: print(f"  However, {model1_name} was faster on average ({avg_steps1_overall:.2f} vs {avg_steps2_overall:.2f} steps).")
        elif avg_steps2_overall < avg_steps1_overall: print(f"  However, {model2_name} was faster on average ({avg_steps2_overall:.2f} vs {avg_steps1_overall:.2f} steps).")
        else: print("  Average steps on successful runs were also similar.")


    logger.info("--- Comparison Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two trained MAPF models (Performance, Size, Speed).")
    # Model Selection
    parser.add_argument("--model1_dir", type=Path, required=True, help="Path to the results directory of the first model.")
    parser.add_argument("--model2_dir", type=Path, required=True, help="Path to the results directory of the second model.")
    # Performance Evaluation Args
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes per setting for SR/Steps.")
    parser.add_argument("--max_steps", type=int, default=120, help="Maximum steps allowed per evaluation episode.")
    parser.add_argument("--eval_dims", type=str, nargs='*', help="Evaluation map dimensions (rows,cols). E.g., 16,16 28,28")
    parser.add_argument("--eval_obstacles", type=str, nargs='*', help="Number of obstacles. E.g., 8 16")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible evaluation scenarios.")
    # Speed Evaluation Args
    parser.add_argument("--speed_iterations", type=int, default=100, help="Number of iterations for inference speed measurement.")
    parser.add_argument("--speed_batch_size", type=int, default=32, help="Batch size for inference speed measurement.")
    parser.add_argument("--speed_device", type=str, default=None, help="Device for speed test (e.g., 'cpu', 'cuda:0'). Auto-detects if None.")
    # Output
    parser.add_argument("--output_dir", type=Path, default=Path("results/model_comparison"), help="Directory to save comparison results (plots, CSVs).")

    parsed_args = parser.parse_args()
    main(parsed_args)