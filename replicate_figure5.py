# File: replicate_figure5.py (Parallelized Version with Spawn Fix and Plot Title Fix)

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
import math
import time
from multiprocessing import Pool, cpu_count, set_start_method # Import set_start_method
from functools import partial

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
# --- --------------- ---

# --- Project Imports ---
try:
    from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
    # Dynamic imports inside worker/main
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Please run this script from the project's root directory.")
    sys.exit(1)
# --- --------------- ---

# --- Helper Functions (run_evaluation_episode_replication - unchanged) ---
def run_evaluation_episode_replication(
    model: torch.nn.Module,
    env: GraphEnv, # Env is already created with correct N, size, obs
    config: dict,
    max_steps_eval: int
) -> tuple[bool, int]:
    """Runs a single evaluation episode with the model and collision shielding."""
    model.eval()
    device = config["device"]
    net_type = config["_net_type"]
    num_eval_agents = env.nb_agents
    idle_action = 0
    try:
        obs = env.getObservations()
        terminated = np.all(env.reached_goal)
        truncated = env.time >= max_steps_eval
    except Exception as e: logger.error(f"\nError getting initial state: {e}"); return False, max_steps_eval
    steps_taken = env.time
    while not terminated and not truncated:
        if steps_taken >= max_steps_eval: truncated = True; break
        try:
            current_fov_np = obs["fov"]; current_gso_np = obs["adj_matrix"]
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device)
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device)
        except Exception as e: logger.error(f"\nError processing obs @ step {steps_taken}: {e}"); return False, max_steps_eval
        with torch.no_grad():
            try:
                if net_type == 'gnn': action_scores = model(fov_tensor, gso_tensor)
                else: action_scores = model(fov_tensor)
                if action_scores.shape[1] != num_eval_agents: raise RuntimeError(f"Model output N ({action_scores.shape[1]}) != Env N ({num_eval_agents})")
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
            except Exception as e: logger.error(f"\nError model forward @ step {steps_taken}: {e}"); return False, max_steps_eval
        # Collision Shielding
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
                      shielded_actions[colliding_agent_indices] = idle_action; next_pos_y[colliding_agent_indices] = current_pos_y[colliding_agent_indices]; next_pos_x[colliding_agent_indices] = current_pos_x[colliding_agent_indices]; active_mask[colliding_agent_indices] = False
        active_indices = np.where(active_mask)[0]
        if len(active_indices) > 1:
            next_coords_check=np.stack([next_pos_y[active_indices], next_pos_x[active_indices]],axis=1); current_coords_check=np.stack([current_pos_y[active_indices], current_pos_x[active_indices]],axis=1)
            unique_coords, unique_map, counts = np.unique(next_coords_check, axis=0, return_inverse=True, return_counts=True); vertex_collision_agents = active_indices[np.isin(unique_map, np.where(counts > 1)[0])]
            swapping_agents_list = []
            rel_idx = np.arange(len(active_indices))
            for i in rel_idx:
                 for j in range(i + 1, len(active_indices)):
                     if np.array_equal(next_coords_check[i], current_coords_check[j]) and np.array_equal(next_coords_check[j], current_coords_check[i]): swapping_agents_list.extend([active_indices[i], active_indices[j]])
            swapping_collision_agents = np.unique(swapping_agents_list)
            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents])).astype(int)
            if agents_to_shield_idx.size > 0: shielded_actions[agents_to_shield_idx] = idle_action
        # Step environment
        try:
            obs, reward, terminated, truncated_env, info = env.step(shielded_actions)
            truncated = truncated_env or (info['time'] >= max_steps_eval)
            steps_taken = info['time']
        except Exception as e: logger.error(f"\nError env.step @ step {steps_taken}: {e}"); return False, max_steps_eval
    is_success = terminated and not truncated
    final_steps = env.time
    return is_success, final_steps

# --- Worker Function (Loads models internally) ---
def run_episode_worker(
    episode_index: int, # Keep track of which episode this is
    n_agents_test: int,
    eval_dims: tuple[int, int],
    eval_obstacles_count: int,
    model_paths_configs: list[tuple[Path, dict]], # List of (model_dir_path, loaded_config)
    max_steps: int,
    base_seed: int | None,
) -> dict:
    """ Worker function to run one episode for all specified models. """
    episode_seed = base_seed + episode_index if base_seed is not None else None
    if episode_seed is not None:
        np.random.seed(episode_seed); torch.manual_seed(episode_seed); random.seed(episode_seed)

    # Load models *inside* worker
    worker_models = {}
    # Need to import model classes here for dynamic loading
    from models.framework_baseline import Network as BaselineNetwork
    from models.framework_gnn import Network as GNNNetwork

    for model_dir, config in model_paths_configs:
        model = None
        NetworkClass = None
        net_type = config.get("net_type", "gnn")
        msg_type = config.get("msg_type", "gcn").lower() # Ensure msg_type exists
        try:
            if net_type == "baseline": NetworkClass = BaselineNetwork
            elif net_type == "gnn": NetworkClass = GNNNetwork; config['msg_type'] = msg_type # Pass msg_type
            else: raise ValueError(f"Unknown net_type: {net_type}")

            model = NetworkClass(config)
            # Determine device string and convert back to device object INSIDE worker
            device_str = config.get("device_str", "cuda:0" if torch.cuda.is_available() else "cpu")
            worker_device = torch.device(device_str)
            config["device"] = worker_device # Set device object for this worker's config copy

            model.to(worker_device)
            model.eval()
            model_best_path = model_dir / "model_best.pt"
            model_final_path = model_dir / "model_final.pt"
            model_load_path = model_best_path if model_best_path.is_file() else model_final_path
            if not model_load_path.is_file(): raise FileNotFoundError(f"No model weights found in {model_dir}")
            state_dict = torch.load(model_load_path, map_location=worker_device) # Load to worker device
            model.load_state_dict(state_dict)
            worker_models[model_dir.name] = {"model": model, "config": config}
        except Exception as e:
            # Log error with process ID for clarity
            logger.error(f"Worker {os.getpid()} failed to load model {model_dir.name}: {e}", exc_info=True)
            continue

    if not worker_models:
        logger.error(f"Worker {os.getpid()} failed to load any models for episode {episode_index}.")
        return {"episode": episode_index, "results": {}}

    # Run the episode
    episode_results = {"episode": episode_index, "results": {}}
    env_eval = None
    try:
        eval_rows, eval_cols = eval_dims
        eval_obstacles_ep = create_obstacles(eval_dims, eval_obstacles_count)
        eval_start_pos_ep = create_goals(eval_dims, n_agents_test, obstacles=eval_obstacles_ep)
        eval_goals_ep = create_goals(eval_dims, n_agents_test, obstacles=eval_obstacles_ep, current_starts=eval_start_pos_ep)

        for model_name, model_data in worker_models.items():
            model = model_data["model"]
            config = model_data["config"]
            env_config_eval = config.copy()
            env_config_eval.update({"board_size": [eval_rows, eval_cols], "num_agents": n_agents_test, "max_time": max_steps, "render_mode": None})
            env_eval = GraphEnv(config=env_config_eval, goal=eval_goals_ep, obstacles=eval_obstacles_ep, starting_positions=eval_start_pos_ep)
            _, _ = env_eval.reset(seed=episode_seed)
            success, steps = run_evaluation_episode_replication(model, env_eval, config, max_steps)
            episode_results["results"][model_name] = (success, steps if success else -1)
            env_eval.close(); env_eval = None
    except Exception as e:
         logger.error(f"Worker {os.getpid()} error episode {episode_index} N={n_agents_test}: {e}", exc_info=True)
         for model_name in worker_models: episode_results["results"].setdefault(model_name, (False, -1))
    finally:
        if env_eval is not None: env_eval.close()
    return episode_results

# --- Map Detail Calculation ---
def get_map_details_for_density(
    num_agents_test: int, base_map_side: int, base_num_agents: int,
    obstacle_density: float = 0.10
) -> tuple[tuple[int,int], int]:
    if num_agents_test <= 0: return (10, 10), 0
    base_area = base_map_side * base_map_side; base_num_obstacles = int(obstacle_density * base_area)
    if base_area <= 0: base_density = obstacle_density # Handle edge case
    else: base_density = (base_num_agents + base_num_obstacles) / base_area
    denominator = base_density - obstacle_density
    if denominator <= 1e-6 or base_num_agents <= 0: target_area = base_area * (num_agents_test / max(1, base_num_agents)); logger.warning(f"Using simple area scaling for N={num_agents_test}.")
    else: target_area = num_agents_test / denominator
    target_side = int(math.ceil(math.sqrt(max(1.0, target_area)))); target_dims = (target_side, target_side)
    target_num_obstacles = int(obstacle_density * target_side * target_side)
    return target_dims, target_num_obstacles

# --- Main Replication Logic ---
def main(args):
    if args.seed is not None: logger.info(f"Using fixed random seed: {args.seed}"); np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed);
    else: logger.info("Using random seed.")

    # Load Model Configs and Paths
    model_paths_configs = []
    model_names_list = []
    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str)
        config_path = model_dir / "config_used.yaml"
        if not model_dir.is_dir() or not config_path.is_file(): logger.warning(f"Skipping invalid model directory: {model_dir}"); continue
        try:
            with open(config_path, "r") as f: config = yaml.safe_load(f)
            if config is None: raise ValueError("Empty config")
            config["device_str"] = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu") # Store device string
            config["device"] = torch.device(config["device_str"]) # Create device object for main process needs
            config['_net_type'] = config.get("net_type", "gnn")
            model_paths_configs.append((model_dir, config))
            model_names_list.append(model_dir.name)
        except Exception as e: logger.error(f"Failed to load config for {model_dir}: {e}")
    if not model_paths_configs: logger.error("No valid model configs found. Exiting."); sys.exit(1)
    logger.info(f"Prepared {len(model_paths_configs)} model configs for evaluation: {model_names_list}")

    # Determine Base Density
    ref_config = model_paths_configs[0][1]
    try:
        ref_map_dims = ref_config.get("board_size", [28, 28]); ref_rows, ref_cols = map(int, ref_map_dims)
        ref_map_side = int(math.sqrt(ref_rows * ref_cols)); ref_num_agents = int(ref_config.get("num_agents", 5)) # Use trained N
        ref_obs_density = args.obstacle_density
        logger.info(f"Calculating maps based on reference: {ref_map_side}x{ref_map_side} map, {ref_num_agents} agents, {ref_obs_density*100:.1f}% obstacles")
    except Exception as e: logger.error(f"Failed get ref params: {e}. Using defaults."); ref_map_side=28; ref_num_agents=5; ref_obs_density=0.10

    # Evaluation Loop
    all_results_list = []
    num_workers = min(args.num_workers, cpu_count())
    logger.info(f"Using {num_workers} parallel workers for episodes.")

    pbar_agents = tqdm(args.agent_counts, desc="Agent Counts")
    for n_agents_test in pbar_agents:
        pbar_agents.set_postfix({"N": n_agents_test})
        eval_dims, eval_obstacles_count = get_map_details_for_density(n_agents_test, ref_map_side, ref_num_agents, ref_obs_density)
        logger.info(f"  Testing N={n_agents_test}: Map Size={eval_dims[0]}x{eval_dims[1]}, Obstacles={eval_obstacles_count}")

        # Run episodes in parallel
        episode_results_map = {}
        with Pool(processes=num_workers) as pool:
            with tqdm(total=args.episodes, desc=f"    Episodes (N={n_agents_test})", leave=False) as pbar_episodes:
                # Fixed arguments for the worker function
                fixed_args = {
                    "n_agents_test": n_agents_test,
                    "eval_dims": eval_dims,
                    "eval_obstacles_count": eval_obstacles_count,
                    "model_paths_configs": model_paths_configs,
                    "max_steps": args.max_steps,
                    "base_seed": args.seed
                }
                # Use partial to fix arguments, map over episode index
                worker_func_partial = partial(run_episode_worker, **fixed_args)
                for result_dict in pool.imap_unordered(worker_func_partial, range(args.episodes)):
                    if result_dict: # Ensure worker didn't return None on total failure
                         episode_results_map[result_dict["episode"]] = result_dict["results"]
                    pbar_episodes.update(1)

        # Aggregate results
        current_n_results_agg = {name: {"success": [], "steps": []} for name in model_names_list}
        for ep_idx in range(args.episodes):
            ep_result = episode_results_map.get(ep_idx, {}) # Get episode results or empty dict
            for model_name in model_names_list:
                model_ep_res = ep_result.get(model_name)
                if model_ep_res:
                    success, steps = model_ep_res
                    current_n_results_agg[model_name]["success"].append(success)
                    if success: current_n_results_agg[model_name]["steps"].append(steps)
                else:
                    current_n_results_agg[model_name]["success"].append(False)

        log_str = f"  N={n_agents_test} Aggregated Results:"
        for model_name, results in current_n_results_agg.items():
            sr = np.mean(results["success"]) if results["success"] else 0.0
            avg_steps = np.mean(results["steps"]) if results["steps"] else np.nan
            all_results_list.append({"Model": model_name, "# Robots": n_agents_test, "Success Rate": sr, "Avg Steps (Success)": avg_steps})
            log_str += f" | {model_name}: SR={sr:.3f}, AvgSteps={avg_steps:.2f}"
        logger.info(log_str)

    # Save and Plot Results
    if not all_results_list: logger.warning("No evaluation results generated."); sys.exit(0)
    results_df = pd.DataFrame(all_results_list); args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "figure5_replication_data.csv"; results_df.to_csv(csv_path, index=False)
    logger.info(f"\nReplication results saved to {csv_path}")
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); model_names = results_df["Model"].unique()
    markers = ['-o', '--x', ':s', '-.^', '--p', ':*']; colors = plt.cm.get_cmap('tab10', len(model_names))
    # SR Plot
    ax = axes[0]
    for i, model_name in enumerate(model_names):
        model_df = results_df[results_df["Model"] == model_name].sort_values("# Robots")
        ax.plot(model_df["# Robots"], model_df["Success Rate"], markers[i % len(markers)], label=model_name, color=colors(i))
    ax.set_xlabel("# Robots Tested"); ax.set_ylabel("Success Rate"); ax.set_title("a: Success Rate vs. # Robots")
    ax.set_ylim(min(0.7, results_df["Success Rate"].min() - 0.05) if not results_df["Success Rate"].empty else 0.7, 1.01)
    ax.legend(title="Model"); ax.grid(True)
    # Steps Plot
    ax = axes[1]
    for i, model_name in enumerate(model_names):
        model_df = results_df[results_df["Model"] == model_name].sort_values("# Robots")
        ax.plot(model_df["# Robots"], model_df["Avg Steps (Success)"], markers[i % len(markers)], label=model_name, color=colors(i))
    ax.set_xlabel("# Robots Tested"); ax.set_ylabel("Avg Steps (Successful Runs)"); ax.set_title("b: Avg Steps vs. # Robots")
    ax.legend(title="Model"); ax.grid(True)
    min_step = results_df["Avg Steps (Success)"].dropna().min(); max_step = results_df["Avg Steps (Success)"].dropna().max()
    if not np.isnan(min_step) and not np.isnan(max_step): ax.set_ylim(min_step * 0.9, max_step * 1.1)
    fig.suptitle("Replication: Performance vs. Number of Robots (Constant Density)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); plot_path = args.output_dir / "figure5_replication_plots.png"; fig.savefig(plot_path)
    logger.info(f"Replication plots saved to {plot_path}"); plt.close(fig)
    logger.info("--- Replication Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate Figure 5 (Performance vs N) from IROS 2020 paper.")
    parser.add_argument("--model_dirs", type=str, nargs='+', required=True, help="Paths to results directories of models to compare.")
    parser.add_argument("--agent_counts", type=int, nargs='+', default=[20, 30, 40, 50, 60], help="List of agent counts to test.")
    parser.add_argument("--obstacle_density", type=float, default=0.10, help="Obstacle density for scaling map size.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes per agent count.")
    parser.add_argument("--max_steps", type=int, default=250, help="Maximum steps per evaluation episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() // 2), help="Number of parallel workers for episodes.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/figure5_replication"), help="Directory to save results.")

    # Set multiprocessing start method early, within the name==main block
    try:
        set_start_method('spawn', force=True)
        logger.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logger.warning("Multiprocessing context already set. Using existing context.")
        # This might happen if you run this in an environment like Jupyter
        # where the context might be implicitly set. 'spawn' is still preferred.

    parsed_args = parser.parse_args()
    main(parsed_args)