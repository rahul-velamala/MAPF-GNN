# File: evaluate_models.py
# (Evaluates with Avg Makespan, Flowtime, and Flowtime Increase)

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
import traceback
import random
import time
from collections import defaultdict

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- --------------- ---

# --- Project Imports ---
try:
    # Use make_env to load specific scenarios
    from data_generation.record import make_env
    # Dynamic imports for models inside load function
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Please run this script from the project's root directory.")
    sys.exit(1)
# --- --------------- ---

def load_model_and_config_eval(model_dir: Path) -> tuple[torch.nn.Module | None, dict | None]:
    """Loads model and its config_used.yaml."""
    logger.debug(f"Loading model and config from: {model_dir}")
    if not model_dir.is_dir():
        logger.error(f"Model directory not found: {model_dir}")
        return None, None

    config_path = model_dir / "config_used.yaml"
    if not config_path.is_file():
        logger.error(f"config_used.yaml not found in {model_dir}")
        # Try loading from main config if used_config is missing (less ideal)
        config_path_guess_str = f"configs/{model_dir.name}.yaml" # Basic guess
        # More robust guess if exp_name in config_used might differ from dir name
        # For example, if config_used.yaml had "exp_name: adc_train10D..."
        # and directory was "adc_train10D_map10x10_r5_p5"
        # This part might need adjustment if your naming scheme is complex
        config_path = Path(config_path_guess_str)
        if not config_path.is_file():
             logger.error(f"Could not find config_used.yaml or guessed config for {model_dir.name}")
             return None, None
        logger.warning(f"Using guessed config path: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None: raise ValueError("Config file is empty or invalid.")
        # Convert device string back to object for loading
        config["device_str"] = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        config["device"] = torch.device(config["device_str"])
        config['_model_name'] = model_dir.name # Store model name in config
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return None, None

    # --- Dynamically Import Model Class ---
    NetworkClass = None
    net_type = config.get("net_type", "gnn")
    msg_type = ""
    try:
        if net_type == "baseline":
            from models.framework_baseline import Network as NetworkClass
        elif net_type == "gnn":
            # framework_gnn handles gcn, message, adc types
            from models.framework_gnn import Network as NetworkClass
            msg_type = config.get("msg_type", "gcn").lower() # Default to gcn if missing
            config['msg_type'] = msg_type # Ensure it's set for model init
        else:
            raise ValueError(f"Unknown net_type in config: {net_type}")
        logger.debug(f"  Using network type: {net_type}" + (f" ({msg_type})" if net_type == "gnn" else ""))
    except (ImportError, ValueError, KeyError) as e:
        logger.error(f"Error importing model class for {model_dir}: {e}")
        return None, config

    # --- Load Model Weights ---
    model = NetworkClass(config)
    model.to(config["device"])
    model.eval()

    model_best_path = model_dir / "model_best.pt"
    model_final_path = model_dir / "model_final.pt"
    model_load_path = None

    if model_best_path.is_file(): model_load_path = model_best_path
    elif model_final_path.is_file(): model_load_path = model_final_path; logger.warning(f"Using model_final.pt for {model_dir.name}")
    else: logger.error(f"No model weights found in {model_dir}"); return None, config

    logger.debug(f"  Loading model weights from: {model_load_path}")
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=config["device"]))
        logger.debug("  Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading state_dict from {model_load_path}: {e}")
        return None, config

    config['_net_type'] = net_type # Add net_type for convenience
    return model, config

def run_evaluation_episode_metrics(
    model: torch.nn.Module,
    env: object, # GraphEnv | None
    config: dict, # Model's training config
    max_steps_eval: int,
    case_path: Path
    ) -> dict:
    """
    Runs one episode, returns detailed metrics.
    """
    num_agents_from_config = int(config.get("num_agents",1)) # Fallback if env is None

    # Default metrics for failure cases
    metrics = {
        'success': False,
        'executed_makespan': max_steps_eval,
        'executed_soc': max_steps_eval * (env.nb_agents if env else num_agents_from_config),
        'optimal_soc': np.nan,
        'optimal_makespan': np.nan,
        'inference_times': []
    }

    if env is None:
        logger.warning(f"Skipping episode evaluation for {case_path.name}: env creation failed.")
        return metrics

    model.eval(); device = config["device"]; net_type = config["_net_type"]
    num_eval_agents = env.nb_agents; idle_action = 0

    try:
        obs, _ = env.reset(seed=np.random.randint(1e6)) # Random seed per episode for robustness
        terminated = np.all(env.reached_goal)
        truncated = env.time >= max_steps_eval
    except Exception as e: logger.error(f"Error reset env for {case_path.name}: {e}", exc_info=True); return metrics

    step_count = 0
    agent_completion_times = np.full(num_eval_agents, max_steps_eval, dtype=int)

    while not terminated and not truncated:
        if step_count >= max_steps_eval: truncated = True; break

        start_inf_time = time.perf_counter()
        try:
            current_fov_np = obs["fov"]; current_gso_np = obs["adj_matrix"]
            fov_tensor = torch.from_numpy(current_fov_np).float().unsqueeze(0).to(device)
            gso_tensor = torch.from_numpy(current_gso_np).float().unsqueeze(0).to(device)
        except Exception as e: logger.error(f"Error processing obs {case_path.name} @ step {step_count}: {e}"); return metrics
        with torch.no_grad():
            try:
                if net_type == 'gnn': action_scores = model(fov_tensor, gso_tensor)
                else: action_scores = model(fov_tensor)
                if action_scores.shape[1] != num_eval_agents:
                    logger.error(f"Model output N ({action_scores.shape[1]}) != Env N ({num_eval_agents}) for {case_path.name}")
                    return metrics # Critical error
                proposed_actions = action_scores.argmax(dim=-1).squeeze(0).cpu().numpy()
            except Exception as e: logger.error(f"Error model forward {case_path.name} @ step {step_count}: {e}"); return metrics
        end_inf_time = time.perf_counter()
        metrics['inference_times'].append((end_inf_time - start_inf_time) * 1000)

        # --- Collision Shielding (Identical to train.py) ---
        shielded_actions = proposed_actions.copy(); current_pos_y = env.positionY.copy(); current_pos_x = env.positionX.copy()
        next_pos_y = current_pos_y.copy(); next_pos_x = current_pos_x.copy(); active_mask = ~env.reached_goal
        for agent_id in np.where(active_mask)[0]: act = proposed_actions[agent_id]; dy, dx = env.action_map_dy_dx.get(act, (0,0)); next_pos_y[agent_id] += dy; next_pos_x[agent_id] += dx
        next_pos_y[active_mask] = np.clip(next_pos_y[active_mask], 0, env.board_rows - 1); next_pos_x[active_mask] = np.clip(next_pos_x[active_mask], 0, env.board_cols - 1)
        if env.obstacles.size > 0: # Obstacle
            active_indices = np.where(active_mask)[0]
            if len(active_indices) > 0:
                 prop_coords_act=np.stack([next_pos_y[active_indices],next_pos_x[active_indices]],axis=1); obs_coll_act_mask=np.any(np.all(prop_coords_act[:,np.newaxis,:] == env.obstacles[np.newaxis,:,:],axis=2),axis=1)
                 coll_agent_idx = active_indices[obs_coll_act_mask]
                 if coll_agent_idx.size > 0: shielded_actions[coll_agent_idx] = idle_action; next_pos_y[coll_agent_idx] = current_pos_y[coll_agent_idx]; next_pos_x[coll_agent_idx] = current_pos_x[coll_agent_idx]; active_mask[coll_agent_idx] = False
        active_indices = np.where(active_mask)[0] # Agent-Agent
        if len(active_indices) > 1:
            next_coords_chk=np.stack([next_pos_y[active_indices], next_pos_x[active_indices]],axis=1); curr_coords_chk=np.stack([current_pos_y[active_indices], current_pos_x[active_indices]],axis=1)
            _, u_map, counts = np.unique(next_coords_chk, axis=0, return_inverse=True, return_counts=True); v_coll_agents = active_indices[np.isin(u_map, np.where(counts > 1)[0])]
            swap_agents_list = []; rel_idx = np.arange(len(active_indices))
            for i in rel_idx:
                 for j in range(i + 1, len(active_indices)):
                     if np.array_equal(next_coords_chk[i], curr_coords_chk[j]) and np.array_equal(next_coords_chk[j], curr_coords_chk[i]): swap_agents_list.extend([active_indices[i], active_indices[j]])
            swap_coll_agents = np.unique(swap_agents_list); agents_to_shield = np.unique(np.concatenate([v_coll_agents, swap_coll_agents])).astype(int)
            if agents_to_shield.size > 0: shielded_actions[agents_to_shield] = idle_action
        # End Shielding

        try:
            obs, _, terminated, truncated_env, info = env.step(shielded_actions)
            truncated = truncated_env or (info['time'] >= max_steps_eval)
            step_count = info['time']
            for i in range(num_eval_agents):
                if env.reached_goal[i] and agent_completion_times[i] == max_steps_eval:
                    agent_completion_times[i] = step_count
        except Exception as e: logger.error(f"Error env.step {case_path.name} @ step {step_count}: {e}"); return metrics

    # --- After loop ---
    metrics['success'] = terminated and not truncated
    metrics['executed_makespan'] = env.time
    metrics['executed_soc'] = np.sum(agent_completion_times) # Sum of actual (or penalty) completion times

    # --- Load Optimal Costs ---
    solution_file = case_path / "solution.yaml"
    if solution_file.is_file():
        try:
            with open(solution_file, 'r') as f_sol: solution_data = yaml.safe_load(f_sol)
            cost_opt = solution_data.get('cost')
            if cost_opt is not None: metrics['optimal_soc'] = float(cost_opt)
            schedule = solution_data.get('schedule')
            if schedule and isinstance(schedule, dict):
                max_t_opt = 0
                if schedule.values(): # Check if schedule is not empty
                    for agent_path in schedule.values():
                        if agent_path and isinstance(agent_path, list) and len(agent_path) > 0:
                            max_t_opt = max(max_t_opt, agent_path[-1]['t'])
                metrics['optimal_makespan'] = float(max_t_opt)
        except Exception as e: logger.warning(f"Error reading {solution_file}: {e}")
    else: logger.warning(f"solution.yaml not found for: {case_path.name}")
    return metrics

# --- Main Evaluation Logic ---
def main(args):
    models_data = {}; model_param_counts = {}
    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str)
        model, config = load_model_and_config_eval(model_dir)
        if model and config:
            model_name = config.get('_model_name', model_dir.name) # Use name from config if set
            models_data[model_name] = {'model': model, 'config': config}
            model_param_counts[model_name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else: logger.warning(f"Failed to load {model_dir}. Skipping.")
    if not models_data: logger.error("No models loaded. Exiting."); sys.exit(1)

    test_sets = defaultdict(list)
    for test_set_path_str in args.test_sets:
        test_set_path = Path(test_set_path_str)
        if not test_set_path.is_dir(): logger.warning(f"Test set dir not found: {test_set_path}. Skip."); continue
        test_set_name = f"{test_set_path.parent.name}_{test_set_path.name}"
        cases = sorted([d for d in test_set_path.glob("case_*") if d.is_dir()], key=lambda x: int(x.name.split('_')[-1]))
        if not cases: logger.warning(f"No cases in test set: {test_set_path}. Skip."); continue
        test_sets[test_set_name].extend(cases); logger.info(f"Found {len(cases)} cases in '{test_set_name}'")
    if not test_sets: logger.error("No valid test sets. Exiting."); sys.exit(1)

    all_results_list = []
    overall_agg_metrics = defaultdict(lambda: defaultdict(list)) # model_name -> metric_name -> list_of_values

    for test_set_name, case_paths in test_sets.items():
        logger.info(f"\n--- Evaluating on Test Set: {test_set_name} ({len(case_paths)} cases) ---")

        for model_name, data in models_data.items():
            model = data['model']; config = data['config']
            max_steps_model_config = int(config.get("eval_max_steps", config.get("max_steps", 120)))
            max_steps_run = args.max_steps if args.max_steps is not None else max_steps_model_config
            num_agents_for_penalty = int(config.get("num_agents", 1))


            # Store results for this model on this test set
            current_model_test_set_metrics = []

            pbar_cases = tqdm(case_paths, desc=f"  Model: {model_name[:20]:<20}", leave=False, unit="case")
            env = None
            for case_path in pbar_cases:
                try:
                    env = make_env(case_path, config)
                    ep_metrics = run_evaluation_episode_metrics(model, env, config, max_steps_run, case_path)
                    current_model_test_set_metrics.append(ep_metrics)
                except Exception as e:
                     logger.error(f"Critical error processing case {case_path.name} for {model_name}: {e}", exc_info=True)
                     # Append a default failure metric
                     current_model_test_set_metrics.append({
                         'success': False, 'executed_makespan': max_steps_run,
                         'executed_soc': max_steps_run * num_agents_for_penalty,
                         'optimal_soc': np.nan, 'optimal_makespan': np.nan, 'inference_times': []
                     })
                finally:
                    if env is not None: env.close(); env = None

            # Calculate aggregated metrics for this model on this test set
            sr = np.mean([m['success'] for m in current_model_test_set_metrics])
            successful_makespans = [m['executed_makespan'] for m in current_model_test_set_metrics if m['success']]
            avg_makespan_succ = np.mean(successful_makespans) if successful_makespans else np.nan
            all_inf_times = [t for m in current_model_test_set_metrics for t in m['inference_times']]
            avg_inf_time = np.mean(all_inf_times) if all_inf_times else np.nan

            # Flowtime and Flowtime Increase
            total_exec_soc_for_ft = 0
            total_opt_soc_for_ft = 0
            num_cases_for_ft = 0

            for m in current_model_test_set_metrics:
                if np.isnan(m['optimal_soc']):
                    # logger.debug(f"Skipping case for FT calc due to missing optimal_soc: {m}") # Too verbose
                    continue # Cannot calculate dFT without optimal SoC

                total_opt_soc_for_ft += m['optimal_soc']
                num_cases_for_ft += 1
                if m['success']:
                    total_exec_soc_for_ft += m['executed_soc']
                else: # Failed case, apply penalty for Flowtime
                    penalty_time = max_steps_run # Default penalty
                    if not np.isnan(m['optimal_makespan']) and m['optimal_makespan'] > 0:
                        penalty_time = 3 * m['optimal_makespan']
                    elif not np.isnan(m['optimal_makespan']) and m['optimal_makespan'] == 0: # Opt is 0 steps
                        penalty_time = 3 * 1 # Small penalty if optimal is 0 but failed
                    total_exec_soc_for_ft += penalty_time * num_agents_for_penalty


            avg_ft = total_exec_soc_for_ft / num_cases_for_ft if num_cases_for_ft > 0 else np.nan
            delta_ft = (total_exec_soc_for_ft - total_opt_soc_for_ft) / total_opt_soc_for_ft if total_opt_soc_for_ft > 0 else np.nan

            all_results_list.append({
                'Test Set': test_set_name, 'Model': model_name,
                'Success Rate': sr, 'Average Makespan (Successful)': avg_makespan_succ,
                'Flowtime (FT)': avg_ft,
                'Flowtime Increase (dFT)': delta_ft,
                'Avg Inference Time (ms/step)': avg_inf_time,
                'Parameters': model_param_counts[model_name]
            })

            # Store detailed metrics for overall aggregation
            overall_agg_metrics[model_name]['all_episode_metrics'].extend(current_model_test_set_metrics)


    # --- Calculate Overall Aggregated Results ---
    logger.info("\n--- Overall Performance (Aggregated across all specified test sets) ---")
    overall_summary_list = []
    for model_name in models_data.keys():
        all_ep_metrics_for_model = overall_agg_metrics[model_name]['all_episode_metrics']
        if not all_ep_metrics_for_model: continue

        sr = np.mean([m['success'] for m in all_ep_metrics_for_model])
        successful_makespans = [m['executed_makespan'] for m in all_ep_metrics_for_model if m['success']]
        avg_makespan_succ = np.mean(successful_makespans) if successful_makespans else np.nan
        all_inf_times = [t for m in all_ep_metrics_for_model for t in m['inference_times']]
        avg_inf_time = np.mean(all_inf_times) if all_inf_times else np.nan

        total_exec_soc_for_ft_overall = 0
        total_opt_soc_for_ft_overall = 0
        num_cases_for_ft_overall = 0
        num_agents_for_penalty_overall = int(models_data[model_name]['config'].get("num_agents",1)) # Get N from this model's config


        for m in all_ep_metrics_for_model:
            if np.isnan(m['optimal_soc']): continue
            total_opt_soc_for_ft_overall += m['optimal_soc']
            num_cases_for_ft_overall +=1
            if m['success']:
                total_exec_soc_for_ft_overall += m['executed_soc']
            else:
                penalty_time = args.max_steps if args.max_steps is not None else int(models_data[model_name]['config'].get("max_steps", 120))
                if not np.isnan(m['optimal_makespan']) and m['optimal_makespan'] > 0:
                    penalty_time = 3 * m['optimal_makespan']
                elif not np.isnan(m['optimal_makespan']) and m['optimal_makespan'] == 0:
                    penalty_time = 3 * 1
                total_exec_soc_for_ft_overall += penalty_time * num_agents_for_penalty_overall

        avg_ft_overall = total_exec_soc_for_ft_overall / num_cases_for_ft_overall if num_cases_for_ft_overall > 0 else np.nan
        delta_ft_overall = (total_exec_soc_for_ft_overall - total_opt_soc_for_ft_overall) / total_opt_soc_for_ft_overall if total_opt_soc_for_ft_overall > 0 else np.nan

        overall_summary_list.append({
            'Model': model_name, 'Success Rate': sr,
            'Average Makespan (Successful)': avg_makespan_succ,
            'Flowtime (FT)': avg_ft_overall,
            'Flowtime Increase (dFT)': delta_ft_overall,
            'Avg Inference Time (ms/step)': avg_inf_time,
            'Parameters': model_param_counts[model_name]
        })

    # --- Save and Print Results ---
    results_df = pd.DataFrame(all_results_list)
    overall_df = pd.DataFrame(overall_summary_list)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "evaluation_metrics_per_testset.csv"
    overall_csv_path = args.output_dir / "evaluation_metrics_overall.csv"
    try:
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"\nDetailed results per test set saved to: {csv_path}")
        overall_df.to_csv(overall_csv_path, index=False, float_format='%.4f')
        logger.info(f"Overall aggregated results saved to: {overall_csv_path}")
        print("\n--- Results per Test Set ---"); print(results_df.to_string(index=False, float_format='%.4f'))
        print("\n--- Overall Aggregated Results ---"); print(overall_df.to_string(index=False, float_format='%.4f'))
    except Exception as e: logger.error(f"Error saving results to CSV: {e}", exc_info=True)
    logger.info("--- Evaluation Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MAPF models.")
    parser.add_argument("--model_dirs", type=str, nargs='+', required=True, help="Paths to model result directories.")
    parser.add_argument("--test_sets", type=str, nargs='+', required=True, help="Paths to test set directories (case_*/).")
    parser.add_argument("--max_steps", type=int, default=None, help="Global override for max steps per eval episode. If None, uses model's config.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/model_evaluation_final"), help="Dir to save results.")
    parsed_args = parser.parse_args()
    main(parsed_args)