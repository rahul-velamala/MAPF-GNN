# File: data_generation/main_data.py
# (Revised for Robustness, Clarity, Pathlib Usage, and MULTIPROCESSING)

import os
import yaml
import numpy as np
from pathlib import Path
import traceback
import logging
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial # To pass fixed arguments to worker functions

# --- Setup Logging ---
logger = logging.getLogger(__name__)
# Configure root logger level, format etc.
# Let worker processes inherit this config or configure them separately if needed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
# --- --------------- ---

# Use relative imports assuming standard project structure
try:
    # These imports need to work in the main process AND the worker processes
    from .dataset_gen import create_solutions as create_solutions_sequential, data_gen, gen_input # Keep sequential for comparison/fallback maybe
    from .trayectory_parser import parse_traject
    from .record import record_env as record_env_sequential, make_env # Keep sequential
    logger.info("Successfully imported data generation submodules.")
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import data generation submodules: {e}", exc_info=True)
    logger.error("Check if you are running from the project root directory and if __init__.py files exist.")
    exit(1)

logger.info("Starting data_generation/main_data.py...")

# --- Worker Functions for Multiprocessing ---

def _generate_one_case_worker(case_index: int, output_dir: Path, config: dict) -> tuple[int, bool, str]:
    """Worker function to generate input and solution for one case."""
    case_path = output_dir / f"case_{case_index}"
    try:
        input_data = gen_input(
            config["map_shape"], # Expects [width, height]
            config["nb_obstacles"],
            config["nb_agents"]
        )
        success, reason = data_gen(
            input_data,
            case_path,
            config.get("cbs_timeout_seconds", 60)
        )
        return case_index, success, reason
    except Exception as e:
        logger.error(f"Unhandled exception in generate worker for case {case_index}: {e}", exc_info=True)
        # Clean up potentially partially created directory
        if case_path.exists():
            try: shutil.rmtree(case_path)
            except OSError: pass
        return case_index, False, f"worker_exception: {type(e).__name__}"

def _record_one_case_worker(case_path: Path, config: dict) -> tuple[str, bool]:
    """Worker function to record states/GSO for one case."""
    env = None
    try:
        # --- Determine expected shapes ---
        N = int(config["num_agents"])
        C = 3 # Fixed channels
        pad_val = int(config["pad"])
        H = W = (pad_val * 2) - 1
        expected_fov_shape = (N, C, H, W)
        expected_gso_shape = (N, N)

        # --- Load Trajectory ---
        trajectory_path = case_path / "trajectory.npy"
        if not trajectory_path.exists():
            # This case might have failed parsing, skip recording
            return case_path.name, False # Indicate failure

        trajectory_actions = np.load(trajectory_path)
        if trajectory_actions.ndim != 2: return case_path.name, False # Invalid traj dim
        num_agents_traj = trajectory_actions.shape[0]
        num_timesteps = trajectory_actions.shape[1]
        if num_timesteps == 0: return case_path.name, False # Empty traj

        # --- Create Env ---
        env = make_env(case_path, config)
        if env is None: return case_path.name, False # Env creation failed
        if num_agents_traj != env.nb_agents: return case_path.name, False # Agent mismatch

        # --- Simulate and Record ---
        initial_obs, _ = env.reset(seed=np.random.randint(1e6))
        initial_fov = initial_obs.get('fov')
        initial_gso = initial_obs.get('adj_matrix')
        if initial_fov is None or initial_fov.shape != expected_fov_shape or \
           initial_gso is None or initial_gso.shape != expected_gso_shape:
            return case_path.name, False # Shape error

        recordings_fov = np.zeros((num_timesteps + 1,) + expected_fov_shape, dtype=np.float32)
        recordings_gso = np.zeros((num_timesteps + 1,) + expected_gso_shape, dtype=np.float32)
        recordings_fov[0] = initial_fov
        recordings_gso[0] = initial_gso

        sim_successful = True
        current_obs = initial_obs
        for i in range(num_timesteps):
            actions_at_step_i = trajectory_actions[:, i]
            current_obs, _, terminated, truncated, _ = env.step(actions_at_step_i)
            obs_fov = current_obs.get('fov')
            obs_gso = current_obs.get('adj_matrix')
            if obs_fov is None or obs_fov.shape != expected_fov_shape or \
               obs_gso is None or obs_gso.shape != expected_gso_shape:
                sim_successful = False; break
            recordings_fov[i + 1] = obs_fov
            recordings_gso[i + 1] = obs_gso
            if terminated or truncated:
                recordings_fov = recordings_fov[:i+2]
                recordings_gso = recordings_gso[:i+2]
                break

        # --- Save ---
        if sim_successful:
            states_save_path = case_path / "states.npy"
            gso_save_path = case_path / "gso.npy"
            np.save(states_save_path, recordings_fov)
            np.save(gso_save_path, recordings_gso)
            return case_path.name, True
        else:
            return case_path.name, False

    except Exception as e:
        logger.error(f"Unhandled exception in record worker for case {case_path.name}: {e}", exc_info=True)
        return case_path.name, False
    finally:
        if env is not None:
            env.close()

# --- Modified Main Steps ---

def create_solutions_parallel(dataset_path: Path, num_target_cases: int, config: dict, num_workers: int):
    """Generates CBS solutions in parallel."""
    logger.info(f"Starting parallel solution generation with {num_workers} workers.")
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create dataset directory {dataset_path}: {e}", exc_info=True)
        return

    # --- Determine existing and needed cases ---
    existing_successful_cases = 0
    highest_existing_index = -1
    if dataset_path.exists():
        # This check can be slow for large directories, consider optimizing if needed
        for item in dataset_path.iterdir():
             if item.is_dir() and item.name.startswith("case_") and (item / "solution.yaml").exists():
                 existing_successful_cases += 1
             try: index = int(item.name.split('_')[-1]); highest_existing_index = max(highest_existing_index, index)
             except: pass # Ignore directories that don't parse

    needed_cases = num_target_cases - existing_successful_cases
    start_index = highest_existing_index + 1

    if needed_cases <= 0:
        logger.info(f"Target of {num_target_cases} successful cases already met in {dataset_path}. Skipping generation.")
        return

    logger.info(f"Found {existing_successful_cases} existing cases. Need to generate {needed_cases} more.")
    logger.info(f"Starting case index: {start_index}")

    tasks_to_submit = needed_cases
    max_attempts = max(needed_cases * 10, 200) # Limit overall attempts
    submitted_indices = set()
    generated_count = 0
    attempt_count = 0
    next_idx = start_index

    # Prepare fixed arguments for worker function
    worker_func = partial(_generate_one_case_worker, output_dir=dataset_path, config=config)
    results_list = []

    # Use Pool for parallel execution
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for potentially better performance as results come in
        # Need to manage which indices are submitted vs completed
        futures = []
        pbar = tqdm(total=needed_cases, desc=f"Generating ({dataset_path.name})", unit="case")

        while generated_count < needed_cases and attempt_count < max_attempts:
            # Submit new tasks if pool has capacity and we haven't submitted enough attempts
            while len(futures) < num_workers * 2 and attempt_count < max_attempts: # Keep pool busy
                if next_idx not in submitted_indices:
                    futures.append(pool.apply_async(_generate_one_case_worker, (next_idx, dataset_path, config)))
                    submitted_indices.add(next_idx)
                    attempt_count += 1
                    next_idx += 1
                else:
                     # This index was submitted but maybe failed? Try next one.
                     # This logic might need refinement if specific retries are needed.
                     next_idx += 1

            # Process completed tasks
            completed_futures = [f for f in futures if f.ready()]
            for future in completed_futures:
                 case_idx, success, reason = future.get()
                 results_list.append((case_idx, success, reason))
                 if success:
                     generated_count += 1
                     pbar.update(1)
                 # Remove completed future
                 futures.remove(future)

            # Avoid busy-waiting if pool is full but no results ready
            if len(futures) >= num_workers * 2 or attempt_count >= max_attempts:
                 time.sleep(0.1)

        # Clean up progress bar
        pbar.close()

        # Process any remaining completed futures after loop ends
        for future in futures:
            if future.ready():
                case_idx, success, reason = future.get()
                results_list.append((case_idx, success, reason))
                if success:
                    generated_count += 1 # Count potentially missed updates


    # --- Final Summary ---
    if attempt_count >= max_attempts and generated_count < needed_cases:
        logger.warning(f"Reached maximum generation attempts ({max_attempts}) but only generated {generated_count}/{needed_cases} cases for {dataset_path.name}.")

    failures = [(idx, reason) for idx, succ, reason in results_list if not succ]
    failure_counts = {}
    for _, reason in failures:
         reason_key = reason.split(":")[0]
         failure_counts[reason_key] = failure_counts.get(reason_key, 0) + 1

    final_successful_cases = sum(1 for item in dataset_path.iterdir() if item.is_dir() and item.name.startswith("case_") and (item / "solution.yaml").exists())
    logger.info(f"\n--- Parallel Generation Finished for: {dataset_path.name} ---")
    logger.info(f"Total successful cases in directory: {final_successful_cases}")
    logger.info(f"Generated {generated_count} successful cases in this run.")

    total_failed = len(failures)
    if total_failed > 0:
        logger.info(f"Failures during this run ({total_failed} total attempts failed):")
        sorted_failures = sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_failures:
            if count > 0:
                logger.info(f"  - {reason}: {count}")


def record_env_parallel(path: Path, config: dict, num_workers: int):
    """Records states/GSO in parallel."""
    if not path.is_dir():
        logger.error(f"Dataset directory not found: {path}. Cannot record.")
        return

    # Find cases that have trajectory.npy but not states.npy/gso.npy
    cases_to_process = []
    try:
        all_cases = [d for d in path.glob("case_*") if d.is_dir()]
        for case_path in all_cases:
            traj_file = case_path / "trajectory.npy"
            state_file = case_path / "states.npy"
            gso_file = case_path / "gso.npy"
            if traj_file.exists() and not (state_file.exists() and gso_file.exists()):
                cases_to_process.append(case_path)
            elif not traj_file.exists() and not (state_file.exists() and gso_file.exists()):
                 logger.debug(f"Skipping {case_path.name}: Missing trajectory.npy for recording.")

    except Exception as e:
        logger.error(f"Error listing cases for recording in {path}: {e}", exc_info=True)
        return

    if not cases_to_process:
        logger.info(f"No cases found in {path.name} needing state/GSO recording.")
        # Check if cases exist but are already recorded
        num_already_recorded = sum(1 for d in path.glob("case_*") if (d / "states.npy").exists())
        logger.info(f"({num_already_recorded} cases appear to be already recorded).")
        return

    logger.info(f"\n--- Recording States/GSOs in Parallel for Dataset: {path.name} ---")
    logger.info(f"Found {len(cases_to_process)} cases to process with {num_workers} workers.")

    # Prepare fixed arguments for worker
    worker_func = partial(_record_one_case_worker, config=config)
    recorded_count = 0
    failed_count = 0
    results_list = []

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(cases_to_process), desc=f"Recording ({path.name})", unit="case") as pbar:
            # Use imap_unordered for efficient processing
            for case_name, success in pool.imap_unordered(worker_func, cases_to_process):
                 results_list.append((case_name, success))
                 if success:
                     recorded_count += 1
                 else:
                     failed_count += 1
                 pbar.update(1)
                 pbar.set_postfix({"RecOK": recorded_count, "RecFail": failed_count})

    logger.info(f"\n--- Recording Finished for: {path.name} ---")
    logger.info(f"Successfully recorded state/GSO for: {recorded_count} cases.")
    if failed_count > 0:
        logger.info(f"Failed to record for: {failed_count} cases.")
        # Optionally log failed case names:
        # failed_cases = [name for name, succ in results_list if not succ]
        # logger.info(f"Failed cases: {failed_cases}")


# --- Main Execution Logic ---
if __name__ == "__main__":

    logger.info("Entered __main__ block.")

    # --- Configuration ---
    try:
        logger.info("Defining configuration...")
        # !!! USER: Define your dataset parameters here !!!
        dataset_name = "5_8_28_fov5_parallel_large" # Example name
        num_agents_global = 5
        board_rows_global = 28
        board_cols_global = 28
        num_obstacles_global = 8
        sensing_range_global = 4
        pad_global = 3
        max_time_env = 120
        cbs_timeout_generation = 30

        # --- Dataset Split Configuration ---
        base_data_dir = Path("dataset") / dataset_name
        num_total_cases = 60000 # Adjust as needed
        val_ratio = 0.2
        test_ratio = 0.001

        if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and (val_ratio + test_ratio) < 1):
             raise ValueError("val_ratio/test_ratio invalid.")
        num_cases_val_target = int(num_total_cases * val_ratio)
        num_cases_test_target = int(num_total_cases * test_ratio)
        num_cases_train_target = num_total_cases - num_cases_val_target - num_cases_test_target

        # --- Parallelism Config ---
        # Use slightly fewer workers than total cores to leave resources for OS/other tasks
        num_parallel_workers = max(1, cpu_count() - 1)
        logger.info(f"Using {num_parallel_workers} parallel workers (CPU cores: {cpu_count()}).")


        base_config = {
            "num_agents": num_agents_global,
            "board_size": [board_rows_global, board_cols_global],
            "sensing_range": sensing_range_global,
            "pad": pad_global,
            "max_time": max_time_env,
            "map_shape": [board_cols_global, board_rows_global],
            "nb_agents": num_agents_global,
            "nb_obstacles": num_obstacles_global,
            "cbs_timeout_seconds": cbs_timeout_generation,
        }

        data_sets = {}
        if num_cases_train_target > 0: data_sets["train"] = {"path": base_data_dir / "train", "cases": num_cases_train_target}
        if num_cases_val_target > 0: data_sets["val"] = {"path": base_data_dir / "val", "cases": num_cases_val_target}
        if num_cases_test_target > 0: data_sets["test"] = {"path": base_data_dir / "test", "cases": num_cases_test_target}

        if not data_sets: logger.warning("No dataset splits configured."); exit(0)

        logger.info(f"Base directory: {base_data_dir.resolve()}")
        logger.info(f"Base config: {base_config}")
        logger.info(f"Datasets: {list(data_sets.keys())}")
        for name, cfg in data_sets.items(): logger.info(f"  - {name}: Target cases = {cfg['cases']}, Path = {cfg['path']}")

    except Exception as e:
        logger.error(f"FATAL ERROR during configuration: {e}", exc_info=True)
        exit(1)

    # --- Generation Loop ---
    logger.info("\n--- Starting Dataset Generation ---")
    overall_success = True
    total_start_time = time.time()

    for set_name, set_config in data_sets.items():
        current_path: Path = set_config["path"]
        num_target = set_config["cases"]
        split_start_time = time.time()

        logger.info(f"\n\n{'='*10} Processing dataset split: {set_name} {'='*10}")
        logger.info(f"Target path: {current_path.resolve()}")
        logger.info(f"Target number of successful cases: {num_target}")

        try:
            current_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"FATAL ERROR creating parent dir for {current_path}: {e}", exc_info=True)
            overall_success = False; continue

        run_config = base_config.copy()

        # --- Step 1: Generate CBS solutions (Parallel) ---
        try:
            logger.info("\n>>> Running Step 1: create_solutions (Parallel CBS Generation)...")
            create_solutions_parallel(current_path, num_target, run_config, num_parallel_workers)
            logger.info("<<< Finished Step 1: create_solutions.")
        except Exception as e:
            logger.error(f"FATAL ERROR during create_solutions_parallel for '{set_name}': {e}", exc_info=True)
            overall_success = False; continue

        # --- Step 2: Parse trajectories (Sequential) ---
        try:
            logger.info("\n>>> Running Step 2: parse_traject (Trajectory Parsing)...")
            parse_traject(current_path) # Expects Path object
            logger.info("<<< Finished Step 2: parse_traject.")
        except Exception as e:
            logger.error(f"FATAL ERROR during parse_traject for '{set_name}': {e}", exc_info=True)
            overall_success = False; continue

        # --- Step 3: Record environment states (Parallel) ---
        try:
            logger.info("\n>>> Running Step 3: record_env (Parallel State/GSO Recording)...")
            record_env_parallel(current_path, run_config, num_parallel_workers) # Expects Path object
            logger.info("<<< Finished Step 3: record_env.")
        except Exception as e:
            logger.error(f"FATAL ERROR during record_env_parallel for '{set_name}': {e}", exc_info=True)
            overall_success = False; continue

        split_duration = time.time() - split_start_time
        logger.info(f"\n--- Successfully finished processing dataset split: {set_name} (Duration: {split_duration:.2f}s) ---")

    # --- End Generation Loop ---
    total_duration = time.time() - total_start_time
    logger.info(f"\n\n--- All dataset generation steps completed (Total Duration: {total_duration:.2f}s) ---")
    if not overall_success:
        logger.warning("One or more dataset splits encountered errors during generation.")