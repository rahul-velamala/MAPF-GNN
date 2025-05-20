# File: data_generation/main_data.py
# (MODIFIED to loop through multiple configurations using parallel functions)

import os
import yaml
import numpy as np
from pathlib import Path
import traceback
import logging
import time
import shutil # Added for potential cleanup in worker exceptions
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method # Import set_start_method
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
    # create_solutions_sequential is removed as we focus on parallel
    from .dataset_gen import data_gen, gen_input
    from .trayectory_parser import parse_traject
    # record_env_sequential is removed
    from .record import make_env
    logger.info("Successfully imported data generation submodules.")
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import data generation submodules: {e}", exc_info=True)
    logger.error("Check if you are running from the project root directory and if __init__.py files exist.")
    exit(1)

logger.info("Starting data_generation/main_data.py...")

# --- Worker Functions for Multiprocessing (Unchanged) ---

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
        # Check consistency after env creation
        if num_agents_traj != env.nb_agents:
             logger.warning(f"Agent mismatch after env creation in {case_path.name}. Traj N={num_agents_traj}, Env N={env.nb_agents}")
             return case_path.name, False # Agent mismatch

        # --- Simulate and Record ---
        initial_obs, _ = env.reset(seed=np.random.randint(1e6))
        initial_fov = initial_obs.get('fov')
        initial_gso = initial_obs.get('adj_matrix')
        if initial_fov is None or initial_fov.shape != expected_fov_shape or \
           initial_gso is None or initial_gso.shape != expected_gso_shape:
            logger.warning(f"Initial obs shape error in {case_path.name}. FOV:{initial_fov.shape if initial_fov is not None else 'None'}(exp {expected_fov_shape}), GSO:{initial_gso.shape if initial_gso is not None else 'None'}(exp {expected_gso_shape})")
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
                logger.warning(f"Obs shape error during sim step {i+1} in {case_path.name}.")
                sim_successful = False; break
            recordings_fov[i + 1] = obs_fov
            recordings_gso[i + 1] = obs_gso
            if terminated or truncated:
                # Trim arrays if simulation ended early
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
            # Clean up potentially created empty files if sim failed midway? Optional.
            return case_path.name, False

    except Exception as e:
        logger.error(f"Unhandled exception in record worker for case {case_path.name}: {e}", exc_info=True)
        return case_path.name, False
    finally:
        if env is not None:
            env.close()

# --- Parallel Runner Functions (Unchanged) ---

def create_solutions_parallel(dataset_path: Path, num_target_cases: int, config: dict, num_workers: int):
    """Generates CBS solutions in parallel."""
    logger.info(f"Starting parallel solution generation with {num_workers} workers for {dataset_path.name}.")
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create dataset directory {dataset_path}: {e}", exc_info=True)
        return

    # --- Determine existing and needed cases ---
    existing_successful_cases = 0
    highest_existing_index = -1
    if dataset_path.exists():
        # Optimization: Check only a sample or rely on index for speed if necessary
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

    # Limit overall attempts, e.g., 3x needed or 100 minimum
    max_attempts = max(needed_cases * 3, 100)
    submitted_indices = set()
    generated_count = 0
    attempt_count = 0
    next_idx = start_index

    # Prepare fixed arguments for worker function
    # Pass only the necessary config items to avoid large object serialization
    worker_config = {
        "map_shape": config["map_shape"],
        "nb_obstacles": config["nb_obstacles"],
        "nb_agents": config["nb_agents"],
        "cbs_timeout_seconds": config.get("cbs_timeout_seconds", 60)
    }
    worker_func = partial(_generate_one_case_worker, output_dir=dataset_path, config=worker_config)
    results_list = []

    # Use Pool for parallel execution
    with Pool(processes=num_workers) as pool:
        futures = []
        pbar = tqdm(total=needed_cases, desc=f"Generating ({dataset_path.name})", unit="case")

        while generated_count < needed_cases and attempt_count < max_attempts:
            # Submit new tasks
            while len(futures) < num_workers * 2 and attempt_count < max_attempts: # Keep pool busy
                # Check if case_idx already exists physically to avoid re-submission if failed partially
                case_check_path = dataset_path / f"case_{next_idx}"
                if not case_check_path.exists() and next_idx not in submitted_indices:
                    futures.append(pool.apply_async(worker_func, (next_idx,))) # Only pass changing arg
                    submitted_indices.add(next_idx)
                    attempt_count += 1
                    next_idx += 1
                else:
                    # If index exists or was submitted, skip to next potential index
                    next_idx += 1
                    # Add a check to break if next_idx grows excessively large compared to attempts
                    if next_idx > start_index + max_attempts * 2:
                        logger.warning(f"next_idx ({next_idx}) seems too high, potential issue finding free indices.")
                        break


            if not futures: # Break if no tasks could be submitted (e.g., next_idx too high)
                break

            # Process completed tasks
            ready_futures = [f for f in futures if f.ready()]
            if not ready_futures and len(futures) >= num_workers * 2:
                 # Wait briefly if pool is full but nothing is ready
                 time.sleep(0.1)
                 continue

            for future in ready_futures:
                 try:
                     case_idx, success, reason = future.get()
                     results_list.append((case_idx, success, reason))
                     if success:
                         generated_count += 1
                         pbar.update(1)
                 except Exception as e:
                      logger.error(f"Error getting result from future: {e}", exc_info=True)
                      # Try to find which case index this might have been (difficult)
                      results_list.append((-1, False, "future_get_error")) # Log failure
                 finally:
                      futures.remove(future)


        # Clean up progress bar
        pbar.close()

        # Process any remaining completed futures after loop ends
        for future in futures:
            if future.ready():
                try:
                    case_idx, success, reason = future.get()
                    results_list.append((case_idx, success, reason))
                    if success: generated_count += 1 # Count potentially missed updates
                except Exception: pass # Error already logged


    # --- Final Summary for this split ---
    if attempt_count >= max_attempts and generated_count < needed_cases:
        logger.warning(f"Reached maximum generation attempts ({max_attempts}) but only generated {generated_count}/{needed_cases} cases for {dataset_path.name}.")

    failures = [(idx, reason) for idx, succ, reason in results_list if not succ]
    failure_counts = {}
    for _, reason in failures:
         reason_key = reason.split(":")[0]
         failure_counts[reason_key] = failure_counts.get(reason_key, 0) + 1

    final_successful_cases = sum(1 for item in dataset_path.iterdir() if item.is_dir() and item.name.startswith("case_") and (item / "solution.yaml").exists())
    logger.info(f"Parallel Generation Finished for: {dataset_path.name}")
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
        # Use iterator for potentially large directories
        for item in path.glob("case_*"):
            if item.is_dir():
                 traj_file = item / "trajectory.npy"
                 state_file = item / "states.npy"
                 gso_file = item / "gso.npy"
                 # Process if traj exists AND either state or gso is missing
                 if traj_file.exists() and not (state_file.exists() and gso_file.exists()):
                     cases_to_process.append(item)
                 elif not traj_file.exists() and not (state_file.exists() and gso_file.exists()):
                      logger.debug(f"Skipping {item.name}: Missing trajectory.npy for recording.")
    except Exception as e:
        logger.error(f"Error listing cases for recording in {path}: {e}", exc_info=True)
        return

    if not cases_to_process:
        logger.info(f"No cases found in {path.name} needing state/GSO recording.")
        num_already_recorded = sum(1 for d in path.glob("case_*") if (d / "states.npy").exists() and (d / "gso.npy").exists())
        logger.info(f"({num_already_recorded} cases appear to be already recorded).")
        return

    logger.info(f"Recording States/GSOs in Parallel for Dataset: {path.name}")
    logger.info(f"Found {len(cases_to_process)} cases to process with {num_workers} workers.")

    # Prepare fixed arguments for worker
    worker_config = {
        "num_agents": config["num_agents"],
        "pad": config["pad"],
        "board_size": config["board_size"],
        "sensing_range": config["sensing_range"],
        "max_time": config["max_time"],
    }
    worker_func = partial(_record_one_case_worker, config=worker_config)
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

    logger.info(f"Recording Finished for: {path.name}")
    logger.info(f"Successfully recorded state/GSO for: {recorded_count} cases.")
    if failed_count > 0:
        logger.warning(f"Failed to record for: {failed_count} cases.")
        # failed_cases = [name for name, succ in results_list if not succ]
        # logger.debug(f"Failed recording cases: {failed_cases[:20]}...") # Log first few failed

# --- Main Execution Logic ---
if __name__ == "__main__":

    # Set start method for multiprocessing (important for some environments)
    try:
        set_start_method('spawn', force=True)
        logger.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logger.warning("Multiprocessing context already set or 'spawn' not available/forced.")

    logger.info("Entered __main__ block.")

    # --- Configuration ---
    try:
        logger.info("Defining configuration parameters...")

        # === Define Parameter Ranges ===
        env_sizes = [(10, 10)] # (rows, cols)
        robot_densities = [0.05]
        obstacle_densities = [0.3]

        # === Define Fixed Parameters ===
        sensing_range_fixed = 5
        pad_fixed = 5           # For 9x9 FOV (FOV_size = 2*pad - 1 = 9)
        max_time_env_fixed = 256 # Max steps for recording simulation
        cbs_timeout_generation_fixed = 60 # Timeout for CBS during generation (seconds)

        # === Dataset Split & Target Config ===
        num_cases_per_config = 5000 # TARGET PER CONFIGURATION
        val_ratio = 0.15
        test_ratio = 0.15
        base_output_dir = Path("./dataset5") # Root directory for all datasets

        # === Parallelism Config ===
        num_parallel_workers = max(1, cpu_count() - 2) # Leave 2 cores free
        logger.info(f"Using {num_parallel_workers} parallel workers (CPU cores: {cpu_count()}).")

    except Exception as e:
        logger.error(f"FATAL ERROR during initial configuration setup: {e}", exc_info=True)
        exit(1)

    # --- Generation Loop ---
    logger.info("\n--- Starting Multi-Configuration Dataset Generation ---")
    overall_success_count = 0
    overall_fail_count = 0
    total_configs_to_generate = len(env_sizes) * len(robot_densities) * len(obstacle_densities)
    config_counter = 0
    total_start_time = time.time()

    for size in env_sizes:
        board_rows_global, board_cols_global = size
        for r_density in robot_densities:
            for o_density in obstacle_densities:
                config_counter += 1
                config_start_time = time.time()
                logger.info(f"\n\n{'='*15} Processing Config {config_counter}/{total_configs_to_generate} {'='*15}")

                # --- Calculate Config-Specific Parameters ---
                try:
                    num_agents_global = int(r_density * board_rows_global * board_cols_global)
                    if num_agents_global < 2: num_agents_global = 2
                    num_obstacles_global = int(o_density * board_rows_global * board_cols_global)
                    dataset_name = f"map{board_rows_global}x{board_cols_global}_r{int(r_density*100)}_o{int(o_density*100)}_p{pad_fixed}"
                    logger.info(f"Config: {dataset_name}, Agents={num_agents_global}, Obstacles={num_obstacles_global}")

                    config_data_dir = base_output_dir / dataset_name
                    train_path = config_data_dir / "train"
                    val_path = config_data_dir / "val"
                    test_path = config_data_dir / "test"

                    # Calculate split sizes for THIS config
                    num_cases_val_target = int(num_cases_per_config * val_ratio)
                    num_cases_test_target = int(num_cases_per_config * test_ratio)
                    num_cases_train_target = num_cases_per_config - num_cases_val_target - num_cases_test_target

                    # Config dict passed to worker/parallel functions FOR THIS ITERATION
                    current_config = {
                        "num_agents": num_agents_global,
                        "board_size": [board_rows_global, board_cols_global],
                        "sensing_range": sensing_range_fixed,
                        "pad": pad_fixed,
                        "max_time": max_time_env_fixed,
                        "map_shape": [board_cols_global, board_rows_global], # W, H for CBS
                        "nb_agents": num_agents_global,
                        "nb_obstacles": num_obstacles_global,
                        "cbs_timeout_seconds": cbs_timeout_generation_fixed,
                    }

                    # Data sets dict FOR THIS ITERATION
                    data_sets_for_config = {}
                    if num_cases_train_target > 0: data_sets_for_config["train"] = {"path": train_path, "cases": num_cases_train_target}
                    if num_cases_val_target > 0: data_sets_for_config["val"] = {"path": val_path, "cases": num_cases_val_target}
                    if num_cases_test_target > 0: data_sets_for_config["test"] = {"path": test_path, "cases": num_cases_test_target}

                    if not data_sets_for_config:
                        logger.warning(f"No dataset splits defined for config {dataset_name} (num_cases_per_config={num_cases_per_config}). Skipping.")
                        continue

                    logger.info(f"Target cases: Train={num_cases_train_target}, Val={num_cases_val_target}, Test={num_cases_test_target}")
                    logger.info(f"Output dir: {config_data_dir.resolve()}")

                except Exception as e:
                     logger.error(f"Error setting up parameters for config {config_counter}: {e}", exc_info=True)
                     overall_fail_count += 1
                     continue # Skip to next configuration

                # --- Process Splits for Current Config ---
                config_fully_successful = True
                for set_name, set_config in data_sets_for_config.items():
                    current_path: Path = set_config["path"]
                    num_target = set_config["cases"]
                    split_start_time = time.time()

                    logger.info(f"\n-- Processing split: {set_name} for {dataset_name} --")
                    logger.info(f"Target path: {current_path.resolve()}")
                    logger.info(f"Target successful cases: {num_target}")

                    try:
                        current_path.parent.mkdir(parents=True, exist_ok=True) # Create dataset_name dir if needed
                    except OSError as e:
                        logger.error(f"FATAL ERROR creating parent dir {current_path.parent}: {e}", exc_info=True)
                        config_fully_successful = False; break # Break from processing splits for this config

                    # --- Step 1: Generate CBS solutions (Parallel) ---
                    try:
                        logger.info(f"Running Step 1: create_solutions_parallel for {set_name}...")
                        create_solutions_parallel(current_path, num_target, current_config, num_parallel_workers)
                        logger.info(f"Finished Step 1 for {set_name}.")
                    except Exception as e:
                        logger.error(f"ERROR during create_solutions_parallel for '{dataset_name}/{set_name}': {e}", exc_info=True)
                        config_fully_successful = False; # Continue to next split or config? Let's continue to next split maybe

                    # --- Step 2: Parse trajectories (Sequential) ---
                    try:
                        logger.info(f"Running Step 2: parse_traject for {set_name}...")
                        parse_traject(current_path) # Expects Path object
                        logger.info(f"Finished Step 2 for {set_name}.")
                    except Exception as e:
                        logger.error(f"ERROR during parse_traject for '{dataset_name}/{set_name}': {e}", exc_info=True)
                        config_fully_successful = False;

                    # --- Step 3: Record environment states (Parallel) ---
                    try:
                        logger.info(f"Running Step 3: record_env_parallel for {set_name}...")
                        record_env_parallel(current_path, current_config, num_parallel_workers) # Expects Path object
                        logger.info(f"Finished Step 3 for {set_name}.")
                    except Exception as e:
                        logger.error(f"ERROR during record_env_parallel for '{dataset_name}/{set_name}': {e}", exc_info=True)
                        config_fully_successful = False;

                    split_duration = time.time() - split_start_time
                    logger.info(f"-- Finished processing split: {set_name} (Duration: {split_duration:.2f}s) --")

                # --- Update Overall Counters ---
                if config_fully_successful:
                    overall_success_count += 1
                else:
                    overall_fail_count += 1
                config_duration = time.time() - config_start_time
                logger.info(f"--- Finished processing config {config_counter}/{total_configs_to_generate}: {dataset_name} (Duration: {config_duration:.2f}s) ---")


    # --- End Generation Loop ---
    total_duration = time.time() - total_start_time
    logger.info(f"\n\n{'='*20} Overall Generation Summary {'='*20}")
    logger.info(f"Total configurations attempted: {total_configs_to_generate}")
    logger.info(f"Configurations completed without fatal errors: {overall_success_count}")
    logger.info(f"Configurations with fatal errors: {overall_fail_count}")
    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    logger.info(f"Generated datasets are located in: {base_output_dir.resolve()}")