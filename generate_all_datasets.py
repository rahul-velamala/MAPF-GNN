# File: generate_all_datasets.py
# Place this file in the project root directory (rahul-velamala-mapf-gnn/)

import os
import yaml
import numpy as np
from pathlib import Path
import traceback
import logging
import time
import shutil
from tqdm import tqdm
import signal # Required for data_gen timeout
import random # For shuffling case assignments

# --- Setup Logging ---
# Configure logging settings here
log_filename = 'dataset_generation_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'), # Log to file, overwrite previous
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# --- Project Imports ---
# Assume this script is in the project root
try:
    from data_generation.dataset_gen import data_gen, gen_input, generate_obstacles_for_map, TimeoutError, handle_timeout, can_use_alarm
    from data_generation.trayectory_parser import parse_traject
    from data_generation.record import make_env, record_env
    logger.info("Successfully imported data generation submodules.")
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import submodules: {e}", exc_info=True)
    logger.error("Ensure this script is in the project root directory and necessary __init__.py files exist.")
    exit(1)

# --- Helper: Count existing FINAL cases (states.npy exists) ---
# Copied from main_data.py for self-containment
def count_final_cases(dataset_path: Path) -> tuple[int, int]:
    """Counts existing cases with states.npy (final stage) and finds max index."""
    count = 0
    max_idx = -1
    if dataset_path.is_dir():
        state_files = list(dataset_path.glob("case_*/states.npy"))
        count = len(state_files)
        for item in state_files:
            try: max_idx = max(max_idx, int(item.parent.name.split('_')[-1]))
            except (ValueError, IndexError): pass # Ignore cases with non-standard names
    return count, max_idx

# --- Core Generation Function for a Single Configuration ---
def generate_single_configuration(
    board_rows: int,
    board_cols: int,
    robot_density: float,
    obstacle_density: float,
    sensing_range: int,
    pad: int,
    cbs_timeout: int,
    max_time_recording: int,
    num_cases_target: int,
    val_ratio: float,
    test_ratio: float,
    base_dataset_dir: Path
    ):
    """
    Generates the complete dataset (CBS -> Parse -> Record) for one specific
    set of environment parameters.
    """
    config_start_time = time.time()
    logger.info(f"\n{'='*20} Processing Config {'='*20}")

    # --- Configuration Calculation ---
    try:
        num_agents_global = int(robot_density * board_rows * board_cols)
        if num_agents_global < 2:
             logger.warning(f"Calculated agents {num_agents_global} for {board_rows}x{board_cols} r={robot_density*100}% is < 2. Setting to 2.")
             num_agents_global = 2

        num_obstacles_global = int(obstacle_density * board_rows * board_cols)

        dataset_name = f"map{board_rows}x{board_cols}_r{int(robot_density*100)}_o{int(obstacle_density*100)}"
        logger.info(f"Current Config: {dataset_name}, Agents={num_agents_global}, Obstacles={num_obstacles_global}")

        # Define paths
        config_data_dir = base_dataset_dir / dataset_name
        train_path = config_data_dir / "train"
        val_path = config_data_dir / "val"
        test_path = config_data_dir / "test"

        # Calculate split sizes
        num_cases_val_target = int(num_cases_target * val_ratio)
        num_cases_test_target = int(num_cases_target * test_ratio)
        num_cases_train_target = num_cases_target - num_cases_val_target - num_cases_test_target

        split_configs = {
            "train": {"path": train_path, "target": num_cases_train_target},
            "val":   {"path": val_path,   "target": num_cases_val_target},
            "test":  {"path": test_path,  "target": num_cases_test_target}
        }

        # Base Config Dictionary for sub-modules
        base_config = {
            "num_agents": num_agents_global,
            "board_size": [board_rows, board_cols],
            "map_shape": [board_cols, board_rows], # W, H for CBS
            "nb_obstacles": num_obstacles_global, # Used by generate_obstacles_for_map
            "sensing_range": sensing_range,
            "pad": pad,
            "max_time": max_time_recording, # For record_env
            "cbs_timeout_seconds": cbs_timeout, # For dataset_gen.data_gen
            "nb_agents": num_agents_global, # Duplicate for convenience
        }

        logger.info(f"Targeting {num_cases_target} total cases for this config.")
        logger.info(f" -> Train: {num_cases_train_target}, Val: {num_cases_val_target}, Test: {num_cases_test_target}")
        logger.info(f"Base directory: {config_data_dir.resolve()}")

    except Exception as e:
        logger.error(f"Config Error for {board_rows}x{board_cols}, r={robot_density}, o={obstacle_density}: {e}", exc_info=True)
        return False # Indicate failure for this config

    # --- STEP 1 & 2 Combined: Generate Case & Run CBS ---
    logger.info(f"\n--- Generating {num_cases_target} Cases & Running CBS for {dataset_name} ---")

    # Assign cases to splits
    global_case_indices = list(range(num_cases_target))
    random.shuffle(global_case_indices)
    case_assignments = {}
    train_indices = global_case_indices[:num_cases_train_target]
    val_indices = global_case_indices[num_cases_train_target : num_cases_train_target + num_cases_val_target]
    test_indices = global_case_indices[num_cases_train_target + num_cases_val_target:]
    for idx in train_indices: case_assignments[idx] = {"split": "train", "split_path": train_path}
    for idx in val_indices:   case_assignments[idx] = {"split": "val",   "split_path": val_path}
    for idx in test_indices:  case_assignments[idx] = {"split": "test",  "split_path": test_path}

    # Ensure split directories exist
    for split_cfg in split_configs.values():
        split_cfg["path"].mkdir(parents=True, exist_ok=True)

    # Track progress
    successful_cbs_counts = {"train": 0, "val": 0, "test": 0}
    cbs_failure_reasons = {}
    total_cases_attempted_this_config = 0
    max_attempts = int(num_cases_target * 1.5) # Allow 50% retries for failures

    pbar_cbs = tqdm(total=num_cases_target, desc=f"CBS {dataset_name}", unit="case", leave=True)

    while sum(successful_cbs_counts.values()) < num_cases_target and total_cases_attempted_this_config < max_attempts:
        total_cases_attempted_this_config += 1
        # Determine target split (simple assignment based on current success counts)
        current_success_idx = sum(successful_cbs_counts.values())
        if successful_cbs_counts["train"] < num_cases_train_target: split_name, split_path = "train", train_path
        elif successful_cbs_counts["val"] < num_cases_val_target: split_name, split_path = "val", val_path
        else: split_name, split_path = "test", test_path

        # Case name based on attempt number ensures uniqueness if previous attempts failed
        case_name = f"case_{total_cases_attempted_this_config:05d}"
        case_output_dir = split_path / case_name

        # Generate obstacles for this case
        obstacle_set = generate_obstacles_for_map(
            tuple(base_config["map_shape"]), base_config["nb_obstacles"]
        )
        if obstacle_set is None:
            reason = "obstacle_gen_fail"
            cbs_failure_reasons[reason] = cbs_failure_reasons.get(reason, 0) + 1
            continue # Try next attempt

        # Generate start/goal
        input_data = gen_input(
            dimensions=tuple(base_config["map_shape"]),
            nb_agents=base_config["nb_agents"],
            fixed_obstacles_xy_set=obstacle_set
        )

        # Run CBS solver (data_gen handles directory cleanup on failure)
        success, reason = data_gen(
            input_dict=input_data,
            output_dir=case_output_dir,
            cbs_timeout_seconds=base_config["cbs_timeout_seconds"]
        )

        if success:
            successful_cbs_counts[split_name] += 1
            pbar_cbs.update(1)
        else:
            reason_key = reason.split(":")[0]
            cbs_failure_reasons[reason_key] = cbs_failure_reasons.get(reason_key, 0) + 1

        pbar_cbs.set_postfix({
            "Success": f"{sum(successful_cbs_counts.values())}/{num_cases_target}",
            "Fail": sum(cbs_failure_reasons.values()),
            "Attempt": f"{total_cases_attempted_this_config}/{max_attempts}"
        })

    pbar_cbs.close()
    total_cbs_success = sum(successful_cbs_counts.values())
    logger.info(f"Finished CBS runs for {dataset_name}. Total successful: {total_cbs_success}/{num_cases_target}")
    logger.info(f"  Success per split: {successful_cbs_counts}")
    if cbs_failure_reasons: logger.warning(f"  CBS Failure Reasons: {cbs_failure_reasons}")
    if total_cbs_success < num_cases_target:
        logger.warning(f"Could not generate target number of cases ({num_cases_target}). Generated {total_cbs_success}.")

    # Proceed only if some cases were successful
    if total_cbs_success == 0:
        logger.error(f"No cases successfully solved by CBS for {dataset_name}. Skipping parsing and recording.")
        return False

    # --- STEP 3: Parse Trajectories ---
    logger.info(f"\n--- STEP 3: Parsing Trajectories for {dataset_name} ---")
    for split_name, split_cfg in split_configs.items():
        if split_cfg["path"].exists() and successful_cbs_counts[split_name] > 0:
            logger.info(f"Parsing trajectories for split: {split_name}")
            parse_traject(split_cfg["path"])

    # --- STEP 4: Record Environment States ---
    logger.info(f"\n--- STEP 4: Recording Environment States & GSO for {dataset_name} ---")
    for split_name, split_cfg in split_configs.items():
        if split_cfg["path"].exists() and successful_cbs_counts[split_name] > 0:
            logger.info(f"Recording states/GSO for split: {split_name}")
            # Pass the config specific to this generation run
            record_env(split_cfg["path"], base_config)

    config_duration = time.time() - config_start_time
    logger.info(f"--- Finished Processing Config: {dataset_name} (Duration: {config_duration:.2f}s) ---")
    return True


# --- Main Script Execution ---
if __name__ == "__main__":
    logger.info("Starting dataset generation process...")
    logger.info(f"Logging detailed output to: {log_filename}")

    # --- Parameter Ranges & Fixed Values ---
    env_sizes = [(10, 10), (20, 20), (30, 30)] # (rows, cols)
    robot_densities = [0.05, 0.10, 0.15, 0.20]
    obstacle_densities = [0.10, 0.20, 0.30]
    sensing_range_fixed = 5
    pad_fixed = 5 # For 9x9 FOV
    cbs_timeout_fixed = 300# START WITH 60s. Increase if too many timeouts. Paper used 300s.
    max_time_recording_fixed = 256 # Max steps for simulation recording
    num_cases_per_config = 100 # Target scenarios per config
    val_ratio = 0.15
    test_ratio = 0.15
    base_dataset_dir = Path("./dataset") # Root directory for all datasets

    overall_start_time = time.time()
    configs_generated = 0
    configs_failed = 0

    # Create the base dataset directory if it doesn't exist
    base_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all configurations
    for size in env_sizes:
        board_r, board_c = size
        for r_density in robot_densities:
            for o_density in obstacle_densities:
                try:
                    success = generate_single_configuration(
                        board_rows=board_r,
                        board_cols=board_c,
                        robot_density=r_density,
                        obstacle_density=o_density,
                        sensing_range=sensing_range_fixed,
                        pad=pad_fixed,
                        cbs_timeout=cbs_timeout_fixed,
                        max_time_recording=max_time_recording_fixed,
                        num_cases_target=num_cases_per_config,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        base_dataset_dir=base_dataset_dir
                    )
                    if success:
                        configs_generated += 1
                    else:
                        configs_failed += 1
                except Exception as e:
                    config_name = f"map{board_r}x{board_c}_r{int(r_density*100)}_o{int(o_density*100)}"
                    logger.error(f"Unhandled exception during generation for {config_name}: {e}", exc_info=True)
                    configs_failed += 1

    overall_duration = time.time() - overall_start_time
    logger.info(f"\n{'='*20} Overall Generation Summary {'='*20}")
    logger.info(f"Total configurations attempted: {len(env_sizes) * len(robot_densities) * len(obstacle_densities)}")
    logger.info(f"Configurations successfully generated (at least partially): {configs_generated}")
    logger.info(f"Configurations failed entirely: {configs_failed}")
    logger.info(f"Total execution time: {overall_duration:.2f} seconds")
    logger.info(f"Detailed logs saved to: {log_filename}")
    logger.info(f"Generated datasets are located in: {base_dataset_dir.resolve()}")