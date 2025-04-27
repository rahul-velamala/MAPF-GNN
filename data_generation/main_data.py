# File: data_generation/main_data.py
# (SEQUENTIAL Version - Modified for IROS 2020 Paper Setup: 600 maps, 50 cases/map)

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
import torch # For GPU detection (informational only)
import random # For shuffling case assignments

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
)

# --- GPU Check (Informational) ---
gpu_available = torch.cuda.is_available()
logger.info(f"GPU detection: {'Available' if gpu_available else 'Not available'} (Note: GPU not used by standard CBS/Env)")

# --- Imports ---
try:
    # Use absolute imports relative to the project structure if needed
    # Assuming main_data.py is in project_root/data_generation
    from data_generation.dataset_gen import data_gen, gen_input, generate_obstacles_for_map
    from data_generation.trayectory_parser import parse_traject
    from data_generation.record import make_env, record_env # Use updated record_env
    logger.info("Successfully imported data generation submodules.")
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import submodules: {e}", exc_info=True)
    logger.error("Ensure you are running this script from the correct directory or have set up PYTHONPATH.")
    exit(1)

# --- Helper: Count existing COMPLETE cases (solution.yaml exists) ---
def count_existing_solutions(dataset_path: Path) -> tuple[int, int]:
    """Counts existing cases with solution.yaml and finds max index."""
    count = 0
    max_idx = -1
    if dataset_path.is_dir():
        solution_files = list(dataset_path.glob("case_*/solution.yaml"))
        count = len(solution_files)
        for item in solution_files:
            try: max_idx = max(max_idx, int(item.parent.name.split('_')[-1]))
            except (ValueError, IndexError): pass # Ignore cases with non-standard names
    return count, max_idx

# --- Helper: Count existing FINAL cases (states.npy exists) ---
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

# --- Main Execution Logic (Sequential) ---
if __name__ == "__main__":

    logger.info("Entered __main__ block - Running SEQUENTIALLY.")

    # --- Configuration ---
    try:
        logger.info("Defining configuration...")
        # !!! PARAMETERS to match Li et al. IROS 2020 !!!
        dataset_name = "IROS2020_N10_20x20_O40_CBS300_30k" # Descriptive name
        num_agents_global = 10                  # Paper uses N in [4..12], N=10 used for scaling tests
        board_rows_global = 20                  # Paper uses 20x20
        board_cols_global = 20
        obstacle_density = 0.10                 # Paper uses 10% density
        num_obstacles_global = int(obstacle_density * board_rows_global * board_cols_global)

        sensing_range_global = 5 # r_comm = 5
        pad_global = 5           # r_fov = 4 implies pad=5 for 9x9 FOV (r_fov+1)

        max_time_env_recording = 256 # Max steps for *recording* simulation (adjust as needed)
        cbs_timeout_generation = 300 # Paper uses 300s timeout for expert

        # --- Dataset Size & Splits (Match paper's 30k cases, 70/15/15) ---
        num_maps_target = 600
        cases_per_map_target = 50
        num_total_cases_target = num_maps_target * cases_per_map_target # 30,000

        val_ratio = 0.15
        test_ratio = 0.15
        if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and (val_ratio + test_ratio) < 1):
             raise ValueError("val_ratio/test_ratio invalid.")
        num_cases_val_target = int(num_total_cases_target * val_ratio)   # 4500
        num_cases_test_target = int(num_total_cases_target * test_ratio)  # 4500
        num_cases_train_target = num_total_cases_target - num_cases_val_target - num_cases_test_target # 21000

        # --- Paths ---
        base_data_dir = Path("dataset") / dataset_name
        train_path = base_data_dir / "train"
        val_path = base_data_dir / "val"
        test_path = base_data_dir / "test"

        split_configs = {
            "train": {"path": train_path, "target": num_cases_train_target},
            "val":   {"path": val_path,   "target": num_cases_val_target},
            "test":  {"path": test_path,  "target": num_cases_test_target}
        }

        # --- Base Config Dictionary ---
        base_config = {
            "num_agents": num_agents_global,
            "board_size": [board_rows_global, board_cols_global], # [rows, cols] for GraphEnv
            "map_shape": [board_cols_global, board_rows_global], # [width, height] for CBS Env
            "nb_obstacles": num_obstacles_global, # For map generation
            "sensing_range": sensing_range_global,
            "pad": pad_global,
            "max_time": max_time_env_recording, # Max steps for GraphEnv recording
            "cbs_timeout_seconds": cbs_timeout_generation, # Timeout for dataset_gen.data_gen
             # Adding these for dataset_gen.gen_input convenience (though map_shape is W,H)
            "nb_agents": num_agents_global,
        }

        logger.info(f"Targeting {num_total_cases_target} total cases.")
        logger.info(f" -> Train: {num_cases_train_target}, Val: {num_cases_val_target}, Test: {num_cases_test_target}")
        logger.info(f"Based on {num_maps_target} maps, {cases_per_map_target} cases/map.")
        logger.info(f"Base directory: {base_data_dir.resolve()}")
        logger.info(f"Base config: {base_config}")

    except Exception as e:
        logger.error(f"FATAL ERROR during configuration: {e}", exc_info=True)
        exit(1)

    # --- STEP 1: Generate Base Maps (Obstacles) ---
    logger.info(f"\n--- STEP 1: Generating {num_maps_target} Base Maps ---")
    generated_maps = {} # Store obstacles: {map_id: set((x,y), (x,y), ...)}
    map_gen_attempts = 0
    max_map_gen_attempts = num_maps_target * 5 # Allow retries for obstacle placement

    pbar_map = tqdm(total=num_maps_target, desc="Generating Maps")
    while len(generated_maps) < num_maps_target and map_gen_attempts < max_map_gen_attempts:
        map_gen_attempts += 1
        map_id_to_gen = len(generated_maps) # Generate maps sequentially 0, 1, 2...
        # Generate obstacles for the current map ID
        obstacle_set_xy = generate_obstacles_for_map(
            tuple(base_config["map_shape"]), # Needs (width, height)
            base_config["nb_obstacles"]
        )
        if obstacle_set_xy is not None:
             # Check for uniqueness (unlikely hash collision, but possible with few obstacles)
             # A simple check: ensure the set hasn't been generated before
             is_unique = True
             for existing_set in generated_maps.values():
                 if obstacle_set_xy == existing_set:
                     is_unique = False
                     logger.debug(f"Duplicate obstacle map generated ({len(obstacle_set_xy)} obs). Retrying.")
                     break
             if is_unique:
                generated_maps[map_id_to_gen] = obstacle_set_xy
                pbar_map.update(1)
        # No else needed, if None or duplicate, the loop continues

    pbar_map.close()
    if len(generated_maps) < num_maps_target:
        logger.error(f"FATAL: Could only generate {len(generated_maps)}/{num_maps_target} unique maps. Exiting.")
        exit(1)
    logger.info(f"Successfully generated {len(generated_maps)} unique base maps.")


    # --- STEP 2: Generate Cases (Start/Goal) & Run CBS ---
    logger.info(f"\n--- STEP 2: Generating {num_total_cases_target} Cases & Running CBS ---")

    # Create mapping from global case index (0 to 29999) to map_id (0 to 599)
    # And assign each global case index to a split
    global_case_indices = list(range(num_total_cases_target))
    random.shuffle(global_case_indices) # Shuffle to distribute maps across splits randomly

    case_assignments = {} # {global_case_idx: {"map_id": int, "split": str, "split_path": Path}}
    train_indices = global_case_indices[:num_cases_train_target]
    val_indices = global_case_indices[num_cases_train_target : num_cases_train_target + num_cases_val_target]
    test_indices = global_case_indices[num_cases_train_target + num_cases_val_target:]

    for idx in train_indices: case_assignments[idx] = {"map_id": idx // cases_per_map_target, "split": "train", "split_path": train_path}
    for idx in val_indices:   case_assignments[idx] = {"map_id": idx // cases_per_map_target, "split": "val",   "split_path": val_path}
    for idx in test_indices:  case_assignments[idx] = {"map_id": idx // cases_per_map_target, "split": "test",  "split_path": test_path}

    # Ensure directories exist
    for split_cfg in split_configs.values():
        split_cfg["path"].mkdir(parents=True, exist_ok=True)

    # Track successful CBS runs per split
    successful_cbs_counts = {"train": 0, "val": 0, "test": 0}
    cbs_failure_reasons = {} # {reason: count}
    max_cbs_attempts = int(num_total_cases_target * 1.2) # Allow 20% CBS failures/retries
    cbs_attempts_count = 0

    # Use tqdm for overall progress
    pbar_cbs = tqdm(total=num_total_cases_target, desc="Running CBS", unit="case")

    # We iterate through assigned indices. If CBS fails for one assignment, we don't easily "retry"
    # with a different start/goal for the *same map* and *same global index*.
    # Instead, we just report the failure rate. The target is the number of assignments.
    for global_case_idx in range(num_total_cases_target):
        if global_case_idx not in case_assignments: continue # Should not happen

        assignment = case_assignments[global_case_idx]
        map_id = assignment["map_id"]
        split_name = assignment["split"]
        split_path = assignment["split_path"]
        # Use the global index for naming consistency
        case_name = f"case_{global_case_idx}"
        case_output_dir = split_path / case_name

        # Skip if solution already exists (from previous partial run)
        if (case_output_dir / "solution.yaml").exists():
            successful_cbs_counts[split_name] += 1
            pbar_cbs.update(1)
            logger.debug(f"Skipping {case_name} in {split_name} - solution.yaml exists.")
            continue

        # Get obstacles for this map
        obstacle_set = generated_maps.get(map_id)
        if obstacle_set is None:
            logger.error(f"Consistency error: Obstacles for map_id {map_id} not found. Skipping case {global_case_idx}.")
            continue

        # Generate start/goal input for this specific map layout
        input_data = gen_input(
            dimensions=tuple(base_config["map_shape"]), # Needs (width, height)
            nb_agents=base_config["nb_agents"],
            fixed_obstacles_xy_set=obstacle_set,
            max_placement_attempts=100 # Attempts for start/goal placement
        )

        # Run CBS solver (data_gen handles file writing and cleanup on failure)
        success, reason = data_gen(
            input_dict=input_data,
            output_dir=case_output_dir,
            cbs_timeout_seconds=base_config["cbs_timeout_seconds"]
        )

        if success:
            successful_cbs_counts[split_name] += 1
        else:
            # Increment failure reason count
            reason_key = reason.split(":")[0] # Get base reason
            cbs_failure_reasons[reason_key] = cbs_failure_reasons.get(reason_key, 0) + 1
            # The case directory should have been cleaned up by data_gen

        pbar_cbs.update(1)
        pbar_cbs.set_postfix({
            "Success": f"{sum(successful_cbs_counts.values())}/{global_case_idx+1}",
            "Fail": sum(cbs_failure_reasons.values())
        })

    pbar_cbs.close()
    total_cbs_success = sum(successful_cbs_counts.values())
    logger.info(f"Finished CBS runs. Total successful: {total_cbs_success}/{num_total_cases_target}")
    logger.info(f"Success per split: {successful_cbs_counts}")
    if cbs_failure_reasons:
        logger.warning(f"CBS Failure Reasons: {cbs_failure_reasons}")
    if total_cbs_success < num_total_cases_target * 0.9: # Example warning threshold
        logger.warning("CBS success rate is low. Consider checking parameters or increasing timeout.")


    # --- STEP 3: Parse Trajectories (solution.yaml -> trajectory.npy) ---
    logger.info(f"\n--- STEP 3: Parsing Trajectories ---")
    for split_name, split_cfg in split_configs.items():
        logger.info(f"Parsing trajectories for split: {split_name}")
        parse_traject(split_cfg["path"])


    # --- STEP 4: Record Environment States (Simulate trajectory.npy -> states.npy, gso.npy) ---
    logger.info(f"\n--- STEP 4: Recording Environment States & GSO ---")
    for split_name, split_cfg in split_configs.items():
        logger.info(f"Recording states/GSO for split: {split_name}")
        # Pass the base_config which contains GraphEnv parameters
        record_env(split_cfg["path"], base_config)

    # --- Final Summary ---
    logger.info(f"\n--- Dataset Generation Complete for {dataset_name} ---")
    total_final_cases = 0
    for split_name, split_cfg in split_configs.items():
         final_count, _ = count_final_cases(split_cfg["path"])
         logger.info(f"Split '{split_name}': Found {final_count} final cases (with states.npy). Target was {split_cfg['target']}.")
         total_final_cases += final_count
         if final_count < split_cfg['target'] * 0.9: # Check if significantly below target
              logger.warning(f"Split '{split_name}' final count is significantly below target.")

    logger.info(f"Total final cases generated across all splits: {total_final_cases}")
    logger.info("Check logs for details on CBS failures or recording issues.")