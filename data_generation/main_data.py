# File: data_generation/main_data.py
# (Modified for Robustness, Clarity, and Pathlib Usage)

import os
import yaml
import numpy as np # Make sure numpy is imported
from pathlib import Path # Use pathlib for better path handling
import traceback

# Use relative imports assuming standard project structure
try:
    from .dataset_gen import create_solutions
    from .trayectory_parser import parse_traject
    from .record import record_env
    print("DEBUG: Successfully imported data generation submodules.")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import data generation submodules: {e}")
    print("Check if you are running from the project root directory and if __init__.py files exist.")
    exit(1)

print("DEBUG: Starting data_generation/main_data.py...")

if __name__ == "__main__":

    print("DEBUG: Entered __main__ block.")

    # --- Configuration ---
    try:
        print("DEBUG: Defining configuration...")
        # --- Core Parameters ---
        dataset_name = "5_8_28_fov5_large_v1" # Give a distinct name, maybe add date/version
        num_agents_global = 5
        board_rows_global = 28 # For GraphEnv
        board_cols_global = 28 # For GraphEnv
        num_obstacles_global = 8
        sensing_range_global = 4 # For GraphEnv adjacency and FOV calculation (if pad not set)
        pad_global = 3           # For GraphEnv FOV size (5x5)
        max_time_env = 120       # Max steps in GraphEnv simulation during recording/training
        cbs_timeout_generation = 30 # Timeout for CBS solver in dataset_gen.py

        # --- Dataset Split Configuration ---
        base_data_dir = Path("dataset") / dataset_name # Root directory for this dataset
        num_total_cases = 2000 # Total successful CBS solutions to generate initially
        val_ratio = 0.15
        test_ratio = 0.15 # Set test_ratio > 0 if you want a test set

        num_cases_train_target = int(num_total_cases * (1 - val_ratio - test_ratio))
        num_cases_val_target = int(num_total_cases * val_ratio)
        num_cases_test_target = int(num_total_cases * test_ratio)

        # --- Base Configuration Dictionary (used by generation steps) ---
        base_config = {
            # Parameters for GraphEnv (used in record.py)
            "num_agents": num_agents_global,
            "board_size": [board_rows_global, board_cols_global], # GraphEnv uses [rows, cols]
            "sensing_range": sensing_range_global,
            "pad": pad_global,
            "max_time": max_time_env, # Used by GraphEnv

            # Parameters specifically for dataset_gen.py (CBS)
            "map_shape": [board_cols_global, board_rows_global], # CBS uses [width, height]
            "nb_agents": num_agents_global,      # Consistent name used by dataset_gen
            "nb_obstacles": num_obstacles_global,# Consistent name used by dataset_gen
            "cbs_timeout_seconds": cbs_timeout_generation,

            # Other misc potentially needed keys (add/remove as necessary)
            # "min_time": 1, # Often set per split in data loader config
            "generation_device": "gpu", # Device likely not needed for generation itself

            # --- REMOVED Dummy Model Params ---
            # These shouldn't be needed by the environment generation/recording steps.
            # Keep them in the *training* config files (e.g., config_gnn.yaml)
            # "encoder_layers": 1, "encoder_dims": [64], "last_convs": [0],
            # "graph_filters": [3], "node_dims": [128], "action_layers": 1, "channels": [16, 16, 16],
        }

        # Define dataset splits
        data_sets = {
            "train": {"path": base_data_dir / "train", "cases": num_cases_train_target},
            "val":   {"path": base_data_dir / "val",   "cases": num_cases_val_target},
        }
        if num_cases_test_target > 0:
             data_sets["test"] = {"path": base_data_dir / "test",  "cases": num_cases_test_target}

        print(f"DEBUG: Base directory: {base_data_dir.resolve()}")
        print(f"DEBUG: Base config defined: {base_config}")
        print(f"DEBUG: Datasets to process: {list(data_sets.keys())}")
        for name, cfg in data_sets.items():
             print(f"  - {name}: Target cases = {cfg['cases']}, Path = {cfg['path']}")

    except Exception as e:
        print(f"FATAL ERROR during configuration setup: {e}")
        traceback.print_exc()
        exit(1)

    # --- Generation Loop ---
    print("\nDEBUG: Starting generation loop...")
    for set_name, set_config in data_sets.items():
        current_path = set_config["path"]
        num_target_cases = set_config["cases"]

        if num_target_cases <= 0:
            print(f"\n--- Skipping dataset: {set_name} (target cases <= 0) ---")
            continue

        print(f"\n\n{'='*10} Processing dataset: {set_name} {'='*10}")
        print(f"Target path: {current_path.resolve()}")
        print(f"Target number of successful CBS cases: {num_target_cases}")

        # Create directory for the split
        try:
            print(f"DEBUG: Ensuring directory exists: {current_path}")
            current_path.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Directory exists/created.")
        except OSError as e:
            print(f"FATAL ERROR: Could not create directory {current_path}: {e}. Skipping this split.")
            continue

        # Prepare config specific to this run (though base_config is usually sufficient)
        run_config = base_config.copy()
        # Add split-specific info if needed, e.g., run_config['split_name'] = set_name

        # --- Step 1: Generate CBS solutions (input.yaml, solution.yaml) ---
        try:
            print("\n>>> Running Step 1: create_solutions (CBS Generation)...")
            create_solutions(current_path, num_target_cases, run_config)
            print("<<< Finished Step 1: create_solutions.")
        except Exception as e:
            print(f"FATAL ERROR during create_solutions for dataset '{set_name}': {e}")
            traceback.print_exc()
            print(f"--- Aborting processing for dataset '{set_name}' ---")
            continue # Skip remaining steps for this split

        # --- Step 2: Parse trajectories (solution.yaml -> trajectory.npy) ---
        try:
            print("\n>>> Running Step 2: parse_traject (Trajectory Parsing)...")
            parse_traject(current_path)
            print("<<< Finished Step 2: parse_traject.")
        except Exception as e:
            print(f"FATAL ERROR during parse_traject for dataset '{set_name}': {e}")
            traceback.print_exc()
            print(f"--- Aborting processing for dataset '{set_name}' ---")
            continue # Skip remaining steps for this split

        # --- Step 3: Record environment states (trajectory.npy -> states.npy, gso.npy) ---
        try:
            print("\n>>> Running Step 3: record_env (State/GSO Recording)...")
            # Pass run_config which contains GraphEnv parameters
            record_env(current_path, run_config)
            print("<<< Finished Step 3: record_env.")
        except Exception as e:
            print(f"FATAL ERROR during record_env for dataset '{set_name}': {e}")
            traceback.print_exc()
            print(f"--- Aborting processing for dataset '{set_name}' ---")
            continue # Skip remaining steps for this split (though this is the last step)

        print(f"\n--- Successfully finished processing dataset: {set_name} ---")

    # --- End Generation Loop ---
    print("\n\n--- All dataset generation steps completed. ---")