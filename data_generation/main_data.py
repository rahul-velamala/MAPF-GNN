# File: data_generation/main_data.py
# (Modified for Debugging Step 1)

# --- Add these imports if not already present ---
import os
import yaml
import numpy as np # Make sure numpy is imported
# --- ----------------------------------------- ---

# Use relative imports if running as part of the package/project structure
try:
    from .dataset_gen import create_solutions
    from .trayectory_parser import parse_traject
    from .record import record_env
    print("DEBUG: Successfully imported submodules.") # <<< ADDED PRINT
except ImportError as e:
    print(f"FATAL ERROR: Failed to import submodules: {e}") # <<< ADDED PRINT
    print("Check if you are running from the project root directory and if __init__.py files exist.")
    exit() # Exit if imports fail

print("DEBUG: Starting data_generation/main_data.py...") # <<< ADDED PRINT

if __name__ == "__main__":

    print("DEBUG: Entered __main__ block.") # <<< ADDED PRINT

    # --- Configuration ---
    try:
        print("DEBUG: Defining configuration...") # <<< ADDED PRINT
        dataset_name = "5_8_28_fov5"
        base_data_dir = os.path.join("dataset", dataset_name)
        num_agents_global = 5
        board_rows_global = 28
        board_cols_global = 28
        num_obstacles_global = 8
        sensing_range_global = 4
        pad_global = 3
        max_time_env = 60
        cbs_timeout_generation = 30
        cases_train = 1000 # Reduced for faster testing initially
        cases_val_ratio = 0.15
        cases_test_ratio = 0.15

        base_config = {
            "num_agents": num_agents_global,
            "board_size": [board_rows_global, board_cols_global],
            "sensing_range": sensing_range_global,
            "pad": pad_global,
            "max_time": max_time_env,
            "min_time": 1,
            "map_shape": [board_rows_global, board_cols_global],
            "nb_agents": num_agents_global,
            "nb_obstacles": num_obstacles_global,
            "cbs_timeout_seconds": cbs_timeout_generation,
            "device": "gpu",
             # Add dummy model config keys if make_env requires them from record.py
             "encoder_layers": 1, "encoder_dims": [64], "last_convs": [0],
             "graph_filters": [3], "node_dims": [128], "action_layers": 1, "channels": [16, 16, 16],
        }

        num_cases_train_target = int(cases_train * (1 - cases_val_ratio - cases_test_ratio))
        num_cases_val_target = int(cases_train * cases_val_ratio)
        # num_cases_test_target = int(cases_train * cases_test_ratio) # Test set commented out

        data_sets = {
            "train": {"path": os.path.join(base_data_dir, "train"), "cases": num_cases_train_target},
            "val":   {"path": os.path.join(base_data_dir, "val"),   "cases": num_cases_val_target},
            # "test":  {"path": os.path.join(base_data_dir, "test"),  "cases": num_cases_test_target},
        }
        print(f"DEBUG: Base config defined: {base_config}") # <<< ADDED PRINT
        print(f"DEBUG: Datasets to process: {data_sets.keys()}") # <<< ADDED PRINT

    except Exception as e:
        print(f"FATAL ERROR during configuration setup: {e}") # <<< ADDED PRINT
        import traceback
        traceback.print_exc()
        exit()

    # --- Generation Loop ---
    print("DEBUG: Starting generation loop...") # <<< ADDED PRINT
    for set_name, set_config in data_sets.items():
        try:
            print(f"\n--- Processing dataset: {set_name} ---") # Original print
            current_path = set_config["path"]
            num_target_cases = set_config["cases"]

            print(f"DEBUG: Creating directory (if needed): {current_path}") # <<< ADDED PRINT
            os.makedirs(current_path, exist_ok=True)
            print(f"DEBUG: Directory exists/created.") # <<< ADDED PRINT

            run_config = base_config.copy()
            run_config["path"] = current_path
            run_config["root_dir"] = current_path

            print(f"Target path: {current_path}") # Original print
            print(f"Target number of successful CBS cases: {num_target_cases}") # Original print

            # 1. Generate CBS solutions
            print("\nDEBUG: Calling Step 1: create_solutions...") # <<< ADDED PRINT
            create_solutions(current_path, num_target_cases, run_config)
            print("DEBUG: Finished Step 1: create_solutions.") # <<< ADDED PRINT

            # 2. Parse trajectories
            print("\nDEBUG: Calling Step 2: parse_traject...") # <<< ADDED PRINT
            parse_traject(current_path)
            print("DEBUG: Finished Step 2: parse_traject.") # <<< ADDED PRINT

            # 3. Record environment states
            print("\nDEBUG: Calling Step 3: record_env...") # <<< ADDED PRINT
            record_env(current_path, run_config)
            print("DEBUG: Finished Step 3: record_env.") # <<< ADDED PRINT

            print(f"\n--- Finished processing dataset: {set_name} ---") # Original print

        except Exception as e:
            print(f"FATAL ERROR during processing of dataset '{set_name}': {e}") # <<< ADDED PRINT
            import traceback
            traceback.print_exc()
            # Decide whether to continue to the next dataset or stop
            # continue # or break or exit()

    # --- End Generation Loop ---
    print("\n--- All dataset generation steps completed. ---") # Original print