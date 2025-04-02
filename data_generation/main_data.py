# File: data_generation/main_data.py
# MODIFIED: Use relative imports
# from dataset_gen import create_solutions
# from trayectory_parser import parse_traject
# from record import record_env
from .dataset_gen import create_solutions
from .trayectory_parser import parse_traject
from .record import record_env
import os

if __name__ == "__main__":
    cases = 2000 # Target number of cases *per directory* (train/val etc.)

    # Define base directory structure
    base_data_dir = "dataset/5_8_28" # Use forward slashes

    # Configuration common to all sets
    base_config = {
        "num_agents": 5, # For environment simulation during recording
        "map_shape": [28, 28], # For dataset_gen
        "nb_agents": 5, # For dataset_gen and recording (should match num_agents ideally)
        "nb_obstacles": 8, # For dataset_gen
        "sensor_range": 4, # For recording environment
        "board_size": [28, 28], # For recording environment
        "max_time": 32, # For recording environment? Check usage.
        "min_time": 9,  # For recording environment? Check usage. Maybe filter trajectories?
        # Add other necessary keys like device if needed by downstream steps
        "device": "gpu",
    }

    # Define different sets (e.g., train, validation)
    data_sets = {
        "train": {"path": os.path.join(base_data_dir, "train"), "cases": cases},
        "val": {"path": os.path.join(base_data_dir, "val"), "cases": int(cases * 0.2)}, # Example: 20% for validation
        # "test": {"path": os.path.join(base_data_dir, "test"), "cases": int(cases * 0.1)} # Example: 10% for test
    }

    for set_name, set_config in data_sets.items():
        print(f"\n--- Processing dataset: {set_name} ---")
        current_path = set_config["path"]
        num_target_cases = set_config["cases"]

        # Update config with the specific path for this set
        run_config = base_config.copy()
        run_config["path"] = current_path # Set the specific path for this run
        run_config["root_dir"] = current_path # Often root_dir is expected

        print(f"Target path: {current_path}")
        print(f"Target number of cases: {num_target_cases}")

        # 1. Generate CBS solutions (input.yaml, solution.yaml)
        print("\nStep 1: Generating CBS solutions...")
        create_solutions(current_path, num_target_cases, run_config)

        # 2. Parse trajectories (solution.yaml -> trajectory.npy)
        print("\nStep 2: Parsing trajectories...")
        parse_traject(current_path) # Assumes parse_traject processes all cases in the dir

        # 3. Record environment states (trajectory.npy -> states.npy, gso.npy, etc.)
        print("\nStep 3: Recording environment states...")
        record_env(current_path, run_config) # Pass the run_config containing board_size etc.

        print(f"\n--- Finished processing dataset: {set_name} ---")

    print("\n--- All dataset generation steps completed. ---")