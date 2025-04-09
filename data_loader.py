# File: data_loader.py
# (Modified for Robustness, Correct File/Shape Handling, Validation Split)

import os
import numpy as np
from tqdm import tqdm # Added tqdm
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset # Ensure Dataset is imported
from pathlib import Path # Use Path object

class GNNDataLoader:
    """Manages the creation of training and validation DataLoader instances."""
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.valid_loader = None

        # --- Validate Essential Config Keys ---
        if 'batch_size' not in self.config:
            raise ValueError("Missing 'batch_size' in top-level configuration.")
        if 'num_workers' not in self.config:
             # Default to 0 if not specified, but warn
             print("Warning: 'num_workers' not specified in config, defaulting to 0.")
             self.config['num_workers'] = 0
        if 'train' not in self.config or not isinstance(self.config['train'], dict) or 'root_dir' not in self.config['train']:
             raise ValueError("Missing or invalid 'train' section (with 'root_dir') in configuration.")

        # --- Initialize Training Loader ---
        print("\n--- Initializing Training DataLoader ---")
        try:
            train_set = CreateDataset(self.config, "train")
        except Exception as e:
            print(f"ERROR: Failed to create training dataset: {e}")
            raise # Re-raise the exception

        # Check if dataset is empty BEFORE creating DataLoader
        if len(train_set) == 0:
             print("\nERROR: CreateDataset('train') resulted in an empty dataset.")
             print("Please check:")
             print(f"  - Path exists and is correct: {self.config['train'].get('root_dir')}")
             print(f"  - Dataset directory contains valid 'case_*' subdirectories with required .npy files (states.npy, trajectory.npy, gso.npy).")
             print(f"  - Filtering parameters (min_time, max_time_dl, nb_agents) match the data.")
             raise RuntimeError("Training dataset is empty after loading attempt.")

        try:
            self.train_loader = DataLoader(
                train_set,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["num_workers"],
                pin_memory=torch.cuda.is_available(), # Pin memory only if using CUDA
                # persistent_workers=True if self.config["num_workers"] > 0 else False # Optional: Can speed up epoch starts
                drop_last=False # Keep partial batches by default
            )
            print(f"Initialized Training DataLoader with {len(train_set)} total samples (timesteps).")
            # Optional: Print stats
            # idle_prop = train_set.statistics()
            # print(f"  - Idle action proportion in training data: {idle_prop:.4f}")
        except Exception as e:
             print(f"ERROR: Failed to create training DataLoader: {e}")
             raise # Re-raise


        # --- Optional: Initialize Validation Loader ---
        if 'valid' in self.config and isinstance(self.config.get('valid'), dict) and 'root_dir' in self.config['valid']:
            print("\n--- Initializing Validation DataLoader ---")
            try:
                valid_set = CreateDataset(self.config, "valid")
                if len(valid_set) > 0:
                    try:
                        self.valid_loader = DataLoader(
                            valid_set,
                            batch_size=self.config["batch_size"], # Or a different validation batch size
                            shuffle=False, # No need to shuffle validation data
                            num_workers=self.config["num_workers"],
                            pin_memory=torch.cuda.is_available(),
                            # persistent_workers=True if self.config["num_workers"] > 0 else False,
                            drop_last=False
                        )
                        print(f"Initialized Validation DataLoader with {len(valid_set)} total samples (timesteps).")
                        # idle_prop_val = valid_set.statistics()
                        # print(f"  - Idle action proportion in validation data: {idle_prop_val:.4f}")
                    except Exception as e:
                        print(f"ERROR: Failed to create validation DataLoader: {e}")
                        # Don't raise here, just warn and continue without validation
                        self.valid_loader = None
                else:
                    print("WARNING: Validation dataset configured but resulted in 0 samples. Skipping validation loader.")
            except Exception as e:
                 print(f"ERROR: Failed to create validation dataset: {e}")
                 # Continue without validation loader
                 self.valid_loader = None
        else:
             print("\nValidation data not configured or 'root_dir' missing. Skipping validation loader.")
        print("--- DataLoader Initialization Complete ---")


class CreateDataset(Dataset):
    """Loads data (FOV, Action, GSO) for each timestep from generated cases."""
    def __init__(self, config, mode):
        """
        Args:
            config (dict): The main configuration dictionary.
            mode (str): 'train' or 'valid'.
        """
        if mode not in config or not isinstance(config[mode], dict):
            raise ValueError(f"Configuration missing or invalid section for mode: '{mode}'")
        mode_config = config[mode]

        self.mode = mode
        self.root_dir = Path(mode_config.get("root_dir"))
        if not self.root_dir:
             raise ValueError(f"'root_dir' not specified in '{mode}' config section.")
        if not self.root_dir.is_dir():
            print(f"ERROR: Dataset directory not found or is not a directory: {self.root_dir}")
            # Initialize empty state to prevent errors later
            self.count = 0
            self.cases = []
            self._initialize_empty_arrays()
            return # Allow creation of empty dataset

        # --- Get Agent Number ---
        # Prioritize 'nb_agents' in mode_config, then 'nb_agents' in main config, then 'num_agents' in main config
        self.nb_agents = mode_config.get("nb_agents", config.get("nb_agents", config.get("num_agents")))
        if self.nb_agents is None:
             raise ValueError("Number of agents ('nb_agents' or 'num_agents') not specified in config.")

        # --- Get Time Filters ---
        # Use filters from mode_config, fallback to main config if needed
        self.min_time_filter = mode_config.get("min_time", config.get("min_time"))
        if self.min_time_filter is None:
             print(f"Warning: 'min_time' not specified for '{mode}'. Defaulting min_time to 0.")
             self.min_time_filter = 0
        self.min_time_filter = int(self.min_time_filter) # Ensure integer

        # Use 'max_time_dl' for data loading filter (max trajectory length T)
        self.max_time_filter = mode_config.get("max_time_dl", config.get("max_time_dl"))
        if self.max_time_filter is None:
             print(f"Warning: 'max_time_dl' not specified for '{mode}'. Defaulting max_time to infinity.")
             self.max_time_filter = float('inf')
        else:
             self.max_time_filter = int(self.max_time_filter) # Ensure integer
        print(f"Applying filters for '{mode}': min_traj_len={self.min_time_filter}, max_traj_len={self.max_time_filter}")

        # --- Find and Load Cases ---
        try:
            self.cases = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith("case_")],
                               key=lambda x: int(x.split('_')[-1])) # Sort cases numerically
        except Exception as e:
             print(f"Error listing or sorting cases in {self.root_dir}: {e}")
             self.count = 0; self.cases = []; self._initialize_empty_arrays(); return

        print(f"Found {len(self.cases)} potential cases in {self.root_dir}")

        valid_states_list = []      # List to hold [N, C, H, W] arrays
        valid_trajectories_list = [] # List to hold [N,] arrays (actions)
        valid_gsos_list = []        # List to hold [N, N] arrays

        # Stats for skipping reasons
        stats = {"processed": 0, "skip_missing": 0, "skip_dim": 0, "skip_agent": 0, "skip_time": 0, "skip_filter": 0, "skip_error": 0}

        pbar_load = tqdm(self.cases, desc=f"Loading Dataset ({mode})", unit="case", leave=False)
        for case_name in pbar_load:
            case_path = self.root_dir / case_name
            state_file = case_path / "states.npy"
            traj_file = case_path / "trajectory.npy" # Corrected filename
            gso_file = case_path / "gso.npy"

            # Check if all required files exist
            if not (state_file.exists() and traj_file.exists() and gso_file.exists()):
                stats["skip_missing"] += 1
                continue

            try:
                # Load data for the case
                # Expected shapes:
                # states: (T+1, N, C, H, W) -> num_steps+1 states
                # trajectory: (N, T) -> num_steps actions
                # gso: (T+1, N, N) -> num_steps+1 gsos
                state_data = np.load(state_file)     # float32
                traj_data = np.load(traj_file)       # int32
                gso_data = np.load(gso_file)         # float32

                # --- Validation and Filtering ---
                # 1. Check dimensions
                if state_data.ndim != 5 or traj_data.ndim != 2 or gso_data.ndim != 3:
                    stats["skip_dim"] += 1; continue

                # 2. Check agent consistency
                # N should match across files and config
                N_state = state_data.shape[1]
                N_traj = traj_data.shape[0]
                N_gso = gso_data.shape[1]
                if not (N_state == N_traj == N_gso == self.nb_agents):
                    # print(f"Agent mismatch {case_name}: State N={N_state}, Traj N={N_traj}, GSO N={N_gso}, Expected N={self.nb_agents}") # Debug
                    stats["skip_agent"] += 1; continue

                # 3. Check time step consistency
                # State/GSO should have T+1 steps, Trajectory should have T steps
                T_traj = traj_data.shape[1] # Number of actions/steps (T)
                T_plus_1_state = state_data.shape[0]
                T_plus_1_gso = gso_data.shape[0]
                if not (T_plus_1_state == T_plus_1_gso == T_traj + 1):
                     stats["skip_time"] += 1; continue

                # 4. Apply trajectory length filtering (based on T, the number of actions)
                if not (self.min_time_filter <= T_traj <= self.max_time_filter):
                    stats["skip_filter"] += 1; continue
                # --- End Validation ---

                # --- Add data for each timestep t from 0 to T-1 ---
                # We use state[t], action[t], gso[t] as one sample instance
                for t in range(T_traj):
                    valid_states_list.append(state_data[t])       # State at time t -> (N, C, H, W)
                    valid_trajectories_list.append(traj_data[:, t]) # Action taken at time t -> (N,)
                    valid_gsos_list.append(gso_data[t])           # GSO corresponding to state at t -> (N, N)

                stats["processed"] += 1
                pbar_load.set_postfix({"LoadOK": stats["processed"], "Skip": sum(stats[k] for k in stats if k.startswith('skip'))})

            except Exception as e:
                print(f"\nWarning: Error processing {case_name}: {e}. Skipping.")
                stats["skip_error"] += 1
                continue
        # --- End Case Loop ---

        self.count = len(valid_states_list) # Total number of samples (timesteps)
        pbar_load.close()

        total_skipped = sum(stats[k] for k in stats if k.startswith('skip'))
        print(f"\nFinished loading dataset '{mode}'.")
        print(f"  Processed {stats['processed']} cases successfully.")
        if total_skipped > 0:
            print(f"  Skipped {total_skipped} cases:")
            print(f"    - Missing Files: {stats['skip_missing']}")
            print(f"    - Dim Mismatch: {stats['skip_dim']}")
            print(f"    - Agent#: {stats['skip_agent']}")
            print(f"    - Time Mismatch: {stats['skip_time']}")
            print(f"    - Filtered Out: {stats['skip_filter']}")
            print(f"    - Load Errors: {stats['skip_error']}")
        print(f"Total individual samples loaded (timesteps): {self.count}")

        if self.count == 0:
            print(f"WARNING: No valid samples found for mode '{mode}' after loading and filtering!")
            self._initialize_empty_arrays()
        else:
            # Use np.stack for efficiency (creates new arrays)
            print("Stacking loaded data...")
            self.states = np.stack(valid_states_list, axis=0)           # Shape: (TotalSamples, N, C, H, W)
            self.trajectories = np.stack(valid_trajectories_list, axis=0) # Shape: (TotalSamples, N)
            self.gsos = np.stack(valid_gsos_list, axis=0)               # Shape: (TotalSamples, N, N)
            print(f"Final shapes: States={self.states.shape}, Trajectories={self.trajectories.shape}, GSOs={self.gsos.shape}")

            # Check state channel dimension (should be 3 based on env)
            if self.states.ndim == 5 and self.states.shape[2] != 3:
                 print(f"WARNING: Loaded states have {self.states.shape[2]} channels (dim 2), but expected 3 channels based on GraphEnv.")
            elif self.states.ndim != 5:
                 print(f"WARNING: Loaded states have unexpected dimensions: {self.states.ndim}. Expected 5.")

    def _initialize_empty_arrays(self):
        """Helper to set empty arrays if loading fails or results in no samples."""
        # Define shapes with 0 samples but correct other dimensions if possible
        # This requires knowing C, H, W, N beforehand. Let's assume N is known.
        # C, H, W might need defaults or reading from config. Assume 3, 5, 5 for now.
        C, H, W = 3, 5, 5 # Default/Example values, adjust if needed
        N = self.nb_agents if hasattr(self, 'nb_agents') else 0
        self.states = np.empty((0, N, C, H, W), dtype=np.float32)
        self.trajectories = np.empty((0, N), dtype=np.int64) # Use int64 for LongTensor compatibility
        self.gsos = np.empty((0, N, N), dtype=np.float32)
        self.count = 0

    def statistics(self):
        """Calculates proportion of action 0 (idle) in trajectories."""
        if self.trajectories.size == 0: return 0.0
        zeros = np.count_nonzero(self.trajectories == 0)
        total_elements = self.trajectories.size
        return zeros / total_elements if total_elements > 0 else 0.0

    def __len__(self):
        """Returns the total number of samples (timesteps) in the dataset."""
        return self.count

    def __getitem__(self, index):
        """
        Returns a single sample (data for one timestep).
        Output order: state, action, gso
        """
        if not 0 <= index < self.count:
             raise IndexError(f"Index {index} out of bounds for dataset with size {self.count}")

        # Extract data for the given index
        states_np = self.states[index]           # (N, C, H, W)
        trayec_np = self.trajectories[index]     # (N,)
        gsos_np = self.gsos[index]               # (N, N)

        # Convert to PyTorch tensors
        states_sample = torch.from_numpy(states_np).float()
        # Actions need to be LongTensor for CrossEntropyLoss
        trayec_sample = torch.from_numpy(trayec_np).long()
        gsos_sample = torch.from_numpy(gsos_np).float()

        # Return order must match the training loop unpacking: state, action, gso
        return states_sample, trayec_sample, gsos_sample

# --- Test Block (Example - uncomment and adapt paths/config) ---
if __name__ == "__main__":
     print("\n--- Running DataLoader Test ---")
     # Define a sample config reflecting your actual setup
     test_config = {
         "train": {
             "root_dir": "dataset/5_8_28_fov5_test/train", # <<<--- UPDATE PATH
             "mode": "train",
             "nb_agents": 5,    # Should match the data
             "min_time": 5,     # Example filter
             "max_time_dl": 55, # Example filter (max traj length T)
         },
         "valid": {
              "root_dir": "dataset/5_8_28_fov5_test/val", # <<<--- UPDATE PATH
              "mode": "valid",
              "nb_agents": 5,
              "min_time": 5,
              "max_time_dl": 55,
         },
         "num_agents": 5, # Global fallback/check
         "batch_size": 4,
         "num_workers": 0, # Use 0 for easier debugging
         # Add any other keys required by CreateDataset/GNNDataLoader
     }
     try:
         data_loader_manager = GNNDataLoader(test_config)

         # Test Train Loader
         if data_loader_manager.train_loader and len(data_loader_manager.train_loader.dataset) > 0:
             print("\n--- Iterating through train_loader (first 2 batches) ---")
             count = 0
             for batch_s, batch_t, batch_g in data_loader_manager.train_loader:
                 print(f"Train Batch {count}:")
                 print(f"  States Shape: {batch_s.shape}, Type: {batch_s.dtype}")
                 print(f"  Traj Shape:   {batch_t.shape}, Type: {batch_t.dtype}") # Should be torch.int64 (Long)
                 print(f"  GSO Shape:    {batch_g.shape}, Type: {batch_g.dtype}")
                 # Check action values are valid (0-4)
                 if torch.any(batch_t < 0) or torch.any(batch_t >= 5):
                      print(f"  ERROR: Invalid action value found in batch {count}! Min: {batch_t.min()}, Max: {batch_t.max()}")
                 count += 1
                 if count >= 2: break # Limit test output
         else:
              print("\nTrain loader is empty or was not created.")

         # Test Validation Loader
         if data_loader_manager.valid_loader and len(data_loader_manager.valid_loader.dataset) > 0:
             print("\n--- Iterating through valid_loader (first 2 batches) ---")
             count = 0
             for batch_s, batch_t, batch_g in data_loader_manager.valid_loader:
                 print(f"Valid Batch {count}:")
                 print(f"  States Shape: {batch_s.shape}, Type: {batch_s.dtype}")
                 print(f"  Traj Shape:   {batch_t.shape}, Type: {batch_t.dtype}")
                 print(f"  GSO Shape:    {batch_g.shape}, Type: {batch_g.dtype}")
                 count += 1
                 if count >= 2: break
         else:
              print("\nValidation loader is empty or was not created.")

     except Exception as e:
         print(f"\nERROR during data loader test: {e}")
         traceback.print_exc()
     print("\n--- DataLoader Test Finished ---")