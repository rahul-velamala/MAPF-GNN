# File: data_loader.py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from torch.utils.data import DataLoader


class GNNDataLoader:
    def __init__(self, config):
        self.config = config

        # Ensure 'train' key exists in config
        if 'train' not in self.config:
            raise ValueError("Missing 'train' section in configuration.")

        # Ensure essential keys exist for CreateDataset
        if 'batch_size' not in self.config or 'num_workers' not in self.config:
             raise ValueError("Missing 'batch_size' or 'num_workers' in top-level configuration.")

        print("Initializing training dataset...") # Added print
        train_set = CreateDataset(self.config, "train")

        # Check if dataset is empty before creating DataLoader
        if len(train_set) == 0:
             print("\nERROR: CreateDataset('train') resulted in an empty dataset.")
             print("Please check:")
             print(f"  - Path exists and is correct: {self.config['train'].get('root_dir')}")
             print(f"  - Dataset directory contains valid 'case_*' subdirectories with required .npy files.")
             print(f"  - Filtering parameters (min_time, max_time_dl, nb_agents) match the data.")
             raise RuntimeError("Training dataset is empty after loading attempt.") # Modified error


        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=torch.cuda.is_available(), # Pin memory only if using CUDA
            # persistent_workers=True if self.config["num_workers"] > 0 else False # Optional
        )
        print(f"Initialized DataLoader with {len(train_set)} total samples (timesteps).") # Modified print

        # --- Optional: Initialize Validation Loader ---
        self.valid_loader = None
        if 'valid' in self.config and self.config.get('valid'):
            print("Initializing validation dataset...")
            valid_set = CreateDataset(self.config, "valid")
            if len(valid_set) > 0:
                self.valid_loader = DataLoader(
                    valid_set,
                    batch_size=self.config["batch_size"], # Or a different batch size for validation
                    shuffle=False, # No need to shuffle validation data
                    num_workers=self.config["num_workers"],
                    pin_memory=torch.cuda.is_available(),
                    # persistent_workers=True if self.config["num_workers"] > 0 else False
                )
                print(f"Initialized Validation DataLoader with {len(valid_set)} total samples (timesteps).")
            else:
                print("WARNING: Validation dataset configured but resulted in 0 samples.")
        # --- End Validation Loader ---


class CreateDataset(data.Dataset):
    def __init__(self, config, mode):
        """
        Args:
            config (dict): The main configuration dictionary.
            mode (string): 'train' or 'valid'.
        """
        mode_config = config.get(mode)
        if mode_config is None:
            raise ValueError(f"Configuration missing section for mode: '{mode}'")

        self.config = mode_config
        self.dir_path = self.config.get("root_dir")
        if not self.dir_path:
             raise ValueError(f"'root_dir' not specified in '{mode}' config section.")

        # Use main config for global num_agents if not in mode_config
        self.nb_agents = self.config.get("nb_agents", config.get("num_agents"))
        if self.nb_agents is None:
             raise ValueError("Number of agents ('nb_agents' or 'num_agents') not specified in config.")

        self.min_time_filter = self.config.get("min_time")
        if self.min_time_filter is None:
             print(f"Warning: 'min_time' not specified in '{mode}' config. Defaulting min_time to 0.")
             self.min_time_filter = 0

        self.max_time_filter = self.config.get("max_time_dl")
        if self.max_time_filter is None:
             print(f"Warning: 'max_time_dl' not specified in '{mode}' config. Defaulting max_time to infinity.")
             self.max_time_filter = float('inf')

        if not os.path.isdir(self.dir_path):
            print(f"ERROR: Dataset directory not found or is not a directory: {self.dir_path}")
            self.count = 0
            self.cases = []
            self._initialize_empty_arrays()
            return

        self.cases = sorted([d for d in os.listdir(self.dir_path) if d.startswith("case_") and os.path.isdir(os.path.join(self.dir_path, d))],
                           key=lambda x: int(x.split('_')[-1])) # Sort cases
        print(f"Found {len(self.cases)} potential cases in {self.dir_path}")

        valid_states_list = []
        valid_trajectories_list = []
        valid_gsos_list = []

        cases_processed = 0
        cases_skipped = 0
        pbar_load = tqdm(self.cases, desc=f"Loading Dataset ({mode})", unit="case", leave=False)
        for case in pbar_load:
            case_path = os.path.join(self.dir_path, case)
            state_file = os.path.join(case_path, "states.npy")
            # ---- FIX 1: Correct trajectory filename ----
            traj_file = os.path.join(case_path, "trajectory.npy") # Changed from trajectory_record.npy
            gso_file = os.path.join(case_path, "gso.npy")

            if not (os.path.exists(state_file) and os.path.exists(traj_file) and os.path.exists(gso_file)):
                cases_skipped += 1
                continue

            try:
                # Load data for the case
                # record.py saves:
                # states: (T+1, N, C, H, W) -> num_steps+1 states
                # trajectory: (N, T) -> num_steps actions
                # gso: (T+1, N, N) -> num_steps+1 gsos
                state_data = np.load(state_file)
                traj_data = np.load(traj_file)
                gso_data = np.load(gso_file)

                # --- Validation and Filtering ---
                if state_data.ndim != 5 or traj_data.ndim != 2 or gso_data.ndim != 3:
                    # print(f"Skipping {case}: Invalid dimensions S:{state_data.ndim}, T:{traj_data.ndim}, G:{gso_data.ndim}") # Debug
                    cases_skipped += 1
                    continue

                if not (state_data.shape[1] == traj_data.shape[0] == gso_data.shape[1] == self.nb_agents):
                    # print(f"Skipping {case}: Agent mismatch (Exp:{self.nb_agents} S:{state_data.shape[1]} T:{traj_data.shape[0]} G:{gso_data.shape[1]})") # Debug
                    cases_skipped += 1
                    continue

                num_steps = traj_data.shape[1] # Number of actions/steps

                # ---- FIX 2: Correct time step consistency check ----
                # Check if states and GSO have T+1 length compared to trajectory T length
                if not (state_data.shape[0] == gso_data.shape[0] == num_steps + 1):
                     # print(f"Skipping {case}: Time step mismatch (Traj: {num_steps}, State: {state_data.shape[0]}, GSO: {gso_data.shape[0]}, Exp State/GSO: {num_steps + 1})") # Debug
                     cases_skipped += 1
                     continue

                # Apply time filtering based on the number of steps (trajectory length)
                if not (self.min_time_filter <= num_steps <= self.max_time_filter):
                    # print(f"Skipping {case}: Trajectory length {num_steps} outside range [{self.min_time_filter}, {self.max_time_filter}]") # Debug
                    cases_skipped += 1
                    continue
                # --- End Validation ---

                # Add data for each timestep as a sample
                # We use state[t], action[t], gso[t] as one sample
                for t in range(num_steps):
                    valid_states_list.append(state_data[t])       # State at time t (N, C, H, W)
                    valid_trajectories_list.append(traj_data[:, t]) # Action taken at time t (N,)
                    valid_gsos_list.append(gso_data[t])           # GSO corresponding to state at t (N, N)

                cases_processed += 1
                pbar_load.set_postfix({"Loaded": cases_processed, "Skipped": cases_skipped})

            except Exception as e:
                print(f"\nWarning: Error processing {case}: {e}. Skipping.")
                cases_skipped += 1
                continue

        self.count = len(valid_states_list)
        print(f"\nFinished loading dataset '{mode}'. Processed {cases_processed} cases, skipped {cases_skipped}.")
        print(f"Total individual samples (timesteps): {self.count}")

        if self.count == 0:
            print(f"WARNING: No valid samples found for mode '{mode}' after loading and filtering!")
            self._initialize_empty_arrays()
        else:
            self.states = np.stack(valid_states_list, axis=0)
            self.trajectories = np.stack(valid_trajectories_list, axis=0)
            self.gsos = np.stack(valid_gsos_list, axis=0)
            print(f"Final shapes: States={self.states.shape}, Trajectories={self.trajectories.shape}, GSOs={self.gsos.shape}")
            # Optional: Calculate statistics only if needed, can be slow
            # zeros_stat = self.statistics()
            # print(f"Trajectory Action '0' (Idle) Proportion: {zeros_stat:.4f}")

    def _initialize_empty_arrays(self):
        """Helper to set empty arrays if loading fails or results in no samples."""
        self.states = np.array([])
        self.trajectories = np.array([])
        self.gsos = np.array([])

    def statistics(self):
        """Calculates proportion of action 0 (idle) in trajectories."""
        if self.trajectories.size == 0:
            return 0.0
        zeros = np.count_nonzero(self.trajectories == 0)
        total_elements = self.trajectories.size # Use .size for total elements
        if total_elements == 0:
            return 0.0
        return zeros / total_elements

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        """
        Returns a single sample (data for one timestep).
        state : (agents, channels, dimX, dimY),
        trayec: (agents,) - Actions should be LongTensor
        gsos: (agents, agents)
        """
        if index >= self.count:
             raise IndexError(f"Index {index} out of bounds for dataset with size {self.count}")

        states_sample = torch.from_numpy(self.states[index]).float()
        # ---- FIX 3: Convert actions to Long ----
        trayec_sample = torch.from_numpy(self.trajectories[index]).long() # Use .long() for actions
        gsos_sample = torch.from_numpy(self.gsos[index]).float()

        return states_sample, trayec_sample, gsos_sample

# --- Test Block (Example - adapt paths/config as needed) ---
# if __name__ == "__main__":
#     test_config = {
#         "train": {
#             "root_dir": "dataset/5_8_28/train", # Point to actual data
#             "mode": "train",
#             "nb_agents": 5,
#             "min_time": 5,
#             "max_time_dl": 55, # Use updated max time
#         },
#         "valid": {
#             "root_dir": "dataset/5_8_28/val",
#             "mode": "valid",
#             "nb_agents": 5,
#             "min_time": 5,
#             "max_time_dl": 55,
#         },
#         "num_agents": 5,
#         "batch_size": 4,
#         "num_workers": 0,
#     }
#     print("--- Testing GNNDataLoader ---")
#     try:
#         data_loader_instance = GNNDataLoader(test_config)
#         if data_loader_instance.train_loader and len(data_loader_instance.train_loader.dataset) > 0:
#             print("\n--- Iterating through train_loader ---")
#             count = 0
#             for batch_s, batch_t, batch_g in data_loader_instance.train_loader:
#                 print(f"Batch {count}:")
#                 print(f"  States Shape: {batch_s.shape}, Type: {batch_s.dtype}")
#                 print(f"  Traj Shape:   {batch_t.shape}, Type: {batch_t.dtype}") # Should be torch.int64 (Long)
#                 print(f"  GSO Shape:    {batch_g.shape}, Type: {batch_g.dtype}")
#                 count += 1
#                 if count >= 2: break
#         else:
#              print("\nTrain loader is empty or not created.")
#     except Exception as e:
#         print(f"\nError during data loader test: {e}")
#         import traceback
#         traceback.print_exc()