# File: data_loader.py
# (Modified for Robustness, Correct File/Shape Handling, Validation Split)

import os
import numpy as np
from tqdm import tqdm # Added tqdm
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset # Ensure Dataset is imported
from pathlib import Path # Use Path object
import logging
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Adjust level as needed

class GNNDataLoader:
    """Manages the creation of training and validation DataLoader instances."""
    def __init__(self, config: dict):
        self.config = config
        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None

        # --- Validate Essential Config Keys ---
        if 'batch_size' not in self.config:
            raise ValueError("Missing 'batch_size' in top-level configuration.")
        batch_size = int(self.config['batch_size']) # Ensure int
        if 'num_workers' not in self.config:
             logger.warning("Config key 'num_workers' not specified, defaulting to 0.")
             self.config['num_workers'] = 0
        num_workers = int(self.config['num_workers'])
        if 'train' not in self.config or not isinstance(self.config['train'], dict) or 'root_dir' not in self.config['train']:
             raise ValueError("Missing or invalid 'train' section (with 'root_dir') in configuration.")

        # --- Initialize Training Loader ---
        logger.info("\n--- Initializing Training DataLoader ---")
        try:
            # Pass the main config down, CreateDataset will extract 'train' part
            train_set = CreateDataset(self.config, "train")
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}", exc_info=True)
            raise # Re-raise the exception

        # Check if dataset is empty BEFORE creating DataLoader
        if len(train_set) == 0:
             logger.error("CreateDataset('train') resulted in an empty dataset.")
             logger.error("Please check:")
             logger.error(f"  - Path exists and is correct: {self.config['train'].get('root_dir')}")
             logger.error(f"  - Dataset directory contains valid 'case_*' subdirectories with required .npy files (states.npy, trajectory.npy, gso.npy).")
             logger.error(f"  - Filtering parameters (min_time, max_time_dl, nb_agents) match the data.")
             raise RuntimeError("Training dataset is empty after loading attempt.")

        try:
            self.train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(), # Pin memory only if using CUDA
                persistent_workers=(num_workers > 0), # Can speed up epoch starts if workers > 0
                drop_last=False # Keep partial batches by default
            )
            logger.info(f"Initialized Training DataLoader with {len(train_set)} total samples (timesteps).")
            idle_prop = train_set.statistics()
            logger.info(f"  - Idle action proportion in training data: {idle_prop:.4f}")
        except Exception as e:
             logger.error(f"Failed to create training DataLoader: {e}", exc_info=True)
             raise # Re-raise


        # --- Optional: Initialize Validation Loader ---
        if 'valid' in self.config and isinstance(self.config.get('valid'), dict) and 'root_dir' in self.config['valid']:
            logger.info("\n--- Initializing Validation DataLoader ---")
            try:
                valid_set = CreateDataset(self.config, "valid")
                if len(valid_set) > 0:
                    try:
                        self.valid_loader = DataLoader(
                            valid_set,
                            batch_size=batch_size, # Or a different validation batch size
                            shuffle=False, # No need to shuffle validation data
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=(num_workers > 0),
                            drop_last=False
                        )
                        logger.info(f"Initialized Validation DataLoader with {len(valid_set)} total samples (timesteps).")
                        idle_prop_val = valid_set.statistics()
                        logger.info(f"  - Idle action proportion in validation data: {idle_prop_val:.4f}")
                    except Exception as e:
                        logger.error(f"Failed to create validation DataLoader: {e}. Skipping validation.", exc_info=True)
                        self.valid_loader = None
                else:
                    logger.warning("Validation dataset configured but resulted in 0 samples. Skipping validation loader.")
            except Exception as e:
                 logger.error(f"Failed to create validation dataset: {e}. Skipping validation loader.", exc_info=True)
                 self.valid_loader = None
        else:
             logger.info("\nValidation data not configured or 'root_dir' missing. Skipping validation loader.")
        logger.info("--- DataLoader Initialization Complete ---")


class CreateDataset(Dataset):
    """Loads data (FOV, Action, GSO) for each timestep from generated cases."""
    def __init__(self, config: dict, mode: str):
        """
        Args:
            config (dict): The main configuration dictionary.
            mode (str): 'train' or 'valid'.
        """
        if mode not in config or not isinstance(config[mode], dict):
            raise ValueError(f"Configuration missing or invalid section for mode: '{mode}'")
        mode_config = config[mode]
        self.mode = mode
        logger.info(f"Creating Dataset for mode: '{mode}'")

        root_dir_str = mode_config.get("root_dir")
        if not root_dir_str:
             raise ValueError(f"'root_dir' not specified in '{mode}' config section.")
        self.root_dir = Path(root_dir_str)

        if not self.root_dir.is_dir():
            logger.error(f"Dataset directory not found or is not a directory: {self.root_dir}")
            # Initialize empty state to prevent errors later
            self._initialize_empty_arrays(config) # Pass config to get agent num etc.
            return # Allow creation of empty dataset

        # --- Get Agent Number ---
        # Prioritize 'nb_agents' in mode_config, fallback to global 'num_agents'
        self.nb_agents = mode_config.get("nb_agents", config.get("num_agents"))
        if self.nb_agents is None:
             raise ValueError("Number of agents ('nb_agents' or 'num_agents') not specified in config.")
        self.nb_agents = int(self.nb_agents)
        logger.info(f"Expecting {self.nb_agents} agents per sample.")

        # --- Get Time Filters ---
        # Use filters from mode_config, fallback to main config if needed
        self.min_time_filter = mode_config.get("min_time", config.get("min_time"))
        if self.min_time_filter is None:
             logger.warning(f"'min_time' not specified for '{mode}'. Defaulting min_time to 0.")
             self.min_time_filter = 0
        self.min_time_filter = int(self.min_time_filter) # Ensure integer

        # Use 'max_time_dl' for data loading filter (max trajectory length T)
        self.max_time_filter = mode_config.get("max_time_dl", config.get("max_time_dl"))
        if self.max_time_filter is None:
             logger.warning(f"'max_time_dl' not specified for '{mode}'. Defaulting max_time to infinity.")
             self.max_time_filter = float('inf')
        else:
             self.max_time_filter = int(self.max_time_filter) # Ensure integer
        logger.info(f"Applying filters for '{mode}': min_traj_len(T)>={self.min_time_filter}, max_traj_len(T)<={self.max_time_filter}")

        # --- Find and Load Cases ---
        try:
            # List directories starting with 'case_' and sort numerically
            self.cases = sorted(
                [d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith("case_")],
                key=lambda x: int(x.name.split('_')[-1])
            )
        except Exception as e:
             logger.error(f"Error listing or sorting cases in {self.root_dir}: {e}", exc_info=True)
             self._initialize_empty_arrays(config); return

        logger.info(f"Found {len(self.cases)} potential cases in {self.root_dir}")
        if not self.cases:
            self._initialize_empty_arrays(config); return

        valid_states_list = []      # List to hold [N, C, H, W] arrays
        valid_trajectories_list = [] # List to hold [N,] arrays (actions)
        valid_gsos_list = []        # List to hold [N, N] arrays

        # Stats for skipping reasons
        stats = {"processed": 0, "skip_missing": 0, "skip_dim": 0, "skip_agent": 0, "skip_time": 0, "skip_filter": 0, "skip_error": 0}

        pbar_load = tqdm(self.cases, desc=f"Loading Dataset ({mode})", unit="case", leave=False)
        for case_path in pbar_load: # case_path is now a Path object
            state_file = case_path / "states.npy"
            traj_file = case_path / "trajectory.npy" # Corrected filename
            gso_file = case_path / "gso.npy"

            # Check if all required files exist
            if not (state_file.exists() and traj_file.exists() and gso_file.exists()):
                # logger.debug(f"Skipping {case_path.name}: Missing one or more .npy files.")
                stats["skip_missing"] += 1
                continue

            try:
                # Load data for the case
                # Expected shapes:
                # states: (T+1, N, C, H, W) -> num_steps+1 states
                # trajectory: (N, T) -> num_steps actions
                # gso: (T+1, N, N) -> num_steps+1 gsos
                state_data = np.load(state_file)     # Should be float32
                traj_data = np.load(traj_file)       # Should be int32/int64
                gso_data = np.load(gso_file)         # Should be float32

                # --- Validation and Filtering ---
                # 1. Check dimensions
                if state_data.ndim != 5 or traj_data.ndim != 2 or gso_data.ndim != 3:
                    # logger.debug(f"Skipping {case_path.name}: Dim mismatch. S:{state_data.ndim} T:{traj_data.ndim} G:{gso_data.ndim}")
                    stats["skip_dim"] += 1; continue

                # 2. Check agent consistency (N should match across files and config)
                N_state, N_traj_agents, N_gso = state_data.shape[1], traj_data.shape[0], gso_data.shape[1]
                if not (N_state == N_traj_agents == N_gso == self.nb_agents):
                    # logger.debug(f"Skipping {case_path.name}: Agent mismatch. S:{N_state} T:{N_traj_agents} G:{N_gso}. Expected:{self.nb_agents}")
                    stats["skip_agent"] += 1; continue

                # 3. Check time step consistency (State/GSO have T+1, Trajectory has T)
                T_traj = traj_data.shape[1] # Number of actions/steps (T)
                T_plus_1_state = state_data.shape[0]
                T_plus_1_gso = gso_data.shape[0]
                if not (T_plus_1_state == T_plus_1_gso == T_traj + 1):
                     # logger.debug(f"Skipping {case_path.name}: Time mismatch. S:{T_plus_1_state} T:{T_traj} G:{T_plus_1_gso}")
                     stats["skip_time"] += 1; continue

                # 4. Apply trajectory length filtering (based on T, the number of actions)
                if not (self.min_time_filter <= T_traj <= self.max_time_filter):
                    # logger.debug(f"Skipping {case_path.name}: Filtered by time T={T_traj}. Range [{self.min_time_filter},{self.max_time_filter}]")
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
                logger.warning(f"Error processing {case_path.name}: {e}. Skipping.", exc_info=True)
                stats["skip_error"] += 1
                continue
        # --- End Case Loop ---

        self.count = len(valid_states_list) # Total number of samples (timesteps)
        pbar_load.close()

        total_skipped = sum(stats[k] for k in stats if k.startswith('skip'))
        logger.info(f"\nFinished loading dataset '{mode}'.")
        logger.info(f"  Processed {stats['processed']} cases successfully.")
        if total_skipped > 0:
            logger.info(f"  Skipped {total_skipped} cases:")
            logger.info(f"    - Missing Files: {stats['skip_missing']}")
            logger.info(f"    - Dim Mismatch: {stats['skip_dim']}")
            logger.info(f"    - Agent#: {stats['skip_agent']}")
            logger.info(f"    - Time Mismatch: {stats['skip_time']}")
            logger.info(f"    - Filtered Out: {stats['skip_filter']}")
            logger.info(f"    - Load Errors: {stats['skip_error']}")
        logger.info(f"Total individual samples loaded (timesteps): {self.count}")

        if self.count == 0:
            logger.warning(f"No valid samples found for mode '{mode}' after loading and filtering!")
            self._initialize_empty_arrays(config)
        else:
            # Use np.stack for efficiency (creates new arrays)
            logger.info("Stacking loaded data...")
            self.states = np.stack(valid_states_list, axis=0).astype(np.float32) # Ensure float32
            # Ensure actions are int64 for LongTensor conversion
            self.trajectories = np.stack(valid_trajectories_list, axis=0).astype(np.int64)
            self.gsos = np.stack(valid_gsos_list, axis=0).astype(np.float32) # Ensure float32
            logger.info(f"Final shapes: States={self.states.shape}, Trajectories={self.trajectories.shape}, GSOs={self.gsos.shape}")

            # --- Final Sanity Checks ---
            if self.states.ndim != 5:
                 logger.error(f"Final states array has unexpected dimensions: {self.states.ndim}. Expected 5.")
            elif self.states.shape[1] != self.nb_agents:
                 logger.error(f"Final states agent dimension mismatch. Expected {self.nb_agents}, got {self.states.shape[1]}")

            # Check state channel dimension (should be 3 based on env)
            expected_channels = 3
            if self.states.shape[2] != expected_channels:
                 logger.warning(f"Loaded states have {self.states.shape[2]} channels (dim 2), but expected {expected_channels} based on GraphEnv.")

            if self.trajectories.ndim != 2 or self.trajectories.shape[1] != self.nb_agents:
                 logger.error(f"Final trajectories array shape error. Expected (samples, {self.nb_agents}), got {self.trajectories.shape}")
            if self.gsos.ndim != 3 or self.gsos.shape[1:] != (self.nb_agents, self.nb_agents):
                 logger.error(f"Final GSOs array shape error. Expected (samples, {self.nb_agents}, {self.nb_agents}), got {self.gsos.shape}")

    def _initialize_empty_arrays(self, config):
        """Helper to set empty arrays if loading fails or results in no samples."""
        logger.warning("Initializing dataset with empty arrays.")
        # Infer shapes from config if possible
        N = int(config.get("num_agents", 0))
        C = 3 # Default channels from GraphEnv
        # Infer H, W from pad
        pad_val = int(config.get("pad", 3)) # Default to pad 3 (5x5)
        H = W = (pad_val * 2) - 1

        self.states = np.empty((0, N, C, H, W), dtype=np.float32)
        self.trajectories = np.empty((0, N), dtype=np.int64) # Use int64 for LongTensor compatibility
        self.gsos = np.empty((0, N, N), dtype=np.float32)
        self.count = 0
        # Add cases attribute for compatibility with OE DAgger in train.py
        self.cases = []
        self.root_dir = Path(config.get(self.mode, {}).get("root_dir", ".")) # Store root_dir even if empty

    def statistics(self) -> float:
        """Calculates proportion of action 0 (idle) in trajectories."""
        if self.trajectories.size == 0: return 0.0
        zeros = np.count_nonzero(self.trajectories == 0)
        total_elements = self.trajectories.size
        return float(zeros / total_elements) if total_elements > 0 else 0.0

    def __len__(self) -> int:
        """Returns the total number of samples (timesteps) in the dataset."""
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single sample (data for one timestep).
        Output order: state, action, gso
        """
        if not 0 <= index < self.count:
             # This should ideally not happen if DataLoader handles indices correctly
             logger.error(f"Index {index} out of bounds for dataset with size {self.count}")
             # Returning dummy data or raising IndexError are options
             # For robustness, let's try to return the first sample, but log error
             if self.count > 0:
                 index = 0
             else: # Cannot return anything if dataset is truly empty
                 raise IndexError(f"Index {index} out of bounds for empty dataset.")

        # Extract data for the given index
        states_np = self.states[index]           # (N, C, H, W)
        trayec_np = self.trajectories[index]     # (N,)
        gsos_np = self.gsos[index]               # (N, N)

        # Convert to PyTorch tensors
        states_sample = torch.from_numpy(states_np).float()
        # Actions need to be LongTensor for CrossEntropyLoss
        trayec_sample = torch.from_numpy(trayec_np).long() # Already ensured int64 in __init__
        gsos_sample = torch.from_numpy(gsos_np).float()

        # Return order must match the training loop unpacking: state, action, gso
        return states_sample, trayec_sample, gsos_sample

# --- Test Block (Example - uncomment and adapt paths/config) ---
if __name__ == "__main__":
     print("\n--- Running DataLoader Test ---")
     # Define a sample config reflecting your actual setup
     # >>> IMPORTANT: UPDATE PATHS AND PARAMETERS TO MATCH YOUR DATASET <<<
     test_config = {
         "num_agents": 5, # Global fallback/check
         "pad": 3,        # Should match data FOV (pad=3 -> 5x5)
         "batch_size": 4,
         "num_workers": 0, # Use 0 for easier debugging
         "train": {
             "root_dir": "dataset/5_8_28_fov5_parallel/train", # <<<--- UPDATE PATH
             "mode": "train",
             "nb_agents": 5,    # Should match the data
             "min_time": 5,     # Example filter
             "max_time_dl": 55, # Example filter (max traj length T)
         },
         "valid": {
              "root_dir": "dataset/5_8_28_fov5_parallel/val", # <<<--- UPDATE PATH
              "mode": "valid",
              "nb_agents": 5,
              "min_time": 5,
              "max_time_dl": 55,
         },
         # Add any other top-level keys required by CreateDataset/GNNDataLoader if needed
     }
     try:
         # Set logging level to DEBUG for detailed output during test
         logging.basicConfig(level=logging.DEBUG)
         logger.info("Initializing GNNDataLoader for testing...")
         data_loader_manager = GNNDataLoader(test_config)

         # Test Train Loader
         if data_loader_manager.train_loader and len(data_loader_manager.train_loader.dataset) > 0:
             logger.info("\n--- Iterating through train_loader (first 2 batches) ---")
             count = 0
             for batch_idx, (batch_s, batch_t, batch_g) in enumerate(data_loader_manager.train_loader):
                 logger.info(f"Train Batch {batch_idx}:")
                 logger.info(f"  States Shape: {batch_s.shape}, Type: {batch_s.dtype}") # Expect B,N,C,H,W float32
                 logger.info(f"  Traj Shape:   {batch_t.shape}, Type: {batch_t.dtype}") # Expect B,N int64 (Long)
                 logger.info(f"  GSO Shape:    {batch_g.shape}, Type: {batch_g.dtype}") # Expect B,N,N float32
                 # Check action values are valid (0-4)
                 if torch.any(batch_t < 0) or torch.any(batch_t >= 5): # Assuming 5 actions
                      logger.error(f"  INVALID ACTION VALUE found in batch {batch_idx}! Min: {batch_t.min()}, Max: {batch_t.max()}")
                 count += 1
                 if count >= 2: break # Limit test output
         else:
              logger.warning("\nTrain loader is empty or was not created.")

         # Test Validation Loader
         if data_loader_manager.valid_loader and len(data_loader_manager.valid_loader.dataset) > 0:
             logger.info("\n--- Iterating through valid_loader (first 2 batches) ---")
             count = 0
             for batch_idx, (batch_s, batch_t, batch_g) in enumerate(data_loader_manager.valid_loader):
                 logger.info(f"Valid Batch {batch_idx}:")
                 logger.info(f"  States Shape: {batch_s.shape}, Type: {batch_s.dtype}")
                 logger.info(f"  Traj Shape:   {batch_t.shape}, Type: {batch_t.dtype}")
                 logger.info(f"  GSO Shape:    {batch_g.shape}, Type: {batch_g.dtype}")
                 if torch.any(batch_t < 0) or torch.any(batch_t >= 5):
                      logger.error(f"  INVALID ACTION VALUE found in validation batch {batch_idx}! Min: {batch_t.min()}, Max: {batch_t.max()}")
                 count += 1
                 if count >= 2: break
         else:
              logger.warning("\nValidation loader is empty or was not created.")

     except Exception as e:
         logger.error(f"\nERROR during data loader test: {e}", exc_info=True)
     finally:
        print("\n--- DataLoader Test Finished ---")
