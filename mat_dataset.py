# File: mat_dataset.py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
import scipy.io # To load .mat files

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MatDataset(Dataset):
    """
    Loads MAPF training data from .mat files provided by Li et al.

    Each .mat file contains sequences for one episode. This Dataset
    extracts all timesteps from all files into flat arrays.
    Expected keys in .mat: 'inputTensor', 'target', 'GSO'.
    """
    def __init__(self, root_dir_str: str, config: dict):
        """
        Args:
            root_dir_str (str): Path to the directory containing .mat files (e.g., '.../train').
            config (dict): Main configuration dictionary (used for consistency checks).
        """
        self.root_dir = Path(root_dir_str)
        self.config = config
        logger.info(f"Initializing MatDataset from: {self.root_dir}")

        if not self.root_dir.is_dir():
            logger.error(f"Dataset directory not found: {self.root_dir}")
            self._initialize_empty()
            return

        # --- Get Expected Config Params ---
        try:
            self.num_agents = int(config["num_agents"])
            self.pad = int(config["pad"])
            self.fov_h = self.fov_w = (self.pad * 2) - 1
            self.fov_c = 3 # Assuming 3 channels based on previous structure
            self.num_actions = 5
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Config missing required keys (num_agents, pad) or invalid value: {e}")
            self._initialize_empty()
            return

        # --- Find and Load .mat Files ---
        try:
            mat_files = sorted(list(self.root_dir.glob("*.mat")))
        except Exception as e:
            logger.error(f"Error finding .mat files in {self.root_dir}: {e}")
            self._initialize_empty()
            return

        if not mat_files:
            logger.error(f"No .mat files found in {self.root_dir}")
            self._initialize_empty()
            return

        logger.info(f"Found {len(mat_files)} .mat files.")

        all_fovs = []
        all_targets = []
        all_gsos = []
        load_errors = 0
        shape_mismatches = 0
        skipped_empty = 0

        pbar = tqdm(mat_files, desc=f"Loading .mat data ({self.root_dir.name})", unit="file")
        for mat_file_path in pbar:
            try:
                mat_data = scipy.io.loadmat(mat_file_path)

                # Extract relevant data - Need to match keys from inspect_mat.py
                # inputTensor: (TimeSteps, N, C, H, W) -> FOV
                # target: (TimeSteps, N, Actions) -> One-hot actions? Need to convert to indices.
                # GSO: (TimeSteps, N, N) -> Adjacency
                fov_seq = mat_data.get('inputTensor')
                target_seq_onehot = mat_data.get('target')
                gso_seq = mat_data.get('GSO')

                if fov_seq is None or target_seq_onehot is None or gso_seq is None:
                    logger.warning(f"Skipping {mat_file_path.name}: Missing one or more required keys.")
                    load_errors += 1
                    continue

                # --- Data Validation ---
                if fov_seq.ndim != 5 or target_seq_onehot.ndim != 3 or gso_seq.ndim != 3:
                     logger.warning(f"Skipping {mat_file_path.name}: Unexpected dimensions.")
                     shape_mismatches += 1; continue

                T_steps, N_fov, C_fov, H_fov, W_fov = fov_seq.shape
                T_tgt, N_tgt, A_tgt = target_seq_onehot.shape
                T_gso, N_gso1, N_gso2 = gso_seq.shape

                # Check consistency across arrays within the file
                if not (T_steps == T_tgt == T_gso and \
                        N_fov == N_tgt == N_gso1 == N_gso2):
                    logger.warning(f"Skipping {mat_file_path.name}: Inconsistent T/N dims within file.")
                    shape_mismatches += 1; continue

                # Check consistency with config
                if N_fov != self.num_agents:
                    logger.warning(f"Skipping {mat_file_path.name}: Agent mismatch file ({N_fov}) vs config ({self.num_agents}).")
                    shape_mismatches += 1; continue
                if C_fov != self.fov_c or H_fov != self.fov_h or W_fov != self.fov_w:
                    logger.warning(f"Skipping {mat_file_path.name}: FOV shape mismatch file ({C_fov},{H_fov},{W_fov}) vs config ({self.fov_c},{self.fov_h},{self.fov_w}).")
                    shape_mismatches += 1; continue
                if A_tgt != self.num_actions:
                    logger.warning(f"Skipping {mat_file_path.name}: Target action dim mismatch file ({A_tgt}) vs expected ({self.num_actions}).")
                    shape_mismatches += 1; continue

                if T_steps == 0: # Skip files with no timesteps
                    skipped_empty += 1; continue

                # --- Process Target Actions ---
                # Assuming 'target' is one-hot (T, N, A). Convert to action indices (T, N).
                target_seq_indices = np.argmax(target_seq_onehot, axis=2).astype(np.int64) # Shape (T, N)

                # Add data from this file (timestep by timestep)
                # .mat files seem to store (Time, N, ...). We need to flatten the time dimension.
                all_fovs.append(fov_seq.astype(np.float32)) # Append full sequence
                all_targets.append(target_seq_indices) # Append full sequence
                all_gsos.append(gso_seq.astype(np.float32)) # Append full sequence

            except Exception as e:
                logger.error(f"Error loading or processing {mat_file_path.name}: {e}", exc_info=True)
                load_errors += 1

        pbar.close()
        logger.info(f"Finished loading files. Errors: {load_errors}, Shape Mismatches: {shape_mismatches}, Empty: {skipped_empty}")

        # --- Concatenate all sequences ---
        if not all_fovs: # Check if any valid data was loaded
            logger.error("No valid data could be loaded from .mat files.")
            self._initialize_empty()
            return

        try:
            logger.info("Concatenating data...")
            # Concatenate along the time dimension (axis=0)
            self.all_fovs = np.concatenate(all_fovs, axis=0)
            self.all_targets = np.concatenate(all_targets, axis=0)
            self.all_gsos = np.concatenate(all_gsos, axis=0)
            self.count = len(self.all_fovs)
            logger.info(f"Concatenated data shapes: FOV={self.all_fovs.shape}, Target={self.all_targets.shape}, GSO={self.all_gsos.shape}")

            # Final sanity check
            if not (self.count == len(self.all_targets) == len(self.all_gsos)):
                logger.error("FATAL: Mismatch in total timesteps after concatenation!")
                min_len = min(self.count, len(self.all_targets), len(self.all_gsos))
                self.all_fovs = self.all_fovs[:min_len]
                self.all_targets = self.all_targets[:min_len]
                self.all_gsos = self.all_gsos[:min_len]
                self.count = min_len
                logger.warning(f"Dataset truncated to {self.count} samples.")

            if self.count == 0:
                 logger.error("Resulting dataset has 0 samples after concatenation.")
                 self._initialize_empty()

        except ValueError as e: # Catch potential concat errors (shape mismatch between files)
            logger.error(f"Error during concatenation: {e}. Check if all .mat files have consistent N, C, H, W dimensions.")
            self._initialize_empty()
        except Exception as e:
            logger.error(f"Unexpected error during concatenation: {e}", exc_info=True)
            self._initialize_empty()


    def _initialize_empty(self):
        """Helper to set empty arrays if loading fails."""
        logger.warning("Initializing dataset with empty arrays.")
        N = self.config.get("num_agents", 0)
        C = 3
        H = W = (self.config.get("pad", 1) * 2) - 1 # Best guess if pad available
        self.all_fovs = np.empty((0, N, C, H, W), dtype=np.float32)
        self.all_targets = np.empty((0, N), dtype=np.int64)
        self.all_gsos = np.empty((0, N, N), dtype=np.float32)
        self.count = 0

    def __len__(self) -> int:
        """Returns the total number of samples (timesteps) in the dataset."""
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single sample (data for one timestep).
        Output order: FOV, Action_Index, GSO
        """
        if not 0 <= index < self.count:
             raise IndexError(f"Index {index} out of bounds for dataset with size {self.count}")

        fov_np = self.all_fovs[index]         # (N, C, H, W) float32
        target_np = self.all_targets[index]   # (N,) int64
        gso_np = self.all_gsos[index]         # (N, N) float32

        # Convert to PyTorch tensors
        fov_sample = torch.from_numpy(fov_np) # Already float32
        target_sample = torch.from_numpy(target_np) # Already int64 -> LongTensor
        gso_sample = torch.from_numpy(gso_np) # Already float32

        # Return order must match the training loop unpacking: state/FOV, action, gso
        return fov_sample, target_sample, gso_sample

# --- Test Block (Example - uncomment and adapt paths/config) ---
# if __name__ == "__main__":
#      print("\n--- Running MatDataset Test ---")
#      # Define a sample config reflecting your actual setup
#      # >>> IMPORTANT: UPDATE PATHS AND PARAMETERS <<<
#      test_config_mat = {
#          "num_agents": 10, # Should match the data in the .mat files
#          "pad": 6,        # Should match the data FOV (e.g., pad=6 -> 11x11)
#          "batch_size": 4, # For DataLoader test
#          "num_workers": 0,
#      }
#      test_data_dir = "/scratch/rahul/v1/project/MAPF-GNN-ADC/dataset/DataSource_DMap_FixedComR/EffectiveDensity/Training/map20x20_density_p1/10_Agent/valid" # <<<--- UPDATE PATH

#      try:
#          logging.basicConfig(level=logging.DEBUG)
#          logger.info("Creating MatDataset...")
#          mat_dataset = MatDataset(test_data_dir, test_config_mat)

#          if len(mat_dataset) > 0:
#              logger.info(f"Dataset loaded successfully with {len(mat_dataset)} samples.")
#              # Test __getitem__
#              logger.info("Testing __getitem__(0):")
#              fov, target, gso = mat_dataset[0]
#              logger.info(f"  FOV shape: {fov.shape}, dtype: {fov.dtype}")
#              logger.info(f"  Target shape: {target.shape}, dtype: {target.dtype}")
#              logger.info(f"  GSO shape: {gso.shape}, dtype: {gso.dtype}")

#              # Test DataLoader
#              logger.info("Testing DataLoader...")
#              from torch.utils.data import DataLoader
#              test_loader = DataLoader(mat_dataset, batch_size=test_config_mat['batch_size'], shuffle=True)
#              for i, (batch_fov, batch_target, batch_gso) in enumerate(test_loader):
#                  logger.info(f" Batch {i}:")
#                  logger.info(f"   FOV shape: {batch_fov.shape}")
#                  logger.info(f"   Target shape: {batch_target.shape}")
#                  logger.info(f"   GSO shape: {batch_gso.shape}")
#                  # Check target values are indices (0-4)
#                  if torch.any(batch_target < 0) or torch.any(batch_target >= 5):
#                      logger.error(f"   INVALID TARGET ACTION INDEX FOUND! Min: {batch_target.min()}, Max: {batch_target.max()}")
#                  if i >= 1: # Show first 2 batches
#                      break
#          else:
#              logger.error("Dataset loading resulted in 0 samples.")

#      except Exception as e:
#          logger.error(f"\nERROR during MatDataset test: {e}", exc_info=True)
#      finally:
#         print("\n--- MatDataset Test Finished ---")