# File: data_loader.py
# (Simplified for MatDataset only)

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset # Ensure Dataset is imported
from pathlib import Path
import logging
import traceback

# --- Setup Logging FIRST ---
logger = logging.getLogger(__name__)
# Configure logger if not already configured by a higher-level script
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- End Logging Setup ---

# --- Import only MatDataset ---
try:
    from mat_dataset import MatDataset # Import the new one for .mat files
except ImportError:
    logger.error("FATAL: Could not import MatDataset from mat_dataset.py. Ensure the file exists.")
    # exit(1) # Exit if MatDataset is essential
    # Or define a dummy class to allow the script to proceed further for debugging other parts:
    class MatDataset(Dataset):
        def __init__(self, *args, **kwargs): self.count = 0; logger.error("Using Dummy MatDataset!")
        def __len__(self): return 0
        def __getitem__(self, index): raise IndexError


# --- MatDataLoader (using MatDataset for .mat files) ---
class MatDataLoader:
    """Manages DataLoaders using MatDataset (.mat files)."""
    def __init__(self, config: dict):
        if MatDataset is None: # Should have been caught by import try/except
            raise RuntimeError("MatDataset class not available.")
        self.config = config
        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None

        # --- Validate Essential Config Keys ---
        if 'batch_size' not in self.config: raise ValueError("Missing 'batch_size'.")
        batch_size = int(self.config['batch_size'])
        num_workers = int(self.config.get('num_workers', 0))
        # Expect 'train' and 'valid' sections with 'root_dir' pointing to folders with .mat files
        if 'train' not in self.config or not isinstance(self.config['train'], dict) or 'root_dir' not in self.config['train']:
             raise ValueError("Config needs 'train' section with 'root_dir' for .mat files.")

        # --- Initialize Training Loader (.mat) ---
        logger.info("\n--- Initializing MAT Training DataLoader (MatDataset) ---")
        train_root_dir = self.config['train']['root_dir']
        try:
            # Pass the main config down, MatDataset extracts necessary info
            train_set_mat = MatDataset(train_root_dir, self.config)
        except Exception as e: logger.error(f"Failed create MatDataset(train) from '{train_root_dir}': {e}", exc_info=True); raise

        if len(train_set_mat) == 0:
             logger.error(f"MatDataset('train') from '{train_root_dir}' is empty. Check path and .mat files.")
             raise RuntimeError("MAT training dataset is empty.")

        try:
            # Consider prefetch_factor for potential speedup with num_workers > 0
            prefetch = 2 if num_workers > 0 else None
            self.train_loader = DataLoader(
                train_set_mat, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0),
                prefetch_factor=prefetch
            )
            logger.info(f"Initialized MAT Training DataLoader: {len(train_set_mat)} samples (timesteps).")
        except Exception as e: logger.error(f"Failed create MAT training DataLoader: {e}", exc_info=True); raise

        # --- Optional: Initialize Validation Loader (.mat) ---
        if 'valid' in self.config and isinstance(self.config.get('valid'), dict) and 'root_dir' in self.config['valid']:
            logger.info("\n--- Initializing MAT Validation DataLoader (MatDataset) ---")
            valid_root_dir = self.config['valid']['root_dir']
            try:
                valid_set_mat = MatDataset(valid_root_dir, self.config)
                if len(valid_set_mat) > 0:
                    prefetch = 2 if num_workers > 0 else None
                    self.valid_loader = DataLoader(
                        valid_set_mat, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0),
                        prefetch_factor=prefetch
                    )
                    logger.info(f"Initialized MAT Validation DataLoader: {len(valid_set_mat)} samples (timesteps).")
                else: logger.warning(f"MAT validation dataset from '{valid_root_dir}' empty."); self.valid_loader = None
            except Exception as e: logger.error(f"Failed create MatDataset(valid) from '{valid_root_dir}': {e}", exc_info=True); self.valid_loader = None
        else:
             logger.info("\nMAT validation data directory not configured in 'valid.root_dir'.")
        logger.info("--- MAT DataLoader Init Complete ---")

# --- Original GNNDataLoader is REMOVED as it's not needed ---
# class GNNDataLoader: ...