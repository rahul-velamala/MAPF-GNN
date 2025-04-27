# File: cleanup_dataset.py
import argparse
from pathlib import Path
import shutil
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_split(split_path: Path):
    """Removes case directories within a split that are missing required files."""
    if not split_path.is_dir():
        logger.warning(f"Directory not found: {split_path}. Skipping cleanup.")
        return 0, 0

    required_files = ["input.yaml", "solution.yaml", "trajectory.npy", "states.npy", "gso.npy"]
    cases_to_check = [d for d in split_path.glob("case_*") if d.is_dir()]
    
    if not cases_to_check:
        logger.info(f"No case directories found in {split_path}.")
        return 0, 0

    removed_count = 0
    kept_count = 0
    
    logger.info(f"Checking {len(cases_to_check)} cases in {split_path}...")
    for case_dir in tqdm(cases_to_check, desc=f"Cleaning {split_path.name}"):
        is_complete = True
        for fname in required_files:
            if not (case_dir / fname).exists():
                is_complete = False
                logger.debug(f"Removing incomplete case {case_dir.name}: Missing {fname}")
                break # No need to check other files for this case
        
        if not is_complete:
            try:
                shutil.rmtree(case_dir)
                removed_count += 1
            except OSError as e:
                logger.error(f"Error removing directory {case_dir}: {e}")
        else:
            kept_count += 1
    
    logger.info(f"Cleanup finished for {split_path}: Kept {kept_count}, Removed {removed_count}")
    return kept_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up incomplete cases in generated datasets.")
    parser.add_argument("dataset_dir", type=str, help="Path to the root dataset directory (e.g., dataset/5_8_28_fov5_parallel).")
    
    args = parser.parse_args()
    
    root_path = Path(args.dataset_dir)
    if not root_path.is_dir():
        logger.error(f"Root dataset directory not found: {root_path}")
        exit(1)
        
    logger.info(f"Starting cleanup for dataset at: {root_path}")
    
    total_kept = 0
    # Iterate through potential splits (train, val, test)
    for split_name in ["train", "val", "test"]:
        split_path = root_path / split_name
        if split_path.is_dir():
            kept = cleanup_split(split_path)
            total_kept += kept
        else:
            logger.info(f"Split directory {split_path} not found, skipping.")
            
    logger.info(f"\nCleanup Complete. Total complete cases across all splits: {total_kept}")