# File: merge_datasets.py
# Merges case directories from a source dataset structure into a target
# dataset structure, renaming cases from the source to continue numbering.

import shutil
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import re # For potentially more robust number extraction

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---

def find_max_case_index(directory: Path) -> int:
    """Finds the highest numerical index from subdirectories named 'case_XXXXX'."""
    max_idx = -1
    if not directory.is_dir():
        return max_idx # Return -1 if directory doesn't exist

    pattern = re.compile(r"case_(\d+)") # Regex to find digits after case_
    for item in directory.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                try:
                    index = int(match.group(1))
                    max_idx = max(max_idx, index)
                except ValueError:
                    logger.warning(f"Could not parse index from directory name: {item.name}")
    return max_idx

def merge_split(source_split_dir: Path, target_split_dir: Path):
    """Merges cases from source_split_dir into target_split_dir, renaming."""
    if not source_split_dir.is_dir():
        logger.warning(f"Source split directory not found, skipping: {source_split_dir}")
        return 0, 0 # Return 0 moved, 0 skipped

    target_split_dir.mkdir(parents=True, exist_ok=True)

    start_index_offset = find_max_case_index(target_split_dir)
    logger.info(f"Target directory '{target_split_dir.name}' max index: {start_index_offset}. New cases will start from {start_index_offset + 1}.")

    # Find and sort source cases numerically
    source_cases = []
    pattern = re.compile(r"case_(\d+)")
    for item in source_split_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                try:
                    index = int(match.group(1))
                    source_cases.append((index, item)) # Store index for sorting
                except ValueError:
                    logger.warning(f"Could not parse index from source directory name: {item.name}")

    source_cases.sort(key=lambda x: x[0]) # Sort by original index

    if not source_cases:
        logger.info(f"No valid 'case_XXXXX' directories found in source: {source_split_dir}")
        return 0, 0

    moved_count = 0
    skipped_count = 0
    current_offset = 1 # Start numbering from 1 after the max index

    pbar = tqdm(source_cases, desc=f"Merging '{source_split_dir.name}' -> '{target_split_dir.name}'", unit="case")
    for _, source_case_path in pbar:
        new_index = start_index_offset + current_offset
        new_case_name = f"case_{new_index:05d}" # Format with leading zeros
        target_case_path = target_split_dir / new_case_name

        # Basic check to prevent overwriting, though start_index_offset should handle this
        if target_case_path.exists():
             logger.warning(f"Target path {target_case_path} already exists! Skipping {source_case_path}. Check max index calculation.")
             skipped_count += 1
             # Don't increment offset if skipped, try next source case with same target index? Risky.
             # Better to increment offset anyway to avoid infinite loop if max index was wrong.
             current_offset += 1
             continue

        try:
            shutil.move(str(source_case_path), str(target_case_path)) # shutil needs strings
            moved_count += 1
            current_offset += 1 # Increment offset for the next case
        except FileNotFoundError:
            logger.error(f"Source case not found during move (unexpected): {source_case_path}")
            skipped_count += 1
        except Exception as e:
            logger.error(f"Error moving {source_case_path} to {target_case_path}: {e}", exc_info=True)
            skipped_count += 1
            # Decide if offset should still increment on other errors? Let's increment to avoid potential loops.
            current_offset += 1

        pbar.set_postfix({"Moved": moved_count, "Skipped": skipped_count})

    logger.info(f"Finished merging for '{target_split_dir.name}'. Moved: {moved_count}, Skipped: {skipped_count}")
    return moved_count, skipped_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge dataset cases from a source directory into a target directory, renaming cases.")
    parser.add_argument(
        "source_base_dir",
        type=str,
        help="Path to the base directory containing the *additional* dataset cases (e.g., 'dataset1')."
    )
    parser.add_argument(
        "target_base_dir",
        type=str,
        help="Path to the base directory where cases should be *merged into* (e.g., 'dataset')."
    )
    parser.add_argument(
        "config_names",
        nargs='+',
        help="Name(s) of the specific configuration subdirectories to merge (e.g., map10x10_r5_o20_p5)."
    )
    parser.add_argument(
        "--splits",
        nargs='+',
        default=['train', 'val', 'test'],
        help="Names of the split subdirectories to process (default: train val test)."
    )

    args = parser.parse_args()

    source_base = Path(args.source_base_dir)
    target_base = Path(args.target_base_dir)

    if not source_base.is_dir():
        logger.error(f"Source base directory not found: {source_base}")
        exit(1)
    if not target_base.is_dir():
        logger.warning(f"Target base directory not found: {target_base}. It will be created if possible.")
        # No exit needed, the script will try to create subdirs

    total_moved_all = 0
    total_skipped_all = 0

    for config_name in args.config_names:
        logger.info(f"\n===== Processing Configuration: {config_name} =====")
        source_config_dir = source_base / config_name
        target_config_dir = target_base / config_name

        if not source_config_dir.is_dir():
            logger.warning(f"Source configuration directory not found, skipping: {source_config_dir}")
            continue

        # Target config directory will be created by merge_split if needed

        for split_name in args.splits:
            logger.info(f"--- Merging split: {split_name} ---")
            source_split = source_config_dir / split_name
            target_split = target_config_dir / split_name

            moved, skipped = merge_split(source_split, target_split)
            total_moved_all += moved
            total_skipped_all += skipped

    logger.info("\n===== Merge Complete =====")
    logger.info(f"Total cases moved across all specified configs/splits: {total_moved_all}")
    logger.info(f"Total cases skipped due to errors or existing target: {total_skipped_all}")