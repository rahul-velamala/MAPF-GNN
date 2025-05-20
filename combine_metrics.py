# File: combine_metrics.py
# Combines training_metrics.xlsx files from multiple model result directories
# into a single Excel workbook with multiple sheets.

import pandas as pd
from pathlib import Path
import argparse
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---

def combine_metrics_sheets(result_dir_paths: list[str], output_excel_path: str):
    """
    Reads training_metrics.xlsx from specified directories and writes them
    as separate sheets into a single output Excel file.

    Args:
        result_dir_paths (list[str]): A list of paths to the result directories
                                      (e.g., ['results/gcn_k1_...', 'results/adc_main_...']).
        output_excel_path (str): Path for the combined output Excel file.
    """
    all_metrics_dfs = {} # Dictionary to hold {sheet_name: DataFrame}

    logger.info(f"Attempting to combine metrics from {len(result_dir_paths)} directories.")

    for dir_path_str in result_dir_paths:
        result_dir = Path(dir_path_str)
        metrics_file = result_dir / "training_metrics.xlsx"
        # Use the result directory name as the sheet name (usually the exp_name)
        sheet_name = result_dir.name

        # Sanitize sheet name for Excel limitations (max 31 chars, no invalid chars)
        # Replace potentially invalid characters (crude replacement)
        invalid_chars = r'[]:*?/\\'
        for char in invalid_chars:
            sheet_name = sheet_name.replace(char, '_')
        # Truncate if too long
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
            logger.warning(f"Sheet name from '{result_dir.name}' was truncated to '{sheet_name}'.")

        if not metrics_file.is_file():
            logger.warning(f"Metrics file not found, skipping: {metrics_file}")
            continue

        try:
            logger.info(f"Reading metrics from: {metrics_file}")
            df = pd.read_excel(metrics_file, engine='openpyxl') # Specify engine

            # Prevent duplicate sheet names if somehow exp_names clash after sanitizing
            original_sheet_name = sheet_name
            counter = 1
            while sheet_name in all_metrics_dfs:
                 suffix = f"_{counter}"
                 max_len = 31 - len(suffix)
                 sheet_name = original_sheet_name[:max_len] + suffix
                 counter += 1
                 if counter > 10: # Safety break
                      logger.error(f"Could not generate unique sheet name for {original_sheet_name}. Skipping.")
                      break
            if sheet_name not in all_metrics_dfs:
                 all_metrics_dfs[sheet_name] = df

        except ImportError:
             logger.error("Module 'openpyxl' not found. Please install it: pip install openpyxl")
             return # Cannot proceed without the engine
        except FileNotFoundError:
             logger.warning(f"File not found error during read (should have been caught earlier), skipping: {metrics_file}")
        except Exception as e:
            logger.error(f"Error reading {metrics_file}: {e}", exc_info=True)

    if not all_metrics_dfs:
        logger.error("No valid metrics files were successfully read. Cannot create combined file.")
        return

    # Write all collected DataFrames to the output Excel file
    output_path = Path(output_excel_path)
    try:
        logger.info(f"Writing {len(all_metrics_dfs)} sheets to combined metrics file: {output_path}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet, df in all_metrics_dfs.items():
                 # Write DataFrame to a sheet named after the experiment
                 df.to_excel(writer, sheet_name=sheet, index=False)
        logger.info(f"Successfully created combined metrics file: {output_path.resolve()}")
    except ImportError:
        # This case should be caught earlier, but good to have redundancy
         logger.error("Module 'openpyxl' not found. Please install it: pip install openpyxl")
    except Exception as e:
        logger.error(f"Error writing combined Excel file {output_path}: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine training_metrics.xlsx from multiple model result directories into one file.")
    parser.add_argument(
        "-o", "--output",
        default="results/combined_training_metrics.xlsx",
        help="Path for the output combined Excel file (default: results/combined_training_metrics.xlsx)"
    )
    parser.add_argument(
        "result_dirs",
        nargs='+', # Expect one or more directory paths
        help="Paths to the result directories containing training_metrics.xlsx (e.g., results/gcn_k1... results/adc_main...)"
    )

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_file_path = Path(args.output)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    combine_metrics_sheets(args.result_dirs, args.output)