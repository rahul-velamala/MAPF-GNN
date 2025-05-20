# File: export_metrics_to_text.py
# Reads training_metrics.xlsx from multiple model result directories
# and exports their contents into a single formatted text file.

import pandas as pd
from pathlib import Path
import argparse
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---

def export_metrics_to_text(result_dir_paths: list[str], output_text_path: str):
    """
    Reads training_metrics.xlsx from specified directories and writes them
    as formatted text blocks into a single output text file.

    Args:
        result_dir_paths (list[str]): A list of paths to the result directories
                                      (e.g., ['results/gcn_k1_...', 'results/adc_main_...']).
        output_text_path (str): Path for the combined output text file.
    """
    logger.info(f"Attempting to export metrics from {len(result_dir_paths)} directories to {output_text_path}")

    output_path = Path(output_text_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Open the output text file in write mode (this will overwrite existing content)
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for dir_path_str in result_dir_paths:
                result_dir = Path(dir_path_str)
                metrics_file = result_dir / "training_metrics.xlsx"
                exp_name = result_dir.name # Use directory name as identifier

                if not metrics_file.is_file():
                    logger.warning(f"Metrics file not found, skipping: {metrics_file}")
                    outfile.write(f"=== SKIPPED: {exp_name} (File Not Found: {metrics_file}) ===\n\n")
                    continue

                try:
                    logger.info(f"Processing: {metrics_file}")
                    # Read the excel file using openpyxl engine
                    df = pd.read_excel(metrics_file, engine='openpyxl')

                    # Write a header for this model's data
                    outfile.write(f"=== Metrics for Model: {exp_name} ===\n")
                    outfile.write(f"Source File: {metrics_file}\n") # Show relative path
                    outfile.write(f"{'=' * (25 + len(exp_name))}\n") # Separator line matching header length

                    # Write DataFrame content as a formatted string table
                    # index=False prevents writing the DataFrame index (0, 1, 2...)
                    # max_rows=None ensures all rows are written (important for many epochs)
                    df_string = df.to_string(index=False, max_rows=None)
                    outfile.write(df_string)
                    outfile.write("\n\n\n") # Add extra blank lines for better separation

                except ImportError:
                    error_msg = "Module 'openpyxl' not found. Cannot read .xlsx. Please install it: pip install openpyxl"
                    logger.error(error_msg)
                    outfile.write(f"=== ERROR: {exp_name} ({error_msg}) ===\n\n")
                    # Stop processing further files if library is missing
                    return
                except FileNotFoundError:
                    logger.warning(f"File not found error during read (should have been caught earlier), skipping: {metrics_file}")
                    outfile.write(f"=== SKIPPED: {exp_name} (File Not Found during read) ===\n\n")
                except Exception as e:
                    logger.error(f"Error processing {metrics_file}: {e}", exc_info=True)
                    outfile.write(f"=== ERROR Processing: {exp_name} ({e}) ===\n\n")

        logger.info(f"Successfully exported metrics to: {output_path.resolve()}")

    except IOError as e:
        logger.error(f"Error opening or writing to output file {output_path}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training_metrics.xlsx from multiple directories to a single text file.")
    parser.add_argument(
        "-o", "--output",
        default="results/exported_training_metrics.txt", # Default output file name
        help="Path for the output combined text file (default: results/exported_training_metrics.txt)"
    )
    parser.add_argument(
        "result_dirs",
        nargs='+', # Expect one or more directory paths
        help="Paths to the result directories containing training_metrics.xlsx (e.g., results/gcn_k1... results/adc_main...)"
    )

    args = parser.parse_args()

    export_metrics_to_text(args.result_dirs, args.output)