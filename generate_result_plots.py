# File: generate_result_plots.py
# (Includes legend placement modifications)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # For enhanced aesthetics
import argparse
from pathlib import Path
import numpy as np
import re # For parsing test set names
from collections import defaultdict

# --- Plotting Style Configuration ---
sns.set_theme(style="whitegrid", palette="muted") # Base style
plt.rcParams['figure.figsize'] = (12, 7) # Adjusted for potentially better legend room
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 15 # Slightly larger
plt.rcParams['axes.labelsize'] = 13 # Slightly larger
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 17 # Slightly larger
# --- ----------------------------- ---

def plot_metric_comparison(df, metric_column, y_label, title, output_path, lower_is_better=False, y_limit=None):
    """
    Generates a grouped bar chart comparing models for a given metric across test sets.
    """
    if df.empty or metric_column not in df.columns:
        print(f"Data for metric '{metric_column}' is missing or DataFrame is empty. Skipping plot: {title}")
        return

    plt.figure()
    x_axis_col = 'Condition_Label' if 'Condition_Label' in df.columns else 'Test Set'
    try:
        ax = sns.barplot(x=x_axis_col, y=metric_column, hue='Model_Short_Name', data=df, errorbar='sd')
    except Exception as e:
        print(f"Error creating barplot for '{title}' with metric '{metric_column}': {e}. Skipping.")
        plt.close()
        return

    ax.set_ylabel(y_label)
    ax.set_xlabel("Test Condition / Obstacle Density")
    ax.set_title(title)
    plt.xticks(rotation=30, ha='right')

    # --- MODIFIED LEGEND PLACEMENT for bar plot ---
    # Try to place inside, e.g., 'upper right' or 'best'.
    # Adjust 'ncol' if many models to make legend wider rather than taller.
    num_hues = len(df['Model_Short_Name'].unique())
    if num_hues <= 4:
        ax.legend(title='Model', loc='best', fontsize='small')
    else: # For more models, placing outside might still be cleaner, or try multi-column inside
        ax.legend(title='Model', loc='upper right', fontsize='x-small', ncol=1, bbox_to_anchor=(1,1))
        # If still bad, revert to: ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    # --- END MODIFIED LEGEND ---

    if y_limit:
        ax.set_ylim(y_limit)
    elif metric_column == 'Success Rate':
        max_val = df[metric_column].max() if not df[metric_column].dropna().empty else 1.0
        ax.set_ylim(0, max(1.05, max_val * 1.1))

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(format(height, '.2f'),
                           (p.get_x() + p.get_width() / 2., height),
                           ha = 'center', va = 'center',
                           xytext = (0, 9),
                           textcoords = 'offset points',
                           fontsize=8)

    # Adjust tight_layout based on legend placement
    if num_hues > 4 and ax.get_legend() and ax.get_legend().get_bbox_to_anchor().x0 > 1: # Check if legend is outside
        plt.tight_layout(rect=[0, 0, 0.80, 1]) # Make more room if legend is outside
    else:
        plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_ablation_study(df, x_column, y_column, hue_column, x_label, y_label, title, output_path, y_limit=None):
    """
    Generates a line plot for ablation studies.
    """
    if df.empty or not all(col in df.columns for col in [x_column, y_column, hue_column]):
        print(f"Missing columns for ablation plot '{title}'. Required: {x_column}, {y_column}, {hue_column}")
        return
    plt.figure()
    ax = sns.lineplot(x=x_column, y=y_column, hue=hue_column, data=df, marker='o', errorbar='sd')
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
    if y_limit: ax.set_ylim(y_limit)
    elif y_column == 'Success Rate':
        max_val = df[y_column].max() if not df[y_column].dropna().empty else 1.0
        ax.set_ylim(0, max(1.05, max_val * 1.1))

    # --- MODIFIED LEGEND PLACEMENT for line plot ---
    ax.legend(title=hue_column.replace('_', ' ').title(), loc='best', fontsize='small')
    # --- END MODIFIED LEGEND ---

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300); print(f"Ablation plot saved to {output_path}"); plt.close()

def parse_test_set_to_condition_label(test_set_name_from_csv_column):
    """ Parses 'Test Set' column into a readable label for x-axis. """
    match_density = re.search(r"o(\d+)", str(test_set_name_from_csv_column))
    if match_density:
        obstacles = int(match_density.group(1))
        map_size_match = re.search(r"map(\d+)x(\d+)", str(test_set_name_from_csv_column))
        if map_size_match:
            rows, cols = int(map_size_match.group(1)), int(map_size_match.group(2))
            density_percent = (obstacles / (rows * cols)) * 100 if (rows*cols) > 0 else obstacles
            return f"{obstacles} Obs ({density_percent:.0f}\%)"
        return f"{obstacles} Obs"
    return str(test_set_name_from_csv_column)

def simplify_model_name(model_name_orig):
    """Creates shorter, more readable names for models for legends."""
    model_name = str(model_name_orig)
    if "adc_main" in model_name: return "ADC (Learn $t$, K=10)"
    if "adc_fixedt" in model_name: return "ADC (Fixed $t=1$, K=10)"
    if "adc_k1" in model_name: return "ADC (Learn $t$, K=1)"
    m = re.match(r"gcn_k(\d+)", model_name)
    if m: return f"GCN (K={m.group(1)})"
    name = model_name
    name = re.sub(r"_map\d+x\d+", "", name)
    name = re.sub(r"_r\d+", "", name)
    name = re.sub(r"_o\d+", "", name)
    name = re.sub(r"_p\d+", "", name)
    name = name.replace("_", " ").replace("10x10", "").strip().title()
    return name if name else model_name_orig

def plot_combined_training_curves(
    training_data_dict: dict,
    metric_column: str,
    y_label: str,
    plot_title: str,
    output_path: Path,
    y_limit=None,
    smoothing_window: int = 0
):
    plt.figure(figsize=(10, 6)) # Adjusted figure size for potentially internal legend
    plotted_something = False
    num_models = len(training_data_dict)
    current_palette = sns.color_palette("husl", num_models) if num_models > 6 else sns.color_palette("muted", num_models)
    line_styles = ['-', '--', '-.', ':'] * ( (num_models // 4) + 1)

    for i, (model_name_orig, df_train) in enumerate(training_data_dict.items()):
        model_short_name_legend = simplify_model_name(model_name_orig)
        if df_train.empty or 'Epoch' not in df_train.columns or metric_column not in df_train.columns:
            print(f"Skipping {model_short_name_legend} for '{plot_title}': missing data for '{metric_column}'.")
            continue
        df_metric = df_train.dropna(subset=[metric_column]).copy()
        if df_metric.empty:
            print(f"Skipping {model_short_name_legend} for '{plot_title}': no valid data for '{metric_column}' after dropna.")
            continue
        
        epochs = df_metric['Epoch']
        metric_values = df_metric[metric_column].copy()
        if metric_column == 'Evaluation Episode Success Rate' and metric_values.max() > 1.1:
                 metric_values = metric_values / 100.0

        color = current_palette[i % len(current_palette)]
        linestyle = line_styles[i % len(line_styles)]

        if smoothing_window > 0 and len(metric_values) >= smoothing_window:
            metric_values_smoothed = metric_values.rolling(window=smoothing_window, center=True, min_periods=1).mean()
            plt.plot(epochs, metric_values_smoothed, linestyle=linestyle, label=f"{model_short_name_legend}", color=color)
        else:
            plt.plot(epochs, metric_values, marker='.', linestyle=linestyle, label=model_short_name_legend, color=color)
        plotted_something = True

    if not plotted_something:
        print(f"No data to plot for any model for '{plot_title}'. Skipping save."); plt.close(); return

    plt.xlabel("Epoch"); plt.ylabel(y_label); plt.title(plot_title)
    if y_limit: plt.ylim(y_limit)
    elif metric_column == 'Evaluation Episode Success Rate': plt.ylim(-0.05, 1.05)
    
    # --- MODIFIED LEGEND PLACEMENT for training curves ---
    # Try 'best' for fewer lines, more controlled for many.
    if num_models <= 4:
        plt.legend(title='Model', loc='best', fontsize='small')
        plt.tight_layout()
    elif num_models <= 7: # For up to 7 models, try to fit inside with multiple columns
        plt.legend(title='Model', loc='best', fontsize='x-small', ncol=2) # ncol=2 might help
        plt.tight_layout()
    else: # Revert to outside for many models as it's usually clearer
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small')
        plt.tight_layout(rect=[0, 0, 0.78, 1]) # Make room if legend is outside
    # --- END MODIFIED LEGEND ---

    plt.grid(True, linestyle=':')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined training plot: {output_path}")
    plt.close()

    sns.set_palette("muted") # Reset palette for other plots

def main():
    parser = argparse.ArgumentParser(description="Generate plots from model evaluation and training CSV/Excel files.")
    parser.add_argument("--per_testset_csv_file", type=Path,
                        help="Path to 'evaluation_metrics_per_testset.csv' (for performance plots).")
    parser.add_argument("--overall_csv_file", type=Path, default=None,
                        help="Optional: Path to 'evaluation_metrics_overall.csv' (for computational plots).")
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper_plots_final_v3"), # New default
                        help="Base directory to save generated plots.")
    parser.add_argument("--plot_title_suffix", type=str, default="",
                        help="Suffix for performance plot titles (e.g., '(Trained on 10% Obstacles)').")
    parser.add_argument("--ablation_csv", type=Path, default=None,
                        help="Optional: Path to a CSV for ablation study plots.")
    parser.add_argument("--training_result_dirs", type=Path, nargs='*', default=[],
                        help="List of ALL model training result directories (e.g., results/exp_name1 results/exp_name2).")
    parser.add_argument("--training_condition_filter", type=str, default=None,
                        help="Substring to filter training_result_dirs by (e.g., '_o10_'). Plots only for this condition.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Performance Plots ---
    if args.per_testset_csv_file and args.per_testset_csv_file.is_file():
        print(f"\n--- Generating Performance Plots from: {args.per_testset_csv_file} ---")
        df_per_testset = pd.read_csv(args.per_testset_csv_file)
        if not df_per_testset.empty:
            df_per_testset['Condition_Label'] = df_per_testset['Test Set'].apply(parse_test_set_to_condition_label)
            df_per_testset['Model_Short_Name'] = df_per_testset['Model'].apply(simplify_model_name)
            if 'Success Rate' in df_per_testset.columns and df_per_testset['Success Rate'].max() > 1.1:
                df_per_testset['Success Rate'] = df_per_testset['Success Rate'] / 100.0
            try:
                df_per_testset['Obstacle_Sort_Key'] = df_per_testset['Condition_Label'].apply(
                    lambda x: int(re.search(r'(\d+)\% Obstacles', x).group(1)) if isinstance(x, str) and re.search(r'(\d+)\% Obstacles', x) else 999)
                df_per_testset.sort_values(by=['Obstacle_Sort_Key', 'Model_Short_Name'], inplace=True)
            except Exception as e: print(f"Could not sort performance data by obstacle density: {e}")
            plot_output_subdir_name = args.per_testset_csv_file.stem.replace("evaluation_metrics_per_testset","perf_plots")
            plot_output_subdir = args.output_dir / plot_output_subdir_name; plot_output_subdir.mkdir(parents=True, exist_ok=True)
            plot_metric_comparison(df_per_testset, 'Success Rate', 'Success Rate (SR)', f'SR vs. Condition {args.plot_title_suffix}', plot_output_subdir / "sr_vs_condition.png")
            plot_metric_comparison(df_per_testset, 'Average Makespan (Successful)', 'Average Makespan', f'Avg. Makespan vs. Condition {args.plot_title_suffix}', plot_output_subdir / "am_vs_condition.png", lower_is_better=True)
            plot_metric_comparison(df_per_testset, 'Flowtime (FT)', 'Flowtime (Total Steps)', f'Flowtime vs. Condition {args.plot_title_suffix}', plot_output_subdir / "ft_vs_condition.png", lower_is_better=True)
            print(f"Performance plots saved to {plot_output_subdir}")
        else: print(f"Performance CSV file {args.per_testset_csv_file} is empty.")
    elif args.per_testset_csv_file: print(f"Warning: Performance CSV file not found at {args.per_testset_csv_file}")

    # --- Ablation Plots ---
    if args.ablation_csv and args.ablation_csv.is_file():
        print(f"\n--- Generating Ablation Plots from: {args.ablation_csv} ---")
        # ... (Your custom ablation plotting logic) ...
        print(f"Ablation CSV provided. Customize plot_ablation_study calls if needed.")
    elif args.ablation_csv: print(f"Warning: Ablation CSV file not found at {args.ablation_csv}")

    # --- Computational Performance Plots ---
    overall_csv_file_to_use = args.overall_csv_file
    if not overall_csv_file_to_use and args.per_testset_csv_file and args.per_testset_csv_file.is_file():
        overall_csv_file_to_use = args.per_testset_csv_file.parent / "evaluation_metrics_overall.csv"
    if overall_csv_file_to_use and overall_csv_file_to_use.is_file():
        print(f"\n--- Generating Computational Plots from: {overall_csv_file_to_use} ---")
        df_overall = pd.read_csv(overall_csv_file_to_use)
        if not df_overall.empty:
            df_overall['Model_Short_Name'] = df_overall['Model'].apply(simplify_model_name)
            comp_plot_output_subdir_name = overall_csv_file_to_use.stem.replace("evaluation_metrics_overall","comp_perf_plots")
            comp_plot_output_subdir = args.output_dir / comp_plot_output_subdir_name; comp_plot_output_subdir.mkdir(parents=True, exist_ok=True)
            if 'Avg Inference Time (ms/step)' in df_overall.columns:
                plt.figure(figsize=(max(8, len(df_overall['Model_Short_Name'].unique()) * 1.2), 6))
                ax = sns.barplot(x='Model_Short_Name', y='Avg Inference Time (ms/step)', data=df_overall, palette="viridis")
                ax.set_ylabel('Avg Inference Time (ms/step)'); ax.set_xlabel('Model'); ax.set_title(f'Inference Time {args.plot_title_suffix}')
                plt.xticks(rotation=30, ha='right')
                for p in ax.patches: ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0,9), textcoords='offset points', fontsize=9)
                plt.tight_layout(); plt.savefig(comp_plot_output_subdir / "inference_time_overall.png", dpi=300)
                print(f"Plot saved to {comp_plot_output_subdir / 'inference_time_overall.png'}"); plt.close()
        else: print(f"Overall CSV file {overall_csv_file_to_use} is empty.")
    elif args.overall_csv_file: print(f"Warning: Overall CSV for computational plots not found: {overall_csv_file_to_use}")

    # --- Generate Combined Training Curves ---
    if args.training_result_dirs:
        print("\n--- Generating Combined Training Curves ---")
        training_plots_base_output_dir = args.output_dir / "training_curves_focused"
        training_plots_base_output_dir.mkdir(parents=True, exist_ok=True)
        all_training_dfs = {}
        for model_training_dir in args.training_result_dirs:
            if not model_training_dir.is_dir(): print(f"W: Skip train dir: {model_training_dir}"); continue
            model_name = model_training_dir.name
            metrics_file = model_training_dir / "training_metrics.xlsx"
            if not metrics_file.is_file(): metrics_file = model_training_dir / "training_metrics.csv"
            if metrics_file.is_file():
                try:
                    df = pd.read_excel(metrics_file) if metrics_file.suffix == '.xlsx' else pd.read_csv(metrics_file)
                    if not df.empty: all_training_dfs[model_name] = df
                except Exception as e: print(f"E: Load metrics for {model_name}: {e}")
            else: print(f"W: No metrics file for {model_name}")

        filtered_dfs_for_training_plot = all_training_dfs.copy() # Start with all loaded DFs
        if args.training_condition_filter:
            filtered_dfs_for_training_plot = {name: df for name, df in all_training_dfs.items() if args.training_condition_filter in name}
            if not filtered_dfs_for_training_plot: print(f"W: No training dirs matched filter '{args.training_condition_filter}'.")
        
        if filtered_dfs_for_training_plot:
            adc_variants_data = {name: df for name, df in filtered_dfs_for_training_plot.items() if "adc_" in name.lower()}
            gcn_variants_data = {name: df for name, df in filtered_dfs_for_training_plot.items() if "gcn_" in name.lower()}
            condition_tag = args.training_condition_filter.replace("_", "") if args.training_condition_filter else "all_loaded"
            output_subdir_for_condition = training_plots_base_output_dir / f"condition_{condition_tag}"
            title_prefix = f"Trained on ~{condition_tag}: " if args.training_condition_filter else "All Loaded Models: "
            smoothing_win = 5
            if adc_variants_data:
                plot_combined_training_curves(adc_variants_data, 'Average Training Loss', 'Avg. Training Loss', f"{title_prefix}ADC Variants Loss", output_subdir_for_condition / "adc_variants_train_loss.png", smoothing_window=smoothing_win)
                plot_combined_training_curves(adc_variants_data, 'Evaluation Episode Success Rate', 'Eval SR', f"{title_prefix}ADC Variants Eval SR", output_subdir_for_condition / "adc_variants_eval_sr.png", y_limit=(-0.05,1.05), smoothing_window=smoothing_win)
                plot_combined_training_curves(adc_variants_data, 'Evaluation Avg Steps (Success)', 'Eval Avg. Makespan', f"{title_prefix}ADC Variants Eval Makespan", output_subdir_for_condition / "adc_variants_eval_am.png", smoothing_window=smoothing_win)
            if gcn_variants_data:
                plot_combined_training_curves(gcn_variants_data, 'Average Training Loss', 'Avg. Training Loss', f"{title_prefix}GCN K-Variants Loss", output_subdir_for_condition / "gcn_variants_train_loss.png", smoothing_window=smoothing_win)
                plot_combined_training_curves(gcn_variants_data, 'Evaluation Episode Success Rate', 'Eval SR', f"{title_prefix}GCN K-Variants Eval SR", output_subdir_for_condition / "gcn_variants_eval_sr.png", y_limit=(-0.05,1.05), smoothing_window=smoothing_win)
                plot_combined_training_curves(gcn_variants_data, 'Evaluation Avg Steps (Success)', 'Eval Avg. Makespan', f"{title_prefix}GCN K-Variants Eval Makespan", output_subdir_for_condition / "gcn_variants_eval_am.png", smoothing_window=smoothing_win)
        else: print("No training data loaded after filtering for training curve plots.")
    else: print("\nNo training result directories provided. Skipping combined training curve plots.")
    print(f"\nAll plot generation tasks complete. Check output in {args.output_dir}")

if __name__ == "__main__":
    main()