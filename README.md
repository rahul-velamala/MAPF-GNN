```markdown
# Multi-Agent Path Finding with Graph Neural Networks (MAPF-GNN)

This project explores the application of Graph Neural Networks (GNNs), including standard Graph Convolutional Networks (GCNs) and an Adaptive Diffusion Convolution (ADC) variant, for solving Multi-Agent Path Finding (MAPF) problems. The project includes tools for dataset generation, model training, comprehensive evaluation, and visualization.

## Overview

The primary goal is to develop and evaluate learning-based policies for decentralized MAPF. Agents make decisions based on local observations (Field of View - FOV) and communication with nearby agents (Graph Shift Operator - GSO). The GNN architecture allows agents to learn to coordinate and navigate to their goals while avoiding collisions.

## Features

*   **Decentralized MAPF Policies**: Learns policies for individual agents.
*   **GNN Architectures**:
    *   Graph Convolutional Networks (GCN) with configurable number of filter taps (K).
    *   Adaptive Diffusion Convolution (ADC) layers with potentially learnable diffusion time.
    *   Baseline CNN-MLP models for comparison.
*   **Dataset Generation**:
    *   Generates MAPF scenarios (maps, agent start/goals, obstacles).
    *   Solves scenarios using Conflict-Based Search (CBS) to get expert trajectories.
    *   Processes trajectories into FOV, GSO, and action sequences.
*   **Model Training**:
    *   Supervised learning from expert trajectories.
    *   Online Expert (DAgger-like) data aggregation to improve policy robustness (optional).
*   **Comprehensive Evaluation**:
    *   Metrics: Success Rate, Average Makespan, Flowtime, Flowtime Increase, Inference Time, Parameter Count.
    *   Comparison scripts for different models and configurations.
    *   Ability to replicate specific experimental setups (e.g., performance vs. number of agents).
*   **Visualization**:
    *   Generates GIFs and videos of MAPF episodes.
    *   Plots training curves and evaluation metrics.
    *   Manim-based animation for conceptual visualization.
    *   Graphviz diagrams for model architectures.
*   **Data Management**:
    *   Tools to convert external `.mat` datasets to the project's `.npy` format.
    *   Dataset cleanup and merging utilities.

## Directory Structure

```
rahul-velamala-mapf-gnn/
├── README.md
├── cbs/                    # Conflict-Based Search (CBS) implementation
├── configs/                # YAML configuration files for experiments
├── data/                   # Example datasets (e.g., map10x10_r5_o30_p5/train/case_10/)
├── data_generation/        # Scripts for dataset generation
├── grid/                   # MAPF environment implementation (GraphEnv)
├── models/                 # Model architectures and network layers
│   ├── networks/           # Specific GNN, ADC layers, utilities
├── results/                # Primary directory for storing training outputs (models, metrics, plots)
├── results1/               # (Likely older results or specific experiment set)
├── results_afterchange/    # (Likely more recent results, e.g., for ADC/GCN K-value comparisons)
├── trained_models/         # Examples of pre-trained models with their configs
├── requirement.txt         # Python dependencies (Note: appears to be a non-text file)
├── train.py                # Main script for training models
├── evaluate_models.py      # Script for evaluating trained models
├── compare_models.py       # Script for comparing different models
├── generate_all_datasets.py # Script to generate datasets for various configs
├── generate_result_plots.py # Script to plot evaluation results
├── create_gif.py           # Script to create GIF visualizations of episodes
├── create_video.py         # Script to create video visualizations of episodes
├── cleanup_dataset.py      # Utility to clean incomplete dataset cases
├── combine_metrics.py      # Utility to combine training metrics from multiple experiments
├── convert_mat_to_npy.py   # Utility to convert .mat datasets to project format
├── export_metrics_to_text.py # Utility to export training metrics to a text file
├── extract_adc_t.py        # Utility to extract learned diffusion parameter 't' from ADC models
├── generate_diagram.py     # Utility to generate model architecture diagrams using Graphviz
├── inspect_mat.py          # Utility to inspect contents of .mat files
├── manim_visualization.py  # Script for conceptual Manim animations
├── mat_dataset.py          # PyTorch Dataset class for loading .mat data
├── merge_datasets.py       # Utility to merge dataset cases
├── replicate_figure5.py    # Script to replicate a specific experimental plot
└── ...                     # Other utility scripts (check_cases.py, read_mat.py, etc.)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rahul-velamala-mapf-gnn
    ```

2.  **Create a Python environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirement.txt` file in this repository appears to be a non-text file. You will need to manually install the required Python libraries. Based on the project's functionality, common dependencies include:
    *   **Core ML/Numerics:** `torch` (PyTorch), `numpy`, `scipy`
    *   **Configuration & Data Handling:** `pyyaml`, `pandas`, `openpyxl` (for `.xlsx`)
    *   **Visualization & Plotting:** `matplotlib`, `seaborn`, `imageio` (and `imageio[ffmpeg]` for video), `graphviz` (Python library)
    *   **Environment & Utilities:** `gymnasium`, `tqdm`
    *   **Conceptual Animation:** `manim` (if using `manim_visualization.py`)

    Example installation using pip:
    ```bash
    pip install torch numpy scipy pyyaml pandas openpyxl matplotlib seaborn tqdm imageio graphviz gymnasium
    pip install imageio[ffmpeg]
    ```
    **Important for Graphviz:**
    In addition to the Python `graphviz` library, you need to install Graphviz at the system level for `generate_diagram.py` to work.
    *   Linux: `sudo apt-get install graphviz` or `sudo dnf install graphviz`
    *   macOS: `brew install graphviz`
    *   Windows: Download from [graphviz.org](https://graphviz.org/download/) and add its `bin` directory to your system's PATH.

    **For Manim:** Follow the official Manim installation guide, as it can be more complex.

## Dataset

### Dataset Structure
The project uses a custom dataset format. Datasets are typically organized as:
`dataset/<dataset_name>/<split>/case_<id>/`
where:
*   `<dataset_name>`: e.g., `map10x10_r5_o10_p5` (map size 10x10, 5% robot density, 10% obstacle density, for 5 agents). The `_pX` often refers to the `pad` value used for FOV generation.
*   `<split>`: `train`, `val`, or `test`.
*   `case_<id>`: Individual scenario directory, where `<id>` is a numerical identifier.

Each `case_<id>` directory should contain:
*   `input.yaml`: Defines the map dimensions, obstacle locations, and agent start/goal locations (in x,y CBS-style coordinates).
*   `solution.yaml`: The expert solution paths from CBS, including cost and schedule for each agent (in x,y CBS-style coordinates).
*   `trajectory.npy`: Parsed expert actions for each agent, ready for use by the GNN. Shape: `(num_agents, num_timesteps_actions)`. Action indices are 0:Idle, 1:Right, 2:Up, 3:Left, 4:Down.
*   `states.npy`: Recorded agent Field of View (FOV) observations. Shape: `(num_timesteps_states, num_agents, num_fov_channels, fov_height, fov_width)`. `num_timesteps_states` is typically `num_timesteps_actions + 1`.
*   `gso.npy`: Recorded Graph Shift Operator (adjacency matrix based on communication range). Shape: `(num_timesteps_states, num_agents, num_agents)`.

An example of this structure is visible under `data/map10x10_r5_o30_p5/train/case_10/`.

### Generating Datasets
Use `generate_all_datasets.py` to create new datasets. This script automates:
1.  Generating random scenarios (`input.yaml`).
2.  Solving these scenarios using the Conflict-Based Search (CBS) algorithm to obtain expert paths (`solution.yaml`).
3.  Parsing the CBS solutions into sequences of discrete actions (`trajectory.npy`).
4.  Simulating the expert paths in the `GraphEnv` environment to record sequences of agent FOVs and GSOs (`states.npy`, `gso.npy`).

Modify the parameter ranges (e.g., `env_sizes`, `robot_densities`, `obstacle_densities`) and fixed values (e.g., `pad_fixed`, `sensing_range_fixed`) within `generate_all_datasets.py` to define the desired dataset configurations.

Example execution (from the project root):
```bash
python generate_all_datasets.py
```
This script will log its progress and save generated datasets into the `dataset/` directory (or as configured within the script).

### Converting External Datasets
The project includes tools to convert datasets from `.mat` format (e.g., from the Prorok Lab IROS 2020 paper):
*   `inspect_mat.py` / `read_mat.py`: Utilities to inspect the content of `.mat` files. The `output.txt` file shows an example output from `read_mat.py`.
*   `convert_mat_to_npy.py`: Converts `.mat` files containing fields like `inputTensor`, `target`, `GSO`, `map`, `goal`, `inputState` into the project's `input.yaml`, `solution.yaml`, and `.npy` format.
    ```bash
    python convert_mat_to_npy.py <path_to_mat_files_dir> <output_dir_for_cases> --num_agents <N> --pad <P>
    ```
    `<P>` is the pad value that corresponds to the FOV size in the `.mat` files (e.g., pad=6 for 11x11 FOV).
*   `mat_dataset.py`: A PyTorch `Dataset` class designed to load data directly from these `.mat` files for training (used in conjunction with `configs/config_train_mat.yaml`).

### Data Utilities
*   `cleanup_dataset.py`: Iterates through dataset splits and removes incomplete `case_` directories (e.g., missing `solution.yaml` or `.npy` files).
    ```bash
    python cleanup_dataset.py dataset/<dataset_name>/
    ```
*   `merge_datasets.py`: Merges case directories from a source dataset structure into a target dataset structure, renaming cases from the source to ensure unique, sequential numbering.
    ```bash
    python merge_datasets.py <source_base_dir> <target_base_dir> <config_name1> [<config_name2> ...] --splits train val test
    ```

## Model Training

### Configuration
Training is primarily controlled by YAML configuration files located in the `configs/` directory. These files specify:
*   `exp_name`: A unique name for the experiment. Results (model checkpoints, metrics, plots) will be saved in a directory named after this under `results_afterchange/`.
*   Model architecture details:
    *   `net_type`: `gnn` or `baseline`.
    *   `msg_type`: For `gnn`, can be `gcn`, `adc`, or `message`.
    *   CNN parameters: `channels`, `strides`, `kernels`, `paddings`.
    *   MLP Encoder parameters: `encoder_layers`, `encoder_dims`.
    *   GNN parameters: `graph_filters` (K-hops/taps/truncation order), `node_dims`.
    *   ADC specific parameters: `adc_initial_t`, `adc_train_t`.
    *   Action MLP parameters: `action_layers`.
*   Training hyperparameters: `epochs`, `learning_rate`, `weight_decay`, `batch_size`, `num_workers`.
*   Evaluation settings during training: `eval_frequency`, `tests_episodes`, parameters for generating evaluation scenarios (`eval_board_size`, `eval_obstacles`, etc.).
*   Dataset paths for training and validation: `train.root_dir`, `valid.root_dir`, and related data loading filters (`min_time`, `max_time_dl`).
*   Environment parameters (must match the dataset): `num_agents`, `board_size`, `obstacles`, `pad`, `sensing_range`, `max_time`.
*   Online Expert (DAgger-like) settings (in `online_expert` sub-section), if DAgger is enabled.

Example configuration files are provided for various GCN and ADC models, often targeting 10x10 maps with different obstacle densities (e.g., `configs/config_10x10_o10_gcn_k3.yaml`, `configs/config_10x10_o20_adc_main.yaml`). The `configs1/` and `configs2/` directories contain copies or variations of these. `config_train_mat.yaml` is an example for training with `.mat` datasets.

### Running Training
The main training script is `train.py`.
To train a model using a specific configuration:
```bash
python train.py --config configs/your_config_file.yaml
```
To disable the Online Expert (DAgger) data aggregation feature:
```bash
python train.py --config configs/your_config_file.yaml --oe_disable
```
You can also set the logging level:
```bash
python train.py --config configs/your_config_file.yaml --log_level DEBUG
```

Training outputs, including model checkpoints (`model_best.pt`, `model_final.pt`) and detailed epoch-wise metrics (`training_metrics.xlsx`), will be saved in `results_afterchange/<exp_name>/`.

## Model Evaluation

Several scripts facilitate the evaluation of trained models:

*   **`evaluate_models.py`**: This script evaluates one or more trained models on specified test sets (collections of pre-generated `case_` directories).
    *   It calculates metrics such as Success Rate, Average Makespan (for successful episodes), Flowtime (FT), Flowtime Increase (dFT), Average Inference Time per step, and the number of trainable parameters.
    *   Results are saved to CSV files (`evaluation_metrics_per_testset.csv` and `evaluation_metrics_overall.csv`) in the specified output directory.
    ```bash
    python evaluate_models.py --model_dirs results_afterchange/model1_dir results_afterchange/model2_dir --test_sets data/dataset_name/test_split_A data/dataset_name/test_split_B --output_dir results/my_evaluation_report
    ```

*   **`compare_models.py`**: Compares two models side-by-side.
    *   Evaluates performance metrics (Success Rate, Average Steps) on randomly generated scenarios.
    *   Measures model size (trainable parameters, file size) and inference speed (average batch time, throughput).
    *   Generates bar plots comparing SR and Avg. Steps.
    *   Outputs a CSV file (`size_speed_comparison.csv`) and a performance comparison CSV.
    ```bash
    python compare_models.py --model1_dir results_afterchange/modelA_dir --model2_dir results_afterchange/modelB_dir --output_dir results/comparison_A_vs_B --episodes 100
    ```

*   **`replicate_figure5.py`**: This script is designed to reproduce experiments similar to Figure 5 from research papers, which typically plot performance (SR, Avg. Steps) against an increasing number of robots while maintaining a constant overall agent/obstacle density by scaling the map size. It uses multiprocessing for faster evaluation across different agent counts.
    ```bash
    python replicate_figure5.py --model_dirs results_afterchange/model_dir1 results_afterchange/model_dir2 --agent_counts 20 30 40 50 60 --obstacle_density 0.10 --output_dir results/figure5_replication_output
    ```

## Visualization

*   **`create_gif.py` / `create_video.py`**: Generate animated visualizations of a trained model executing its policy in a randomly generated MAPF scenario.
    *   `create_gif.py` also saves the specific scenario details (map, agent starts/goals, obstacles) and the agent's executed path to a YAML file (e.g., `mapf_scenario_path.yaml`).
    ```bash
    # For GIF and scenario/path YAML
    python create_gif.py --config configs/your_model_config.yaml --model_path results_afterchange/your_model_dir/model_best.pt --output_gif my_mapf_episode.gif --output_yaml my_episode_details.yaml --seed 123

    # For MP4 video
    python create_video.py --config configs/your_model_config.yaml --model_path results_afterchange/your_model_dir/model_best.pt --output_file my_mapf_episode.mp4 --seed 456
    ```
    Example scenario/path YAML outputs from `create_gif.py` can be found in `results_afterchange/adc_main_10x10_o30_p5/scenario_and_path.yaml` and `results_afterchange/gcn_k3_10x10_o30_p5/scenario_and_path.yaml`.

*   **`generate_result_plots.py`**: Creates various plots from evaluation CSV files (generated by `evaluate_models.py`) and training metrics Excel files (from `train.py`).
    *   Performance plots: Success Rate, Average Makespan, Flowtime vs. Test Condition/Obstacle Density.
    *   Computational performance plots: Average Inference Time, Number of Parameters.
    *   Combined training curves: Training Loss, Evaluation Success Rate, Evaluation Average Makespan vs. Epoch, comparing multiple models.
    ```bash
    python generate_result_plots.py \
        --per_testset_csv_file results_afterchange/final_evaluation_10x10_o10/evaluation_metrics_per_testset.csv \
        --overall_csv_file results_afterchange/final_evaluation_10x10_o10/evaluation_metrics_overall.csv \
        --training_result_dirs results_afterchange/gcn_k*_10x10_o10_p5 results_afterchange/adc_*_10x10_o10_p5 \
        --output_dir results/paper_plots/TRAINED_ON_10_OBS \
        --plot_title_suffix "(Trained on 10% Obstacles)" \
        --training_condition_filter "_o10_"
    ```
    The `results/paper_plots/` directory contains examples of generated plots for different training conditions (10%, 20%, 30% obstacles).

*   **`generate_diagram.py`**: Uses Graphviz to create a block diagram of a model's architecture based on its YAML configuration file.
    ```bash
    python generate_diagram.py --config configs/config_10x10_o10_adc_main.yaml -o results/diagrams/adc_main_arch -f png
    ```

*   **`manim_visualization.py`**: A script using the Manim library for creating more conceptual, high-quality animations related to the MAPF training process (e.g., explaining FOV, GSO, loss calculation). Requires a separate Manim installation and environment setup.

## Implemented Models

*   **Baseline CNN-MLP (`models/framework_baseline.py`)**: A standard Convolutional Neural Network (CNN) processes each agent's local Field of View (FOV). The flattened CNN output is then fed through Multi-Layer Perceptrons (MLPs) to predict actions. This model operates purely on local observations and does not use the GSO for communication.
*   **GCN (Graph Convolutional Network) (`models/framework_gnn.py`, `models/networks/gnn.py`)**: Agents first process their FOV using a CNN encoder. The resulting features are then propagated and aggregated among neighboring agents using GCN layers, guided by the GSO. The number of GCN layers and the filter taps (K, representing K-hop neighborhood aggregation) are configurable.
*   **ADC (Adaptive Diffusion Convolution) (`models/framework_gnn.py`, `models/networks/adc_layer.py`)**: Similar to GCN, ADC uses a CNN encoder followed by GNN-style message passing. ADC layers are based on a heat kernel diffusion process, where the diffusion time 't' can be a fixed hyperparameter or a learnable parameter per layer. The Taylor expansion of the heat kernel is truncated at order K.
*   **Message Passing GNN (`models/framework_gnn_message.py`, `models/networks/gnn.py`)**: A basic Message Passing Neural Network (MPNN) structure. (Note: `models/framework_gnn.py` can be configured with `msg_type: 'message'` to use this layer type, making `framework_gnn_message.py` potentially redundant).

All models output logits for a discrete set of actions (typically 5: Idle, Right, Up, Left, Down).

## Other Key Scripts

*   **`extract_adc_t.py`**: Loads a trained ADC model and prints the learned diffusion parameter 't' for each ADCLayer instance. Useful for analyzing the learned diffusion behavior.
    ```bash
    python extract_adc_t.py results_afterchange/adc_main_10x10_o20_p5/ --checkpoint model_best.pt
    ```
*   **`combine_metrics.py`**: Takes multiple `training_metrics.xlsx` files from different experiment result directories and combines them into a single Excel workbook, with each original file as a separate sheet.
    ```bash
    python combine_metrics.py -o results/combined_o10_metrics.xlsx results_afterchange/*_10x10_o10_p5
    ```
*   **`export_metrics_to_text.py`**: Similar to `combine_metrics.py`, but exports the data from multiple `training_metrics.xlsx` files into a single formatted text file for easier review or inclusion in reports.
    ```bash
    python export_metrics_to_text.py -o results/exported_o10_metrics.txt results_afterchange/*_10x10_o10_p5
    ```
*   `check_cases.py`: A simple script to count the number of subdirectories (cases) within a specified dataset split.
*   `dataset_generation_log.txt`: An example log file produced during dataset generation, showing the kind of output and progress information logged.
*   `output.txt`: Example output from `read_mat.py`, showing the structure and keys of a `.mat` dataset file.

## Usage Notes

*   Most scripts expect to be run from the root directory (`rahul-velamala-mapf-gnn/`).
*   Paths in configuration files and command-line arguments may need to be adjusted based on your local file structure.
*   Ensure consistency between dataset generation parameters, model configuration parameters (e.g., `num_agents`, `pad`), and data loading parameters.
*   The `results/`, `results1/`, and `results_afterchange/` directories contain various experiment outputs. `results_afterchange/` seems to hold the most structured and recent set of results, particularly for the comparisons between GCN (K=1 to K=4) and ADC (main, fixed-t, K=1) models trained on datasets with 10%, 20%, and 30% obstacle densities.
*   The provided CSV files in `results_afterchange/final_evaluation_10x10_oXX/` show aggregated performance metrics and are likely inputs for `generate_result_plots.py`.

This project provides a comprehensive framework for research into GNN-based MAPF. Good luck!
```