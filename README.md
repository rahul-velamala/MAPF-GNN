# Multi-Agent Path Finding with Graph Neural Networks (MAPF-GNN)

This project explores the application of Graph Neural Networks (GNNs), including standard Graph Convolutional Networks (GCNs) and an Adaptive Diffusion Convolution (ADC) variant, for solving Multi-Agent Path Finding (MAPF) problems. The project includes tools for dataset generation, model training, comprehensive evaluation, and visualization.

## Overview

The primary goal is to develop and evaluate learning-based policies for decentralized MAPF. Agents make decisions based on local observations (Field of View - FOV) and communication with nearby agents (Graph Shift Operator - GSO). The GNN architecture allows agents to learn to coordinate and navigate to their goals while avoiding collisions.

## Features

- **Decentralized MAPF Policies**: Learns policies for individual agents.
- **GNN Architectures**:
  - Graph Convolutional Networks (GCN) with configurable number of filter taps (K).
  - Adaptive Diffusion Convolution (ADC) layers with potentially learnable diffusion time.
  - Baseline CNN-MLP models for comparison.
- **Dataset Generation**:
  - Generates MAPF scenarios (maps, agent start/goals, obstacles).
  - Solves scenarios using Conflict-Based Search (CBS) to get expert trajectories.
  - Processes trajectories into FOV, GSO, and action sequences.
- **Model Training**:
  - Supervised learning from expert trajectories.
  - Online Expert (DAgger-like) data aggregation to improve policy robustness (optional).
- **Comprehensive Evaluation**:
  - Metrics: Success Rate, Average Makespan, Flowtime, Flowtime Increase, Inference Time, Parameter Count.
  - Comparison scripts for different models and configurations.
  - Ability to replicate specific experimental setups (e.g., performance vs. number of agents).
- **Visualization**:
  - Generates GIFs and videos of MAPF episodes.
  - Plots training curves and evaluation metrics.
  - Manim-based animation for conceptual visualization.
  - Graphviz diagrams for model architectures.
- **Data Management**:
  - Tools to convert external `.mat` datasets to the project's `.npy` format.
  - Dataset cleanup and merging utilities.

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
├── requirement.txt         # Python dependencies
├── train.py                # Main script for training models
├── evaluate_models.py      # Script for evaluating trained models
├── compare_models.py       # Script for comparing different models
├── generate_all_datasets.py # Script to generate datasets for various configs
├── generate_result_plots.py # Script to plot evaluation results
├── create_gif.py           # Script to create GIF visualizations of episodes
├── create_video.py         # Script to create video visualizations of episodes
└── ... (utility scripts)   # Various utilities for dataset management, visualization, etc.
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd rahul-velamala-mapf-gnn
   ```

2. **Create a Python environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   Based on the project's functionality, common dependencies include:
   - **Core ML/Numerics:** `torch` (PyTorch), `numpy`, `scipy`
   - **Configuration & Data Handling:** `pyyaml`, `pandas`, `openpyxl` (for `.xlsx`)
   - **Visualization & Plotting:** `matplotlib`, `seaborn`, `imageio` (and `imageio[ffmpeg]` for video), `graphviz` (Python library)
   - **Environment & Utilities:** `gymnasium`, `tqdm`
   - **Conceptual Animation:** `manim` (if using `manim_visualization.py`)

   Example installation using pip:
   ```bash
   pip install torch numpy scipy pyyaml pandas openpyxl matplotlib seaborn tqdm imageio graphviz gymnasium
   pip install imageio[ffmpeg]
   ```

   **Important for Graphviz:**
   In addition to the Python `graphviz` library, you need to install Graphviz at the system level for `generate_diagram.py` to work:
   - Linux: `sudo apt-get install graphviz` or `sudo dnf install graphviz`
   - macOS: `brew install graphviz`
   - Windows: Download from [graphviz.org](https://graphviz.org/download/) and add its `bin` directory to your system's PATH.

   **For Manim:** Follow the official Manim installation guide as it requires additional setup.

## Dataset

### Dataset Structure
The project uses a custom dataset format. Datasets are typically organized as:
`dataset/<dataset_name>/<split>/case_<id>/`
where:
- `<dataset_name>`: e.g., `map10x10_r5_o10_p5` (map size 10x10, 5% robot density, 10% obstacle density, for 5 agents). The `_pX` often refers to the `pad` value used for FOV generation.
- `<split>`: `train`, `val`, or `test`.
- `case_<id>`: Individual scenario directory, where `<id>` is a numerical identifier.

Each `case_<id>` directory contains:
- `input.yaml`: Defines the map dimensions, obstacle locations, and agent start/goal locations (in x,y CBS-style coordinates).
- `solution.yaml`: The expert solution paths from CBS, including cost and schedule for each agent (in x,y CBS-style coordinates).
- `trajectory.npy`: Parsed expert actions for each agent, ready for use by the GNN. Shape: `(num_agents, num_timesteps_actions)`. Action indices are 0:Idle, 1:Right, 2:Up, 3:Left, 4:Down.
- `states.npy`: Recorded agent Field of View (FOV) observations. Shape: `(num_timesteps_states, num_agents, num_fov_channels, fov_height, fov_width)`. `num_timesteps_states` is typically `num_timesteps_actions + 1`.
- `gso.npy`: Recorded Graph Shift Operator (adjacency matrix based on communication range). Shape: `(num_timesteps_states, num_agents, num_agents)`.

### Generating Datasets
Use `generate_all_datasets.py` to create new datasets. This script automates:
1. Generating random scenarios (`input.yaml`).
2. Solving these scenarios using the Conflict-Based Search (CBS) algorithm to obtain expert paths (`solution.yaml`).
3. Parsing the CBS solutions into sequences of discrete actions (`trajectory.npy`).
4. Simulating the expert paths in the `GraphEnv` environment to record sequences of agent FOVs and GSOs (`states.npy`, `gso.npy`).

Example execution:
```bash
python generate_all_datasets.py
```

### Converting External Datasets
The project includes tools to convert datasets from `.mat` format:
```bash
python convert_mat_to_npy.py <path_to_mat_files_dir> <output_dir_for_cases> --num_agents <N> --pad <P>
```

### Data Utilities
- **Cleanup datasets:**
  ```bash
  python cleanup_dataset.py dataset/<dataset_name>/
  ```
- **Merge datasets:**
  ```bash
  python merge_datasets.py <source_base_dir> <target_base_dir> <config_name1> [<config_name2> ...] --splits train val test
  ```

## Model Training

### Configuration
Training is primarily controlled by YAML configuration files in the `configs/` directory. These files specify:
- Model architecture details (`net_type`, `msg_type`, network parameters)
- Training hyperparameters (`epochs`, `learning_rate`, etc.)
- Evaluation settings
- Dataset paths
- Environment parameters

### Running Training
The main training script is `train.py`:
```bash
# Basic training
python train.py --config configs/your_config_file.yaml

# Disable Online Expert (DAgger)
python train.py --config configs/your_config_file.yaml --oe_disable

# Set logging level
python train.py --config configs/your_config_file.yaml --log_level DEBUG
```

Training outputs will be saved in `results_afterchange/<exp_name>/`.

## Model Evaluation

Several scripts facilitate the evaluation of trained models:

- **Evaluate Models:**
  ```bash
  python evaluate_models.py --model_dirs results_afterchange/model1_dir results_afterchange/model2_dir --test_sets data/dataset_name/test_split_A data/dataset_name/test_split_B --output_dir results/my_evaluation_report
  ```

- **Compare Models:**
  ```bash
  python compare_models.py --model1_dir results_afterchange/modelA_dir --model2_dir results_afterchange/modelB_dir --output_dir results/comparison_A_vs_B --episodes 100
  ```

- **Replicate Figure5 (Performance vs. Number of Agents):**
  ```bash
  python replicate_figure5.py --model_dirs results_afterchange/model_dir1 results_afterchange/model_dir2 --agent_counts 20 30 40 50 60 --obstacle_density 0.10 --output_dir results/figure5_replication_output
  ```

## Visualization

- **Create GIF/Video of Model Execution:**
  ```bash
  # For GIF
  python create_gif.py --config configs/your_model_config.yaml --model_path results_afterchange/your_model_dir/model_best.pt --output_gif my_mapf_episode.gif --output_yaml my_episode_details.yaml --seed 123

  # For MP4 video
  python create_video.py --config configs/your_model_config.yaml --model_path results_afterchange/your_model_dir/model_best.pt --output_file my_mapf_episode.mp4 --seed 456
  ```

- **Generate Result Plots:**
  ```bash
  python generate_result_plots.py --per_testset_csv_file results_afterchange/final_evaluation_10x10_o10/evaluation_metrics_per_testset.csv --overall_csv_file results_afterchange/final_evaluation_10x10_o10/evaluation_metrics_overall.csv --training_result_dirs results_afterchange/gcn_k*_10x10_o10_p5 results_afterchange/adc_*_10x10_o10_p5 --output_dir results/paper_plots/TRAINED_ON_10_OBS --plot_title_suffix "(Trained on 10% Obstacles)" --training_condition_filter "_o10_"
  ```

- **Generate Architecture Diagram:**
  ```bash
  python generate_diagram.py --config configs/config_10x10_o10_adc_main.yaml -o results/diagrams/adc_main_arch -f png
  ```

## Implemented Models

- **Baseline CNN-MLP**: A standard CNN processes each agent's local FOV, followed by MLPs to predict actions.
- **GCN (Graph Convolutional Network)**: Agents process their FOV using a CNN encoder, then propagate and aggregate features among neighboring agents using GCN layers.
- **ADC (Adaptive Diffusion Convolution)**: Similar to GCN, but uses heat kernel diffusion for message passing, with diffusion time 't' as a fixed or learnable parameter.
- **Message Passing GNN**: A basic Message Passing Neural Network structure.

All models output logits for a discrete set of actions (typically 5: Idle, Right, Up, Left, Down).

## Other Useful Scripts

- **Extract ADC Diffusion Parameter:**
  ```bash
  python extract_adc_t.py results_afterchange/adc_main_10x10_o20_p5/ --checkpoint model_best.pt
  ```

- **Combine Training Metrics:**
  ```bash
  python combine_metrics.py -o results/combined_o10_metrics.xlsx results_afterchange/*_10x10_o10_p5
  ```

- **Export Metrics to Text:**
  ```bash
  python export_metrics_to_text.py -o results/exported_o10_metrics.txt results_afterchange/*_10x10_o10_p5
  ```

## Usage Notes

- Most scripts expect to be run from the root directory (`rahul-velamala-mapf-gnn/`).
- Ensure consistency between dataset generation parameters, model configuration parameters, and data loading parameters.
- The `results_afterchange/` directory contains the most structured and recent set of results, particularly for comparisons between GCN and ADC models.

This project provides a comprehensive framework for research into GNN-based Multi-Agent Path Finding.
