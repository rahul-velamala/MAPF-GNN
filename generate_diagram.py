# File: generate_diagram.py
# Creates a block diagram of the GNN/ADC model architecture using Graphviz.

import graphviz
import yaml
from pathlib import Path
import argparse
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---

def generate_model_diagram(config: dict, output_filename: str = "model_architecture", output_format: str = "png"):
    """
    Generates a Graphviz diagram representing the model architecture defined in the config.

    Args:
        config (dict): The loaded model configuration dictionary.
        output_filename (str): Base name for the output file (without extension).
        output_format (str): Output format (e.g., 'png', 'pdf', 'svg').
    """
    dot = graphviz.Digraph(comment='Model Architecture', format=output_format)
    dot.attr(rankdir='TB', label=f'Model Architecture ({config.get("exp_name", "Unknown")})', fontsize='20')
    dot.attr('node', shape='box', style='filled', color='skyblue', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # --- Extract Key Config Params ---
    try:
        net_type = config.get('net_type', 'gnn')
        msg_type = config.get('msg_type', 'gcn') if net_type == 'gnn' else 'N/A'
        pad = int(config.get('pad', 3))
        fov_size = (pad * 2) - 1
        cnn_channels_list = config.get('channels', [16, 16, 16])
        num_cnn_layers = len(cnn_channels_list)
        encoder_layers = int(config.get('encoder_layers', 1))
        gnn_k = config.get('graph_filters', [3])[0] # Assuming single GNN layer for simplicity in diagram
        num_gnn_layers = len(config.get('graph_filters', [3]))
        action_layers = int(config.get('action_layers', 1))
        cnn_in_c = 3 # Default FOV channels
    except Exception as e:
        logger.error(f"Error parsing config for diagram: {e}. Using defaults.")
        # Use defaults if config parsing fails
        net_type, msg_type, pad, fov_size = 'gnn', 'gcn', 3, 5
        num_cnn_layers, encoder_layers, gnn_k, num_gnn_layers, action_layers = 3, 1, 3, 1, 1
        cnn_in_c = 3

    # --- Define Nodes ---

    # Inputs
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', style='filled', color='lightgrey')
        c.node('fov_input', f'FOV Input\n(B, N, {cnn_in_c}, {fov_size}, {fov_size})', shape='ellipse', color='lightyellow')
        if net_type == 'gnn':
            c.node('gso_input', 'GSO Input\n(B, N, N)', shape='ellipse', color='lightcoral')
        c.attr(rank='same') # Align inputs horizontally

    # CNN Encoder
    with dot.subgraph(name='cluster_cnn') as c:
        c.attr(label='CNN Encoder (Per Agent)', style='filled', color='lightblue')
        c.node('cnn_reshape_in', 'Reshape\n(B*N, C, H, W)', shape='plaintext')
        c.node('cnn_block', f'(Conv2D -> ReLU) * {num_cnn_layers}', color='blue')
        c.node('cnn_flatten', 'Flatten\n(B*N, cnn_flat_dim)', shape='plaintext')

    # MLP Encoder
    with dot.subgraph(name='cluster_mlp_encoder') as c:
        c.attr(label='MLP Encoder (Per Agent)', style='filled', color='lightgreen')
        c.node('mlp_encoder_block', f'(Linear -> ReLU) * {encoder_layers}', color='green')
        c.node('mlp_encoder_out', f'Encoded Features\n(B*N, gnn_in_dim)', shape='plaintext')

    # GNN Block (only if gnn type)
    if net_type == 'gnn':
        gnn_label = f'GNN Layers\n({msg_type.upper()}, K={gnn_k})\n(GNN -> ReLU) * {num_gnn_layers}'
        with dot.subgraph(name='cluster_gnn') as c:
            c.attr(label='Graph Neural Network', style='filled', color='lightgrey')
            c.node('gnn_reshape_in', 'Reshape & Permute\n(B, gnn_in_dim, N)', shape='plaintext')
            c.node('gnn_block', gnn_label, color='grey')
            c.node('gnn_reshape_out', 'Permute & Reshape\n(B*N, gnn_out_dim)', shape='plaintext')

    # Action MLP
    with dot.subgraph(name='cluster_action_mlp') as c:
        c.attr(label='Action MLP (Per Agent)', style='filled', color='lightskyblue')
        action_mlp_label = f'(Linear -> ReLU) * {action_layers-1} + Linear' if action_layers > 1 else 'Linear'
        c.node('action_mlp_block', action_mlp_label, color='deepskyblue')

    # Output
    dot.node('action_logits', 'Action Logits\n(B, N, A)', shape='ellipse', color='orange')

    # --- Define Edges (Data Flow) ---

    # CNN Path
    dot.edge('fov_input', 'cnn_reshape_in', label='FOV Data')
    dot.edge('cnn_reshape_in', 'cnn_block')
    dot.edge('cnn_block', 'cnn_flatten')

    # MLP Encoder Path
    dot.edge('cnn_flatten', 'mlp_encoder_block', label='CNN Features')
    dot.edge('mlp_encoder_block', 'mlp_encoder_out')

    # GNN Path (Conditional)
    if net_type == 'gnn':
        dot.edge('mlp_encoder_out', 'gnn_reshape_in')
        dot.edge('gnn_reshape_in', 'gnn_block', label='Node Features')
        dot.edge('gso_input', 'gnn_block', label='Graph Structure (GSO)', style='dashed') # GSO feeds into GNN
        dot.edge('gnn_block', 'gnn_reshape_out', label='Aggregated Features')
        # Connect GNN output to Action MLP
        dot.edge('gnn_reshape_out', 'action_mlp_block')
    else: # Baseline: connect MLP Encoder directly to Action MLP
        dot.edge('mlp_encoder_out', 'action_mlp_block', label='Encoded Features')

    # Action MLP to Output
    dot.edge('action_mlp_block', 'action_logits', label='Action Logits')

    # --- Render and Save ---
    try:
        output_path = Path(output_filename)
        dot.render(output_path, view=False, cleanup=True) # view=False prevents opening the file automatically
        logger.info(f"Diagram saved to {output_path}.{output_format}")
    except Exception as e:
        logger.error(f"Error rendering diagram: {e}")
        logger.error("Ensure Graphviz is installed and in your system's PATH.")
        logger.error("On Linux/macOS, you might need to install it via package manager (apt, dnf, brew).")
        logger.error("On Windows, download from graphviz.org and add bin folder to PATH.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a model architecture diagram from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model YAML configuration file (e.g., configs/config_10x10_o10_adc_main.yaml)."
    )
    parser.add_argument(
        "-o", "--output",
        default="results/model_architecture_diagram", # Default output name in results/
        help="Base name for the output diagram file (without extension)."
    )
    parser.add_argument(
        "-f", "--format",
        default="png",
        choices=['png', 'pdf', 'svg', 'dot'],
        help="Output file format (default: png)."
    )
    args = parser.parse_args()

    # --- Load Config ---
    config_file_path = Path(args.config)
    logger.info(f"Loading configuration from: {config_file_path}")
    if not config_file_path.is_file():
        logger.error(f"Configuration file not found: {config_file_path}")
        sys.exit(1)

    try:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None: raise ValueError("Config file is empty or invalid.")
        # Add exp_name to config if it's not already there (using filename as fallback)
        if 'exp_name' not in config:
            config['exp_name'] = config_file_path.stem
    except Exception as e:
        logger.error(f"Could not load or parse config file '{config_file_path}': {e}")
        sys.exit(1)

    # --- Generate Diagram ---
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_model_diagram(config, args.output, args.format)