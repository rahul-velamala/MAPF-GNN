# File: extract_adc_t.py
# Loads a trained ADC model and prints the learned diffusion parameter(s) 't'.

import torch
import yaml
from pathlib import Path
import argparse
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# --- --------------- ---

# --- Import Model Components ---
# Need these imports here to initialize the model correctly
try:
    from models import GNNNetwork # Assumes framework_gnn.py uses this name
    from models.networks.adc_layer import ADCLayer
except ImportError as e:
    logger.error(f"Failed to import model classes: {e}")
    logger.error("Ensure you are running this script from the project root directory.")
    exit(1)
# --- ----------------------- ---

def get_learned_t(model_dir: Path, checkpoint_filename: str = "model_best.pt") -> list[tuple[str, float]]:
    """
    Loads a trained ADC model and extracts the learned 't' parameters.

    Args:
        model_dir (Path): Path to the result directory of the trained ADC model.
        checkpoint_filename (str): Name of the checkpoint file to load
                                  (e.g., "model_best.pt", "model_final.pt").

    Returns:
        list[tuple[str, float]]: A list of tuples, where each tuple contains
                                 the name of the ADC layer and its learned 't' value.
                                 Returns an empty list if errors occur or no ADC layers found.
    """
    learned_t_values = []
    model_path = model_dir / checkpoint_filename

    # --- 1. Load Config ---
    config_path = model_dir / "config_used.yaml"
    if not config_path.is_file():
        logger.error(f"Config file not found: {config_path}")
        return learned_t_values

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None or config.get('net_type') != 'gnn' or config.get('msg_type') != 'adc':
             logger.error(f"Loaded config is invalid or not for an ADC model: {config_path}")
             return learned_t_values
        # Load to CPU by default for inspection
        config['device'] = torch.device('cpu')
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return learned_t_values

    # --- 2. Initialize Model ---
    try:
        model = GNNNetwork(config) # Assumes this is the correct class
        model.to(config['device'])
        logger.info(f"Initialized model structure from {config_path.name}")
    except Exception as e:
        logger.error(f"Error initializing model architecture: {e}")
        return learned_t_values

    # --- 3. Load State Dict ---
    if not model_path.is_file():
        logger.error(f"Model checkpoint file not found: {model_path}")
        return learned_t_values

    try:
        state_dict = torch.load(model_path, map_location=config['device'])
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode
        logger.info(f"Loaded model weights from {model_path.name}")
    except Exception as e:
        logger.error(f"Error loading model state_dict from {model_path}: {e}")
        return learned_t_values

    # --- 4. Find ADC Layers and Extract 't' ---
    adc_layer_found = False
    # Iterate through named modules to find ADCLayer instances
    for name, module in model.named_modules():
        if isinstance(module, ADCLayer):
            adc_layer_found = True
            # Check if 't' is a parameter (meaning train_t=True was used)
            if hasattr(module, 't') and isinstance(module.t, torch.nn.Parameter):
                t_value = module.t.item()
                learned_t_values.append((name, t_value))
                logger.info(f"Found ADCLayer '{name}' with learned t = {t_value:.6f}")
            elif hasattr(module, 't'): # It exists but is not a Parameter (likely a buffer, train_t=False)
                t_value = module.t.item() if isinstance(module.t, torch.Tensor) else module.t
                logger.warning(f"Found ADCLayer '{name}' but 't' is not a Parameter (train_t=False?). Value = {t_value:.6f}")
                # Optionally still append it if you want to report fixed values too
                # learned_t_values.append((name, t_value))
            else:
                logger.warning(f"Found ADCLayer '{name}' but it does not have a 't' attribute.")

    if not adc_layer_found:
        logger.warning(f"No ADCLayer instances found within the loaded model '{model_dir.name}'.")

    return learned_t_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract learned diffusion parameter 't' from a trained ADC model.")
    parser.add_argument(
        "model_result_dir",
        type=str,
        help="Path to the result directory of the trained ADC model (e.g., results/adc_main_10x10_o10_p5)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_best.pt",
        help="Name of the checkpoint file to load (default: model_best.pt)."
    )

    args = parser.parse_args()
    model_directory = Path(args.model_result_dir)

    if not model_directory.is_dir():
        print(f"Error: Directory not found: {model_directory}")
        exit(1)

    extracted_t_values = get_learned_t(model_directory, args.checkpoint)

    if extracted_t_values:
        print("\n--- Learned 't' Values ---")
        for layer_name, t_val in extracted_t_values:
            print(f"Layer: {layer_name}, t = {t_val:.6f}")
        # You can now use the first value (if only one layer) for your table
        print(f"\nValue for Table 3 (assuming first ADC layer): {extracted_t_values[0][1]:.4f}")
    else:
        print(f"\nCould not extract learned 't' values from {model_directory}.")