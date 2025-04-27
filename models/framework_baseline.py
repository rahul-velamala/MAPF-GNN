# File: models/framework_baseline.py
# (Revised: Removed inplace=True from ReLUs, added logging)

import torch
import torch.nn as nn
import logging
import numpy as np

from .networks.utils_weights import weights_init

logger = logging.getLogger(__name__)

class Network(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        required_keys = ["num_agents", "pad", "channels", "encoder_layers",
                         "encoder_dims", "action_layers"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config missing required key for Framework Baseline: '{key}'")

        self.num_agents = int(config["num_agents"])
        self.num_actions = 5

        # --- Determine FOV size ---
        self.pad = int(config["pad"])
        if self.pad <= 0: raise ValueError("'pad' must be >= 1")
        self.fov_size = (self.pad * 2) - 1
        logger.info(f"Framework Baseline: Using FOV size {self.fov_size}x{self.fov_size} (pad={self.pad})")

        # --- Model Config ---
        self.cnn_input_channels = 3 # FOV: (Obstacles/Agents, Goal, Self)
        cnn_channels = [self.cnn_input_channels] + config["channels"]
        num_conv_layers = len(cnn_channels) - 1
        cnn_strides = config.get("strides", [1] * num_conv_layers)
        cnn_kernels = config.get("kernels", [3] * num_conv_layers)
        cnn_paddings = config.get("paddings", [1] * num_conv_layers)

        if not (len(cnn_strides) == len(cnn_kernels) == len(cnn_paddings) == num_conv_layers):
            raise ValueError("Length of CNN 'strides'/'kernels'/'paddings' doesn't match 'channels'.")

        mlp_encoder_layers_count = int(config["encoder_layers"])
        mlp_encoder_dims = config["encoder_dims"]

        action_mlp_layers_count = int(config["action_layers"])
        action_mlp_output_dim = self.num_actions

        ############################################################
        # 1. CNN Encoder
        ############################################################
        conv_layers_list = []
        H_out, W_out = self.fov_size, self.fov_size
        current_channels = self.cnn_input_channels
        logger.info("Building CNN:")
        for i in range(num_conv_layers):
            out_channels = cnn_channels[i+1]
            kernel, stride, padding = cnn_kernels[i], cnn_strides[i], cnn_paddings[i]
            logger.info(f"  Layer {i}: Conv2d({current_channels}, {out_channels}, k={kernel}, s={stride}, p={padding}) -> ReLU")
            conv_layers_list.append(nn.Conv2d(current_channels, out_channels, kernel, stride, padding, bias=True))
            # Optional BatchNorm: nn.BatchNorm2d(out_channels)
            conv_layers_list.append(nn.ReLU(inplace=False)) # Use inplace=False
            H_out = int((H_out + 2 * padding - kernel) / stride) + 1
            W_out = int((W_out + 2 * padding - kernel) / stride) + 1
            current_channels = out_channels
        self.convLayers = nn.Sequential(*conv_layers_list)
        self.cnn_flat_feature_dim = current_channels * H_out * W_out
        logger.info(f"CNN Output: H={H_out}, W={W_out}, C={current_channels}, FlatDim={self.cnn_flat_feature_dim}")

        ############################################################
        # 2. MLP Encoder
        ############################################################
        config_cnn_flat_dim = config.get("last_convs", [self.cnn_flat_feature_dim])[0]
        if config_cnn_flat_dim != self.cnn_flat_feature_dim:
             logger.warning(f"Config 'last_convs' ({config_cnn_flat_dim}) mismatch with calculated CNN flat dim ({self.cnn_flat_feature_dim}). Using calculated value.")
        mlp_encoder_input_dim = self.cnn_flat_feature_dim

        if len(mlp_encoder_dims) != mlp_encoder_layers_count:
            raise ValueError(f"MLP Encoder 'encoder_dims' length ({len(mlp_encoder_dims)}) must match 'encoder_layers' count ({mlp_encoder_layers_count}).")
        mlp_encoder_full_dims = [mlp_encoder_input_dim] + mlp_encoder_dims

        mlp_encoder_list = []
        logger.info("Building MLP Encoder:")
        for i in range(mlp_encoder_layers_count):
            in_dim, out_dim = mlp_encoder_full_dims[i], mlp_encoder_full_dims[i+1]
            logger.info(f"  Layer {i}: Linear({in_dim}, {out_dim}) -> ReLU")
            mlp_encoder_list.append(nn.Linear(in_dim, out_dim))
            mlp_encoder_list.append(nn.ReLU(inplace=False)) # Use inplace=False
        self.compressMLP = nn.Sequential(*mlp_encoder_list)
        self.action_mlp_input_dim = mlp_encoder_full_dims[-1]
        logger.info(f"MLP Encoder Output Dim (Action MLP input): {self.action_mlp_input_dim}")

        ############################################################
        # 3. MLP Action Policy
        ############################################################
        if action_mlp_layers_count == 1:
             action_mlp_full_dims = [self.action_mlp_input_dim, action_mlp_output_dim]
        elif action_mlp_layers_count > 1:
             action_mlp_hidden_dims = config.get("action_hidden_dims", [self.action_mlp_input_dim]) # Example hidden dim
             hidden_dim = action_mlp_hidden_dims[0]
             action_mlp_full_dims = [self.action_mlp_input_dim] + [hidden_dim] * (action_mlp_layers_count - 1) + [action_mlp_output_dim]
        else: raise ValueError("action_layers must be >= 1")

        action_mlp_list = []
        logger.info("Building Action MLP:")
        for i in range(action_mlp_layers_count):
            in_dim, out_dim = action_mlp_full_dims[i], action_mlp_full_dims[i+1]
            is_last_layer = (i == action_mlp_layers_count - 1)
            logger.info(f"  Layer {i}: Linear({in_dim}, {out_dim})" + ("" if is_last_layer else " -> ReLU"))
            action_mlp_list.append(nn.Linear(in_dim, out_dim))
            if not is_last_layer:
                action_mlp_list.append(nn.ReLU(inplace=False)) # Use inplace=False
        self.actionMLP = nn.Sequential(*action_mlp_list)
        logger.info(f"Action MLP Output Dim: {action_mlp_output_dim}")

        logger.info("Applying weight initialization...")
        self.apply(weights_init)
        logger.info("--- Framework Baseline Initialization Complete ---")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the baseline model (CNN -> MLP -> ActionMLP).
        Args:
            states (Tensor): FOV observations, shape [B, N, C, H, W]
        Returns:
            Tensor: Action logits, shape [B, N, A]
        """
        batch_size = states.shape[0]
        N = states.shape[1]
        if N != self.num_agents: logger.warning(f"Input N={N} != config num_agents={self.num_agents}")

        expected_state_shape_suffix = (self.cnn_input_channels, self.fov_size, self.fov_size)
        if states.shape[2:] != expected_state_shape_suffix:
            raise ValueError(f"Input states shape error. Expected suffix {expected_state_shape_suffix}, got {states.shape[2:]}")

        # 1. CNN Encoder
        # Reshape: (B, N, C, H, W) -> (B*N, C, H, W)
        states_reshaped = states.reshape(batch_size * N, self.cnn_input_channels, self.fov_size, self.fov_size)
        cnn_features = self.convLayers(states_reshaped)
        cnn_features_flat = cnn_features.view(batch_size * N, -1) # Shape: [B*N, cnn_flat_dim]
        if cnn_features_flat.shape[1] != self.cnn_flat_feature_dim:
             raise RuntimeError(f"CNN flattened feature dimension mismatch.")

        # 2. MLP Encoder
        encoded_features = self.compressMLP(cnn_features_flat) # Shape [B*N, action_mlp_input_dim]

        # 3. MLP Action Policy
        action_logits_flat = self.actionMLP(encoded_features) # Shape [B*N, num_actions]

        # Reshape back to per-agent logits: [B*N, num_actions] -> [B, N, num_actions]
        action_logits = action_logits_flat.view(batch_size, N, self.num_actions)

        return action_logits