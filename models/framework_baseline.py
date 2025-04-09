# File: models/framework_baseline.py
# (Modified: Removed inplace=True from ReLUs)

import torch
import torch.nn as nn
from .networks.utils_weights import weights_init
from copy import copy
import numpy as np

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_agents = self.config["num_agents"] # Not strictly needed for forward, but good for consistency

        # --- Determine FOV size ---
        pad = self.config.get("pad", 3)
        self.fov_size = (pad * 2) - 1
        print(f"Framework Baseline: Using FOV size {self.fov_size}x{self.fov_size} (pad={pad})")

        self.num_actions = 5

        # --- MLP Encoder Config ---
        dim_encoder_mlp = self.config["encoder_layers"]
        self.compress_Features_dim = self.config["encoder_dims"]

        # --- Action MLP Config ---
        dim_action_mlp = self.config["action_layers"]
        action_features_out = [self.num_actions]

        ############################################################
        # CNN Encoder (Input: Agent FOV)
        ############################################################
        cnn_input_channels = 3 # Expects: Obstacles/Agents, Goal, Self
        channels = [cnn_input_channels] + self.config["channels"]
        num_conv_layers = len(channels) - 1
        strides = self.config.get("strides", [1] * num_conv_layers)
        padding_size = self.config.get("padding", [1] * num_conv_layers)
        filter_taps = self.config.get("kernels", [3] * num_conv_layers)

        if not (len(strides) == len(padding_size) == len(filter_taps) == num_conv_layers):
             raise ValueError("CNN layer parameter lengths mismatch")

        conv_layers = []
        H_out, W_out = self.fov_size, self.fov_size
        for l in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(channels[l], channels[l + 1], filter_taps[l], strides[l], padding_size[l], bias=True))
            conv_layers.append(nn.BatchNorm2d(num_features=channels[l + 1]))
            # --- MODIFICATION: Removed inplace=True ---
            conv_layers.append(nn.ReLU(inplace=False))
            H_out = int((H_out + 2 * padding_size[l] - filter_taps[l]) / strides[l]) + 1
            W_out = int((W_out + 2 * padding_size[l] - filter_taps[l]) / strides[l]) + 1

        self.convLayers = nn.Sequential(*conv_layers)
        self.cnn_flat_feature_dim = channels[-1] * H_out * W_out
        print(f"Framework Baseline: CNN output feature dim calculated: {self.cnn_flat_feature_dim}")

        ############################################################
        # MLP Encoder (Input: Flattened CNN Features)
        ############################################################
        mlp_encoder_input_dim_config = self.config.get("last_convs", [self.cnn_flat_feature_dim])[0]
        if mlp_encoder_input_dim_config != self.cnn_flat_feature_dim:
            print(f"Framework Baseline WARNING: Config 'last_convs' mismatch. Using calculated value.")
        mlp_encoder_input_dim = self.cnn_flat_feature_dim

        mlp_encoder_dims = [mlp_encoder_input_dim] + self.compress_Features_dim

        mlp_encoder_layers = []
        for l in range(dim_encoder_mlp):
            mlp_encoder_layers.append(nn.Linear(mlp_encoder_dims[l], mlp_encoder_dims[l+1]))
            # --- MODIFICATION: Removed inplace=True ---
            mlp_encoder_layers.append(nn.ReLU(inplace=False)) # Apply activation

        self.compressMLP = nn.Sequential(*mlp_encoder_layers)
        # Feature dimension after MLP encoder (input to Action MLP)
        self.action_mlp_input_dim = mlp_encoder_dims[-1]
        print(f"Framework Baseline: MLP Encoder output dim (Action MLP input): {self.action_mlp_input_dim}")


        ############################################################
        # MLP Action Policy (Input: Features after MLP Encoder)
        ############################################################
        action_mlp_dims = [self.action_mlp_input_dim] + action_features_out

        action_mlp_layers = []
        for l in range(dim_action_mlp):
            action_mlp_layers.append(nn.Linear(action_mlp_dims[l], action_mlp_dims[l+1]))
            if l < dim_action_mlp - 1: # Add ReLU except after last layer
                # --- MODIFICATION: Removed inplace=True ---
                action_mlp_layers.append(nn.ReLU(inplace=False))

        self.actionMLP = nn.Sequential(*action_mlp_layers)

        # Initialize weights
        self.apply(weights_init)
        print("Framework Baseline: Model initialized.")


    def forward(self, states): # Baseline doesn't use GSO
        """
        Forward pass for the baseline model (CNN -> MLP -> ActionMLP).
        Args:
            states (Tensor): FOV observations, shape [B, N, C, H, W]
        Returns:
            Tensor: Action logits, shape [B, N, A]
        """
        batch_size = states.shape[0]
        expected_state_shape = (self.num_agents, 3, self.fov_size, self.fov_size)
        if states.shape[1:] != expected_state_shape:
            raise ValueError(f"Input states shape error. Expected [B, {self.num_agents}, 3, {self.fov_size}, {self.fov_size}], "
                             f"Got {states.shape}")

        # 1. CNN Encoder
        states_reshaped = states.reshape(batch_size * self.num_agents, 3, self.fov_size, self.fov_size)
        cnn_features = self.convLayers(states_reshaped)
        cnn_features_flat = cnn_features.view(batch_size * self.num_agents, -1)

        # 2. MLP Encoder
        encoded_features = self.compressMLP(cnn_features_flat) # Shape [B*N, action_mlp_input_dim]

        # 3. MLP Action Policy
        action_logits_flat = self.actionMLP(encoded_features) # Shape [B*N, num_actions]

        # Reshape back to per-agent logits: [B*N, num_actions] -> [B, N, num_actions]
        action_logits = action_logits_flat.view(batch_size, self.num_agents, self.num_actions)

        return action_logits