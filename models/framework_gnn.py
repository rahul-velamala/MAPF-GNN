# File: models/framework_gnn.py
# (Revised: Dynamic GNN layer loading including ADC, other cleanups)

import torch
import torch.nn as nn
import numpy as np
import logging

# Import utility
from .networks.utils_weights import weights_init
# GNN layer class is dynamically imported based on config

logger = logging.getLogger(__name__)

class Network(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        required_keys = ["num_agents", "pad", "channels", "encoder_layers",
                         "encoder_dims", "graph_filters", "node_dims",
                         "action_layers"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config missing required key for Framework GNN: '{key}'")

        self.num_agents = int(config["num_agents"])
        self.num_actions = 5 # Assuming 5 actions (Idle, R, U, L, D)

        # --- Determine FOV size from config ---
        self.pad = int(config["pad"])
        if self.pad <= 0: raise ValueError("'pad' must be >= 1")
        self.fov_size = (self.pad * 2) - 1
        logger.info(f"Framework GNN: Using FOV size {self.fov_size}x{self.fov_size} (pad={self.pad})")

        # --- Model Configuration ---
        self.cnn_input_channels = 3 # FOV: (Obstacles/Agents, Goal, Self)
        cnn_channels = [self.cnn_input_channels] + config["channels"]
        num_conv_layers = len(cnn_channels) - 1
        cnn_strides = config.get("strides", [1] * num_conv_layers)
        cnn_kernels = config.get("kernels", [3] * num_conv_layers)
        cnn_paddings = config.get("paddings", [1] * num_conv_layers)

        if not (len(cnn_strides) == len(cnn_kernels) == len(cnn_paddings) == num_conv_layers):
            raise ValueError("Length of CNN 'strides'/'kernels'/'paddings' in config doesn't match number of layers implied by 'channels'.")

        mlp_encoder_layers_count = int(config["encoder_layers"])
        mlp_encoder_dims = config["encoder_dims"]

        gnn_K_values = config["graph_filters"] # Represents K (hops/taps/truncation)
        gnn_node_dims = config["node_dims"]
        self.num_gnn_layers = len(gnn_K_values)
        if len(gnn_node_dims) != self.num_gnn_layers:
            raise ValueError("Length of 'node_dims' must match length of 'graph_filters'.")
        msg_type = config.get("msg_type", "gcn").lower() # Default to GCN

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
            mlp_encoder_list.append(nn.ReLU(inplace=False))
        self.compressMLP = nn.Sequential(*mlp_encoder_list)
        self.gnn_input_feature_dim = mlp_encoder_full_dims[-1]
        logger.info(f"MLP Encoder Output Dim (GNN Input Dim): {self.gnn_input_feature_dim}")

        ############################################################
        # 3. GNN Layers (Dynamic Type Selection)
        ############################################################
        gnn_layers_list = []
        gnn_feature_dims = [self.gnn_input_feature_dim] + gnn_node_dims

        # --- Dynamically Import GNN Layer Class ---
        GNNLayerClass = None
        logger.info(f"Building GNN (Type: {msg_type}):")
        try:
            if msg_type == 'gcn':
                from .networks.gnn import GCNLayer
                GNNLayerClass = GCNLayer
                k_arg_name = "filter_number" # GCNLayer expects this arg name
            elif msg_type == 'message':
                from .networks.gnn import MessagePassingLayer
                GNNLayerClass = MessagePassingLayer
                k_arg_name = "filter_number" # MessagePassingLayer expects this arg name
            elif msg_type == 'adc':
                from .networks.adc_layer import ADCLayer
                GNNLayerClass = ADCLayer
                k_arg_name = "K" # ADCLayer expects 'K'
            else:
                raise ValueError(f"Unsupported msg_type in config: {msg_type}. Choose 'gcn', 'message', or 'adc'.")
        except ImportError as e:
             logger.error(f"Error importing GNN layer type '{msg_type}': {e}", exc_info=True)
             raise

        # --- Instantiate GNN Layers ---
        for i in range(self.num_gnn_layers):
            in_dim, out_dim = gnn_feature_dims[i], gnn_feature_dims[i+1]
            k_value = gnn_K_values[i] # This is K (hops/taps/truncation)

            layer_args = {
                "in_features": in_dim,
                "out_features": out_dim,
                k_arg_name: k_value, # Use the correct argument name for K
                "activation": None, # Activation applied separately
                "bias": True,
            }
            # Add ADC specific args if needed
            if GNNLayerClass == ADCLayer:
                 layer_args["initial_t"] = config.get("adc_initial_t", 1.0)
                 layer_args["train_t"] = config.get("adc_train_t", True)

            logger.info(f"  Layer {i}: {GNNLayerClass.__name__}(in={in_dim}, out={out_dim}, {k_arg_name}={k_value}) -> ReLU")
            gnn_layers_list.append( GNNLayerClass(**layer_args) )
            gnn_layers_list.append(nn.ReLU(inplace=False)) # Separate activation

        self.GNNLayers = nn.ModuleList(gnn_layers_list) # Use ModuleList
        self.action_mlp_input_dim = gnn_feature_dims[-1]
        logger.info(f"GNN Output Dim (Action MLP Input Dim): {self.action_mlp_input_dim}")

        ############################################################
        # 4. MLP Action Policy
        ############################################################
        if action_mlp_layers_count == 1:
             action_mlp_full_dims = [self.action_mlp_input_dim, action_mlp_output_dim]
        elif action_mlp_layers_count > 1:
             action_mlp_hidden_dims = config.get("action_hidden_dims", [self.action_mlp_input_dim])
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
                action_mlp_list.append(nn.ReLU(inplace=False))
        self.actionMLP = nn.Sequential(*action_mlp_list)
        logger.info(f"Action MLP Output Dim: {action_mlp_output_dim}")

        logger.info("Applying weight initialization...")
        self.apply(weights_init)
        logger.info("--- Framework GNN Initialization Complete ---")

    def forward(self, states: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN framework.
        Args:
            states (torch.Tensor): FOV observations, shape (B, N, C, H, W).
            gso (torch.Tensor): Graph Shift Operator (Adjacency), shape (B, N, N).
        Returns:
            torch.Tensor: Action logits, shape (B, N, A).
        """
        batch_size = states.shape[0]
        N = states.shape[1]
        # Allow dynamic N if needed, but log warning if different from config
        if N != self.num_agents:
            logger.warning(f"Input N ({N}) differs from config num_agents ({self.num_agents}). Using N={N}.")

        expected_state_shape_suffix = (self.cnn_input_channels, self.fov_size, self.fov_size)
        if states.shape[2:] != expected_state_shape_suffix: raise ValueError(f"Input states shape error")
        expected_gso_shape_suffix = (N, N)
        if gso.shape[1:] != expected_gso_shape_suffix: raise ValueError(f"Input GSO shape error")
        if gso.shape[0] != batch_size:
             if gso.shape[0] == 1 and batch_size > 1: gso = gso.expand(batch_size, -1, -1)
             else: raise ValueError(f"Batch size mismatch states vs GSO")

        # 1. CNN Encoder
        states_reshaped = states.reshape(batch_size * N, self.cnn_input_channels, self.fov_size, self.fov_size)
        cnn_features = self.convLayers(states_reshaped)
        cnn_features_flat = cnn_features.view(batch_size * N, -1)
        if cnn_features_flat.shape[1] != self.cnn_flat_feature_dim: raise RuntimeError(f"CNN flatten dim error")

        # 2. MLP Encoder
        encoded_features = self.compressMLP(cnn_features_flat)

        # 3. GNN Layers
        gnn_input_features = encoded_features.view(batch_size, N, self.gnn_input_feature_dim).permute(0, 2, 1)
        gnn_output_features = gnn_input_features
        num_gnn_modules = len(self.GNNLayers)
        for i in range(0, num_gnn_modules, 2):
             gnn_layer_op = self.GNNLayers[i]
             activation_op = self.GNNLayers[i+1]
             if hasattr(gnn_layer_op, 'addGSO'): gnn_layer_op.addGSO(gso)
             gnn_output_features = gnn_layer_op(gnn_output_features)
             gnn_output_features = activation_op(gnn_output_features)

        # 4. Action MLP
        action_mlp_input = gnn_output_features.permute(0, 2, 1)
        action_mlp_input_flat = action_mlp_input.reshape(batch_size * N, self.action_mlp_input_dim)
        action_logits_flat = self.actionMLP(action_mlp_input_flat)
        action_logits = action_logits_flat.view(batch_size, N, self.num_actions)

        return action_logits
