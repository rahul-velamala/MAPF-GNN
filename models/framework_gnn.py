# File: models/framework_gnn.py
# (Complete Code - Revised for Shape Consistency, Dynamic GNN Type, Corrected CNN/MLP dims, ReLU inplace=False, cnn_input_channels fix)

import torch
import torch.nn as nn
import numpy as np

# Import necessary layer types
from .networks.utils_weights import weights_init
# GNN layer is dynamically imported based on config

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_agents = self.config["num_agents"]
        self.num_actions = 5 # Assuming 5 actions (Idle, R, U, L, D)

        # --- Determine FOV size from config ---
        pad = self.config.get("pad")
        if pad is None:
            sensing_range = self.config.get("sensing_range")
            if sensing_range is not None:
                 pad = int(np.ceil(sensing_range)) + 1
                 print(f"Framework WARNING: 'pad' not in config. Inferring pad={pad} from sensing_range={sensing_range}. Ensure this matches env!")
            else:
                 pad = 3 # Default pad for 5x5 FOV
                 print(f"Framework WARNING: 'pad' and 'sensing_range' not in config. Using default pad={pad} (5x5 FOV). Ensure this matches env!")
        self.pad = pad
        self.fov_size = (self.pad * 2) - 1
        print(f"Framework GNN: Using FOV size {self.fov_size}x{self.fov_size} (pad={self.pad})")

        # --- Model Configuration ---
        # CNN Encoder
        self.cnn_input_channels = 3 # Make it an attribute. Based on GraphEnv FOV: (Obstacles/Agents, Goal, Self)
        cnn_channels = [self.cnn_input_channels] + self.config.get("channels", [16, 16, 16])
        num_conv_layers = len(cnn_channels) - 1
        cnn_strides = self.config.get("strides", [1] * num_conv_layers)
        cnn_kernels = self.config.get("kernels", [3] * num_conv_layers)
        cnn_paddings = self.config.get("paddings", [1] * num_conv_layers)

        if len(cnn_strides) != num_conv_layers or len(cnn_kernels) != num_conv_layers or len(cnn_paddings) != num_conv_layers:
            raise ValueError("Length of 'strides', 'kernels', or 'paddings' in config does not match number of CNN layers implied by 'channels'.")

        # MLP Encoder (after CNN)
        mlp_encoder_layers_count = self.config.get("encoder_layers", 1)
        mlp_encoder_dims = self.config.get("encoder_dims", [64])

        # GNN
        gnn_filter_taps = self.config.get("graph_filters", [3])
        gnn_node_dims = self.config.get("node_dims", [128])
        self.num_gnn_layers = len(gnn_filter_taps)
        if len(gnn_node_dims) != self.num_gnn_layers:
            raise ValueError("Length of 'node_dims' must match length of 'graph_filters' (number of GNN layers).")
        msg_type = self.config.get("msg_type", "gcn").lower()

        # Action MLP (after GNN)
        action_mlp_layers_count = self.config.get("action_layers", 1)
        action_mlp_output_dim = self.num_actions


        ############################################################
        # 1. CNN Encoder -> Extracts features from FOV
        ############################################################
        conv_layers_list = []
        H_out, W_out = self.fov_size, self.fov_size
        current_channels = self.cnn_input_channels # Use attribute
        print("Building CNN:")
        for i in range(num_conv_layers):
            out_channels = cnn_channels[i+1]
            kernel = cnn_kernels[i]
            stride = cnn_strides[i]
            padding = cnn_paddings[i]
            print(f"  Layer {i}: Conv2d({current_channels}, {out_channels}, kernel={kernel}, stride={stride}, padding={padding}) -> ReLU")

            conv_layers_list.append(nn.Conv2d(current_channels, out_channels, kernel, stride, padding, bias=True))
            # Optional: Add BatchNorm here if desired
            # conv_layers_list.append(nn.BatchNorm2d(out_channels))
            conv_layers_list.append(nn.ReLU(inplace=False)) # Use inplace=False

            # Calculate output dimensions after this layer
            H_out = int((H_out - kernel + 2 * padding) / stride) + 1
            W_out = int((W_out - kernel + 2 * padding) / stride) + 1
            current_channels = out_channels

        self.convLayers = nn.Sequential(*conv_layers_list)
        self.cnn_flat_feature_dim = current_channels * H_out * W_out
        print(f"CNN Output: H={H_out}, W={W_out}, Channels={current_channels}, FlatDim={self.cnn_flat_feature_dim}")

        ############################################################
        # 2. MLP Encoder -> Compresses CNN features per agent
        ############################################################
        config_cnn_flat_dim = self.config.get("last_convs", [self.cnn_flat_feature_dim])[0]
        if config_cnn_flat_dim != self.cnn_flat_feature_dim:
             print(f"Framework WARNING: Config 'last_convs' ({config_cnn_flat_dim}) mismatch with calculated CNN flat dim ({self.cnn_flat_feature_dim}). Using calculated value.")
        mlp_encoder_input_dim = self.cnn_flat_feature_dim

        mlp_encoder_full_dims = [mlp_encoder_input_dim] + mlp_encoder_dims
        if len(mlp_encoder_full_dims) != mlp_encoder_layers_count + 1:
             if mlp_encoder_layers_count == 1 and len(mlp_encoder_dims) == 1:
                  mlp_encoder_full_dims = [mlp_encoder_input_dim] + mlp_encoder_dims
             else:
                  raise ValueError(f"MLP Encoder 'encoder_dims' length mismatch. Check 'encoder_layers' ({mlp_encoder_layers_count}) and 'encoder_dims' ({mlp_encoder_dims}).")

        mlp_encoder_list = []
        print("Building MLP Encoder:")
        for i in range(mlp_encoder_layers_count):
            in_dim = mlp_encoder_full_dims[i]
            out_dim = mlp_encoder_full_dims[i+1]
            print(f"  Layer {i}: Linear({in_dim}, {out_dim}) -> ReLU")
            mlp_encoder_list.append(nn.Linear(in_dim, out_dim))
            # Assuming ReLU after every Linear layer in the encoder MLP
            mlp_encoder_list.append(nn.ReLU(inplace=False)) # Use inplace=False

        self.compressMLP = nn.Sequential(*mlp_encoder_list)
        self.gnn_input_feature_dim = mlp_encoder_full_dims[-1]
        print(f"MLP Encoder Output Dim (GNN Input Dim): {self.gnn_input_feature_dim}")

        ############################################################
        # 3. GNN Layers -> Share information between agents
        ############################################################
        gnn_layers_list = []
        gnn_feature_dims = [self.gnn_input_feature_dim] + gnn_node_dims

        # --- Select GNN Layer Class ---
        GNNLayerClass = None
        print(f"Building GNN (Type: {msg_type}):")
        try:
            if msg_type == 'gcn':
                from .networks.gnn import GCNLayer
                GNNLayerClass = GCNLayer
            elif msg_type == 'message':
                from .networks.gnn import MessagePassingLayer
                GNNLayerClass = MessagePassingLayer
            else:
                raise ValueError(f"Unsupported msg_type in config: {msg_type}. Choose 'gcn' or 'message'.")
        except ImportError as e:
             print(f"Error importing GNN layer type '{msg_type}': {e}")
             raise

        for i in range(self.num_gnn_layers):
            in_dim = gnn_feature_dims[i]
            out_dim = gnn_feature_dims[i+1]
            filter_taps_k = gnn_filter_taps[i]
            print(f"  Layer {i}: {GNNLayerClass.__name__}(in={in_dim}, out={out_dim}, K={filter_taps_k}) -> ReLU")

            gnn_layers_list.append(
                GNNLayerClass(
                    n_nodes=self.num_agents,
                    in_features=in_dim,
                    out_features=out_dim,
                    filter_number=filter_taps_k,
                    activation=None,
                    bias=True,
                )
            )
            gnn_layers_list.append(nn.ReLU(inplace=False)) # Use inplace=False

        self.GNNLayers = nn.Sequential(*gnn_layers_list)
        self.action_mlp_input_dim = gnn_feature_dims[-1]
        print(f"GNN Output Dim (Action MLP Input Dim): {self.action_mlp_input_dim}")

        ############################################################
        # 4. MLP Action Policy -> Produces action logits per agent
        ############################################################
        # Construct dimensions list based on layer count
        action_mlp_hidden_dims = self.config.get("action_hidden_dims", [64]) # Example hidden dim
        if action_mlp_layers_count == 1:
             action_mlp_full_dims = [self.action_mlp_input_dim, action_mlp_output_dim]
        elif action_mlp_layers_count > 1:
             # Assuming single hidden dim used for all hidden layers if multiple layers specified
             if len(action_mlp_hidden_dims) != 1: print("Warning: Using first dimension from 'action_hidden_dims' for all hidden layers.")
             hidden_dim = action_mlp_hidden_dims[0]
             action_mlp_full_dims = [self.action_mlp_input_dim] + [hidden_dim] * (action_mlp_layers_count - 1) + [action_mlp_output_dim]
        else:
             raise ValueError("action_layers must be >= 1")

        action_mlp_list = []
        print("Building Action MLP:")
        for i in range(action_mlp_layers_count):
            in_dim = action_mlp_full_dims[i]
            out_dim = action_mlp_full_dims[i+1]
            print(f"  Layer {i}: Linear({in_dim}, {out_dim})" + (" -> ReLU" if i < action_mlp_layers_count - 1 else ""))
            action_mlp_list.append(nn.Linear(in_dim, out_dim))
            if i < action_mlp_layers_count - 1:
                action_mlp_list.append(nn.ReLU(inplace=False)) # Use inplace=False

        self.actionMLP = nn.Sequential(*action_mlp_list)
        print(f"Action MLP Output Dim: {action_mlp_output_dim}")

        # Initialize weights using the specified function
        print("Applying weight initialization...")
        self.apply(weights_init)
        print("--- Framework GNN Initialization Complete ---")

    def forward(self, states, gso):
        """
        Forward pass through the GNN framework.
        Args:
            states (torch.Tensor): FOV observations, shape (B, N, C, H, W).
            gso (torch.Tensor): Graph Shift Operator (Adjacency), shape (B, N, N).
        Returns:
            torch.Tensor: Action logits, shape (B, N, A).
        """
        batch_size = states.shape[0]
        # --- Input Shape Validation ---
        expected_state_shape = (self.num_agents, self.cnn_input_channels, self.fov_size, self.fov_size)
        if states.shape[1:] != expected_state_shape:
             raise ValueError(f"Input states shape error. Expected (B, {self.num_agents}, {self.cnn_input_channels}, {self.fov_size}, {self.fov_size}), Got {states.shape}")
        expected_gso_shape = (self.num_agents, self.num_agents)
        if gso.shape[1:] != expected_gso_shape:
             if batch_size == 1 and gso.shape == expected_gso_shape:
                  gso = gso.unsqueeze(0)
             else:
                  raise ValueError(f"Input GSO shape error. Expected (B, {self.num_agents}, {self.num_agents}) or ({self.num_agents}, {self.num_agents}) if B=1, Got {gso.shape}")
        if gso.shape[0] != batch_size:
             if gso.shape[0] == 1 and batch_size > 1:
                  gso = gso.expand(batch_size, -1, -1)
             else:
                  raise ValueError(f"Batch size mismatch between states ({batch_size}) and GSO ({gso.shape[0]}).")


        # 1. CNN Encoder: Process each agent's FOV independently
        # Reshape: (B, N, C, H, W) -> (B*N, C, H, W)
        states_reshaped = states.reshape(batch_size * self.num_agents, self.cnn_input_channels, self.fov_size, self.fov_size)
        cnn_features = self.convLayers(states_reshaped)
        # Flatten: (B*N, C_out, H_out, W_out) -> (B*N, C_out*H_out*W_out)
        cnn_features_flat = cnn_features.view(batch_size * self.num_agents, -1)
        # Check flattened size
        if cnn_features_flat.shape[1] != self.cnn_flat_feature_dim:
             raise RuntimeError(f"CNN flattened feature dimension mismatch. Expected {self.cnn_flat_feature_dim}, got {cnn_features_flat.shape[1]}")

        # 2. MLP Encoder: Compress features per agent
        # Input: (B*N, cnn_flat_dim), Output: (B*N, gnn_input_dim)
        encoded_features = self.compressMLP(cnn_features_flat)

        # 3. GNN Layers: Share features across agents
        # Reshape for GNN: (B*N, gnn_input_dim) -> (B, N, gnn_input_dim) -> (B, gnn_input_dim, N)
        gnn_input_features = encoded_features.view(batch_size, self.num_agents, self.gnn_input_feature_dim).permute(0, 2, 1)

        # Iterate through GNN layers (each layer is GNN op + ReLU activation)
        gnn_output_features = gnn_input_features
        for i in range(self.num_gnn_layers):
             gnn_layer_index = i * 2          # Index of the GNN layer
             activation_index = gnn_layer_index + 1 # Index of the ReLU activation

             gnn_layer_op = self.GNNLayers[gnn_layer_index]
             activation_op = self.GNNLayers[activation_index]

             # Add GSO to the layer state *before* calling its forward pass
             # This assumes GNNLayerClass has an addGSO method
             if hasattr(gnn_layer_op, 'addGSO'):
                 gnn_layer_op.addGSO(gso)
             else:
                 # Should not happen with the provided GCNLayer/MessagePassingLayer
                 print(f"Warning: GNN layer {type(gnn_layer_op).__name__} does not have addGSO method.")

             # Apply GNN layer and then activation
             gnn_output_features = gnn_layer_op(gnn_output_features)
             gnn_output_features = activation_op(gnn_output_features) # Apply ReLU

        # 4. Action MLP: Generate action logits per agent
        # Reshape GNN output for MLP: (B, gnn_output_dim, N) -> (B, N, gnn_output_dim)
        action_mlp_input = gnn_output_features.permute(0, 2, 1)
        # Flatten for MLP: (B, N, gnn_output_dim) -> (B*N, gnn_output_dim)
        action_mlp_input_flat = action_mlp_input.reshape(batch_size * self.num_agents, self.action_mlp_input_dim)

        # Apply Action MLP: (B*N, gnn_output_dim) -> (B*N, num_actions)
        action_logits_flat = self.actionMLP(action_mlp_input_flat)

        # Reshape back to per-agent logits: (B*N, num_actions) -> (B, N, num_actions)
        action_logits = action_logits_flat.view(batch_size, self.num_agents, self.num_actions)

        return action_logits