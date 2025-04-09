# File: models/networks/gnn.py
# (Complete Code - Revised GCNLayer/MessagePassingLayer with correct normalization and aggregation)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import copy

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer based on Kipf & Welling (2017).
    Uses symmetric normalization: D^-0.5 * A_hat * D^-0.5 * X * W
    where A_hat = A + I. Aggregates features over K hops.
    """
    def __init__(
        self,
        n_nodes, # Usually determined dynamically in forward pass based on input N
        in_features,
        out_features,
        filter_number, # K (number of filter taps / hops)
        bias=True,
        activation=None, # Activation applied *after* the layer in the framework
        name="GCN_Layer",
    ):
        super().__init__()
        if filter_number < 1:
            raise ValueError("Filter number (K) must be at least 1.")
        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number # K
        # Weight matrix: Combines features across K hops.
        # Shape: [InFeatures * K, OutFeatures] for efficient computation later.
        self.W = nn.parameter.Parameter(
            torch.Tensor(self.in_features * self.filter_number, self.out_features)
        )
        if bias:
            self.b = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("b", None) # Correct way to register no bias
        self.activation = activation # Store activation if needed (though likely applied outside)
        self.name = name
        self._current_gso = None # To store the GSO set by addGSO

        self.init_params()

    def init_params(self):
        # Xavier initialization is common for GCNs
        nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('relu') if self.activation == F.relu else 1.0)
        if self.b is not None:
             # Initialize bias to zero or small constant
             nn.init.zeros_(self.b.data)

    def extra_repr(self):
        reprString = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"filter_taps(K)={self.filter_number}, bias={self.b is not None}"
        )
        return reprString

    def addGSO(self, GSO):
        """Stores the Graph Shift Operator (Adjacency Matrix) for the forward pass.
           Expects GSO shape [B, N, N].
        """
        if GSO is None or GSO.ndim != 3:
             raise ValueError("GSO must be a 3D tensor [B, N, N].")
        self._current_gso = GSO

    def forward(self, node_feats):
        """
        Processes graph features using GCN layer.
        Assumes self._current_gso has been set via addGSO().

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N].

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        # --- Device Handling & Input Checks ---
        if self._current_gso is None:
            raise RuntimeError("Adjacency matrix (GSO) has not been set. Call addGSO() before forward.")

        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3:
             raise ValueError(f"Expected node_feats dim 3 (B, F_in, N), got {node_feats.ndim}")
        batch_size, F_in, n_nodes = node_feats.shape # N = number of nodes
        if F_in != self.in_features:
             raise ValueError(f"Input feature dimension mismatch. Expected {self.in_features}, got {F_in}")

        # --- Adjacency Matrix Check & Preparation ---
        adj_matrix = self._current_gso
        if adj_matrix.device != input_device:
             # print(f"Warning: GCNLayer GSO device ({adj_matrix.device}) differs from input device ({input_device}). Moving GSO.")
             adj_matrix = adj_matrix.to(input_device)
        if adj_matrix.shape != (batch_size, n_nodes, n_nodes):
            raise ValueError(f"GSO shape mismatch. Expected ({batch_size}, {n_nodes}, {n_nodes}), Got {adj_matrix.shape}")

        # === Add self-loops (Identity Matrix) for GCN ===
        identity = torch.eye(n_nodes, device=input_device, dtype=adj_matrix.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, n_nodes, n_nodes)
        adj_matrix_with_loops = adj_matrix + identity # A_hat = A + I
        # === -------------------------------------- ===

        # --- Symmetrically Normalize Adjacency Matrix ---
        degree_hat = adj_matrix_with_loops.sum(dim=2).clamp(min=1e-6)
        degree_hat_inv_sqrt = torch.pow(degree_hat, -0.5)
        # Use the correct variable name here
        D_hat_inv_sqrt_diag = torch.diag_embed(degree_hat_inv_sqrt) # Shape: [B, N, N]
        # Use the correct variable name in the next line
        # Calculate normalized adjacency: A_norm = D_hat^-0.5 * A_hat * D_hat^-0.5
        adj_normalized = D_hat_inv_sqrt_diag @ adj_matrix_with_loops @ D_hat_inv_sqrt_diag # <--- CORRECTED
        # --- ---------------------------------------- ---

        # --- K-hop Aggregation ---
        current_hop_feats = node_feats # Shape: [B, F_in, N] (Hop 0, represents X)
        z_hops = [current_hop_feats] # List to store features from each hop [X, AX, A^2X, ...]

        if self.filter_number > 1:
            hop_features = current_hop_feats
            for k in range(1, self.filter_number):
                hop_features_permuted = hop_features.permute(0, 2, 1)
                aggregated_feats = adj_normalized @ hop_features_permuted
                hop_features = aggregated_feats.permute(0, 2, 1)
                z_hops.append(hop_features)

        z = torch.cat(z_hops, dim=1) # Concatenate along the feature dimension (dim=1)

        # --- Linear Transformation ---
        z_permuted = z.permute(0, 2, 1)
        output_node_feats = z_permuted @ self.W
        # --- --------------------- ---

        # --- Add Bias ---
        if self.b is not None:
            output_node_feats = output_node_feats + self.b

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = output_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
             output_node_feats = self.activation(output_node_feats)

        return output_node_feats


class MessagePassingLayer(nn.Module):
    """
    Basic Message Passing Layer (MPNN) Update.
    Aggregates normalized neighbor features and combines with transformed self-features.
    Typically assumes K=1 round of message passing per layer application.
    Update rule similar to: h_v' = Update(h_v, Aggregate(h_u for u in N(v)))
    Here: h_v' = ReLU( W_self * h_v + W_agg * sum_norm(h_u) + b )
    """
    def __init__(
        self,
        n_nodes, # Dynamic based on input N
        in_features,
        out_features,
        filter_number, # K: Typically 1 for standard MPNN layer. If > 1, interpretation needs care.
        bias=True,
        activation=None, # Applied outside
        name="MP_Layer",
    ):
        super().__init__()
        if filter_number > 1:
             print(f"Warning: MessagePassingLayer created with K={filter_number}. Standard MPNN update usually uses K=1 (one round of aggregation per layer call). Ensure framework handles K>1 correctly if intended.")

        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number # K (Nominally stored, but logic below assumes K=1 round)

        self.W_agg = nn.parameter.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.W_self = nn.parameter.Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.activation = activation
        self.name = name
        self._current_gso = None
        self.init_params()

    def init_params(self):
        gain = nn.init.calculate_gain('relu') if self.activation == F.relu else 1.0
        nn.init.xavier_uniform_(self.W_agg.data, gain=gain)
        nn.init.xavier_uniform_(self.W_self.data, gain=gain)
        if self.bias is not None:
             nn.init.zeros_(self.bias.data)

    def extra_repr(self):
        reprString = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rounds(K)={self.filter_number}, bias={self.bias is not None}"
        )
        return reprString

    def addGSO(self, GSO):
        """Stores the Graph Shift Operator (Adjacency Matrix). Expects [B, N, N]."""
        if GSO is None or GSO.ndim != 3:
             raise ValueError("GSO must be a 3D tensor [B, N, N].")
        self._current_gso = GSO

    def forward(self, node_feats):
        """
        Message passing forward pass (1 round).
        Update rule: h_v' = ReLU( W_self * h_v + W_agg * sum_norm(h_u for u in N(v)) + b )

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N].

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        # --- Device/Input Checks ---
        if self._current_gso is None: raise RuntimeError("GSO has not been set.")

        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3: raise ValueError("Expected node_feats dim 3 (B, F_in, N)")
        batch_size, F_in, n_nodes = node_feats.shape
        if F_in != self.in_features: raise ValueError("Input feature dimension mismatch.")

        # --- Adjacency Matrix Check & Preparation ---
        adj_matrix = self._current_gso
        if adj_matrix.device != input_device: adj_matrix = adj_matrix.to(input_device)
        if adj_matrix.shape != (batch_size, n_nodes, n_nodes): raise ValueError("GSO shape mismatch.")

        # --- Normalization (Symmetric normalization of A, *without* self-loops) ---
        degree = adj_matrix.sum(dim=2).clamp(min=1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt_diag = torch.diag_embed(degree_inv_sqrt)
        adj_normalized_neighbors = D_inv_sqrt_diag @ adj_matrix @ D_inv_sqrt_diag
        # --- ------------------------------------------------------------------- ---

        # --- Aggregate Neighbor Features (1 hop using A_norm_neighbors) ---
        node_feats_permuted = node_feats.permute(0, 2, 1)
        aggregated_neighbor_feats = adj_normalized_neighbors @ node_feats_permuted
        # --- ----------------------------------------------------------- ---

        # --- Apply Transformations ---
        transformed_self = node_feats_permuted @ self.W_self
        transformed_agg = aggregated_neighbor_feats @ self.W_agg
        updated_node_feats = transformed_self + transformed_agg
        # --- --------------------- ---

        # --- Add Bias ---
        if self.bias is not None:
            updated_node_feats = updated_node_feats + self.bias

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = updated_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
            output_node_feats = self.activation(output_node_feats)

        return output_node_feats