# File: models/networks/gnn.py
# (Incorporating previous fixes for normalization, self-loops, device handling)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Removed sqrtm import
# from scipy.linalg import sqrtm
# from scipy.special import softmax # Unused?
import math
from copy import copy


class GCNLayer(nn.Module):
    def __init__(
        self,
        n_nodes, # Can be dynamic, usually determined in forward pass
        in_features,
        out_features,
        filter_number, # K (number of filter taps / hops)
        bias=True,     # Usually True for GCN layers
        activation=None, # Activation applied *after* the layer in the framework
        name="GCN_Layer",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number # K
        # Weight matrix: combines features across K hops
        # Shape: [InFeatures * K, OutFeatures] for efficient computation later
        self.W = nn.parameter.Parameter(
            torch.Tensor(self.in_features * self.filter_number, self.out_features)
        )
        if bias:
            self.b = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("b", None) # Correct way to register no bias
        self.activation = activation # Store activation if needed (though applied outside)
        self.name = name
        self._current_gso = None # To store the GSO set by addGSO

        self.init_params()

    def init_params(self):
        # Xavier initialization is common for GCNs
        nn.init.xavier_uniform_(self.W.data)
        if self.b is not None:
             # Initialize bias to zero or small constant
             nn.init.zeros_(self.b.data)

    def extra_repr(self):
        # Use registered self.b check
        reprString = (
            "in_features=%d, out_features=%d, " % (self.in_features, self.out_features)
            + "filter_taps=%d, " % (self.filter_number)
            + "bias=%s" % (self.b is not None)
        )
        return reprString

    def addGSO(self, GSO):
        """Stores the Graph Shift Operator (Adjacency Matrix) for the forward pass."""
        # GSO shape expected: [B, N, N]
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
        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3:
             raise ValueError(f"Expected node_feats dim 3 (B, F_in, N), got {node_feats.ndim}")
        batch_size, F_in, n_nodes = node_feats.shape # N = number of nodes
        if F_in != self.in_features:
             raise ValueError(f"Input feature dimension mismatch. Expected {self.in_features}, got {F_in}")

        # --- Adjacency Matrix Check & Preparation ---
        if self._current_gso is None:
            raise RuntimeError("Adjacency matrix (GSO) has not been set. Call addGSO() before forward.")
        adj_matrix = self._current_gso # Use the stored GSO
        if adj_matrix.device != input_device:
             # This can happen if GSO wasn't moved correctly before addGSO
             # print(f"Warning: GCNLayer GSO device ({adj_matrix.device}) differs from input device ({input_device}). Moving GSO.")
             adj_matrix = adj_matrix.to(input_device)
        # Ensure adj_matrix has correct dimensions [B, N, N]
        if adj_matrix.shape != (batch_size, n_nodes, n_nodes):
            raise ValueError(f"GSO shape mismatch. Expected ({batch_size}, {n_nodes}, {n_nodes}), Got {adj_matrix.shape}")

        # === Add self-loops (Identity Matrix) for GCN ===
        identity = torch.eye(n_nodes, device=input_device, dtype=adj_matrix.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, n_nodes, n_nodes)
        adj_matrix_with_loops = adj_matrix + identity
        # === -------------------------------------- ===

        # --- Symmetrically Normalize Adjacency Matrix ---
        degree = adj_matrix_with_loops.sum(dim=2).clamp(min=1e-6) # Avoid div by zero
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt) # Shape: [B, N, N]
        adj_normalized = D_inv_sqrt @ adj_matrix_with_loops @ D_inv_sqrt # A_norm = D^-0.5 * A_hat * D^-0.5
        # --- ---------------------------------------- ---

        # --- K-hop Aggregation ---
        # Efficient implementation: Z = [X, A_norm*X, A_norm^2*X, ..., A_norm^(K-1)*X]
        # Then apply weight matrix W across all K features: Reshape Z and W.
        current_hop_feats = node_feats # Shape: [B, F_in, N] (Hop 0)
        z_hops = [current_hop_feats] # List to store features from each hop

        for k in range(1, self.filter_number): # Hops 1 to K-1
            # Aggregate features: A_norm * X (matmul needs shapes [B,N,N] @ [B,N,F] or [B,F,N] @ [B,N,N])
            # Current feats: [B, F_in, N]. Need to permute for matmul with A_norm [B, N, N]
            current_hop_feats_permuted = current_hop_feats.permute(0, 2, 1) # -> [B, N, F_in]
            # Matmul: [B, N, N] @ [B, N, F_in] -> [B, N, F_in]
            aggregated_feats = adj_normalized @ current_hop_feats_permuted
            # Permute back: [B, N, F_in] -> [B, F_in, N]
            current_hop_feats = aggregated_feats.permute(0, 2, 1)
            z_hops.append(current_hop_feats)

        # Concatenate features across hops: List of [B, F_in, N] -> [B, F_in * K, N]
        z = torch.cat(z_hops, dim=1) # Concatenate along the feature dimension

        # --- Linear Transformation ---
        # Reshape features for matmul with weights: [B, F_in * K, N] -> [B, N, F_in * K]
        z_permuted = z.permute(0, 2, 1)

        # Apply linear transformation: (B, N, F_in*K) @ (F_in*K, F_out) -> (B, N, F_out)
        # self.W shape is already [F_in*K, F_out]
        output_node_feats = z_permuted @ self.W
        # --- --------------------- ---

        # --- Add Bias ---
        if self.b is not None:
            # Bias shape: (F_out), needs broadcasting to (B, N, F_out)
            output_node_feats = output_node_feats + self.b # Broadcasting handles this

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = output_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
            # This shouldn't be needed if activation is applied in the Sequential container
            output_node_feats = self.activation(output_node_feats)
        # --- --------------------------------------------- ---

        # Reset stored GSO for next forward pass if stateful design is kept
        # self._current_gso = None # Optional: Reset if GSO must be added each time

        return output_node_feats


class MessagePassingLayer(nn.Module):
    """
    Basic Message Passing Layer (Sums messages from neighbors).
    Self features are transformed separately and added.
    Normalization is applied to the adjacency matrix before aggregation.
    """
    def __init__(
        self,
        n_nodes, # Dynamic
        in_features,
        out_features,
        filter_number, # K: Number of message passing rounds (usually 1 for basic MPNN)
                       # If K>1, it means messages are propagated K times before update.
        bias=True,
        activation=None, # Applied outside
        name="MP_Layer",
    ):
        super().__init__()
        if filter_number != 1:
             print("Warning: MessagePassingLayer with filter_number > 1 is non-standard. Usually K=1 for basic MPNN update.")
             # Consider if this implementation correctly handles K > 1 message steps.
             # For now, assume K=1: one round of message passing/aggregation.

        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number # K (interpreted as rounds of message passing)
        # Weight for transforming aggregated neighbor messages
        self.W_agg = nn.parameter.Parameter(torch.Tensor(self.in_features, self.out_features))
        # Weight for transforming node's own features
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
        nn.init.xavier_uniform_(self.W_agg.data)
        nn.init.xavier_uniform_(self.W_self.data)
        if self.bias is not None:
             nn.init.zeros_(self.bias.data)

    def extra_repr(self):
        reprString = (
            "in_features=%d, out_features=%d, " % (self.in_features, self.out_features)
            + "rounds(K)=%d, " % (self.filter_number) # Clarify K meaning
            + "bias=%s" % (self.bias is not None)
        )
        return reprString

    def addGSO(self, GSO):
        """Stores the Graph Shift Operator (Adjacency Matrix)."""
        self._current_gso = GSO

    def forward(self, node_feats):
        """
        Message passing forward pass (assuming K=1 round).
        Update rule: h_v' = ReLU( W_self * h_v + W_agg * sum(h_u for u in N(v)) + b )
                      (Using normalized adjacency for the sum part)

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N].

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        # --- Device/Input Checks ---
        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3: raise ValueError("Expected node_feats dim 3 (B, F_in, N)")
        batch_size, F_in, n_nodes = node_feats.shape
        if F_in != self.in_features: raise ValueError("Input feature dimension mismatch.")

        # --- Adjacency Matrix Check & Preparation ---
        if self._current_gso is None: raise RuntimeError("GSO has not been set.")
        adj_matrix = self._current_gso
        if adj_matrix.device != input_device: adj_matrix = adj_matrix.to(input_device)
        if adj_matrix.shape != (batch_size, n_nodes, n_nodes): raise ValueError("GSO shape mismatch.")

        # --- Normalization (No self-loops needed for basic message passing aggregation) ---
        # We only want to aggregate from neighbors (A), not A_hat = A+I
        degree = adj_matrix.sum(dim=2).clamp(min=1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        adj_normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt # Use A, not A+I
        # --- ------------------------------------------------------------------------- ---

        # --- Aggregate Neighbor Features (1 hop) ---
        # node_feats: [B, F_in, N] -> permute to [B, N, F_in] for matmul
        node_feats_permuted = node_feats.permute(0, 2, 1)
        # Matmul: [B, N, N] @ [B, N, F_in] -> [B, N, F_in] (Represents sum(h_u))
        aggregated_feats = adj_normalized @ node_feats_permuted
        # --- ------------------------------- ---

        # --- Apply Transformations ---
        # 1. Transform self features: (B, N, F_in) @ (F_in, F_out) -> (B, N, F_out)
        transformed_self = node_feats_permuted @ self.W_self

        # 2. Transform aggregated features: (B, N, F_in) @ (F_in, F_out) -> (B, N, F_out)
        transformed_agg = aggregated_feats @ self.W_agg

        # 3. Combine self and aggregated features
        updated_node_feats = transformed_self + transformed_agg
        # --- --------------------- ---

        # --- Add Bias ---
        if self.bias is not None:
            updated_node_feats = updated_node_feats + self.bias # Broadcasting [F_out] -> [B, N, F_out]

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = updated_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
            output_node_feats = self.activation(output_node_feats)

        return output_node_feats