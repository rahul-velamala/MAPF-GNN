# File: models/networks/gnn.py
# (Revised GCNLayer/MessagePassingLayer with correct normalization and aggregation)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer based on Kipf & Welling (2017).
    Uses symmetric normalization: D_hat^-0.5 * A_hat * D_hat^-0.5 * X * W
    where A_hat = A + I. Aggregates features over K hops.
    """
    def __init__(
        self,
        # n_nodes is now determined dynamically in forward pass based on input N
        in_features: int,
        out_features: int,
        filter_number: int, # K (number of filter taps / hops)
        bias: bool = True,
        activation=None, # Activation applied *after* the layer in the framework
        name: str ="GCN_Layer",
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
        self._current_gso: torch.Tensor | None = None # To store the GSO set by addGSO

        self.init_params()
        logger.debug(f"Initialized {self.name}: In={in_features}, Out={out_features}, K={filter_number}")

    def init_params(self):
        """ Initializes layer parameters using Xavier uniform initialization. """
        # Xavier initialization is common for GCNs
        gain = nn.init.calculate_gain('relu') if self.activation == F.relu else 1.0
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        if self.b is not None:
             nn.init.zeros_(self.b.data)

    def extra_repr(self):
        """ String representation for print(model). """
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"filter_taps(K)={self.filter_number}, bias={self.b is not None}"
        )

    def addGSO(self, GSO: torch.Tensor):
        """Stores the Graph Shift Operator (Adjacency Matrix) for the forward pass.
           Expects GSO shape [B, N, N].
        """
        if GSO is None or GSO.ndim != 3:
             raise ValueError(f"GSO must be a 3D tensor [B, N, N], got shape {GSO.shape if GSO is not None else 'None'}")
        self._current_gso = GSO

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
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

        batch_size, F_in, n_nodes = node_feats.shape
        if F_in != self.in_features:
             raise ValueError(f"Input feature dimension mismatch. Expected {self.in_features}, got {F_in}")

        # --- Adjacency Matrix Check & Preparation ---
        adj_matrix = self._current_gso
        if adj_matrix.device != input_device:
             logger.debug(f"{self.name}: GSO device ({adj_matrix.device}) differs from input ({input_device}). Moving GSO.")
             adj_matrix = adj_matrix.to(device=input_device, dtype=input_dtype) # Ensure matching device and dtype
        if adj_matrix.shape != (batch_size, n_nodes, n_nodes):
            # Allow broadcasting if adj_matrix has B=1
            if adj_matrix.shape == (1, n_nodes, n_nodes) and batch_size > 1:
                adj_matrix = adj_matrix.expand(batch_size, -1, -1)
            else:
                raise ValueError(f"GSO shape mismatch. Expected ({batch_size}, {n_nodes}, {n_nodes}) or (1, {n_nodes}, {n_nodes}), Got {adj_matrix.shape}")

        # === Add self-loops (Identity Matrix) for GCN ===
        identity = torch.eye(n_nodes, device=input_device, dtype=adj_matrix.dtype)
        identity = identity.expand(batch_size, n_nodes, n_nodes) # Expand to batch size
        adj_matrix_with_loops = adj_matrix + identity # A_hat = A + I
        # === -------------------------------------- ===

        # --- Symmetrically Normalize Adjacency Matrix (A_hat_norm = D_hat^-0.5 * A_hat * D_hat^-0.5)---
        # Calculate degree D_hat from A_hat
        degree_hat = torch.sum(adj_matrix_with_loops, dim=2).clamp(min=1e-6) # Sum over columns for in-degree, clamp to avoid div by zero
        degree_hat_inv_sqrt = torch.pow(degree_hat, -0.5)
        D_hat_inv_sqrt_diag = torch.diag_embed(degree_hat_inv_sqrt) # Shape: [B, N, N]
        # Calculate normalized adjacency
        adj_normalized = D_hat_inv_sqrt_diag @ adj_matrix_with_loops @ D_hat_inv_sqrt_diag
        # --- ------------------------------------------------------------------------------------ ---

        # --- K-hop Aggregation ---
        # Note: node_feats shape is [B, F_in, N]
        current_hop_feats = node_feats
        z_hops = [current_hop_feats] # List to store features from each hop [X, A_norm*X, A_norm^2*X, ...]

        # Check if adj_normalized needs dtype conversion (should match node_feats)
        if adj_normalized.dtype != current_hop_feats.dtype:
            adj_normalized = adj_normalized.to(current_hop_feats.dtype)

        if self.filter_number > 1:
            # Store A_norm^k * X efficiently
            # Here, we apply A_norm to the features X. Since A_norm is [B, N, N] and X is [B, F_in, N],
            # we need to permute X or use batch matrix multiplication carefully.
            # Let's use bmm: X shape [B, F_in, N], A_norm shape [B, N, N]
            # Result = X @ A_norm.T ? No, it's A_norm @ X (operating on nodes)
            # We need to treat features as batch dim or iterate?
            # Easier: A_norm @ X.permute(0,2,1) -> [B, N, F_in], then permute back?
            # Or matmul X with A_norm? torch.matmul([B,F,N], [B,N,N]) -> [B, F, N] -- This is right!

            hop_features = current_hop_feats # X (shape B, F_in, N)
            for _ in range(1, self.filter_number):
                # hop_features = adj_normalized @ hop_features # Incorrect shape multiply
                hop_features = torch.matmul(hop_features, adj_normalized) # Correct: (B, F_in, N) @ (B, N, N) -> (B, F_in, N)
                z_hops.append(hop_features)

        # Concatenate features from all hops along the feature dimension (dim=1)
        # Result shape: [B, F_in * K, N]
        z = torch.cat(z_hops, dim=1)

        # --- Linear Transformation ---
        # We need to apply W which has shape [F_in * K, F_out]
        # Input z is [B, F_in * K, N]. Output should be [B, F_out, N]
        # We can do this with batched matmul: z.permute(0, 2, 1) @ W -> [B, N, F_out]
        z_permuted = z.permute(0, 2, 1) # Shape: [B, N, F_in * K]
        output_node_feats = torch.matmul(z_permuted, self.W) # Shape: [B, N, F_out]
        # --- --------------------- ---

        # --- Add Bias ---
        if self.b is not None:
            # Bias shape [F_out], needs broadcasting to [B, N, F_out]
            output_node_feats = output_node_feats + self.b # Broadcasting handles this

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = output_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
             logger.warning(f"{self.name}: Activation is set but usually applied outside the layer in this framework.")
             output_node_feats = self.activation(output_node_feats)

        return output_node_feats


class MessagePassingLayer(nn.Module):
    """
    Basic Message Passing Layer (MPNN) Update.
    Aggregates normalized neighbor features and combines with transformed self-features.
    Update rule similar to: h_v' = Update(h_v, Aggregate(h_u for u in N(v)))
    Here: h_v' = ReLU( W_self * h_v + W_agg * sum_norm(h_u) + b )
    Uses symmetric normalization of A (neighbor adjacency) for aggregation.
    Assumes K=1 round of message passing per layer application, regardless of filter_number value.
    """
    def __init__(
        self,
        # n_nodes is now dynamic
        in_features: int,
        out_features: int,
        filter_number: int, # K: Nominally stored, but logic below assumes K=1 round.
        bias: bool = True,
        activation=None, # Applied outside
        name: str ="MP_Layer",
    ):
        super().__init__()
        # filter_number (K) is stored but not directly used in the K=1 MPNN update logic below.
        # If K>1 was intended to mean multi-round MP, the framework's sequential application handles it.
        if filter_number > 1:
             logger.debug(f"{name}: Initialized with K={filter_number}. Logic performs 1 round; framework handles multiple rounds.")

        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number

        # Weight for aggregating neighbor messages
        self.W_agg = nn.parameter.Parameter(torch.Tensor(self.in_features, self.out_features))
        # Weight for transforming self features
        self.W_self = nn.parameter.Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.activation = activation
        self.name = name
        self._current_gso: torch.Tensor | None = None
        self.init_params()
        logger.debug(f"Initialized {self.name}: In={in_features}, Out={out_features}, K(nominal)={filter_number}")


    def init_params(self):
        """ Initializes layer parameters using Xavier uniform initialization. """
        gain = nn.init.calculate_gain('relu') if self.activation == F.relu else 1.0
        nn.init.xavier_uniform_(self.W_agg.data, gain=gain)
        nn.init.xavier_uniform_(self.W_self.data, gain=gain)
        if self.bias is not None:
             nn.init.zeros_(self.bias.data)

    def extra_repr(self):
        """ String representation for print(model). """
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rounds(nominal K)={self.filter_number}, bias={self.bias is not None}"
        )

    def addGSO(self, GSO: torch.Tensor):
        """Stores the Graph Shift Operator (Adjacency Matrix). Expects [B, N, N]."""
        if GSO is None or GSO.ndim != 3:
             raise ValueError(f"GSO must be a 3D tensor [B, N, N], got shape {GSO.shape if GSO is not None else 'None'}")
        self._current_gso = GSO

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Message passing forward pass (1 round).
        Update rule: h_v' = ReLU( W_self * h_v + W_agg * sum_norm(h_u for u in N(v)) + b )

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N].

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        # --- Device/Input Checks ---
        if self._current_gso is None: raise RuntimeError(f"{self.name}: GSO has not been set.")

        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3: raise ValueError(f"{self.name}: Expected node_feats dim 3 (B, F_in, N)")
        batch_size, F_in, n_nodes = node_feats.shape
        if F_in != self.in_features: raise ValueError(f"{self.name}: Input feature dimension mismatch.")

        # --- Adjacency Matrix Check & Preparation ---
        adj_matrix = self._current_gso # This is A (neighbors only, no self-loops needed for message passing)
        if adj_matrix.device != input_device:
            logger.debug(f"{self.name}: GSO device ({adj_matrix.device}) differs from input ({input_device}). Moving GSO.")
            adj_matrix = adj_matrix.to(device=input_device, dtype=input_dtype)
        if adj_matrix.shape != (batch_size, n_nodes, n_nodes):
             # Allow broadcasting if adj_matrix has B=1
            if adj_matrix.shape == (1, n_nodes, n_nodes) and batch_size > 1:
                adj_matrix = adj_matrix.expand(batch_size, -1, -1)
            else:
                raise ValueError(f"{self.name}: GSO shape mismatch. Expected ({batch_size}, {n_nodes}, {n_nodes}) or (1, {n_nodes}, {n_nodes}), Got {adj_matrix.shape}")

        # --- Normalization (Symmetric normalization of A, *without* self-loops) ---
        # D_ii = sum_j A_ij
        degree = torch.sum(adj_matrix, dim=2).clamp(min=1e-6) # Use original A degree
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt_diag = torch.diag_embed(degree_inv_sqrt)
        # A_norm = D^-0.5 * A * D^-0.5
        adj_normalized_neighbors = D_inv_sqrt_diag @ adj_matrix @ D_inv_sqrt_diag
        # --- ------------------------------------------------------------------- ---

        # --- Aggregate Neighbor Features (1 hop using A_norm_neighbors) ---
        # node_feats is [B, F_in, N], adj_norm is [B, N, N]
        # aggregated = matmul(node_feats, adj_norm) -> [B, F_in, N]
        # Ensure matching dtypes
        if adj_normalized_neighbors.dtype != node_feats.dtype:
            adj_normalized_neighbors = adj_normalized_neighbors.to(node_feats.dtype)

        aggregated_neighbor_feats = torch.matmul(node_feats, adj_normalized_neighbors) # Shape: [B, F_in, N]
        # --- ----------------------------------------------------------- ---

        # --- Apply Transformations ---
        # Need shapes like [B, N, F_in] to multiply with W [F_in, F_out] -> [B, N, F_out]
        node_feats_permuted = node_feats.permute(0, 2, 1)               # Shape: [B, N, F_in]
        aggregated_neighbor_feats_permuted = aggregated_neighbor_feats.permute(0, 2, 1) # Shape: [B, N, F_in]

        transformed_self = torch.matmul(node_feats_permuted, self.W_self) # Shape: [B, N, F_out]
        transformed_agg = torch.matmul(aggregated_neighbor_feats_permuted, self.W_agg) # Shape: [B, N, F_out]

        # Combine self and aggregated features
        updated_node_feats = transformed_self + transformed_agg
        # --- --------------------- ---

        # --- Add Bias ---
        if self.bias is not None:
            # Bias shape [F_out], broadcast to [B, N, F_out]
            updated_node_feats = updated_node_feats + self.bias

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = updated_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
            logger.warning(f"{self.name}: Activation is set but usually applied outside the layer in this framework.")
            output_node_feats = self.activation(output_node_feats)

        return output_node_feats