# File: models/networks/gnn.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
# from scipy.special import softmax # Unused?
import math
from copy import copy


class GCNLayer(nn.Module):
    def __init__(
        self,
        n_nodes, # Note: n_nodes might vary per batch if graphs are different sizes; usually derived dynamically
        in_features,
        out_features,
        filter_number,
        bias=False,
        activation=None,
        name="GCN_Layer",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number
        # Consider LazyLinear if input features/nodes aren't fixed beforehand
        self.W = nn.parameter.Parameter(
            torch.Tensor(self.in_features, self.filter_number, self.out_features)
        )
        if bias:
            self.b = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None) # Correct way to register no bias
            # self.b = None # No need for this instance variable if registered as None
        self.activation = activation
        self.name = name
        # self.n_nodes = n_nodes # Store n_nodes if fixed, otherwise get from input
        self.init_params()

    def init_params(self):
        # Consider Kaiming or Xavier initialization for better convergence
        stdv = 1.0 / math.sqrt(self.W.size(0) * self.W.size(1)) # Use actual W dimensions
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None: # Check registered parameter
            # stdv_b = 1.0 / math.sqrt(self.out_features) # Bias init can differ
            self.bias.data.uniform_(-stdv, stdv) # Use self.bias

    def extra_repr(self):
        # Use registered self.bias check
        reprString = (
            "in_features=%d, out_features=%d, " % (self.in_features, self.out_features)
            + "filter_taps=%d, " % (self.filter_number)
            + "bias=%s" % (self.bias is not None) # Check registered parameter
        )
        return reprString

    def addGSO(self, GSO):
        # This method might be redundant if GSO is passed directly to forward
        # If it's stateful, be careful about batching if GSO changes per batch
        self.adj_matrix = GSO

    def forward(self, node_feats):
        """
        Processes graph features using GCN layer.
        Assumes self.adj_matrix has been set or is passed appropriately.

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N] or [B, N, F_in].
                                       Let's assume [B, F_in, N] based on original code's reshape.

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        # --- Device Handling ---
        # Infer device and dtype from input features for robustness
        input_device = node_feats.device
        input_dtype = node_feats.dtype

        # --- Dynamic Node Count ---
        # Get n_nodes dynamically from input shape (assuming B, F_in, N)
        if node_feats.ndim != 3:
             raise ValueError(f"Expected node_feats dim 3 (B, F, N), got {node_feats.ndim}")
        batch_size = node_feats.shape[0]
        self.n_nodes = node_feats.shape[2] # N = number of nodes

        # --- Adjacency Matrix Check ---
        if not hasattr(self, 'adj_matrix') or self.adj_matrix is None:
            raise RuntimeError("Adjacency matrix (GSO) has not been set for GCNLayer. Call addGSO or pass it.")
        if self.adj_matrix.device != input_device:
             # This shouldn't happen if train.py moves GSO correctly, but good check
             print(f"Warning: GCNLayer adj_matrix device ({self.adj_matrix.device}) differs from input device ({input_device}). Moving adj_matrix.")
             self.adj_matrix = self.adj_matrix.to(input_device)
        # Ensure adj_matrix has correct dimensions [B, N, N]
        if self.adj_matrix.shape != (batch_size, self.n_nodes, self.n_nodes):
            raise ValueError(f"adj_matrix shape mismatch. Expected ({batch_size}, {self.n_nodes}, {self.n_nodes}), Got {self.adj_matrix.shape}")
        # --- ---------------------- ---


        # === FIX: Add self-loops (Identity Matrix) on the correct device ===
        # Create identity matrix on the same device and dtype as adj_matrix
        identity = torch.eye(self.n_nodes, device=input_device, dtype=input_dtype)
        # Expand identity to match batch dimension
        identity = identity.unsqueeze(0).expand(batch_size, self.n_nodes, self.n_nodes)

        # Add self-loops *before* normalization
        adj_matrix_with_loops = self.adj_matrix + identity
        # ===================================================================

        # --- Normalization ---
        # Calculate degree matrix D (diagonal)
        # Use clamp to avoid division by zero for isolated nodes
        degree = adj_matrix_with_loops.sum(dim=2).clamp(min=1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        # Create diagonal matrix from vector
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt) # Shape: [B, N, N]

        # Calculate normalized adjacency matrix: D^-0.5 * A * D^-0.5
        adj_normalized = D_inv_sqrt @ adj_matrix_with_loops @ D_inv_sqrt
        # --- ------------- ---


        # --- K-hop Aggregation ---
        # Reshape node_feats for matmul if needed: [B, F_in, N]
        current_hop_feats = node_feats # Shape: [B, F_in, N]
        # Store features for each hop (including original hop 0)
        z_hops = [current_hop_feats.unsqueeze(2)] # Add filter dim: [B, F_in, 1, N]

        for k in range(1, self.filter_number):
            # Aggregate features from neighbors: feats @ adj_norm^T (or adj_norm @ feats if feats is N x F)
            # Note: adj_normalized is symmetric here. Matmul: (B, F_in, N) @ (B, N, N) -> (B, F_in, N)
            current_hop_feats = current_hop_feats @ adj_normalized
            z_hops.append(current_hop_feats.unsqueeze(2)) # Add filter dim: [B, F_in, 1, N]

        # Concatenate features across hops
        z = torch.cat(z_hops, dim=2) # Shape: [B, F_in, K, N] K=filter_number
        # --- ----------------- ---

        # --- Linear Transformation ---
        # Reshape z for linear layer: [B, N, F_in * K]
        # Input: (B, F_in, K, N) -> (B, N, F_in, K) -> (B, N, F_in * K)
        z = z.permute(0, 3, 1, 2).reshape(batch_size, self.n_nodes, self.in_features * self.filter_number)

        # Reshape weights for efficient matmul: [F_in * K, F_out]
        W_reshaped = self.W.reshape(self.in_features * self.filter_number, self.out_features)

        # Apply linear transformation: (B, N, F_in*K) @ (F_in*K, F_out) -> (B, N, F_out)
        output_node_feats = z @ W_reshaped
        # --- --------------------- ---

        # --- Add Bias and Activation ---
        if self.bias is not None:
            # Bias shape: (F_out), needs broadcasting to (B, N, F_out)
            output_node_feats = output_node_feats + self.bias.unsqueeze(0).unsqueeze(0) # Add batch and node dims for broadcasting

        # Permute back to expected output format [B, F_out, N]
        output_node_feats = output_node_feats.permute(0, 2, 1)

        if self.activation is not None:
            output_node_feats = self.activation(output_node_feats)
        # --- ----------------------- ---

        return output_node_feats


class MessagePassingLayer(nn.Module):
     # --- This layer seems more complex and wasn't the source of the immediate error ---
     # --- Review its device handling carefully if you intend to use it.          ---
     # --- Specifically, check operations involving self.adj_matrix and D_mod.   ---
    def __init__(
        self,
        n_nodes,
        in_features,
        out_features,
        filter_number,
        bias=False,
        activation=None,
        name="MP_Layer", # Changed default name
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number
        # Separate weights for aggregated and self features
        self.W_agg = nn.parameter.Parameter(
            torch.Tensor(self.in_features, self.filter_number, self.out_features)
        )
        self.W_self = nn.parameter.Parameter(
            torch.Tensor(self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.activation = activation
        self.name = name
        # self.n_nodes = n_nodes # Better to get dynamically
        self.init_params()

    def init_params(self):
        # Use consistent initialization
        stdv_agg = 1.0 / math.sqrt(self.W_agg.size(0) * self.W_agg.size(1))
        self.W_agg.data.uniform_(-stdv_agg, stdv_agg)
        stdv_self = 1.0 / math.sqrt(self.W_self.size(0))
        self.W_self.data.uniform_(-stdv_self, stdv_self)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv_agg, stdv_agg) # Or use stdv_self? Consistent is better.

    def extra_repr(self):
        reprString = (
            "in_features=%d, out_features=%d, " % (self.in_features, self.out_features)
            + "filter_taps=%d, " % (self.filter_number)
            + "bias=%s" % (self.bias is not None)
        )
        return reprString

    def addGSO(self, GSO):
        # Again, consider passing GSO to forward if it changes per batch
        self.adj_matrix = GSO

    def forward(self, node_feats):
        """
        Message passing layer forward pass.

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N].

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        # --- Device/Input Checks (similar to GCNLayer) ---
        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3:
             raise ValueError(f"Expected node_feats dim 3 (B, F, N), got {node_feats.ndim}")
        batch_size = node_feats.shape[0]
        self.n_nodes = node_feats.shape[2]
        if not hasattr(self, 'adj_matrix') or self.adj_matrix is None:
            raise RuntimeError("Adjacency matrix (GSO) has not been set for MessagePassingLayer.")
        if self.adj_matrix.device != input_device:
             self.adj_matrix = self.adj_matrix.to(input_device)
        if self.adj_matrix.shape != (batch_size, self.n_nodes, self.n_nodes):
            raise ValueError(f"adj_matrix shape mismatch. Expected ({batch_size}, {self.n_nodes}, {self.n_nodes}), Got {self.adj_matrix.shape}")
        # --- ------------------------------------------ ---

        # --- Normalization (No self-loops added here, assuming pure message passing) ---
        # Note: If you *do* want self-loops influencing messages, add identity before normalization
        degree = self.adj_matrix.sum(dim=2).clamp(min=1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        adj_normalized = D_inv_sqrt @ self.adj_matrix @ D_inv_sqrt
        # --- ------------------------------------------------------------------------- ---

        # --- K-hop Aggregation (Messages) ---
        # Keep original features for self-transformation
        node_feats_self = node_feats # Shape: [B, F_in, N]

        current_hop_feats = node_feats # Shape: [B, F_in, N]
        z_hops = [] # Store aggregated features only (hops 1 to K)
                    # If filter_number=1, this loop won't run, z_hops remains empty

        # Start aggregation from hop 1
        for k in range(self.filter_number): # Iterate K times for K filter taps
            current_hop_feats = current_hop_feats @ adj_normalized
            z_hops.append(current_hop_feats.unsqueeze(2)) # Add filter dim: [B, F_in, 1, N]

        # --- Apply Transformations ---
        # 1. Transform self features
        # Reshape self features: [B, N, F_in]
        node_feats_self_reshaped = node_feats_self.permute(0, 2, 1)
        # Apply W_self: (B, N, F_in) @ (F_in, F_out) -> (B, N, F_out)
        transformed_self_feats = node_feats_self_reshaped @ self.W_self

        # 2. Transform aggregated features (if any hops were computed)
        if z_hops: # Check if list is not empty
            z = torch.cat(z_hops, dim=2) # Shape: [B, F_in, K, N] K=filter_number
            # Reshape z: [B, N, F_in * K]
            z = z.permute(0, 3, 1, 2).reshape(batch_size, self.n_nodes, self.in_features * self.filter_number)
            # Reshape W_agg: [F_in * K, F_out]
            W_agg_reshaped = self.W_agg.reshape(self.in_features * self.filter_number, self.out_features)
            # Apply W_agg: (B, N, F_in*K) @ (F_in*K, F_out) -> (B, N, F_out)
            transformed_agg_feats = z @ W_agg_reshaped
            # Combine self and aggregated features
            output_node_feats = transformed_self_feats + transformed_agg_feats
        else:
            # If filter_number is 0 or 1 (meaning only self features matter or no aggregation)
            # Output is just the transformed self features
             output_node_feats = transformed_self_feats

        # --- Add Bias and Activation ---
        if self.bias is not None:
            output_node_feats = output_node_feats + self.bias.unsqueeze(0).unsqueeze(0)

        # Permute back to expected output format [B, F_out, N]
        output_node_feats = output_node_feats.permute(0, 2, 1)

        if self.activation is not None:
            output_node_feats = self.activation(output_node_feats)
        # --- ----------------------- ---

        return output_node_feats


# --- Numpy Layer (Keep for reference/comparison if needed, but not used in PyTorch training) ---
class GCNLayerNumpy:
    # ... (Keep original numpy code if needed for testing/prototyping outside PyTorch) ...
    pass

def glorot_init(nin, nout):
    # ... (Keep original numpy code if needed) ...
    pass
# --- ----------------------------------------------------------------------------------- ---


# --- Example Usage / Test Block ---
if __name__ == "__main__":
    # Setup test tensors on CPU for simplicity first
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 2
    n_nodes = 4
    in_features = 3
    out_features = 2
    K = 2 # Filter taps

    # Create random input data
    node_feats = torch.randn(batch_size, in_features, n_nodes, device=device, dtype=torch.float32)
    # Create random adjacency matrices (ensure they are plausible, e.g., symmetric)
    adj_matrix = torch.randint(0, 2, (batch_size, n_nodes, n_nodes), device=device, dtype=torch.float32)
    # Ensure symmetry
    adj_matrix = (adj_matrix + adj_matrix.transpose(1, 2)) / 2
    adj_matrix = (adj_matrix > 0.5).float() # Make it binary 0/1 again
    adj_matrix.fill_diagonal_(0) # No self-loops initially in GSO

    print(f"Input Node features shape: {node_feats.shape}")
    print(f"Input Adjacency matrix shape: {adj_matrix.shape}")

    activation = nn.LeakyReLU()

    # --- Test GCNLayer ---
    print("\n--- Testing GCNLayer ---")
    gcn = GCNLayer(
        n_nodes=n_nodes, # Pass n_nodes or let it be dynamic
        in_features=in_features,
        out_features=out_features,
        filter_number=K,
        activation=activation,
        bias=True,
    ).to(device) # Move layer parameters to device
    print(gcn)
    # Set the adjacency matrix for the layer (stateful way)
    gcn.addGSO(adj_matrix) # This might need adjustment if GSO changes per batch
    try:
        output_gcn = gcn(node_feats)
        print("GCNLayer Output shape:", output_gcn.shape) # Expected: [B, F_out, N]
        print("GCNLayer Output sample:\n", output_gcn[0, :, :])
    except Exception as e:
        print(f"GCNLayer forward pass failed: {e}")
        import traceback
        traceback.print_exc()


    # --- Test MessagePassingLayer ---
    print("\n--- Testing MessagePassingLayer ---")
    # Reset input features if modified by GCNLayer test
    node_feats = torch.randn(batch_size, in_features, n_nodes, device=device, dtype=torch.float32)

    mp_layer = MessagePassingLayer(
        n_nodes=n_nodes,
        in_features=in_features,
        out_features=out_features,
        filter_number=K,
        activation=activation,
        bias=True,
    ).to(device)
    print(mp_layer)
    mp_layer.addGSO(adj_matrix)
    try:
        output_mp = mp_layer(node_feats)
        print("MessagePassingLayer Output shape:", output_mp.shape) # Expected: [B, F_out, N]
        print("MessagePassingLayer Output sample:\n", output_mp[0, :, :])
    except Exception as e:
        print(f"MessagePassingLayer forward pass failed: {e}")
        import traceback
        traceback.print_exc()   