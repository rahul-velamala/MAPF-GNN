import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

# Log factorial calculation helper using torch.lgamma for numerical stability
def _log_factorial(k: torch.Tensor):
    # torch.lgamma(k + 1) computes log(k!)
    # Ensure k is non-negative float for lgamma
    # Add small epsilon for k=0 case if needed, although lgamma(1) handles log(0!)
    return torch.lgamma(k.float() + 1.0)

class ADCLayer(nn.Module):
    """
    Adaptive Diffusion Convolution Layer (Heat Kernel based).
    Learns a diffusion time 't' per layer.
    Propagation: H_out = sum_{k=0}^{K} (exp(-t) * t^k / k!) * T^k * H * W
    where T is the normalized adjacency matrix with self-loops (T = D_hat^-0.5 * A_hat * D_hat^-0.5).

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        K (int): Truncation order for the heat kernel series expansion.
        initial_t (float): Initial value for the learnable diffusion time 't'. Defaults to 1.0.
        bias (bool): If True, adds a learnable bias to the output. Defaults to True.
        activation: Activation function to apply (typically applied outside in the framework). Defaults to None.
        train_t (bool): If True, the diffusion time 't' is a learnable parameter.
                        If False, 't' is a fixed buffer. Defaults to True.
        name (str): Name for the layer instance. Defaults to "ADC_Layer".
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        initial_t: float = 1.0,
        bias: bool = True,
        activation=None,
        train_t: bool = True,
        name: str = "ADC_Layer",
    ):
        super().__init__()
        if K < 1:
            raise ValueError("Truncation order K must be at least 1.")
        self.in_features = in_features
        self.out_features = out_features
        self.K = K # Truncation order / Max power of T
        self.activation = activation # Store activation if needed (often applied outside)
        self.name = name
        self._current_gso: torch.Tensor | None = None # Adjacency matrix A (set via addGSO)

        # Learnable diffusion time 't'
        if initial_t <= 0:
            logger.warning(f"{self.name}: initial_t ({initial_t}) <= 0. Resetting to 1.0.")
            initial_t = 1.0
        if train_t:
            # Initialize 't' as a learnable parameter
            self.t = nn.Parameter(torch.tensor([initial_t], dtype=torch.float32))
            self._train_t = True
        else:
            # Initialize 't' as a fixed buffer
            self.register_buffer('t', torch.tensor([initial_t], dtype=torch.float32))
            self._train_t = False

        # Learnable weight matrix W (applied *after* propagation)
        # Input features to W will be F_in, output F_out
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)

        self.init_params()
        logger.debug(f"Initialized {self.name}: In={in_features}, Out={out_features}, K={K}, initial_t={initial_t:.4f}, train_t={self._train_t}")

    def init_params(self):
        """ Initializes layer parameters using Xavier uniform initialization. """
        gain = nn.init.calculate_gain('relu') if self.activation == F.relu else 1.0
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        if self.b is not None:
             nn.init.zeros_(self.b.data)
        # No specific init needed for t beyond its initial value

    def extra_repr(self):
        """ String representation for print(model). """
        # Access t value correctly whether it's a Parameter or buffer
        t_val = self.t.item() if self._train_t else self.t[0].item()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"K={self.K}, current_t={t_val:.4f}, train_t={self._train_t}, bias={self.b is not None}"
        )

    def addGSO(self, GSO: torch.Tensor):
        """Stores the raw Graph Shift Operator (Adjacency Matrix A). Expects [B, N, N]."""
        if GSO is None or GSO.ndim != 3:
             raise ValueError(f"GSO must be a 3D tensor [B, N, N], got shape {GSO.shape if GSO is not None else 'None'}")
        self._current_gso = GSO

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Processes graph features using ADC layer.
        Assumes self._current_gso (Adjacency A) has been set via addGSO().

        Args:
            node_feats (torch.Tensor): Node features tensor shape [B, F_in, N].

        Returns:
            torch.Tensor: Output node features tensor shape [B, F_out, N].
        """
        if self._current_gso is None:
            raise RuntimeError(f"{self.name}: Adjacency matrix (GSO) has not been set. Call addGSO().")

        input_device = node_feats.device
        input_dtype = node_feats.dtype
        if node_feats.ndim != 3: raise ValueError(f"{self.name}: Expected node_feats dim 3 (B, F_in, N)")
        batch_size, F_in, n_nodes = node_feats.shape
        if F_in != self.in_features: raise ValueError(f"{self.name}: Input feature dimension mismatch.")

        # --- Adjacency Matrix Check & Normalization ---
        adj_matrix_A = self._current_gso # Raw adjacency A
        if adj_matrix_A.device != input_device or adj_matrix_A.dtype != input_dtype:
            adj_matrix_A = adj_matrix_A.to(device=input_device, dtype=input_dtype)
        if adj_matrix_A.shape != (batch_size, n_nodes, n_nodes):
            if adj_matrix_A.shape == (1, n_nodes, n_nodes) and batch_size > 1:
                adj_matrix_A = adj_matrix_A.expand(batch_size, -1, -1)
            else: raise ValueError(f"{self.name}: GSO shape mismatch.")

        # 1. Calculate A_hat = A + I
        identity = torch.eye(n_nodes, device=input_device, dtype=adj_matrix_A.dtype)
        identity = identity.expand(batch_size, n_nodes, n_nodes)
        adj_matrix_A_hat = adj_matrix_A + identity

        # 2. Calculate D_hat = diag(sum(A_hat))
        degree_hat = torch.sum(adj_matrix_A_hat, dim=2).clamp(min=1e-6)
        degree_hat_inv_sqrt = torch.pow(degree_hat, -0.5)
        D_hat_inv_sqrt_diag = torch.diag_embed(degree_hat_inv_sqrt) # Shape: [B, N, N]

        # 3. Calculate Normalized Adjacency T = D_hat^-0.5 * A_hat * D_hat^-0.5
        T = D_hat_inv_sqrt_diag @ adj_matrix_A_hat @ D_hat_inv_sqrt_diag
        # --- End Normalization ---

        # --- Calculate Heat Kernel Coefficients ---
        # theta_k = exp(-t) * t^k / k!
        current_t = self.t.clamp(min=1e-6) # Ensure t is positive for log and powers

        k_indices = torch.arange(0, self.K + 1, device=input_device, dtype=input_dtype) # 0, 1, ..., K

        # Calculate log(theta_k) for stability: log(theta_k) = -t + k*log(t) - log(k!)
        try:
            log_t = torch.log(current_t)
            log_k_factorial = _log_factorial(k_indices)
            # Use broadcasting: current_t [1], log_t [1], k_indices [K+1], log_k_factorial [K+1]
            log_thetas = -current_t + k_indices * log_t - log_k_factorial # Shape [K+1]
        except Exception as e:
             logger.error(f"Error calculating log_thetas: t={current_t.item()}, k={k_indices}", exc_info=True)
             raise e

        # Exponentiate to get thetas
        thetas = torch.exp(log_thetas) # Shape [K+1]
        # Ensure thetas sum reasonably close to 1 (or normalize if desired)
        # logger.debug(f"{self.name} - thetas sum: {thetas.sum().item()}")
        # --- End Coefficient Calculation ---


        # --- Perform Propagation: sum_{k=0}^{K} theta_k * (H @ T^k) ---
        # H shape is [B, F_in, N]
        # T shape is [B, N, N]
        # Output should be [B, F_in, N]

        H_prop = torch.zeros_like(node_feats) # Initialize output propagation result
        H_curr_pow = node_feats # H @ T^0 = H @ I = H

        # Iterate 0 to K
        for k in range(self.K + 1):
            # Add weighted contribution: theta_k * (H @ T^k)
            # thetas[k] is scalar, H_curr_pow is [B, F_in, N]
            H_prop = H_prop + thetas[k] * H_curr_pow

            # Update H_curr_pow for next iteration: H @ T^(k+1) = (H @ T^k) @ T
            if k < self.K: # Don't need to compute T^(K+1)
                 H_curr_pow = torch.matmul(H_curr_pow, T) # [B, F_in, N] @ [B, N, N] -> [B, F_in, N]
        # --- End Propagation ---

        # --- Linear Transformation ---
        # H_prop is [B, F_in, N]. W is [F_in, F_out]. Output should be [B, F_out, N]
        # Need: H_prop.permute(0, 2, 1) @ W -> [B, N, F_out]
        H_prop_permuted = H_prop.permute(0, 2, 1) # Shape [B, N, F_in]
        output_node_feats = torch.matmul(H_prop_permuted, self.W) # Shape [B, N, F_out]

        # --- Add Bias ---
        if self.b is not None:
            output_node_feats = output_node_feats + self.b # Broadcasting [F_out]

        # --- Permute back to expected output format [B, F_out, N] ---
        output_node_feats = output_node_feats.permute(0, 2, 1)

        # --- Activation (Applied outside in the framework) ---
        if self.activation is not None:
            # This usually isn't applied here in this framework structure
            output_node_feats = self.activation(output_node_feats)

        return output_node_feats
