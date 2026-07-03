# Our

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SmoothTopKGate(nn.Module):
    def __init__(self, model_dim: int, num_global_experts: int, k: int = 1, fp32_gate: bool = False,
                 eta_init: float = -9.0, boundary_alpha: float = 1.0, h_offset: float = 0.5,
                 smooth_a: float = 1.0, smooth_b: float = 50.0, g_blance: bool = True,
                 b_loss_flag: bool = True, gate_noise: float = 0.0, **options):
        super().__init__()
        self.wg = nn.Linear(model_dim, num_global_experts, bias=False,
                           dtype=torch.float32 if fp32_gate else None)
        self.num_global_experts = num_global_experts
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.gate_noise = gate_noise
        self.eta_parameterized = nn.Parameter(torch.tensor(eta_init))
        self.boundary_alpha = boundary_alpha
        self.h_offset = h_offset
        self.smooth_a = smooth_a
        self.smooth_b = smooth_b
        self.g_blance = g_blance
        self.b_loss_flag = b_loss_flag
        self.activated_expert = float(self.top_k)  # EMA for tracking
        self.loss = None  # Load balance (gshard) loss
        self.b_loss = None  # Boundary loss
        self.gate_type = 'smooth_top'  # For routing selection
        self.last_logits = None  # Store logits for loss

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception(f'Unrecognized argument provided to Gating module: {opt}')


    @staticmethod
    def smooth(u: torch.Tensor, a: float = 1.0, b: float = 50.0) -> torch.Tensor:
        """
        Smooth function S(u) with dtype consistency.
        Args:
            u: Input tensor.
            a, b: Parameters controlling curve shape.
        Returns:
            Tensor with same dtype as u: -inf if u <= 0, 0 if u >= 1, else -softplus(b*log(1-u) - a*log(u)).
        """
        result = torch.empty_like(u)  # Inherits u.dtype (e.g., float16 if AMP)
        mask_le_0 = u <= 0
        result[mask_le_0] = -float('inf')  # Compatible with any dtype
        mask_ge_1 = u >= 1
        result[mask_ge_1] = 0.0  # Compatible with any dtype
        mask_between = ~(mask_le_0 | mask_ge_1)
        u_between = u[mask_between]

        # Ensure inner_expr matches u.dtype to avoid mismatch
        inner_expr = b * torch.log(1 - u_between.clamp(min=1e-6)) - a * torch.log(u_between.clamp(min=1e-6))
        inner_expr = inner_expr.to(u.dtype)  # Convert to u.dtype (e.g., float16)
        result[mask_between] = -F.softplus(inner_expr)  # Now dtype matches
        return result

    def boundary_loss(self, gate: torch.Tensor, eta: torch.Tensor, k: int) -> torch.Tensor:
        """
        Boundary loss to keep number of activated experts near expected budget.
        Args:
            gate: Logits tensor [B, N].
            eta: Positive threshold (softplus(eta_parameterized)).
            k: Reference top_k.
        Returns:
            Scalar loss: alpha * eta * (mean_activated - (k + h_offset)).
        """
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if k < 1 or k > gate.shape[1]:
            raise ValueError("k must be between 1 and N")
        B, N = gate.shape
        topk_values, _ = torch.topk(gate, k, dim=1, largest=True)
        g_k = topk_values[:, -1].unsqueeze(1)  # [B, 1]
        threshold = g_k - eta
        activated_counts = (gate > threshold).sum(dim=1).float()  # [B]
        X = activated_counts.mean()
        self.activated_expert = 0.99 * self.activated_expert + 0.01 * X.item()
        h_prime = k + self.h_offset
        return self.boundary_alpha * eta * (X - h_prime), self.activated_expert

    def set_load_balance(self, gate: torch.Tensor, gate_score: torch.Tensor) -> torch.Tensor:
        """
        Load balance loss (equivalent to gshard_loss).
        Args:
            gate: Logits tensor [B, N].
            gate_score: Smoothed softmax scores [B, N].
        Returns:
            Scalar loss: sum(fraction_expert * prob_expert) * num_experts.
        """
        score = F.softmax(gate, dim=-1)  # [B, N]
        positive_mask = gate_score > 0
        batch_indices, expert_indices = torch.nonzero(positive_mask, as_tuple=True)
        all_valid_idx = expert_indices
        if all_valid_idx.numel() == 0:
            return torch.tensor(0.0, device=gate.device)
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.num_global_experts, device=all_valid_idx.device),
                0,
                all_valid_idx,
                torch.ones_like(all_valid_idx, dtype=torch.float),
            ) / all_valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / all_valid_idx.numel()
        return (fraction_expert * prob_expert).sum() * self.num_global_experts


    def forward(self, x):
        wg = self.wg.float() if self.fp32_gate else self.wg
        return wg(x.to(dtype=wg.weight.dtype)), F.softplus(self.eta_parameterized), self.smooth, self.boundary_loss



Gate = SmoothTopKGate
