"""Neural network layers used by MyLLM-Tiny."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization, without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulating the variance in fp32 is important when training in fp16.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings for query and key tensors."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q/k: [batch, heads, seq_len, head_dim]
        seq_len = q.size(-2)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device)
        if position_ids.max().item() >= self.cos.size(0):
            raise ValueError("position_ids exceed configured max_seq_len")

        cos = self.cos[position_ids].to(device=q.device, dtype=q.dtype)
        sin = self.sin[position_ids].to(device=q.device, dtype=q.dtype)
        if position_ids.ndim == 1:
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half_dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif position_ids.ndim == 2:
            cos = cos.unsqueeze(1)  # [batch, 1, seq, half_dim]
            sin = sin.unsqueeze(1)
        else:
            raise ValueError("position_ids must have shape [seq] or [batch, seq]")

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        q_rotated = torch.stack(
            (q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1
        ).flatten(-2)
        k_rotated = torch.stack(
            (k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1
        ).flatten(-2)
        return q_rotated, k_rotated


class SwiGLU(nn.Module):
    """The gated feed-forward network used in the Transformer blocks."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
