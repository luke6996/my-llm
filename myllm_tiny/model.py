"""Decoder-only Transformer language model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.feed_forward = SwiGLU(config.d_model, config.ffn_dim, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), position_ids=position_ids)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MyLLM(nn.Module):
    """A small pre-norm decoder-only Transformer.

    When ``targets`` is omitted, ``forward`` returns logits with shape
    ``[batch, seq_len, vocab_size]``. When targets are supplied it returns
    ``(logits, loss)`` using next-token cross entropy.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")

        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids=position_ids)
        logits = self.lm_head(self.final_norm(x))

        if targets is None:
            return logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
            targets[:, 1:].contiguous().view(-1),
        )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

