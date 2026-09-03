"""Small evaluation helpers for validation loss and generation."""

from __future__ import annotations

from typing import Iterable

import torch

from .model import MyLLM
from .train import causal_lm_loss


@torch.no_grad()
def estimate_loss(
    model: MyLLM,
    batches: Iterable[torch.Tensor],
    *,
    max_batches: int = 100,
    device: torch.device | str = "cpu",
) -> float:
    model.eval()
    device = torch.device(device)
    losses: list[float] = []
    for index, input_ids in enumerate(batches):
        if index >= max_batches:
            break
        losses.append(float(causal_lm_loss(model, input_ids.to(device))))
    if not losses:
        raise ValueError("no batches were provided")
    return sum(losses) / len(losses)

