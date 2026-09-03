"""Autoregressive text generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import MyLLM
from .tokenizer import MyTokenizer, load_tokenizer


@torch.no_grad()
def generate(
    model: MyLLM,
    tokenizer: MyTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | str = "cpu",
) -> str:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    device = torch.device(device)
    model.eval().to(device)
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.bos_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.max_seq_len :]
        logits = model(context)[:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_id:
            break
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with MyLLM-Tiny")
    parser.add_argument("--checkpoint", type=Path, required=True)
    tokenizer_source = parser.add_mutually_exclusive_group(required=True)
    tokenizer_source.add_argument("--tokenizer", type=Path)
    tokenizer_source.add_argument("--pretrained-tokenizer")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = MyLLM(config)
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(args.tokenizer, args.pretrained_tokenizer)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
