"""Pretraining loop, optimizer setup and checkpoint/resume support."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn

from .config import ModelConfig
from .data import batch_sequences, packed_sequences
from .model import MyLLM
from .tokenizer import MyTokenizer, fineweb_documents, load_tokenizer


def causal_lm_loss(model: MyLLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross-entropy for a batch of packed token IDs."""

    _, loss = model(input_ids, targets=input_ids)
    return loss


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def save_checkpoint(
    path: str | Path,
    model: MyLLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    *,
    step: int,
    tokens_seen: int,
) -> None:
    """Save enough state to continue training after a Colab interruption."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: MyLLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    *,
    device: torch.device | str = "cpu",
) -> tuple[int, int]:
    """Load checkpoint and return ``(step, tokens_seen)``."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config")
    if saved_config is not None and saved_config != model.config.to_dict():
        raise ValueError("checkpoint config does not match the current model")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("tokens_seen", 0))


def train(
    model: MyLLM,
    batch_factory: Callable[[], Iterator[torch.Tensor]],
    *,
    total_steps: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    device: torch.device | str = "cuda",
    precision: str = "fp16",
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 1000,
    resume: str | Path | None = None,
    log_interval: int = 10,
) -> tuple[int, int]:
    """Train for a fixed number of optimizer steps.

    The data factory is restarted if a finite iterator reaches its end. This
    is useful for short smoke tests and also makes dataset exhaustion safe.
    """

    device = torch.device(device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    use_amp = device.type == "cuda" and precision != "fp32"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    tokens_seen = 0
    if resume is not None:
        step, tokens_seen = load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device=device,
        )

    batch_iterator = iter(batch_factory())
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            try:
                input_ids = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(batch_factory())
                input_ids = next(batch_iterator)
            input_ids = input_ids.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                loss = causal_lm_loss(model, input_ids)
                scaled_loss = loss / gradient_accumulation_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += float(loss.detach())
            tokens_seen += input_ids.numel()

        if use_amp:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        step += 1

        if log_interval > 0 and step % log_interval == 0:
            average_loss = running_loss / gradient_accumulation_steps
            lr = scheduler.get_last_lr()[0]
            print(
                f"step={step} loss={average_loss:.4f} lr={lr:.3e} "
                f"tokens_seen={tokens_seen}"
            )

        if checkpoint_dir is not None and (
            step % checkpoint_interval == 0 or step == total_steps
        ):
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                step=step,
                tokens_seen=tokens_seen,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step=step,
                tokens_seen=tokens_seen,
            )

    return step, tokens_seen


def _synthetic_documents() -> Iterator[str]:
    text = (
        "The small language model learns from tokens. "
        "Artificial intelligence studies language and computation."
    )
    while True:
        yield text


def _make_batch_factory(
    tokenizer: MyTokenizer,
    args: argparse.Namespace,
) -> Callable[[], Iterator[torch.Tensor]]:
    if args.synthetic:
        document_factory = _synthetic_documents
    else:
        document_factory = lambda: fineweb_documents(
            args.dataset,
            args.dataset_config,
        )

    def factory() -> Iterator[torch.Tensor]:
        sequences = packed_sequences(
            document_factory(), tokenizer, args.seq_len, drop_remainder=True
        )
        return batch_sequences(sequences, args.micro_batch_size, drop_remainder=True)

    return factory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyLLM-Tiny")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--pretrained-tokenizer",
        help="Hugging Face tokenizer identifier, e.g. gpt2; tokenizer only, no model weights",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--total-tokens", type=int, default=1_000_000)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.synthetic and args.tokenizer is None and args.pretrained_tokenizer is None:
        raise SystemExit(
            "--tokenizer or --pretrained-tokenizer is required unless --synthetic is used"
        )
    if args.tokenizer is not None and args.pretrained_tokenizer is not None:
        raise SystemExit("choose only one of --tokenizer and --pretrained-tokenizer")
    if args.seq_len > 256:
        raise SystemExit("the first version expects seq_len <= 256")

    tokenizer = None
    if args.tokenizer is not None or args.pretrained_tokenizer is not None:
        tokenizer = load_tokenizer(args.tokenizer, args.pretrained_tokenizer)
        config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len)
    else:
        # A tiny local tokenizer-free smoke test. IDs are already generated
        # below by the deterministic character tokenizer.
        config = ModelConfig(vocab_size=256, max_seq_len=args.seq_len)

    if args.synthetic and tokenizer is None:
        class CharacterTokenizer:
            eos_id = 0

            def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
                ids = [ord(char) % 255 + 1 for char in text]
                if add_eos:
                    ids.append(self.eos_id)
                return ids

        tokenizer = CharacterTokenizer()

    model = MyLLM(config)
    tokens_per_step = args.micro_batch_size * args.seq_len * args.gradient_accumulation_steps
    total_steps = max(1, math.ceil(args.total_tokens / tokens_per_step))
    warmup_steps = max(1, math.ceil(total_steps * args.warmup_ratio))
    print(f"parameters={model.num_parameters():,} device={args.device} steps={total_steps}")
    train(
        model,
        _make_batch_factory(tokenizer, args),
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        precision=args.precision,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
