"""Streaming tokenization, EOS insertion and fixed-length sequence packing."""

from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Callable, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset

from .tokenizer import MyTokenizer


def token_stream(
    documents: Iterable[str],
    tokenizer: MyTokenizer,
    *,
    add_bos: bool = False,
    add_eos: bool = True,
) -> Iterator[int]:
    """Convert documents to one continuous stream of token IDs."""

    for document in documents:
        yield from tokenizer.encode(document, add_bos=add_bos, add_eos=add_eos)


def packed_sequences(
    documents: Iterable[str],
    tokenizer: MyTokenizer,
    seq_len: int,
    *,
    add_bos: bool = False,
    add_eos: bool = True,
    drop_remainder: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield non-overlapping ``[seq_len]`` tensors from a token stream."""

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    buffer: deque[int] = deque()
    for token_id in token_stream(
        documents,
        tokenizer,
        add_bos=add_bos,
        add_eos=add_eos,
    ):
        buffer.append(token_id)
        if len(buffer) == seq_len:
            yield torch.tensor(list(buffer), dtype=torch.long)
            buffer.clear()

    if buffer and not drop_remainder:
        result = torch.full((seq_len,), tokenizer.eos_id, dtype=torch.long)
        result[: len(buffer)] = torch.tensor(list(buffer), dtype=torch.long)
        yield result


def batch_sequences(
    sequences: Iterable[torch.Tensor],
    batch_size: int,
    *,
    drop_remainder: bool = True,
) -> Iterator[torch.Tensor]:
    """Group packed sequences into ``[batch, seq_len]`` tensors."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    iterator = iter(sequences)
    while True:
        batch = list(islice(iterator, batch_size))
        if len(batch) < batch_size:
            if not batch or drop_remainder:
                return
        yield torch.stack(batch)


class PackedTextDataset(IterableDataset[torch.Tensor]):
    """A repeatable IterableDataset driven by a document factory.

    A factory is used instead of storing a one-shot streaming iterator, so a
    training loop can restart the stream after it reaches its end.
    """

    def __init__(
        self,
        documents_factory: Callable[[], Iterable[str]],
        tokenizer: MyTokenizer,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.documents_factory = documents_factory
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        return packed_sequences(
            self.documents_factory(), self.tokenizer, self.seq_len
        )

