"""BPE tokenizer utilities.

The optional ``tokenizers`` dependency is imported only when these functions
are used, so model-only tests stay lightweight.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class MyTokenizer:
    _SPECIAL_CANDIDATES = {
        "pad": ("<pad>", "<|pad|>", "<|endoftext|>"),
        "unk": ("<unk>", "<|unk|>"),
        "bos": ("<bos>", "<s>", "<|begin_of_text|>", "<|endoftext|>"),
        "eos": ("<eos>", "</s>", "<|end_of_text|>", "<|endoftext|>"),
    }

    def __init__(
        self,
        tokenizer_file: str | Path,
        *,
        special_tokens: dict[str, str] | None = None,
    ) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "MyTokenizer requires tokenizers; run pip install -r requirements.txt"
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._special_tokens = self._resolve_special_tokens(special_tokens)

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        *,
        special_tokens: dict[str, str] | None = None,
    ) -> "MyTokenizer":
        """Load only a tokenizer artifact from the Hugging Face Hub.

        This does not load model weights. For example, ``identifier='gpt2'``
        downloads the GPT-2 tokenizer JSON and uses its end-of-text token as
        both BOS and EOS when no explicit mapping is supplied.
        """

        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Pretrained tokenizer loading requires tokenizers; run "
                "pip install -r requirements.txt"
            ) from exc
        instance = cls.__new__(cls)
        instance._tokenizer = Tokenizer.from_pretrained(identifier)
        instance._special_tokens = instance._resolve_special_tokens(special_tokens)
        return instance

    def _resolve_special_tokens(
        self,
        overrides: dict[str, str] | None,
    ) -> dict[str, str]:
        overrides = overrides or {}
        resolved: dict[str, str] = {}
        for name, candidates in self._SPECIAL_CANDIDATES.items():
            token = overrides.get(name)
            if token is not None and self._tokenizer.token_to_id(token) is None:
                raise ValueError(f"Tokenizer does not contain special token {token!r}")
            if token is None:
                token = next(
                    (
                        candidate
                        for candidate in candidates
                        if self._tokenizer.token_to_id(candidate) is not None
                    ),
                    None,
                )
            if token is not None:
                resolved[name] = token
        return resolved

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Tokenizer does not contain special token {token!r}")
        return token_id

    @property
    def eos_id(self) -> int:
        token = self._special_tokens.get("eos")
        if token is None:
            raise KeyError("Tokenizer has no usable EOS token")
        return self.token_id(token)

    @property
    def bos_id(self) -> int:
        token = self._special_tokens.get("bos")
        if token is None:
            raise KeyError("Tokenizer has no usable BOS token")
        return self.token_id(token)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


def train_bpe_tokenizer(
    documents: Iterable[str],
    output_path: str | Path,
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> None:
    """Train a byte-level BPE tokenizer from an iterable of documents."""

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Tokenizer training requires tokenizers; run pip install -r requirements.txt"
        ) from exc

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))


def fineweb_documents(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    split: str = "train",
    *,
    max_documents: int | None = None,
) -> Iterator[str]:
    """Yield document text from FineWeb-Edu in streaming mode."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "FineWeb loading requires datasets; run pip install -r requirements.txt"
        ) from exc

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )
    for index, row in enumerate(dataset):
        if max_documents is not None and index >= max_documents:
            break
        text = row.get("text")
        if text:
            yield text


def line_documents(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def load_tokenizer(
    tokenizer_file: str | Path | None = None,
    pretrained_identifier: str | None = None,
) -> MyTokenizer:
    """Load either a local tokenizer JSON or a Hub tokenizer, but not both."""

    if (tokenizer_file is None) == (pretrained_identifier is None):
        raise ValueError("provide exactly one of tokenizer_file or pretrained_identifier")
    if tokenizer_file is not None:
        return MyTokenizer(tokenizer_file)
    return MyTokenizer.from_pretrained(pretrained_identifier)  # type: ignore[arg-type]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MyLLM-Tiny BPE tokenizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", type=Path)
    source.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--max-documents", type=int, default=50_000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.text_file is not None:
        documents = line_documents(args.text_file)
    else:
        documents = fineweb_documents(
            args.dataset,
            args.dataset_config,
            max_documents=args.max_documents,
        )
    train_bpe_tokenizer(
        documents,
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"saved tokenizer to {args.output}")


if __name__ == "__main__":
    main()
