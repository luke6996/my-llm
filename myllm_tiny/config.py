"""Configuration objects for MyLLM-Tiny."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Architecture configuration.

    The defaults match the first model described in MyLLM_Tiny_Plan.md.
    ``head_dim`` is explicit so the relationship between Q/K/V shapes is
    visible while learning the architecture.
    """

    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn_dim: int = 768
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                "d_model must equal n_heads * head_dim; got "
                f"{self.d_model} != {self.n_heads} * {self.head_dim}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot be greater than n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.vocab_size <= 0 or self.n_layers <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size, n_layers and max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        return cls(**values)

