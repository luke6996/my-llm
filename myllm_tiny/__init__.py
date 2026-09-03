"""A small from-scratch decoder-only language model."""

from .config import ModelConfig
from .model import MyLLM

__all__ = ["ModelConfig", "MyLLM"]

