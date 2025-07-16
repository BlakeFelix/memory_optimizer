"""Convenience imports for ai_memory package."""

# Import core modules without loading the CLI to avoid circular dependencies
from .api import app
from .model_config import get_model_budget
from .dream_lord import DreamLord

__all__ = [
    "app",
    "get_model_budget",
    "DreamLord",
]
