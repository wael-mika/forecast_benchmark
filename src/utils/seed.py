"""Reproducibility helpers for Python, NumPy, and optional PyTorch runs.

This module contains the small seed utility shared by the training pipelines.
The goal is not perfect determinism in every backend, but a single place to
apply the lightweight reproducibility controls used in this benchmark.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the random seed for Python, NumPy, and PyTorch when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover - optional runtime dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
