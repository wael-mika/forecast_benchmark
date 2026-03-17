"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set lightweight reproducibility controls used later in the benchmark."""
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover - optional runtime dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
