"""Abstract base class that every benchmark model must implement.

All models in this benchmark inherit from `BaseForecastModel` and must implement
two methods: `fit` (training) and `predict` (inference). This keeps the evaluation
pipeline model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecastModel(ABC):
    """Minimal interface shared by every benchmark model.

    Subclasses implement `fit` and `predict` so the evaluation pipeline can
    call them uniformly regardless of the underlying architecture.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data.

        Args:
            df: Long-format DataFrame with columns ``unique_id``, ``ds`` (date),
                and ``y`` (discharge), plus any feature columns.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate forecasts for all samples in ``df``.

        Args:
            df: Same format as passed to ``fit``.

        Returns:
            DataFrame with at minimum columns ``unique_id``, ``ds``, and one
            prediction column per forecast horizon.
        """
