"""Base model interface for future forecasting implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecastModel(ABC):
    """Small explicit interface for future benchmark models."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train on canonical long-format data."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions in a future milestone."""
