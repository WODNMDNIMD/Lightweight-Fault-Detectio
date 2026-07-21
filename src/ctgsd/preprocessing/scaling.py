"""Standardization with an explicit train-only fit contract."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class TrainOnlyStandardScaler:
    """Small NumPy scaler that rejects fitting on validation or test data."""

    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None

    def fit(
        self, values: NDArray[np.floating], *, split: str
    ) -> "TrainOnlyStandardScaler":
        if split != "train":
            raise ValueError("scaler fitting is permitted only for split='train'")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("values must be a non-empty two-dimensional array")
        if not np.isfinite(matrix).all():
            raise ValueError("values must contain only finite entries")
        self.mean_ = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean_.shape[0]:
            raise ValueError("values have an incompatible feature dimension")
        return (matrix - self.mean_) / self.scale_

    def fit_transform(
        self, values: NDArray[np.floating], *, split: str
    ) -> NDArray[np.float64]:
        return self.fit(values, split=split).transform(values)
