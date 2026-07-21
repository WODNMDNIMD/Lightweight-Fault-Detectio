"""Materialize raw/feature pairs from one manifest-defined window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .schema import PairedSampleMetadata


@dataclass(frozen=True)
class PairedWindow:
    metadata: PairedSampleMetadata
    raw_signal: NDArray[np.float64]
    features: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected_length = self.metadata.window_end - self.metadata.window_start
        if self.raw_signal.shape != (1, expected_length):
            raise ValueError(
                f"raw_signal must have shape (1, {expected_length}), got {self.raw_signal.shape}"
            )
        if self.features.shape != (151,):
            raise ValueError(f"features must have shape (151,), got {self.features.shape}")
        if not np.isfinite(self.raw_signal).all() or not np.isfinite(self.features).all():
            raise ValueError("raw_signal and features must contain only finite values")


def materialize_paired_window(
    metadata: PairedSampleMetadata,
    parent_signal: NDArray[np.floating],
    feature_extractor: Callable[[NDArray[np.float64]], NDArray[np.floating]],
) -> PairedWindow:
    """Slice once, then derive both model inputs from that exact window."""

    signal = np.asarray(parent_signal, dtype=np.float64).reshape(-1)
    if metadata.window_end > signal.size:
        raise ValueError(
            f"window end {metadata.window_end} exceeds source length {signal.size}"
        )
    raw_window = signal[metadata.window_start : metadata.window_end].copy()
    features = np.asarray(feature_extractor(raw_window), dtype=np.float64).reshape(-1)
    return PairedWindow(
        metadata=metadata,
        raw_signal=raw_window[np.newaxis, :],
        features=features,
    )
