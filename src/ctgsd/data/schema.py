"""Shared metadata schema for paired raw-signal/feature samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class PairedSampleMetadata:
    sample_id: str
    source_id: str
    condition_id: str
    window_start: int
    split: SplitName
    label: int
    sampling_rate: int
    raw_path: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.source_id or not self.condition_id:
            raise ValueError("sample_id, source_id, and condition_id are required")
        if self.window_start < 0:
            raise ValueError("window_start cannot be negative")
        if self.label < 0:
            raise ValueError("label cannot be negative")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

