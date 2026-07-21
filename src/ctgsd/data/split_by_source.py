"""Deterministic source/time splitting performed before window generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import SplitName


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    raw_path: str
    label: int
    condition_id: str
    sampling_rate: int
    num_samples: int

    def __post_init__(self) -> None:
        if not self.source_id or not self.raw_path or not self.condition_id:
            raise ValueError("source identity, path, and condition are required")
        if self.label < 0 or self.sampling_rate <= 0 or self.num_samples <= 0:
            raise ValueError("label, sampling rate, and sample count are invalid")


@dataclass(frozen=True)
class TimeSegment:
    """A non-overlapping time block assigned to exactly one split."""

    segment_id: str
    parent_source_id: str
    raw_path: str
    label: int
    condition_id: str
    sampling_rate: int
    split: SplitName
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("time segment boundaries are invalid")


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, val, and test fractions")
    train, val, test = (float(value) for value in ratios)
    if min(train, val, test) <= 0:
        raise ValueError("all split ratios must be positive")
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    return train, val, test


def split_sources_by_time(
    sources: Iterable[SourceRecord],
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    boundary_gap: int = 0,
) -> list[TimeSegment]:
    """Split every continuous source into train/val/test blocks before windows.

    `boundary_gap` removes samples around split boundaries when an experiment
    requires a guard region. The segment ID, rather than the parent file ID, is
    used as each sample's `source_id`, so train/val/test source IDs are disjoint
    while `parent_source_id` retains provenance.
    """

    train_ratio, val_ratio, _ = _validate_ratios(ratios)
    if boundary_gap < 0:
        raise ValueError("boundary_gap cannot be negative")

    segments: list[TimeSegment] = []
    split_names: tuple[SplitName, ...] = ("train", "val", "test")
    for source in sorted(sources, key=lambda item: item.source_id):
        first = int(source.num_samples * train_ratio)
        second = int(source.num_samples * (train_ratio + val_ratio))
        boundaries = (
            (0, first - boundary_gap),
            (first + boundary_gap, second - boundary_gap),
            (second + boundary_gap, source.num_samples),
        )
        for split, (start, end) in zip(split_names, boundaries, strict=True):
            if end <= start:
                raise ValueError(
                    f"source {source.source_id!r} is too short for ratios and boundary_gap"
                )
            segments.append(
                TimeSegment(
                    segment_id=f"{source.source_id}:{split}",
                    parent_source_id=source.source_id,
                    raw_path=source.raw_path,
                    label=source.label,
                    condition_id=source.condition_id,
                    sampling_rate=source.sampling_rate,
                    split=split,
                    start=start,
                    end=end,
                )
            )
    return segments


def split_sources_by_condition(
    sources: Iterable[SourceRecord],
    *,
    validation_conditions: Iterable[str],
    test_conditions: Iterable[str],
) -> list[TimeSegment]:
    """Assign complete source files by condition for cross-load evaluation.

    Test and validation conditions are explicit. Every other condition is used
    for training, so a held-out load can never contribute a training window.
    """

    validation = frozenset(validation_conditions)
    test = frozenset(test_conditions)
    if not validation or not test:
        raise ValueError("validation_conditions and test_conditions cannot be empty")
    overlap = validation & test
    if overlap:
        raise ValueError(f"validation and test conditions overlap: {sorted(overlap)}")

    records = sorted(sources, key=lambda item: item.source_id)
    known_conditions = {record.condition_id for record in records}
    unknown = (validation | test) - known_conditions
    if unknown:
        raise ValueError(f"unknown held-out conditions: {sorted(unknown)}")

    segments: list[TimeSegment] = []
    counts: dict[SplitName, int] = {"train": 0, "val": 0, "test": 0}
    for source in records:
        split: SplitName
        if source.condition_id in test:
            split = "test"
        elif source.condition_id in validation:
            split = "val"
        else:
            split = "train"
        counts[split] += 1
        segments.append(
            TimeSegment(
                segment_id=f"{source.source_id}:{split}",
                parent_source_id=source.source_id,
                raw_path=source.raw_path,
                label=source.label,
                condition_id=source.condition_id,
                sampling_rate=source.sampling_rate,
                split=split,
                start=0,
                end=source.num_samples,
            )
        )
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"condition split produced empty subsets: {empty}")
    return segments
