"""Leakage-safe data contracts and splitting utilities."""

from .paired_dataset import PairedWindow, materialize_paired_window
from .schema import PairedSampleMetadata, SplitName
from .split_by_source import (
    SourceRecord,
    TimeSegment,
    split_sources_by_condition,
    split_sources_by_time,
)
from .windowing import (
    build_window_manifest,
    read_manifest_csv,
    validate_manifest,
    write_split_manifests,
)

__all__ = [
    "PairedSampleMetadata",
    "PairedWindow",
    "SourceRecord",
    "SplitName",
    "TimeSegment",
    "build_window_manifest",
    "materialize_paired_window",
    "read_manifest_csv",
    "split_sources_by_condition",
    "split_sources_by_time",
    "validate_manifest",
    "write_split_manifests",
]
