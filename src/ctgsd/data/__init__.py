"""Leakage-safe data contracts and splitting utilities."""

from .schema import PairedSampleMetadata, SplitName
from .split_by_source import SourceRecord, TimeSegment, split_sources_by_time
from .windowing import build_window_manifest, validate_manifest

__all__ = [
    "PairedSampleMetadata",
    "SourceRecord",
    "SplitName",
    "TimeSegment",
    "build_window_manifest",
    "split_sources_by_time",
    "validate_manifest",
]

