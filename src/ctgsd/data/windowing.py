"""Window-manifest generation and leakage assertions."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Iterable

from .schema import PairedSampleMetadata
from .split_by_source import TimeSegment


def _step_size(window_length: int, overlap_ratio: float) -> int:
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    step = int(round(window_length * (1.0 - overlap_ratio)))
    if step <= 0:
        raise ValueError("overlap_ratio produces a zero step")
    return step


def build_window_manifest(
    segments: Iterable[TimeSegment],
    window_length: int = 2048,
    overlap_ratio: float = 0.5,
) -> list[PairedSampleMetadata]:
    """Generate windows independently inside already assigned time segments."""

    step = _step_size(window_length, overlap_ratio)
    manifest: list[PairedSampleMetadata] = []
    for segment in sorted(segments, key=lambda item: (item.parent_source_id, item.start)):
        if segment.length < window_length:
            continue
        for start in range(segment.start, segment.end - window_length + 1, step):
            end = start + window_length
            sample_id = f"{segment.segment_id}:{start:09d}"
            manifest.append(
                PairedSampleMetadata(
                    sample_id=sample_id,
                    source_id=segment.segment_id,
                    parent_source_id=segment.parent_source_id,
                    condition_id=segment.condition_id,
                    window_start=start,
                    window_end=end,
                    split=segment.split,
                    label=segment.label,
                    sampling_rate=segment.sampling_rate,
                    raw_path=segment.raw_path,
                )
            )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Iterable[PairedSampleMetadata]) -> None:
    """Raise when identities collide or windows overlap across split boundaries."""

    records = list(manifest)
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")

    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sources_by_split[record.split].add(record.source_id)
    names = tuple(sources_by_split)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = sources_by_split[left_name] & sources_by_split[right_name]
            if overlap:
                raise ValueError(f"source_id leakage between {left_name} and {right_name}: {overlap}")

    by_parent: dict[str, list[PairedSampleMetadata]] = defaultdict(list)
    for record in records:
        by_parent[record.parent_source_id].append(record)
    for parent_source_id, parent_records in by_parent.items():
        for index, left in enumerate(parent_records):
            for right in parent_records[index + 1 :]:
                if left.split == right.split:
                    continue
                intervals_overlap = max(left.window_start, right.window_start) < min(
                    left.window_end, right.window_end
                )
                if intervals_overlap:
                    raise ValueError(
                        f"cross-split window overlap in parent source {parent_source_id}: "
                        f"{left.sample_id} vs {right.sample_id}"
                    )


def write_split_manifests(
    manifest: Iterable[PairedSampleMetadata], output_dir: str | Path
) -> dict[str, Path]:
    """Write deterministic train/val/test CSV files after validation."""

    records = list(manifest)
    validate_manifest(records)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(PairedSampleMetadata)]
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        path = destination / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            for record in records:
                if record.split == split:
                    writer.writerow({name: getattr(record, name) for name in field_names})
        paths[split] = path
    return paths


def read_manifest_csv(path: str | Path) -> list[PairedSampleMetadata]:
    """Read one manifest CSV with explicit numeric conversion and validation."""

    records: list[PairedSampleMetadata] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                PairedSampleMetadata(
                    sample_id=row["sample_id"],
                    source_id=row["source_id"],
                    parent_source_id=row["parent_source_id"],
                    condition_id=row["condition_id"],
                    window_start=int(row["window_start"]),
                    window_end=int(row["window_end"]),
                    split=row["split"],  # type: ignore[arg-type]
                    label=int(row["label"]),
                    sampling_rate=int(row["sampling_rate"]),
                    raw_path=row["raw_path"],
                )
            )
    validate_manifest(records)
    return records
