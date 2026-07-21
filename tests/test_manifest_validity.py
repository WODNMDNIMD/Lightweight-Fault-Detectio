from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ctgsd.data import (
    PairedSampleMetadata,
    SourceRecord,
    build_window_manifest,
    read_manifest_csv,
    split_sources_by_time,
    validate_manifest,
    write_split_manifests,
)


class ManifestValidityTest(unittest.TestCase):
    def test_split_csv_round_trip_preserves_identifiers_and_types(self) -> None:
        source = SourceRecord(
            source_id="cwru:normal:load0:de12k",
            raw_path="Normal Baseline/normal_0.mat",
            label=0,
            condition_id="load_0",
            sampling_rate=12000,
            num_samples=20000,
        )
        manifest = build_window_manifest(split_sources_by_time([source]))
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_split_manifests(manifest, temporary_directory)
            restored = [record for split in ("train", "val", "test") for record in read_manifest_csv(paths[split])]
        self.assertEqual(
            {record.sample_id for record in restored},
            {record.sample_id for record in manifest},
        )
        validate_manifest(restored)

    def test_duplicate_sample_id_is_rejected(self) -> None:
        common = dict(
            sample_id="duplicate",
            source_id="source:train",
            parent_source_id="source",
            condition_id="load_0",
            window_start=0,
            window_end=2048,
            split="train",
            label=0,
            sampling_rate=12000,
            raw_path="normal_0.mat",
        )
        record = PairedSampleMetadata(**common)
        with self.assertRaisesRegex(ValueError, "sample_id values must be unique"):
            validate_manifest([record, record])

    def test_repository_manifests_have_expected_balanced_counts_when_present(self) -> None:
        manifest_dir = Path(__file__).parents[1] / "data" / "manifests"
        paths = [manifest_dir / f"{split}.csv" for split in ("train", "val", "test")]
        if not all(path.exists() for path in paths):
            self.skipTest("tracked CWRU manifests have not been materialized")
        records = [record for path in paths for record in read_manifest_csv(path)]
        validate_manifest(records)
        split_counts = {
            split: sum(record.split == split for record in records)
            for split in ("train", "val", "test")
        }
        self.assertEqual(split_counts, {"train": 2760, "val": 880, "test": 880})
        for split in ("train", "val", "test"):
            class_counts = {
                label: sum(record.split == split and record.label == label for record in records)
                for label in range(10)
            }
            expected_per_class = {"train": 276, "val": 88, "test": 88}[split]
            self.assertEqual(set(class_counts.values()), {expected_per_class})


if __name__ == "__main__":
    unittest.main()
