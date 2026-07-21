from __future__ import annotations

import unittest

from ctgsd.data import SourceRecord, build_window_manifest, split_sources_by_time, validate_manifest
from ctgsd.data.cwru import parse_cwru_identity
from ctgsd.data.schema import PairedSampleMetadata


class DataIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SourceRecord(
            source_id="cwru:b007:load0:de12k",
            raw_path="B007_0.mat",
            label=1,
            condition_id="load_0",
            sampling_rate=12000,
            num_samples=20000,
        )

    def test_split_happens_before_windowing(self) -> None:
        segments = split_sources_by_time([self.source], ratios=(0.6, 0.2, 0.2))
        self.assertEqual([(item.start, item.end) for item in segments], [(0, 12000), (12000, 16000), (16000, 20000)])
        manifest = build_window_manifest(segments, window_length=2048, overlap_ratio=0.5)
        self.assertTrue(manifest)
        for record in manifest:
            expected = next(item for item in segments if item.segment_id == record.source_id)
            self.assertGreaterEqual(record.window_start, expected.start)
            self.assertLessEqual(record.window_end, expected.end)

    def test_segment_source_ids_are_disjoint_across_splits(self) -> None:
        manifest = build_window_manifest(split_sources_by_time([self.source]))
        by_split = {
            split: {record.source_id for record in manifest if record.split == split}
            for split in ("train", "val", "test")
        }
        self.assertFalse(by_split["train"] & by_split["val"])
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["val"] & by_split["test"])

    def test_cross_split_overlap_is_rejected(self) -> None:
        common = dict(
            parent_source_id="parent",
            condition_id="load_0",
            label=0,
            sampling_rate=12000,
            raw_path="normal_0.mat",
        )
        records = [
            PairedSampleMetadata(
                sample_id="train:0",
                source_id="parent:train",
                window_start=0,
                window_end=2048,
                split="train",
                **common,
            ),
            PairedSampleMetadata(
                sample_id="test:1024",
                source_id="parent:test",
                window_start=1024,
                window_end=3072,
                split="test",
                **common,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "cross-split window overlap"):
            validate_manifest(records)

    def test_cwru_parser_uses_exact_patterns(self) -> None:
        self.assertEqual(parse_cwru_identity("B007_3.mat"), ("B007", 1, 3))
        self.assertEqual(parse_cwru_identity("OR021@6_2.mat"), ("OR021", 9, 2))
        self.assertEqual(parse_cwru_identity("normal_1.mat"), ("NORMAL", 0, 1))
        self.assertIsNone(parse_cwru_identity("298.mat"))
        self.assertIsNone(parse_cwru_identity("B028_0.mat"))


if __name__ == "__main__":
    unittest.main()

