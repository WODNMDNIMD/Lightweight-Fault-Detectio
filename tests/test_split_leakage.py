from __future__ import annotations

import unittest

from ctgsd.data import (
    SourceRecord,
    build_window_manifest,
    split_sources_by_condition,
    split_sources_by_time,
)


def _source(load: int, label: int = 0) -> SourceRecord:
    return SourceRecord(
        source_id=f"cwru:class{label}:load{load}:de12k",
        raw_path=f"class{label}_{load}.mat",
        label=label,
        condition_id=f"load_{load}",
        sampling_rate=12000,
        num_samples=20000,
    )


class SplitLeakageTest(unittest.TestCase):
    def test_temporal_windows_never_cross_split_boundaries(self) -> None:
        segments = split_sources_by_time([_source(0)])
        windows = build_window_manifest(segments, window_length=2048, overlap_ratio=0.5)
        boundaries = {segment.segment_id: (segment.start, segment.end) for segment in segments}
        for window in windows:
            start, end = boundaries[window.source_id]
            self.assertGreaterEqual(window.window_start, start)
            self.assertLessEqual(window.window_end, end)

    def test_leave_one_load_test_sources_are_absent_from_train_and_val(self) -> None:
        sources = [_source(load, label) for load in range(4) for label in range(2)]
        segments = split_sources_by_condition(
            sources,
            validation_conditions={"load_2"},
            test_conditions={"load_3"},
        )
        parent_ids = {
            split: {segment.parent_source_id for segment in segments if segment.split == split}
            for split in ("train", "val", "test")
        }
        self.assertFalse(parent_ids["test"] & parent_ids["train"])
        self.assertFalse(parent_ids["test"] & parent_ids["val"])
        self.assertTrue(
            all(segment.condition_id == "load_3" for segment in segments if segment.split == "test")
        )
        self.assertTrue(
            all(segment.condition_id != "load_3" for segment in segments if segment.split != "test")
        )


if __name__ == "__main__":
    unittest.main()
