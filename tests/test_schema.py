from __future__ import annotations

import unittest

from ctgsd.data import PairedSampleMetadata


class PairedSampleMetadataTest(unittest.TestCase):
    def test_valid_metadata(self) -> None:
        metadata = PairedSampleMetadata(
            sample_id="cwru:file-1:0",
            source_id="file-1",
            condition_id="load-0",
            window_start=0,
            split="train",
            label=1,
            sampling_rate=12000,
            raw_path="data/raw/file-1.mat",
        )
        self.assertEqual(metadata.split, "train")

    def test_negative_window_start_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PairedSampleMetadata(
                sample_id="bad",
                source_id="source",
                condition_id="condition",
                window_start=-1,
                split="test",
                label=0,
                sampling_rate=12000,
                raw_path="x.mat",
            )


if __name__ == "__main__":
    unittest.main()

