from __future__ import annotations

import unittest

import numpy as np

from ctgsd.data import PairedSampleMetadata, materialize_paired_window


class WindowPairingTest(unittest.TestCase):
    def test_raw_and_features_are_derived_from_the_same_manifest_slice(self) -> None:
        signal = np.arange(5000, dtype=np.float64)
        metadata = PairedSampleMetadata(
            sample_id="source:train:000001024",
            source_id="source:train",
            parent_source_id="source",
            condition_id="load_0",
            window_start=1024,
            window_end=3072,
            split="train",
            label=0,
            sampling_rate=12000,
            raw_path="normal_0.mat",
        )
        extractor_inputs: list[np.ndarray] = []

        def stub_extractor(window: np.ndarray) -> np.ndarray:
            extractor_inputs.append(window.copy())
            return np.linspace(window[0], window[-1], 151)

        paired = materialize_paired_window(metadata, signal, stub_extractor)
        expected = signal[1024:3072]
        np.testing.assert_array_equal(paired.raw_signal[0], expected)
        np.testing.assert_array_equal(extractor_inputs[0], expected)
        self.assertEqual(paired.features.shape, (151,))

    def test_incorrect_feature_dimension_is_rejected(self) -> None:
        metadata = PairedSampleMetadata(
            sample_id="source:train:000000000",
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
        with self.assertRaisesRegex(ValueError, "features must have shape"):
            materialize_paired_window(metadata, np.zeros(2048), lambda _: np.zeros(150))


if __name__ == "__main__":
    unittest.main()
