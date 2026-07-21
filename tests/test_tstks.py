from __future__ import annotations

import unittest

import numpy as np

from ctgsd.features.tstks import (
    SlidingTSTKSDetector,
    TSTKSConfig,
    TSTKS_FEATURE_NAMES,
    extract_tstks_features,
)


class SlidingTSTKSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TSTKSConfig(
            analysis_window_size=256,
            analysis_hop_size=128,
            min_segment_size=24,
            candidate_step=1,
            ks_threshold=1.36,
            min_distance=40,
            branch_score="ks",
        )
        self.detector = SlidingTSTKSDetector(self.config)

    def test_trigeminal_branches_overlap_and_cover_parent(self) -> None:
        branches = self.detector.branch_intervals(0, 256)
        self.assertEqual(branches, ((0, 128), (64, 192), (128, 256)))
        self.assertGreater(branches[0][1], branches[1][0])
        self.assertGreater(branches[1][1], branches[2][0])

    def test_constant_signal_has_no_change_point(self) -> None:
        points = self.detector.detect(np.zeros(1024, dtype=np.float64))
        self.assertEqual(points, [])

    def test_single_mean_change_is_localized(self) -> None:
        rng = np.random.default_rng(7)
        signal = np.concatenate((rng.normal(0.0, 0.2, 512), rng.normal(3.0, 0.2, 512)))
        points = self.detector.detect(signal)
        self.assertTrue(any(abs(point.index - 512) <= 8 for point in points), points)

    def test_multiple_mean_changes_are_detected(self) -> None:
        rng = np.random.default_rng(11)
        signal = np.concatenate(
            (
                rng.normal(0.0, 0.15, 320),
                rng.normal(2.5, 0.15, 320),
                rng.normal(-2.0, 0.15, 384),
            )
        )
        points = self.detector.detect(signal)
        self.assertTrue(any(abs(point.index - 320) <= 10 for point in points), points)
        self.assertTrue(any(abs(point.index - 640) <= 10 for point in points), points)

    def test_feature_contract_is_finite_and_twenty_dimensional(self) -> None:
        rng = np.random.default_rng(13)
        signal = np.concatenate((rng.normal(0.0, 0.2, 512), rng.normal(2.0, 0.2, 512)))
        features = extract_tstks_features(signal, self.detector)
        self.assertEqual(features.shape, (20,))
        self.assertEqual(len(TSTKS_FEATURE_NAMES), 20)
        self.assertTrue(np.isfinite(features).all())

    def test_detection_is_deterministic(self) -> None:
        rng = np.random.default_rng(17)
        signal = np.concatenate((rng.normal(0.0, 0.2, 512), rng.normal(2.0, 0.2, 512)))
        first = self.detector.detect(signal)
        second = self.detector.detect(signal)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

