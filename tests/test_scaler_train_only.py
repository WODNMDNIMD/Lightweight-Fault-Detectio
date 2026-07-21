from __future__ import annotations

import unittest

import numpy as np

from ctgsd.preprocessing import TrainOnlyStandardScaler


class ScalerTrainOnlyTest(unittest.TestCase):
    def test_validation_and_test_fit_are_rejected(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        for split in ("val", "test"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "only for split='train'"):
                    TrainOnlyStandardScaler().fit(values, split=split)

    def test_validation_transform_uses_training_statistics(self) -> None:
        train = np.array([[0.0, 10.0], [2.0, 14.0]])
        validation = np.array([[11.0, 112.0]])
        scaler = TrainOnlyStandardScaler().fit(train, split="train")
        transformed = scaler.transform(validation)
        np.testing.assert_allclose(scaler.mean_, np.array([1.0, 12.0]))
        np.testing.assert_allclose(scaler.scale_, np.array([1.0, 2.0]))
        np.testing.assert_allclose(transformed, np.array([[10.0, 50.0]]))


if __name__ == "__main__":
    unittest.main()
