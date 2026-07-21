"""Paper-aligned sliding-window trigeminal KS change statistics.

The supplied paper specifies the sliding-window framework, overlapping
trigeminal branches, and the scaled two-sample KS decision statistic, but does
not publish executable code. This module is therefore a documented clean-room
implementation rather than a claim of byte-identical reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BranchScore = Literal["ks", "variance", "hybrid"]


TSTKS_FEATURE_NAMES: tuple[str, ...] = (
    "tstks_count",
    "tstks_density",
    "tstks_score_mean",
    "tstks_score_std",
    "tstks_score_max",
    "tstks_score_min",
    "tstks_score_skew",
    "tstks_score_kurtosis",
    "tstks_amplitude_mean",
    "tstks_amplitude_std",
    "tstks_amplitude_max",
    "tstks_amplitude_min",
    "tstks_amplitude_skew",
    "tstks_amplitude_kurtosis",
    "tstks_interval_mean",
    "tstks_interval_std",
    "tstks_interval_max",
    "tstks_interval_min",
    "tstks_interval_skew",
    "tstks_interval_kurtosis",
)


@dataclass(frozen=True)
class TSTKSConfig:
    """Configuration for local sliding-window TSTKS detection."""

    analysis_window_size: int = 256
    analysis_hop_size: int = 128
    min_segment_size: int = 32
    branch_overlap_ratio: float = 0.5
    max_depth: int = 6
    candidate_step: int = 2
    ks_threshold: float = 1.36
    variance_threshold: float = 0.5
    branch_score: BranchScore = "hybrid"
    min_distance: int = 30
    peak_alignment_radius: int = 0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.analysis_window_size < 2 * self.min_segment_size:
            raise ValueError("analysis_window_size must fit two minimum segments")
        if self.analysis_hop_size <= 0:
            raise ValueError("analysis_hop_size must be positive")
        if self.min_segment_size < 2:
            raise ValueError("min_segment_size must be at least 2")
        if not 0.0 <= self.branch_overlap_ratio < 1.0:
            raise ValueError("branch_overlap_ratio must be in [0, 1)")
        if self.max_depth < 0 or self.candidate_step <= 0:
            raise ValueError("max_depth and candidate_step must be non-negative/positive")
        if self.ks_threshold <= 0 or self.variance_threshold <= 0:
            raise ValueError("thresholds must be positive")
        if self.branch_score not in {"ks", "variance", "hybrid"}:
            raise ValueError(f"unsupported branch_score: {self.branch_score}")
        if self.min_distance < 0 or self.peak_alignment_radius < 0:
            raise ValueError("distance and alignment radius cannot be negative")


@dataclass(frozen=True)
class ChangePoint:
    """One accepted change-point candidate in global signal coordinates."""

    index: int
    score: float
    window_start: int
    local_index: int
    search_depth: int


@dataclass(frozen=True)
class _SplitScore:
    index: int
    ks: float
    variance: float


class SlidingTSTKSDetector:
    """Detect local distribution changes with overlapping trigeminal search."""

    def __init__(self, config: TSTKSConfig | None = None) -> None:
        self.config = config or TSTKSConfig()

    @staticmethod
    def scaled_ks(left: Sequence[float], right: Sequence[float]) -> float:
        """Return sqrt(m*n/(m+n)) times the two-sample empirical KS distance."""

        x = np.asarray(left, dtype=np.float64).reshape(-1)
        y = np.asarray(right, dtype=np.float64).reshape(-1)
        if x.size == 0 or y.size == 0:
            raise ValueError("scaled_ks requires two non-empty samples")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("scaled_ks input must be finite")

        support = np.sort(np.concatenate((x, y)))
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        cdf_x = np.searchsorted(x_sorted, support, side="right") / x.size
        cdf_y = np.searchsorted(y_sorted, support, side="right") / y.size
        distance = float(np.max(np.abs(cdf_x - cdf_y)))
        scale = np.sqrt((x.size * y.size) / (x.size + y.size))
        return float(scale * distance)

    def branch_intervals(self, start: int, end: int) -> tuple[tuple[int, int], ...]:
        """Return left/middle/right child intervals with configured overlap."""

        length = end - start
        if length <= 0:
            raise ValueError("end must be greater than start")

        denominator = 3.0 - 2.0 * self.config.branch_overlap_ratio
        child_size = min(length, max(2, int(round(length / denominator))))
        offsets = (0, int(round((length - child_size) / 2)), length - child_size)
        intervals: list[tuple[int, int]] = []
        for offset in offsets:
            interval = (start + offset, start + offset + child_size)
            if interval not in intervals:
                intervals.append(interval)
        return tuple(intervals)

    def _variance_fluctuation(self, left: FloatArray, right: FloatArray) -> float:
        left_var = float(np.var(left))
        right_var = float(np.var(right))
        return abs(left_var - right_var) / (left_var + right_var + self.config.epsilon)

    def _best_split(self, signal: FloatArray, start: int, end: int) -> _SplitScore | None:
        min_size = self.config.min_segment_size
        first = start + min_size
        last = end - min_size
        if first > last:
            return None

        candidates = list(range(first, last + 1, self.config.candidate_step))
        if candidates[-1] != last:
            candidates.append(last)

        best: _SplitScore | None = None
        for split in candidates:
            left = signal[start:split]
            right = signal[split:end]
            score = _SplitScore(
                index=split,
                ks=self.scaled_ks(left, right),
                variance=self._variance_fluctuation(left, right),
            )
            if best is None or self._navigation_score(score) > self._navigation_score(best):
                best = score
        return best

    def _navigation_score(self, score: _SplitScore) -> float:
        if self.config.branch_score == "ks":
            return score.ks
        if self.config.branch_score == "variance":
            return score.variance
        return max(
            score.ks / self.config.ks_threshold,
            score.variance / self.config.variance_threshold,
        )

    def _search_window(self, signal: FloatArray, start: int, end: int) -> tuple[_SplitScore, int] | None:
        current_start, current_end = start, end
        depth = 0

        while depth < self.config.max_depth:
            branches = self.branch_intervals(current_start, current_end)
            scored: list[tuple[float, tuple[int, int]]] = []
            for branch_start, branch_end in branches:
                branch_best = self._best_split(signal, branch_start, branch_end)
                if branch_best is not None:
                    scored.append((self._navigation_score(branch_best), (branch_start, branch_end)))
            if not scored:
                break

            _, selected = max(scored, key=lambda item: item[0])
            if selected == (current_start, current_end):
                break
            current_start, current_end = selected
            depth += 1

        best = self._best_split(signal, current_start, current_end)
        return (best, depth) if best is not None else None

    def _window_starts(self, length: int) -> list[int]:
        width = self.config.analysis_window_size
        if length < width:
            return []
        starts = list(range(0, length - width + 1, self.config.analysis_hop_size))
        final_start = length - width
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def _align_to_peak(self, signal: FloatArray, index: int) -> int:
        radius = self.config.peak_alignment_radius
        if radius == 0:
            return index
        start = max(0, index - radius)
        end = min(signal.size, index + radius + 1)
        return int(start + np.argmax(np.abs(signal[start:end])))

    def _merge_candidates(self, candidates: list[ChangePoint]) -> list[ChangePoint]:
        kept: list[ChangePoint] = []
        for candidate in sorted(candidates, key=lambda item: (-item.score, item.index)):
            if all(abs(candidate.index - item.index) >= self.config.min_distance for item in kept):
                kept.append(candidate)
        return sorted(kept, key=lambda item: item.index)

    def detect(self, signal: Sequence[float]) -> list[ChangePoint]:
        """Detect accepted change points in a finite one-dimensional signal."""

        values = np.asarray(signal, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("signal must contain only finite values")

        candidates: list[ChangePoint] = []
        width = self.config.analysis_window_size
        for window_start in self._window_starts(values.size):
            result = self._search_window(values, window_start, window_start + width)
            if result is None:
                continue
            best, depth = result
            if best.ks < self.config.ks_threshold:
                continue
            aligned = self._align_to_peak(values, best.index)
            candidates.append(
                ChangePoint(
                    index=aligned,
                    score=best.ks,
                    window_start=window_start,
                    local_index=best.index - window_start,
                    search_depth=depth,
                )
            )
        return self._merge_candidates(candidates)


def _six_statistics(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return np.zeros(6, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    if std <= 1e-12:
        skew = 0.0
        kurtosis = 0.0
    else:
        centered = (values - mean) / std
        skew = float(np.mean(centered**3))
        kurtosis = float(np.mean(centered**4) - 3.0)
    return np.asarray((mean, std, maximum, minimum, skew, kurtosis), dtype=np.float64)


def extract_tstks_features(
    signal: Sequence[float], detector: SlidingTSTKSDetector | None = None
) -> FloatArray:
    """Return the legacy-compatible 20-dimensional TSTKS feature vector."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot extract TSTKS features from an empty signal")
    if not np.isfinite(values).all():
        raise ValueError("signal must contain only finite values")

    active_detector = detector or SlidingTSTKSDetector()
    points = active_detector.detect(values)
    indices = np.asarray([point.index for point in points], dtype=np.int64)
    scores = np.asarray([point.score for point in points], dtype=np.float64)
    amplitudes = np.abs(values[indices]) if indices.size else np.asarray([], dtype=np.float64)
    intervals = np.diff(indices).astype(np.float64) if indices.size > 1 else np.asarray([], dtype=np.float64)

    features = np.concatenate(
        (
            np.asarray((float(indices.size), float(indices.size / values.size))),
            _six_statistics(scores),
            _six_statistics(amplitudes),
            _six_statistics(intervals),
        )
    )
    if features.shape != (20,) or not np.isfinite(features).all():
        raise RuntimeError("TSTKS feature contract violated")
    return features

