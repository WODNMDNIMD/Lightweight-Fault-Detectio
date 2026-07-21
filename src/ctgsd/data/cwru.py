"""Strict CWRU 12 kHz drive-end source discovery and loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .split_by_source import SourceRecord

_FAULT_LABELS = {
    "B007": 1,
    "B014": 2,
    "B021": 3,
    "IR007": 4,
    "IR014": 5,
    "IR021": 6,
    "OR007": 7,
    "OR014": 8,
    "OR021": 9,
}


def parse_cwru_identity(path: str | Path) -> tuple[str, int, int] | None:
    """Return fault code, label, and load without substring-based mislabeling."""

    file_path = Path(path)
    stem = file_path.stem.upper()
    normal = re.fullmatch(r"NORMAL_([0-3])", stem)
    if normal:
        return "NORMAL", 0, int(normal.group(1))

    fault = re.fullmatch(r"(B|IR|OR)(007|014|021)(?:@6)?_([0-3])", stem)
    if not fault:
        return None
    code = f"{fault.group(1)}{fault.group(2)}"
    return code, _FAULT_LABELS[code], int(fault.group(3))


def load_cwru_drive_end(
    path: str | Path, expected_key: str | None = None
) -> NDArray[np.float64]:
    """Load exactly one finite drive-end time series from a CWRU MAT file."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - exercised in configured runtime
        raise RuntimeError("scipy is required to load CWRU MAT files") from exc

    file_path = Path(path)
    payload = loadmat(file_path)
    keys = sorted(key for key in payload if "DE_TIME" in key.upper())
    if expected_key is not None:
        matching = [key for key in keys if key.upper() == expected_key.upper()]
        if len(matching) != 1:
            raise ValueError(
                f"expected key {expected_key!r} in {file_path}, found DE keys {keys}"
            )
        selected_key = matching[0]
    elif len(keys) == 1:
        selected_key = keys[0]
    else:
        raise ValueError(f"expected one DE_time array in {file_path}, found {keys}")
    signal = np.asarray(payload[selected_key], dtype=np.float64).reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"invalid drive-end signal in {file_path}")
    return signal


def _candidate_paths(root: Path) -> Iterable[Path]:
    normal_dir = root / "Normal Baseline"
    drive_dir = root / "12k Drive End Bearing Fault Data"
    yield from sorted(normal_dir.glob("normal_[0-3].mat"))
    yield from sorted((drive_dir / "Ball").glob("00*/B0??_[0-3].mat"))
    yield from sorted((drive_dir / "Inner Race").glob("00*/IR0??_[0-3].mat"))
    # Use the centered (6 o'clock) outer-race location for one balanced source
    # per fault size/load. Other positions are separate experimental conditions.
    yield from sorted((drive_dir / "Outer Race" / "Centered").glob("00*/OR0??@6_[0-3].mat"))


def discover_cwru_sources(
    root: str | Path, max_samples_per_source: int | None = None
) -> list[SourceRecord]:
    """Discover the balanced 10-class, 12 kHz drive-end protocol (40 files).

    CWRU normal recordings are substantially longer than most fault records.
    `max_samples_per_source` provides an explicit, reproducible prefix limit;
    the configured experiment protocol uses 120000 samples for every source.
    """

    data_root = Path(root)
    if max_samples_per_source is not None and max_samples_per_source <= 0:
        raise ValueError("max_samples_per_source must be positive or None")
    records: list[SourceRecord] = []
    seen: set[tuple[str, int]] = set()
    for path in _candidate_paths(data_root):
        identity = parse_cwru_identity(path)
        if identity is None:
            continue
        fault_code, label, load = identity
        key = (fault_code, load)
        if key in seen:
            raise ValueError(f"duplicate CWRU source for {key}: {path}")
        seen.add(key)
        # The local normal_2.mat contains both X098 and X099 arrays. Select the
        # load-specific official key explicitly instead of relying on MAT key order.
        expected_key = f"X{97 + load:03d}_DE_time" if fault_code == "NORMAL" else None
        signal = load_cwru_drive_end(path, expected_key=expected_key)
        records.append(
            SourceRecord(
                source_id=f"cwru:{fault_code.lower()}:load{load}:de12k",
                raw_path=str(path.resolve()),
                label=label,
                condition_id=f"load_{load}",
                sampling_rate=12000,
                num_samples=int(
                    min(signal.size, max_samples_per_source)
                    if max_samples_per_source is not None
                    else signal.size
                ),
            )
        )

    expected = {(fault, load) for fault in ("NORMAL", *_FAULT_LABELS) for load in range(4)}
    missing = expected - seen
    if missing:
        raise FileNotFoundError(f"CWRU protocol is incomplete; missing {sorted(missing)}")
    if len(records) != 40:
        raise RuntimeError(f"expected 40 balanced sources, found {len(records)}")
    return sorted(records, key=lambda item: item.source_id)
