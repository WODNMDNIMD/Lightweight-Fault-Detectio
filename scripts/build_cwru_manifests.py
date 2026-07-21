"""Build leakage-safe CWRU in-domain manifests from local MAT files."""

from __future__ import annotations

import argparse
from collections import Counter

from ctgsd.data import build_window_manifest, split_sources_by_time, write_split_manifests
from ctgsd.data.cwru import discover_cwru_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--max-samples", type=int, default=120000)
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    args = parser.parse_args()

    sources = discover_cwru_sources(args.data_root, args.max_samples)
    segments = split_sources_by_time(sources, ratios=(0.6, 0.2, 0.2))
    manifest = build_window_manifest(
        segments,
        window_length=args.window_length,
        overlap_ratio=args.overlap_ratio,
    )
    paths = write_split_manifests(manifest, args.output_dir)
    counts = Counter(record.split for record in manifest)
    for split in ("train", "val", "test"):
        print(f"{split}: {counts[split]} -> {paths[split]}")


if __name__ == "__main__":
    main()
