#!/usr/bin/env python3
"""
seed_legacy_dataset.py
----------------------
Prepare and optionally copy legacy synthetic dataset files into the new layout.

Design goals:
- Legacy files remain untouched in-place.
- Copy is one-time and gated by an explicit success marker.
- IDs are preserved during copy.
- Manifest captures plan + execution details.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime

from data_utils.id_allocator import build_id_space_state, SINGLE_PATTERN, BENCH_PATTERN


@dataclass
class CopySummary:
    copied_single: int
    copied_bench: int
    skipped_existing_single: int
    skipped_existing_bench: int


def _copy_matching_files(src_dir: str, dst_dir: str, pattern, overwrite: bool) -> tuple[int, int]:
    """Copy files that match *pattern* from src_dir to dst_dir."""
    copied = 0
    skipped_existing = 0

    if not os.path.isdir(src_dir):
        return copied, skipped_existing

    os.makedirs(dst_dir, exist_ok=True)

    for name in sorted(os.listdir(src_dir)):
        if not pattern.match(name):
            continue

        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)

        if os.path.exists(dst) and not overwrite:
            skipped_existing += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    return copied, skipped_existing


def _count_matching_files(directory: str, pattern) -> int:
    if not os.path.isdir(directory):
        return 0
    count = 0
    for name in os.listdir(directory):
        if pattern.match(name):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed legacy synthetic files into new dataset roots")
    parser.add_argument("--legacy-single-dir", required=True)
    parser.add_argument("--legacy-bench-dir", required=True)
    parser.add_argument("--synthetic-single-dir", required=True)
    parser.add_argument("--synthetic-bench-dir", required=True)

    parser.add_argument(
        "--manifest",
        required=True,
        help="Output JSON manifest path",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Perform copy after validating success marker",
    )
    parser.add_argument(
        "--success-marker",
        default=None,
        help="Required marker file path when --copy is enabled",
    )
    parser.add_argument(
        "--once-marker",
        default=None,
        help="Marker created after successful copy to enforce one-time behavior",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing destination files",
    )

    args = parser.parse_args()

    state = build_id_space_state(
        legacy_single_dir=args.legacy_single_dir,
        legacy_bench_dir=args.legacy_bench_dir,
        synthetic_single_dir=args.synthetic_single_dir,
        synthetic_bench_dir=args.synthetic_bench_dir,
    )

    legacy_counts = {
        "single": _count_matching_files(args.legacy_single_dir, SINGLE_PATTERN),
        "benchs": _count_matching_files(args.legacy_bench_dir, BENCH_PATTERN),
    }

    planned_next_ids = {
        "single_next": state.single_next,
        "bench_next": state.bench_next,
    }

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "legacy_single_dir": os.path.abspath(args.legacy_single_dir),
            "legacy_bench_dir": os.path.abspath(args.legacy_bench_dir),
            "synthetic_single_dir": os.path.abspath(args.synthetic_single_dir),
            "synthetic_bench_dir": os.path.abspath(args.synthetic_bench_dir),
            "copy": args.copy,
            "success_marker": os.path.abspath(args.success_marker) if args.success_marker else None,
            "once_marker": os.path.abspath(args.once_marker) if args.once_marker else None,
            "overwrite": args.overwrite,
        },
        "id_space": asdict(state),
        "legacy_counts": legacy_counts,
        "planned_next_ids": planned_next_ids,
        "copy_summary": None,
        "status": "planned",
    }

    if args.copy:
        if not args.success_marker:
            raise SystemExit("--copy requires --success-marker")
        if not os.path.isfile(args.success_marker):
            raise SystemExit(f"success marker not found: {args.success_marker}")

        if args.once_marker and os.path.exists(args.once_marker):
            raise SystemExit(
                f"one-time copy already completed (once marker exists): {args.once_marker}"
            )

        copied_single, skipped_single = _copy_matching_files(
            args.legacy_single_dir,
            args.synthetic_single_dir,
            SINGLE_PATTERN,
            args.overwrite,
        )
        copied_bench, skipped_bench = _copy_matching_files(
            args.legacy_bench_dir,
            args.synthetic_bench_dir,
            BENCH_PATTERN,
            args.overwrite,
        )

        summary = CopySummary(
            copied_single=copied_single,
            copied_bench=copied_bench,
            skipped_existing_single=skipped_single,
            skipped_existing_bench=skipped_bench,
        )
        manifest["copy_summary"] = asdict(summary)
        manifest["status"] = "copied"

        if args.once_marker:
            marker_parent = os.path.dirname(args.once_marker)
            if marker_parent:
                os.makedirs(marker_parent, exist_ok=True)
            with open(args.once_marker, "w", encoding="utf-8") as fh:
                fh.write("legacy copy completed\n")

    manifest_parent = os.path.dirname(args.manifest)
    if manifest_parent:
        os.makedirs(manifest_parent, exist_ok=True)

    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote manifest: {args.manifest}")
    print(json.dumps({
        "single_next_id": state.single_next,
        "bench_next_id": state.bench_next,
        "status": manifest["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
