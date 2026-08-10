"""Restore the corrected 100% burst carpets to the CarSim datasets.

Run from any working directory with:

    python -m src.utils.restore_full_burst_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.generate_burst_force_tables import (
    DEFAULT_OUTPUT_DIR,
    LEGACY_FULL_BURST_SOURCES,
)
from src.utils.utils import replace_tire_force_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the corrected 100% Fx/Fy/Mz burst tables into CarSim."
        )
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "burst_100",
    )
    for component, default_path in LEGACY_FULL_BURST_SOURCES.items():
        parser.add_argument(
            f"--target-{component.lower()}",
            type=Path,
            default=default_path,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = {
        component: getattr(args, f"target_{component.lower()}")
        for component in LEGACY_FULL_BURST_SOURCES
    }
    written = replace_tire_force_tables(args.table_dir, targets)
    print("Restored corrected 100% burst tables:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
