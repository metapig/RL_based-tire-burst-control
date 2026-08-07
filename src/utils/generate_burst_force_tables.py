"""Generate intermediate CarSim tire-force tables by linear interpolation.

The source files must use the CarSim ``*_TIRE_CARPET 2D_FROM_ZERO`` format.
Only the force/moment cells are interpolated; the slip/load axes must match.
Generated CSV files retain the CarSim table layout:

    first row:    0, Fz_1, ..., Fz_n
    later rows:   slip, value_1, ..., value_n
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_SOURCES = {
    "Fx": (
        Path(r"D:\CarSim2023\Databases\Tires\Fx\TireFx_e2eea0b6-a18f-49e9-acae-e78d4587afe6.par"),
        Path(r"D:\CarSim2023\Databases\Tires\Fx\TireFx_59ff8167-24e0-484d-845e-6946e5597b34.par"),
    ),
    "Fy": (
        Path(r"D:\CarSim2023\Databases\Tires\Fy\TireFy_4a6a64d4-89b0-4460-b52b-6b8720461ac6.par"),
        Path(r"D:\CarSim2023\Databases\Tires\Fy\TireFy_efd067b1-5b7f-406a-830a-324252abc172.par"),
    ),
    "Mz": (
        Path(r"D:\CarSim2023\Databases\Tires\Mz\TireMz_9147d723-8515-454c-a219-32675f79a027.par"),
        Path(r"D:\CarSim2023\Databases\Tires\Mz\TireMz_c3c085d8-b8db-4f26-a6e4-032014e75fa1.par"),
    ),
}


def parse_carsim_table(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if "_TIRE_CARPET" in line
    ) + 1
    end = next(
        index
        for index, line in enumerate(lines[start:], start=start)
        if line.strip() == "ENDTABLE"
    )
    return np.asarray(
        [
            [float(value.strip()) for value in line.split(",")]
            for line in lines[start:end]
        ],
        dtype=np.float64,
    )


def interpolate_tables(
    normal: np.ndarray, burst: np.ndarray, severity: float
) -> np.ndarray:
    if normal.shape != burst.shape:
        raise ValueError(
            f"Table shapes differ: normal={normal.shape}, burst={burst.shape}."
        )
    if not np.allclose(normal[:, 0], burst[:, 0], rtol=0.0, atol=1e-12):
        raise ValueError("Slip-angle/slip-ratio axes do not match.")
    if not np.allclose(normal[0, :], burst[0, :], rtol=0.0, atol=1e-12):
        raise ValueError("Vertical-load axes do not match.")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"Severity must be in [0, 1], got {severity}.")

    interpolated = normal.copy()
    interpolated[1:, 1:] = (
        (1.0 - severity) * normal[1:, 1:]
        + severity * burst[1:, 1:]
    )
    return interpolated


def format_number(value: float) -> str:
    if abs(value) < 5e-13:
        return "0"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def write_csv(path: Path, table: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows(
            [[format_number(value) for value in row] for row in table]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate matching normal and full-burst CarSim tables."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=(0.25, 0.50, 0.75),
        help="Burst severities in [0, 1].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_tire_tables"),
    )
    for component, (normal_default, burst_default) in DEFAULT_SOURCES.items():
        parser.add_argument(
            f"--normal-{component.lower()}",
            type=Path,
            default=normal_default,
        )
        parser.add_argument(
            f"--burst-{component.lower()}",
            type=Path,
            default=burst_default,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = {
        component: (
            getattr(args, f"normal_{component.lower()}"),
            getattr(args, f"burst_{component.lower()}"),
        )
        for component in DEFAULT_SOURCES
    }
    source_tables = {
        component: (parse_carsim_table(normal), parse_carsim_table(burst))
        for component, (normal, burst) in source_paths.items()
    }

    manifest = {
        "definition": "severity=0 is normal; severity=1 is full burst",
        "formula": "Q(severity)=(1-severity)*Q_normal+severity*Q_full_burst",
        "levels": args.levels,
        "sources": {
            component: {
                "normal": str(normal),
                "full_burst": str(burst),
            }
            for component, (normal, burst) in source_paths.items()
        },
    }

    for severity in args.levels:
        level_dir = args.output_dir / f"burst_{round(100 * severity):03d}"
        for component, (normal, burst) in source_tables.items():
            table = interpolate_tables(normal, burst, severity)
            write_csv(level_dir / f"{component}.csv", table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
