"""Generate CarSim tire-force carpets for several burst severities.

The CarSim full-burst datasets are overwritten at every environment reset, so
they are not reliable source files after training has started.  Fx and Fy for
the original 100% burst are therefore recovered from one of the previously
generated intermediate tables.  The recovery is the inverse of the linear
interpolation used to create that table.

The original full-burst Mz table was ten times the normal-tire Mz table.  That
is not a physically plausible consequence of losing most lateral force.  The
corrected full-burst Mz keeps the normal curve shape and scales it to 25%,
matching the full-burst Fy scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated_tire_tables"
DEFAULT_REFERENCE_SEVERITY = 0.75
DEFAULT_MZ_FULL_BURST_SCALE = 0.25

DEFAULT_NORMAL_SOURCES = {
    "Fx": Path(
        r"D:\CarSim2023\Databases\Tires\Fx\TireFx_e2eea0b6-"
        r"a18f-49e9-acae-e78d4587afe6.par"
    ),
    "Fy": Path(
        r"D:\CarSim2023\Databases\Tires\Fy\TireFy_4a6a64d4-"
        r"89b0-4460-b52b-6b8720461ac6.par"
    ),
    "Mz": Path(
        r"D:\CarSim2023\Databases\Tires\Mz\TireMz_9147d723-"
        r"8515-454c-a219-32675f79a027.par"
    ),
}

LEGACY_FULL_BURST_SOURCES = {
    "Fx": Path(
        r"D:\CarSim2023\Databases\Tires\Fx\TireFx_59ff8167-"
        r"24e0-484d-845e-6946e5597b34.par"
    ),
    "Fy": Path(
        r"D:\CarSim2023\Databases\Tires\Fy\TireFy_efd067b1-"
        r"5b7f-406a-830a-324252abc172.par"
    ),
    "Mz": Path(
        r"D:\CarSim2023\Databases\Tires\Mz\TireMz_c3c085d8-"
        r"b8db-4f26-a6e4-032014e75fa1.par"
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


def read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def validate_matching_axes(first: np.ndarray, second: np.ndarray) -> None:
    if first.shape != second.shape:
        raise ValueError(
            f"Table shapes differ: first={first.shape}, second={second.shape}."
        )
    if not np.allclose(first[:, 0], second[:, 0], rtol=0.0, atol=1e-9):
        raise ValueError("Slip-angle/slip-ratio axes do not match.")
    if not np.allclose(first[0, :], second[0, :], rtol=0.0, atol=1e-9):
        raise ValueError("Vertical-load axes do not match.")


def interpolate_tables(
    normal: np.ndarray, full_burst: np.ndarray, severity: float
) -> np.ndarray:
    validate_matching_axes(normal, full_burst)
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"Severity must be in [0, 1], got {severity}.")

    interpolated = normal.copy()
    interpolated[1:, 1:] = (
        (1.0 - severity) * normal[1:, 1:]
        + severity * full_burst[1:, 1:]
    )
    return interpolated


def recover_full_burst_table(
    normal: np.ndarray,
    intermediate: np.ndarray,
    intermediate_severity: float,
) -> np.ndarray:
    """Invert linear interpolation to recover the original full-burst table."""
    validate_matching_axes(normal, intermediate)
    if not 0.0 < intermediate_severity <= 1.0:
        raise ValueError(
            "Reference severity must be in (0, 1], got "
            f"{intermediate_severity}."
        )

    full_burst = normal.copy()
    full_burst[1:, 1:] = (
        intermediate[1:, 1:]
        - (1.0 - intermediate_severity) * normal[1:, 1:]
    ) / intermediate_severity
    return full_burst


def corrected_full_burst_mz(
    normal_mz: np.ndarray, scale: float = DEFAULT_MZ_FULL_BURST_SCALE
) -> np.ndarray:
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"Mz scale must be in [0, 1], got {scale}.")
    full_burst_mz = normal_mz.copy()
    full_burst_mz[1:, 1:] *= scale
    return full_burst_mz


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
        description=(
            "Generate 25/50/75/100% burst tables, with a corrected Mz curve."
        )
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=(0.25, 0.50, 0.75, 1.00),
        help="Burst severities in [0, 1].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "burst_075",
        help=(
            "Existing intermediate tables from which the original 100% Fx/Fy "
            "are recovered."
        ),
    )
    parser.add_argument(
        "--reference-severity",
        type=float,
        default=DEFAULT_REFERENCE_SEVERITY,
    )
    parser.add_argument(
        "--mz-full-burst-scale",
        type=float,
        default=DEFAULT_MZ_FULL_BURST_SCALE,
        help="Corrected full-burst Mz divided by normal-tire Mz.",
    )
    for component, default_path in DEFAULT_NORMAL_SOURCES.items():
        parser.add_argument(
            f"--normal-{component.lower()}",
            type=Path,
            default=default_path,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normal_paths = {
        component: getattr(args, f"normal_{component.lower()}")
        for component in DEFAULT_NORMAL_SOURCES
    }
    normal_tables = {
        component: parse_carsim_table(path)
        for component, path in normal_paths.items()
    }

    full_burst_tables = {
        component: recover_full_burst_table(
            normal_tables[component],
            read_csv(args.reference_dir / f"{component}.csv"),
            args.reference_severity,
        )
        for component in ("Fx", "Fy")
    }
    full_burst_tables["Mz"] = corrected_full_burst_mz(
        normal_tables["Mz"], args.mz_full_burst_scale
    )

    for severity in args.levels:
        level_dir = args.output_dir / f"burst_{round(100 * severity):03d}"
        for component, full_burst in full_burst_tables.items():
            table = interpolate_tables(
                normal_tables[component], full_burst, severity
            )
            write_csv(level_dir / f"{component}.csv", table)

    manifest = {
        "definition": "severity=0 is normal; severity=1 is full burst",
        "formula": (
            "Q(severity)=(1-severity)*Q_normal+severity*Q_full_burst"
        ),
        "levels": list(args.levels),
        "normal_sources": {
            component: str(path) for component, path in normal_paths.items()
        },
        "full_burst_sources": {
            "Fx": (
                f"recovered from {args.reference_dir / 'Fx.csv'} at "
                f"severity={args.reference_severity}"
            ),
            "Fy": (
                f"recovered from {args.reference_dir / 'Fy.csv'} at "
                f"severity={args.reference_severity}"
            ),
            "Mz": (
                "corrected from normal Mz: "
                f"Mz_full_burst={args.mz_full_burst_scale}*Mz_normal"
            ),
        },
        "legacy_carsim_full_burst_datasets": {
            component: str(path)
            for component, path in LEGACY_FULL_BURST_SOURCES.items()
        },
        "note": (
            "The legacy full-burst CarSim datasets are runtime targets and may "
            "have been overwritten by an intermediate-severity episode."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
