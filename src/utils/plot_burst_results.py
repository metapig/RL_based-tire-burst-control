from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl


METRIC_NAMES = [
    "AVz",   # 横摆角速度 deg/s
    "Ay",    # 侧向加速度 g
    "Beta",  # 质心侧偏角 deg
    "Steer_SW",  # 方向盘转角 deg（本题不画）
    "Yaw",   # 偏航角 deg
    "Yo",    # 侧向误差 m
]

DISPLAY_NAMES = {
    "AVz": "横摆角速度 (deg/s)",
    "Ay": "侧向加速度 (g)",
    "Beta": "质心侧偏角 (deg)",
    "Yaw": "偏航角 (deg)",
    "Yo": "侧向误差 (m)",
}

CONTROL_LABELS = ["无控制", "LPV-MPC", "RL-based MPC"]
CONTROL_COLORS = ["C0", "C1", "C2"]


def setup_matplotlib() -> None:
    """Try to enable Chinese labels and consistent plotting style."""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def split_row_text(cell_value) -> List[float]:
    """Split a comma-separated row into floats."""
    if cell_value is None:
        return []
    if isinstance(cell_value, (int, float, np.number)):
        return [float(cell_value)]
    text = str(cell_value).strip()
    if not text:
        return []
    return [float(x) for x in text.split(",")]


def load_xlsx_case(xlsx_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, float], np.ndarray]:
    """
    Read the CarSim xlsx file.

    Expected format:
    - 3 columns: no control / LPV-MPC / RL-based MPC
    - first row: header repeated in each column
    - rows 2..end-1: comma-separated metric values
    - last row: rewards for the 3 controllers
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))

    if len(rows) < 3:
        raise ValueError(f"{xlsx_path} rows are too few: {len(rows)}")
    if len(rows[0]) < 3:
        raise ValueError(f"{xlsx_path} should contain 3 columns, got {len(rows[0])}")

    series = {k: [] for k in ["no_ctrl", "lpv_mpc", "rl_mpc"]}
    rewards = np.zeros(3, dtype=float)

    # Parse the metric rows.
    data_rows = rows[1:-1]
    for row in data_rows:
        for key, cell in zip(series.keys(), row[:3]):
            vals = split_row_text(cell)
            if len(vals) != 6:
                raise ValueError(
                    f"Unexpected row format in {xlsx_path}: expected 6 values, got {len(vals)}\ncell={cell!r}"
                )
            series[key].append(vals)

    # Last row is reward.
    last_row = rows[-1][:3]
    for i, cell in enumerate(last_row):
        rewards[i] = float(cell)

    for key in series:
        series[key] = np.asarray(series[key], dtype=float)

    reward_dict = {
        CONTROL_LABELS[0]: rewards[0],
        CONTROL_LABELS[1]: rewards[1],
        CONTROL_LABELS[2]: rewards[2],
    }
    return series, reward_dict, np.asarray(rewards, dtype=float)


def load_control_csv(csv_path: Path, total_seconds: float = 12.0, burst_time: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read control output csv and rebuild a full 0..12s time axis by padding zeros
    for the first 2 seconds.

    Expected format:
    Wall time, Step, Value
    Only the 3rd column is plotted.
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        raise ValueError(f"{csv_path} should have at least 3 columns")

    step = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    value = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
    if np.isnan(step).any() or np.isnan(value).any():
        raise ValueError(f"{csv_path} contains non-numeric values in Step/Value columns")

    # Infer time step from the Step column if possible (e.g., 2039, 2059, ... => 0.02s).
    step_diff = np.diff(step)
    if len(step_diff) == 0:
        raise ValueError(f"{csv_path} has too few data rows")
    dt = float(np.median(step_diff) / 1000.0)
    if dt <= 0:
        raise ValueError(f"Invalid dt inferred from {csv_path}: {dt}")

    # Build the full axis from 0 to total_seconds with the inferred dt.
    t_full = np.arange(0.0, total_seconds + 1e-12, dt)
    y_full = np.zeros_like(t_full)

    # Put the post-burst data starting at burst_time.
    n_full = len(y_full)
    n_val = len(value)

    # ===== 从尾部对齐 =====
    start_idx = max(n_full - n_val, 0)

    valid_len = min(n_val, n_full)
    y_full[start_idx:] = value[-valid_len:]

    return t_full, y_full


def find_case_files(case_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Find the xlsx and three csv files in one case folder."""
    xlsx_files = sorted(case_dir.glob("*.xlsx")) + sorted(case_dir.glob("*.xls")) + sorted(case_dir.glob("*.xlsm"))
    if not xlsx_files:
        raise FileNotFoundError(f"No xlsx/xls file found in {case_dir}")
    xlsx_path = xlsx_files[0]

    csv_files = list(case_dir.glob("*.csv"))
    if len(csv_files) < 3:
        raise FileNotFoundError(f"Need at least 3 csv files in {case_dir}, found {len(csv_files)}")

    def pick(substring: str) -> Path:
        for f in csv_files:
            if substring.lower() in f.name.lower():
                return f
        raise FileNotFoundError(f"Cannot find csv containing '{substring}' in {case_dir}")

    return xlsx_path, pick("without_ctrl"), pick("with_MPC"), pick("with_RL")


def plot_case_summary(case_dir: Path, out_dir: Path) -> None:
    """Create the 6-panel summary figure for one case (IEEE style)."""

    xlsx_path, _, _, _ = find_case_files(case_dir)
    series, _, _ = load_xlsx_case(xlsx_path)

    # ===== 时间轴 =====
    n_steps = series["no_ctrl"].shape[0]
    t = np.linspace(0.0, 12.0, n_steps)

    # ===== 读取控制量 =====
    _, csv_without, csv_mpc, csv_rl = find_case_files(case_dir)
    t_ctrl_0, y0 = load_control_csv(csv_without)
    t_ctrl_1, y1 = load_control_csv(csv_mpc)
    t_ctrl_2, y2 = load_control_csv(csv_rl)

    # ===== subplot =====
    metric_keys = ["AVz", "Ay", "Beta", "Yaw", "Yo"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    axs = axs.ravel()

    # ===== 前5个状态 =====
    for i, key in enumerate(metric_keys):
        ax = axs[i]
        col_idx = METRIC_NAMES.index(key)

        for ctrl_key, label, color in zip(
            series.keys(),
            CONTROL_LABELS,
            CONTROL_COLORS,
        ):
            ax.plot(
                t,
                series[ctrl_key][:, col_idx],
                label=label,
                linewidth=1.8,
                color=color
            )

        # 爆胎时刻（关键！）
        ax.axvline(2.0, color="gray", linestyle=":", linewidth=1)

        # ax.set_title(DISPLAY_NAMES[key])
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel(DISPLAY_NAMES[key])
        ax.grid(True, alpha=0.3)

    # ===== 第6个：控制量 =====
    ax = axs[5]

    ax.plot(t_ctrl_0, y0, color=CONTROL_COLORS[0], linewidth=1.8)
    ax.plot(t_ctrl_1, y1, color=CONTROL_COLORS[1], linewidth=1.8)
    ax.plot(t_ctrl_2, y2, color=CONTROL_COLORS[2], linewidth=1.8)

    ax.axvline(2.0, color="gray", linestyle=":", linewidth=1)

    # ax.set_title("附加横摆力矩")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("附加横摆力矩(Nm)")
    ax.grid(True, alpha=0.3)

    # ===== 全局 legend（关键改进）=====
    handles = [
        plt.Line2D([0], [0], color=CONTROL_COLORS[0]),
        plt.Line2D([0], [0], color=CONTROL_COLORS[1]),
        plt.Line2D([0], [0], color=CONTROL_COLORS[2]),
    ]
    labels = CONTROL_LABELS

    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11,bbox_to_anchor=(0.5, -0.05) )

    # ===== 标题 =====
    case_name = case_dir.name
    # fig.suptitle(f"{case_name} 轮爆胎工况控制效果对比", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.92])

    # ===== 保存 =====
    out_path = out_dir / f"{case_name}_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()

    parser = argparse.ArgumentParser(description="Plot burst-control results from multiple case folders.")
    parser.add_argument("base_dir", type=str, help="Root directory that contains case folders.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to save figures. Default: <base_dir>/plots",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process every subfolder that contains at least one xlsx.
    case_dirs = [p for p in base_dir.iterdir() if p.is_dir() and list(p.glob("*.xlsx"))]
    if not case_dirs:
        raise FileNotFoundError(f"No case folders with xlsx found under {base_dir}")

    for case_dir in sorted(case_dirs):
        plot_case_summary(case_dir, out_dir)
        print(f"Saved figures for: {case_dir.name}")


if __name__ == "__main__":
    main()
