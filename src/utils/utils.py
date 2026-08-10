import csv
import math
import os
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

def set_seed(seed=259):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Normalize(state):
    training_state = [0.0] * 3
    training_state[0] = state[0] / 70
    training_state[1] = state[1] / 0.4 
    training_state[2] = state[2] / 4
    return training_state

# def compute_advantage(gamma, lmbda, td_delta):

#     advantages = []
#     advantage = 0.0

#     for delta in reversed(td_delta):

#         advantage = gamma * lmbda * advantage + delta
#         advantages.append(advantage)

#     advantages.reverse()

#     return torch.stack(advantages).squeeze()

def compute_advantage(gamma, lmbda, td_delta, episode_ends):
    td_delta = td_delta.detach().flatten()
    episode_ends = episode_ends.detach().flatten()

    advantages = torch.empty_like(td_delta)
    advantage = torch.zeros(
        (),
        dtype=td_delta.dtype,
        device=td_delta.device,
    )

    for index in range(td_delta.numel() - 1, -1, -1):
        continuation = 1.0 - episode_ends[index]

        advantage = (
            td_delta[index]
            + gamma * lmbda * continuation * advantage
        )
        advantages[index] = advantage

    return advantages

def modify_run_all_par(v, mu, m, Iz, file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # 修改初速度
        if re.match(r"\s*SV_VXS", line):
            new_lines.append(f"SV_VXS {v}\n")
        
        # 修改路面附着
        elif re.match(r"\s*MU_ROAD_CONSTANT", line):
            new_lines.append(f"MU_ROAD_CONSTANT {mu}\n")

        elif re.match(r"\s*M_SU", line):
            new_lines.append(f"M_SU {m}\n")

        elif re.match(r"\s*IZZ_SU", line):
            new_lines.append(f"IZZ_SU {Iz}\n")
        
        else:
            new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)


TIRE_TABLE_KEYWORDS = {
    "Fx": "FX_TIRE_CARPET",
    "Fy": "FY_TIRE_CARPET",
    "Mz": "MZ_TIRE_CARPET",
}


def _read_numeric_csv(csv_path: Path):
    """Read and validate one rectangular CarSim carpet CSV."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(stream)
            if row and any(cell.strip() for cell in row)
        ]

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(
            f"Tire table must contain at least 2 rows and 2 columns: "
            f"{csv_path}"
        )

    column_count = len(rows[0])
    for row_index, row in enumerate(rows, start=1):
        if len(row) != column_count:
            raise ValueError(
                f"Non-rectangular tire table {csv_path}: row {row_index} "
                f"has {len(row)} columns, expected {column_count}."
            )
        for column_index, cell in enumerate(row, start=1):
            try:
                value = float(cell)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in {csv_path} at row {row_index}, "
                    f"column {column_index}: {cell!r}."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite value in {csv_path} at row {row_index}, "
                    f"column {column_index}: {cell!r}."
                )
    return rows


def _locate_carsim_table(lines, keyword, par_path):
    keyword_pattern = re.compile(rf"^\s*{re.escape(keyword)}\b")
    starts = [
        index for index, line in enumerate(lines)
        if keyword_pattern.match(line)
    ]
    if len(starts) != 1:
        raise ValueError(
            f"Expected exactly one {keyword} table in {par_path}, "
            f"found {len(starts)}."
        )

    start = starts[0]
    try:
        end = next(
            index for index in range(start + 1, len(lines))
            if lines[index].strip() == "ENDTABLE"
        )
    except StopIteration as exc:
        raise ValueError(
            f"Missing ENDTABLE after {keyword} in {par_path}."
        ) from exc
    return start, end


def _parse_existing_carsim_table(lines, start, end, par_path):
    rows = []
    for line_number, line in enumerate(lines[start + 1:end], start=start + 2):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append([float(cell.strip()) for cell in stripped.split(",")])
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric table row in {par_path} at line "
                f"{line_number}: {line!r}."
            ) from exc
    if not rows:
        raise ValueError(f"Empty CarSim table in {par_path}.")
    return np.asarray(rows, dtype=np.float64)


def _prepare_carsim_table_replacement(par_path, csv_path, keyword):
    with par_path.open("r", encoding="utf-8", newline="") as stream:
        original_text = stream.read()

    newline = "\r\n" if "\r\n" in original_text else "\n"
    lines = original_text.splitlines()
    start, end = _locate_carsim_table(lines, keyword, par_path)
    existing = _parse_existing_carsim_table(lines, start, end, par_path)
    csv_rows = _read_numeric_csv(csv_path)
    replacement = np.asarray(csv_rows, dtype=np.float64)

    if existing.shape != replacement.shape:
        raise ValueError(
            f"Table shape mismatch for {keyword}: CarSim has "
            f"{existing.shape}, CSV has {replacement.shape}."
        )
    if not np.allclose(existing[0, :], replacement[0, :], rtol=0.0, atol=1e-9):
        raise ValueError(f"Vertical-load axis mismatch for {keyword}.")
    if not np.allclose(existing[:, 0], replacement[:, 0], rtol=0.0, atol=1e-9):
        raise ValueError(f"Slip axis mismatch for {keyword}.")

    table_lines = [", ".join(row) for row in csv_rows]
    updated_lines = lines[:start + 1] + table_lines + lines[end:]
    updated_text = newline.join(updated_lines)
    if original_text.endswith(("\n", "\r")):
        updated_text += newline
    return original_text, updated_text


def _atomic_write_text(path: Path, text: str):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def replace_tire_force_tables(
    table_dir,
    target_par_files: Mapping[str, str | Path],
    *,
    create_backup=True,
):
    """Install Fx/Fy/Mz CSV carpets into linked burst-tire datasets.

    ``table_dir`` must contain ``Fx.csv``, ``Fy.csv`` and ``Mz.csv``. Each
    target .par file must contain exactly one matching ``*_TIRE_CARPET``
    block. All files are validated before any target is changed.
    """
    table_dir = Path(table_dir)
    expected_components = set(TIRE_TABLE_KEYWORDS)
    provided_components = set(target_par_files)
    if provided_components != expected_components:
        raise ValueError(
            "target_par_files must have exactly the keys "
            f"{sorted(expected_components)}, got {sorted(provided_components)}."
        )

    replacements = []
    for component, keyword in TIRE_TABLE_KEYWORDS.items():
        csv_path = table_dir / f"{component}.csv"
        par_path = Path(target_par_files[component])
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing generated tire table: {csv_path}")
        if not par_path.is_file():
            raise FileNotFoundError(f"Missing CarSim tire dataset: {par_path}")
        original_text, updated_text = _prepare_carsim_table_replacement(
            par_path, csv_path, keyword
        )
        replacements.append((par_path, original_text, updated_text))

    for par_path, _, _ in replacements:
        if create_backup:
            backup_path = par_path.with_suffix(
                par_path.suffix + ".pre_randomization.bak"
            )
            if not backup_path.exists():
                shutil.copy2(par_path, backup_path)

    written_paths = []
    try:
        for par_path, _, updated_text in replacements:
            _atomic_write_text(par_path, updated_text)
            written_paths.append(par_path)
    except Exception:
        # Keep all three linked datasets consistent if a later write fails.
        originals = {path: text for path, text, _ in replacements}
        for written_path in written_paths:
            _atomic_write_text(written_path, originals[written_path])
        raise

    return tuple(path for path, _, _ in replacements)
