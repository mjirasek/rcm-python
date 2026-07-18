from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import transform_connectivity


ATOMIC_NUMBER_TO_SYMBOL = {
    "1": "H",
    "6": "C",
    "7": "N",
    "8": "O",
    "9": "F",
    "16": "S",
    "30": "Zn",
}


@dataclass(frozen=True)
class Geometry:
    symbols: list[str]
    xyz: np.ndarray


def _symbol(token: str) -> str:
    token = token.strip()
    if token.endswith(".0"):
        token = token[:-2]
    return ATOMIC_NUMBER_TO_SYMBOL.get(token, token)


def read_xyz(path: str | Path) -> Geometry:
    rows: list[tuple[str, float, float, float]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue
        rows.append((_symbol(parts[0]), x, y, z))
    if not rows:
        raise ValueError(f"No coordinates found in {path}")
    return Geometry(
        symbols=[row[0] for row in rows],
        xyz=np.ascontiguousarray([row[1:] for row in rows], dtype=np.float64),
    )


def read_csv_matrix(path: str | Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float64, filling_values=0.0)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    return np.nan_to_num(arr, nan=0.0)


def read_connectivity(path: str | Path, reverse: bool = False) -> np.ndarray:
    conn = read_csv_matrix(path)
    if conn.ndim == 1:
        conn = conn.reshape(1, -1)
    if conn.shape[1] < 3:
        raise ValueError("Connectivity must have columns: weight,start_atom,end_atom")
    conn = np.ascontiguousarray(conn[:, :3], dtype=np.float64)
    if reverse:
        conn = transform_connectivity(conn, "swap_rows")
    return conn


def read_atom_groups(path: str | Path) -> list[np.ndarray]:
    idx = read_csv_matrix(path)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    groups: list[np.ndarray] = []
    for col in range(idx.shape[1]):
        atoms = idx[:, col]
        atoms = atoms[atoms > 0].astype(np.int64) - 1
        if atoms.size:
            groups.append(atoms)
    return groups


def read_nmr(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    y = read_csv_matrix(p).astype(np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    return y if y.size else None
