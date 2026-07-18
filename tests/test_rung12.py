from __future__ import annotations

from pathlib import Path

import numpy as np

from rcm_python.core import calculate_rcm
from rcm_python.io import read_atom_groups, read_connectivity, read_nmr, read_xyz


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "rung12"


def test_rung12_legacy_fit_matches_matlab() -> None:
    geom = read_xyz(DATA / "rung12.xyz")
    conn = read_connectivity(DATA / "rung12connectivity.csv")
    groups = read_atom_groups(DATA / "rung12_probes.csv")
    nmr = read_nmr(DATA / "rung12_shifts.csv")

    result = calculate_rcm(geom.xyz, conn, groups, mode="legacy", backend="auto", nmr=nmr)

    assert result.slope is not None
    assert result.r2 is not None
    assert np.isclose(result.slope, -14.896217729, rtol=0, atol=1e-8)
    assert np.isclose(result.r2, 0.968434296, rtol=0, atol=1e-8)


def test_rung12_interior_positive_flips_sign_without_changing_r2() -> None:
    geom = read_xyz(DATA / "rung12.xyz")
    conn = read_connectivity(DATA / "rung12connectivity.csv")
    groups = read_atom_groups(DATA / "rung12_probes.csv")
    nmr = read_nmr(DATA / "rung12_shifts.csv")

    result = calculate_rcm(
        geom.xyz,
        conn,
        groups,
        mode="legacy",
        backend="auto",
        nmr=nmr,
        sign_convention="interior_positive",
    )

    assert result.slope is not None
    assert result.r2 is not None
    assert result.sign_factor == -1.0
    assert np.isclose(result.slope, 14.896217729, rtol=0, atol=1e-8)
    assert np.isclose(result.r2, 0.968434296, rtol=0, atol=1e-8)
