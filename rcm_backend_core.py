from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import plotly.graph_objects as go

import rcm_python as rcm
from rcm_python.core import (
    adaptive_grid_points,
    area_finder,
    build_m_spec_chem_sc,
    build_m_spec_legacy,
    calculate_rcm,
    transform_connectivity,
)
from rcm_python.field import scalar_for_points


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "examples" / "rung12"

ATOM_COLORS = {
    "H": "#f2f2f2",
    "C": "#1f2933",
    "N": "#2351d9",
    "Ni": "#0e9f9a",
    "O": "#d72638",
    "S": "#d88914",
    "Zn": "#0e9f9a",
}
WEIGHT_COLORS = {
    1.0: "#e11d48",
    0.5: "#2563eb",
    0.25: "#16a34a",
}
GROUP_COLORS = (
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#0f766e",
    "#be185d",
    "#475569",
)
ISO_BASE_PERCENTILE = 90.0


@dataclass(frozen=True)
class LoadedCase:
    symbols: list[str]
    xyz: np.ndarray
    conn: np.ndarray
    groups: list[np.ndarray]
    nmr: np.ndarray | None


CASE = LoadedCase(
    symbols=rcm.read_xyz(DATA / "rung12.xyz").symbols,
    xyz=rcm.read_xyz(DATA / "rung12.xyz").xyz,
    conn=rcm.read_connectivity(DATA / "rung12connectivity.csv"),
    groups=rcm.read_atom_groups(DATA / "rung12_probes.csv"),
    nmr=rcm.read_nmr(DATA / "rung12_shifts.csv"),
)
FIELD_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, tuple[int, int, int], tuple[np.ndarray, np.ndarray], float]] = {}


def build_m_spec(xyz: np.ndarray, conn: np.ndarray, mode: str, separation: float, special_rotation: bool) -> np.ndarray:
    if mode == "legacy":
        return build_m_spec_legacy(xyz, conn, separation)
    return build_m_spec_chem_sc(xyz, conn, separation, special_rotation)


def build_bonds(symbols: list[str], xyz: np.ndarray, cutoff: float = 2.0) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            if symbols[i] == "H" and symbols[j] == "H":
                continue
            if np.linalg.norm(xyz[i] - xyz[j]) < cutoff:
                bonds.append((i, j))
    return bonds


BONDS = build_bonds(CASE.symbols, CASE.xyz)


def add_molecule_layers(fig: go.Figure, layers: set[str]) -> None:
    xyz = CASE.xyz

    if "bonds" in layers:
        xb: list[float | None] = []
        yb: list[float | None] = []
        zb: list[float | None] = []
        for i, j in BONDS:
            xb += [xyz[i, 0], xyz[j, 0], None]
            yb += [xyz[i, 1], xyz[j, 1], None]
            zb += [xyz[i, 2], xyz[j, 2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xb,
                y=yb,
                z=zb,
                mode="lines",
                line=dict(color="#9aa4b2", width=2),
                hoverinfo="skip",
                name="bonds",
            )
        )

    if "atoms" in layers:
        for symbol in sorted(set(CASE.symbols)):
            idx = [i for i, s in enumerate(CASE.symbols) if s == symbol]
            pts = xyz[idx]
            labels = [f"{symbol}{i + 1}" for i in idx]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers+text" if "labels" in layers else "markers",
                    marker=dict(size=4 if symbol == "H" else 6, color=ATOM_COLORS.get(symbol, "#64748b"), line=dict(color="#111827", width=0.5)),
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=10, color="#0f172a"),
                    hovertemplate="%{text}<extra></extra>",
                    name=symbol,
                )
            )


def add_connectivity_layers(fig: go.Figure, conn: np.ndarray, layers: set[str]) -> None:
    if "connectivity" not in layers and "arrows" not in layers:
        return
    xyz = CASE.xyz

    for weight in sorted(set(conn[:, 0]), reverse=True):
        rows = conn[np.isclose(conn[:, 0], weight)]
        xp: list[float | None] = []
        yp: list[float | None] = []
        zp: list[float | None] = []
        arrow_x: list[float | None] = []
        arrow_y: list[float | None] = []
        arrow_z: list[float | None] = []
        for _, start_atom, stop_atom in rows:
            a = xyz[int(start_atom) - 1]
            b = xyz[int(stop_atom) - 1]
            xp += [a[0], b[0], None]
            yp += [a[1], b[1], None]
            zp += [a[2], b[2], None]
            direction = b - a
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                unit = direction / norm
                tail = a + 0.42 * direction
                tip = a + 0.74 * direction
                head_length = min(max(0.18 * norm, 0.22), 0.52)
                back = tip - head_length * unit
                reference = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(unit, reference))) > 0.85:
                    reference = np.array([0.0, 1.0, 0.0])
                side = np.cross(unit, reference)
                side_norm = np.linalg.norm(side)
                if side_norm <= 1e-12:
                    continue
                side = side / side_norm
                half_width = min(max(0.09 * norm, 0.10), 0.26)
                left = back + half_width * side
                right = back - half_width * side
                arrow_x += [tail[0], tip[0], None, left[0], tip[0], right[0], None]
                arrow_y += [tail[1], tip[1], None, left[1], tip[1], right[1], None]
                arrow_z += [tail[2], tip[2], None, left[2], tip[2], right[2], None]
        color = WEIGHT_COLORS.get(float(weight), "#475569")
        if "connectivity" in layers:
            fig.add_trace(
                go.Scatter3d(
                    x=xp,
                    y=yp,
                    z=zp,
                    mode="lines",
                    line=dict(color=color, width=7),
                    hoverinfo="skip",
                    name=f"connectivity {weight:g}",
                )
            )
        if "arrows" in layers:
            fig.add_trace(
                go.Scatter3d(
                    x=arrow_x,
                    y=arrow_y,
                    z=arrow_z,
                    mode="lines",
                    line=dict(color="rgba(15,23,42,0.72)", width=9),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"arrow contrast {weight:g}",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=arrow_x,
                    y=arrow_y,
                    z=arrow_z,
                    mode="lines",
                    line=dict(color=color, width=5),
                    hoverinfo="skip",
                    name=f"arrows {weight:g}",
                )
            )


def add_offset_path_layer(fig: go.Figure, m_spec: np.ndarray, layers: set[str]) -> None:
    if "offset" not in layers:
        return

    xo: list[float | None] = []
    yo: list[float | None] = []
    zo: list[float | None] = []
    for row in m_spec:
        xo += [row[1], row[4], None]
        yo += [row[2], row[5], None]
        zo += [row[3], row[6], None]
    fig.add_trace(
        go.Scatter3d(
            x=xo,
            y=yo,
            z=zo,
            mode="lines",
            line=dict(color="rgba(15,23,42,0.24)", width=2),
            hoverinfo="skip",
            name="offset RCM",
        )
    )


def add_probe_layer(fig: go.Figure, layers: set[str]) -> None:
    if "probes" not in layers:
        return
    for group_i, probe_atoms in enumerate(CASE.groups):
        if probe_atoms.size == 0:
            continue
        pts = CASE.xyz[probe_atoms]
        color = GROUP_COLORS[group_i % len(GROUP_COLORS)]
        labels = [f"G{group_i + 1}: {CASE.symbols[i]}{i + 1}" for i in probe_atoms]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers+text" if "labels" in layers else "markers",
                marker=dict(size=8, color=color, symbol="diamond", line=dict(color="#0f172a", width=1)),
                text=labels,
                textposition="bottom center",
                textfont=dict(size=10, color=color),
                hovertemplate="%{text}<extra></extra>",
                name=f"group {group_i + 1}",
            )
        )


def adaptive_iso_base(values: np.ndarray) -> float:
    finite_abs = np.abs(values[np.isfinite(values)])
    if finite_abs.size == 0:
        return 1.0
    base = float(np.percentile(finite_abs, ISO_BASE_PERCENTILE)) / 2.0
    return base if np.isfinite(base) and base > 0 else 1.0


def iso_factor(iso_log10: float) -> float:
    return 10.0 ** float(iso_log10)


def add_field_layers(fig: go.Figure, points: np.ndarray, values: np.ndarray, iso_log10: float, max_points: int = 1_000_000) -> None:
    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step]
        values = values[::step]

    base_step = adaptive_iso_base(values)
    factor = iso_factor(iso_log10)
    level_1 = base_step * factor
    level_2 = 2.0 * level_1
    common = dict(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        value=values,
        surface_count=2,
        opacity=0.24,
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        hoverinfo="skip",
    )
    fig.add_trace(
        go.Isosurface(
            **common,
            isomin=-level_2,
            isomax=-level_1,
            colorscale=[[0.0, "#7c3aed"], [1.0, "#2563eb"]],
            name=f"field -{level_2:.3g}..-{level_1:.3g}",
        )
    )
    fig.add_trace(
        go.Isosurface(
            **common,
            isomin=level_1,
            isomax=level_2,
            colorscale=[[0.0, "#f97316"], [1.0, "#e11d48"]],
            name=f"field {level_1:.3g}..{level_2:.3g}",
        )
    )


def scene_figure(
    conn: np.ndarray,
    m_spec: np.ndarray,
    layers: set[str],
    field_data: tuple[np.ndarray, np.ndarray] | None = None,
    iso_log10: float = 0.0,
) -> go.Figure:
    fig = go.Figure()
    if field_data is not None:
        add_field_layers(fig, field_data[0], field_data[1], iso_log10)
    add_molecule_layers(fig, layers)
    add_probe_layer(fig, layers)
    add_offset_path_layer(fig, m_spec, layers)
    add_connectivity_layers(fig, conn, layers)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data",
            dragmode="orbit",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        uirevision="keep-camera",
        legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.82)"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def fit_figure(rcgf: np.ndarray, nmr: np.ndarray | None, slope: float | None, y_calc: np.ndarray | None) -> go.Figure:
    fig = go.Figure()
    if nmr is None or slope is None or y_calc is None:
        fig.add_annotation(text="No NMR data", x=0.5, y=0.5, showarrow=False)
    else:
        labels = [f"group {i + 1}" for i in range(rcgf.size)]
        fig.add_trace(
            go.Scatter(
                x=rcgf,
                y=nmr,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=11,
                    color=[GROUP_COLORS[i % len(GROUP_COLORS)] for i in range(rcgf.size)],
                    line=dict(color="#0f172a", width=1),
                ),
                hovertemplate="%{text}<br>RCGF %{x:.5g}<br>NMR %{y:.5g}<extra></extra>",
                name="groups",
            )
        )
        x0, x1 = float(np.min(rcgf)), float(np.max(rcgf))
        xs = np.linspace(x0, x1, 100)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs, mode="lines", line=dict(color="#e11d48", width=2), name="fit"))
    fig.update_layout(
        margin=dict(l=50, r=10, t=15, b=45),
        xaxis_title="RCGF",
        yaxis_title="NMR shift",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def rcgf_figure(rcgf: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[f"G{i + 1}" for i in range(rcgf.size)],
            y=rcgf,
            marker_color=[GROUP_COLORS[i % len(GROUP_COLORS)] for i in range(rcgf.size)],
            hovertemplate="%{x}<br>RCGF %{y:.6g}<extra></extra>",
        )
    )
    fig.update_layout(margin=dict(l=45, r=10, t=15, b=35), yaxis_title="RCGF", paper_bgcolor="white", plot_bgcolor="white")
    return fig


def cache_key(mode: str, transform: str, separation: float, special_rotation: bool, grid_size: int, field_span: float, backend: str) -> tuple:
    return (mode, transform, round(float(separation), 5), bool(special_rotation), int(grid_size), round(float(field_span), 4), backend)


def get_field(mode: str, transform: str, separation: float, special_rotation: bool, grid_size: int, field_span: float, backend: str):
    key = cache_key(mode, transform, separation, special_rotation, grid_size, field_span, backend)
    if key in FIELD_CACHE:
        return (*FIELD_CACHE[key], True)
    conn = transform_connectivity(CASE.conn, transform)
    m_spec = build_m_spec(CASE.xyz, conn, mode, separation, special_rotation)
    area = area_finder(CASE.xyz, conn)
    points, shape, limits = adaptive_grid_points(CASE.xyz, grid_size, span_factor=field_span)
    t0 = time.perf_counter()
    values = scalar_for_points(m_spec, points, area, backend=backend)
    elapsed = time.perf_counter() - t0
    if len(FIELD_CACHE) > 8:
        FIELD_CACHE.pop(next(iter(FIELD_CACHE)))
    FIELD_CACHE[key] = (points, values, shape, limits, elapsed)
    return points, values, shape, limits, elapsed, False
