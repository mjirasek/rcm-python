from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import field_for_points, scalar_for_points


@dataclass(frozen=True)
class RCMResult:
    m_spec: np.ndarray
    group_components: list[np.ndarray]
    rcgf_raw: np.ndarray
    rcgf: np.ndarray
    area: np.ndarray
    sign_factor: float
    sign_note: str
    sign_reference_point: np.ndarray | None
    sign_reference_value: float | None
    slope: float | None
    r2: float | None
    y_calc: np.ndarray | None


def _polyarea(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_poly_clockwise(x: np.ndarray, y: np.ndarray) -> bool:
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return signed < 0


def area_finder(xyz: np.ndarray, conn: np.ndarray) -> np.ndarray:
    edge = conn[np.isclose(conn[:, 0], 1.0)]
    if edge.size == 0:
        edge = conn

    starts = edge[:, 1].astype(np.int64) - 1
    pts = xyz[starts]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    ax = _polyarea(y, z)
    ay = _polyarea(x, z)
    az = _polyarea(x, y)

    if not _is_poly_clockwise(z, y):
        ax *= -1
    if not _is_poly_clockwise(x, z):
        ay *= -1
    if not _is_poly_clockwise(y, x):
        az *= -1

    area = np.array([ax, ay, az], dtype=np.float64)
    norm = np.linalg.norm(area)
    return area / norm if norm else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def transform_connectivity(conn: np.ndarray, transform: str = "none") -> np.ndarray:
    """Apply explicit diagnostic orientation transforms to a connectivity table.

    The connectivity convention is [weight, start_atom, end_atom] with 1-based
    atom indices. Swapping start/end reverses segment current direction.
    Reversing rows changes polygon traversal order and, in the legacy path
    builder, also changes how displaced-path normals are propagated.
    """

    conn_out = np.ascontiguousarray(conn, dtype=np.float64).copy()
    if transform in {"swap", "swap_rows"}:
        conn_out[:, [1, 2]] = conn_out[:, [2, 1]]
    elif transform not in {"none", "rows"}:
        raise ValueError(f"Unknown connectivity transform: {transform}")

    if transform in {"rows", "swap_rows"}:
        conn_out = np.ascontiguousarray(conn_out[::-1])
    return conn_out


def _local_plane_offset(points: np.ndarray, separation: float) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([0.0, 0.0, separation], dtype=np.float64)
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    n = vt[-1]
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0.0, 0.0, separation], dtype=np.float64)
    return n * (separation / norm)


def build_m_spec_legacy(xyz: np.ndarray, conn: np.ndarray, separation: float) -> np.ndarray:
    offsets: list[np.ndarray] = []
    for endpoint_col in (1, 2):
        path_points = xyz[conn[:, endpoint_col].astype(np.int64) - 1]
        endpoint_offsets = np.zeros_like(path_points)
        for i, point in enumerate(path_points):
            neighbours = path_points[np.linalg.norm(path_points - point, axis=1) < 2.35]
            n = _local_plane_offset(neighbours, separation)
            if i != 0:
                dis1 = np.linalg.norm(n + endpoint_offsets[i - 1])
                dis2 = np.linalg.norm(n - endpoint_offsets[i - 1])
                if dis1 < dis2:
                    n = -n
            endpoint_offsets[i] = n
        offsets.append(endpoint_offsets)

    starts = xyz[conn[:, 1].astype(np.int64) - 1]
    stops = xyz[conn[:, 2].astype(np.int64) - 1]
    weights = 0.5 * conn[:, 0:1]
    upper = np.hstack([weights, starts + offsets[0], stops + offsets[1]])
    lower = np.hstack([weights, starts - offsets[0], stops - offsets[1]])
    return np.ascontiguousarray(np.vstack([upper, lower]), dtype=np.float64)


def build_m_spec_chem_sc(
    xyz: np.ndarray,
    conn: np.ndarray,
    separation: float,
    special_rotation: bool,
) -> np.ndarray:
    if separation == 0:
        starts = xyz[conn[:, 1].astype(np.int64) - 1]
        stops = xyz[conn[:, 2].astype(np.int64) - 1]
        return np.ascontiguousarray(np.hstack([conn[:, 0:1], starts, stops]), dtype=np.float64)

    offsets = np.zeros((conn.shape[0], 2, 3), dtype=np.float64)
    center = xyz.mean(axis=0)
    for i in range(conn.shape[0]):
        for j, endpoint_col in enumerate((1, 2)):
            atom = int(conn[i, endpoint_col]) - 1
            point = xyz[atom]
            neighbours = xyz[np.linalg.norm(xyz - point, axis=1) < 2.35]
            n = _local_plane_offset(neighbours, separation)
            if special_rotation:
                radial = point - center
                radial_norm = np.linalg.norm(radial)
                if radial_norm:
                    n = separation * radial / radial_norm
            offsets[i, j] = n

        start = xyz[int(conn[i, 1]) - 1]
        stop = xyz[int(conn[i, 2]) - 1]
        dis1 = np.linalg.norm(start + offsets[i, 0] - stop - offsets[i, 1])
        dis2 = np.linalg.norm(start + offsets[i, 0] - stop + offsets[i, 1])
        if dis1 > dis2:
            offsets[i, 1] *= -1

    starts = xyz[conn[:, 1].astype(np.int64) - 1]
    stops = xyz[conn[:, 2].astype(np.int64) - 1]
    weights = 0.5 * conn[:, 0:1]
    upper = np.hstack([weights, starts + offsets[:, 0, :], stops + offsets[:, 1, :]])
    lower = np.hstack([weights, starts - offsets[:, 0, :], stops - offsets[:, 1, :]])
    return np.ascontiguousarray(np.vstack([upper, lower]), dtype=np.float64)


def fit_no_intercept(x: np.ndarray, y: np.ndarray | None) -> tuple[float | None, float | None, np.ndarray | None]:
    if y is None or y.size != x.size or x.size == 0:
        return None, None, None
    denom = float(np.dot(x, x))
    if denom == 0:
        return None, None, None
    slope = float(np.dot(x, y) / denom)
    y_calc = slope * x
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot == 0 else float(1 - np.sum((y - y_calc) ** 2) / ss_tot)
    return slope, r2, y_calc


def rcm_path_center(m_spec: np.ndarray) -> np.ndarray:
    starts = m_spec[:, 1:4]
    stops = m_spec[:, 4:7]
    midpoints = 0.5 * (starts + stops)
    lengths = np.linalg.norm(stops - starts, axis=1)
    weights = np.abs(m_spec[:, 0]) * lengths
    valid = weights > 1e-12
    if np.any(valid):
        return np.average(midpoints[valid], axis=0, weights=weights[valid])
    if midpoints.size:
        return midpoints.mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def apply_sign_convention(
    rcgf_raw: np.ndarray,
    convention: str = "raw",
    interior_groups: np.ndarray | list[int] | None = None,
    reference_value: float | None = None,
    reference_label: str = "internal RCM-path centre",
) -> tuple[np.ndarray, float, str]:
    if convention == "raw":
        return rcgf_raw.copy(), 1.0, "raw path-orientation convention"
    if convention != "interior_positive":
        raise ValueError(f"Unknown sign convention: {convention}")

    if interior_groups is not None and len(interior_groups) != 0:
        interior_idx = np.asarray(interior_groups, dtype=np.int64)
        reference = float(np.nanmean(rcgf_raw[interior_idx]))
        reference_label = "selected interior probe groups"
    elif reference_value is not None:
        reference = float(reference_value)
    else:
        reference = float(np.nanmean(rcgf_raw))
        reference_label = "all probe groups fallback"
    sign_factor = -1.0 if reference < 0 else 1.0
    note = f"interior-positive convention, {reference_label} scalar {reference:.8g}, applied sign {sign_factor:+g}"
    return sign_factor * rcgf_raw, sign_factor, note


def calculate_rcm(
    xyz: np.ndarray,
    conn: np.ndarray,
    atom_groups: list[np.ndarray],
    separation: float = 0.7,
    mode: str = "legacy",
    special_rotation: bool = False,
    backend: str = "python",
    nmr: np.ndarray | None = None,
    sign_convention: str = "raw",
    interior_groups: np.ndarray | list[int] | None = None,
) -> RCMResult:
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    conn = np.ascontiguousarray(conn, dtype=np.float64)
    if mode == "legacy":
        m_spec = build_m_spec_legacy(xyz, conn, separation)
    elif mode == "chem_sc":
        m_spec = build_m_spec_chem_sc(xyz, conn, separation, special_rotation)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    area = area_finder(xyz, conn)
    valid_groups = [atoms[(atoms >= 0) & (atoms < xyz.shape[0])] for atoms in atom_groups]
    non_empty = [atoms for atoms in valid_groups if atoms.size]
    all_atoms = np.unique(np.concatenate(non_empty)) if non_empty else np.array([], dtype=np.int64)
    all_components = field_for_points(m_spec, xyz[all_atoms], backend=backend) if all_atoms.size else np.zeros((0, 3))
    lookup = {atom: all_components[i] for i, atom in enumerate(all_atoms)}

    group_components: list[np.ndarray] = []
    rcgf_raw = np.zeros(len(atom_groups), dtype=np.float64)
    for group_i, atoms in enumerate(valid_groups):
        components = np.array([lookup[atom] for atom in atoms], dtype=np.float64) if atoms.size else np.zeros((0, 3))
        group_components.append(components)
        if components.size:
            rcgf_raw[group_i] = float(np.dot(area, components.mean(axis=0)) / 3.0)

    sign_reference_point = None
    sign_reference_value = None
    if sign_convention == "interior_positive" and (interior_groups is None or len(interior_groups) == 0):
        sign_reference_point = rcm_path_center(m_spec)
        sign_reference_value = float(scalar_for_points(m_spec, sign_reference_point[None, :], area, backend=backend)[0])

    rcgf, sign_factor, sign_note = apply_sign_convention(
        rcgf_raw,
        sign_convention,
        interior_groups,
        reference_value=sign_reference_value,
    )
    slope, r2, y_calc = fit_no_intercept(rcgf, nmr)
    return RCMResult(
        m_spec=m_spec,
        group_components=group_components,
        rcgf_raw=rcgf_raw,
        rcgf=rcgf,
        area=area,
        sign_factor=sign_factor,
        sign_note=sign_note,
        sign_reference_point=sign_reference_point,
        sign_reference_value=sign_reference_value,
        slope=slope,
        r2=r2,
        y_calc=y_calc,
    )


def grid_points(xyz: np.ndarray, grid_size: int, padding: float = 2.5) -> tuple[np.ndarray, tuple[int, int, int]]:
    mins, maxs = grid_limits(xyz, padding=padding)
    axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return np.ascontiguousarray(points, dtype=np.float64), (grid_size, grid_size, grid_size)


def molecular_diameter(xyz: np.ndarray, chunk_size: int = 512) -> float:
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 2:
        return 0.0

    max_d2 = 0.0
    for start in range(0, xyz.shape[0], chunk_size):
        block = xyz[start : start + chunk_size]
        d2 = np.sum((block[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        max_d2 = max(max_d2, float(np.max(d2)))
    return float(np.sqrt(max_d2))


def grid_limits(
    xyz: np.ndarray,
    padding: float = 2.5,
    span_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    if span_factor is None:
        return xyz.min(axis=0) - padding, xyz.max(axis=0) + padding

    factor = max(float(span_factor), 1.0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    diameter = molecular_diameter(xyz)
    side = max(factor * diameter, float(np.max(maxs - mins)) + 2.0 * padding)
    half_side = 0.5 * side
    return center - half_side, center + half_side


def adaptive_grid_points(
    xyz: np.ndarray,
    grid_size: int,
    span_factor: float = 1.5,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    mins, maxs = grid_limits(xyz, span_factor=span_factor)
    axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return np.ascontiguousarray(points, dtype=np.float64), (grid_size, grid_size, grid_size), (mins, maxs)


def scalar_field_grid(
    xyz: np.ndarray,
    m_spec: np.ndarray,
    area: np.ndarray,
    grid_size: int,
    padding: float = 2.5,
    backend: str = "python",
) -> tuple[np.ndarray, tuple[int, int, int]]:
    points, shape = grid_points(xyz, grid_size=grid_size, padding=padding)
    return scalar_for_points(m_spec, points, area, backend=backend), shape
