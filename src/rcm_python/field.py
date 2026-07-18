from __future__ import annotations

import numpy as np

def _field_for_points(m_spec: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0 or m_spec.size == 0:
        return np.zeros((points.shape[0], 3), dtype=np.float64)

    weights = m_spec[:, 0]
    start = m_spec[:, 1:4]
    stop = m_spec[:, 4:7]
    direction = stop - start
    length = np.linalg.norm(direction, axis=1)
    valid = length > 1e-12
    if not np.any(valid):
        return np.zeros((points.shape[0], 3), dtype=np.float64)

    weights = weights[valid]
    start = start[valid]
    direction = direction[valid]
    length = length[valid]
    unit = direction / length[:, None]

    q = points[:, None, :] - start[None, :, :]
    q_parallel = np.sum(q * unit[None, :, :], axis=2)
    q_perp = q - q_parallel[:, :, None] * unit[None, :, :]
    rho2 = np.maximum(np.sum(q_perp * q_perp, axis=2), 1e-24)

    r_start = np.sqrt(rho2 + q_parallel * q_parallel)
    end_parallel = length[None, :] - q_parallel
    r_stop = np.sqrt(rho2 + end_parallel * end_parallel)
    integral = end_parallel / (rho2 * r_stop) + q_parallel / (rho2 * r_start)

    direction_cross_q = np.cross(unit[None, :, :], q)
    field = direction_cross_q * integral[:, :, None] * weights[None, :, None]
    return np.ascontiguousarray(np.sum(field, axis=1), dtype=np.float64)


def field_for_points(m_spec: np.ndarray, points: np.ndarray, backend: str = "python", chunk_size: int = 20_000) -> np.ndarray:
    m_spec = np.ascontiguousarray(m_spec, dtype=np.float64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    if backend not in {"auto", "python"}:
        raise ValueError("This package provides only the Python backend")

    out = np.empty((points.shape[0], 3), dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        out[start:stop] = _field_for_points(m_spec, points[start:stop])
    return out


def scalar_for_points(
    m_spec: np.ndarray,
    points: np.ndarray,
    area: np.ndarray,
    backend: str = "python",
    chunk_size: int = 20_000,
) -> np.ndarray:
    m_spec = np.ascontiguousarray(m_spec, dtype=np.float64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    area = np.ascontiguousarray(area, dtype=np.float64)
    if backend not in {"auto", "python"}:
        raise ValueError("This package provides only the Python backend")

    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        out[start:stop] = field_for_points(m_spec, points[start:stop], backend="python") @ area / 3.0
    return out
