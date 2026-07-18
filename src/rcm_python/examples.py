from __future__ import annotations

import numpy as np


def circular_loop_points(
    radius: float = 1.0,
    n_segments: int = 100,
    z: float = 0.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return equally spaced points on a circular loop in the xy plane."""

    if n_segments < 3:
        raise ValueError("n_segments must be at least 3")
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cx, cy, cz = center
    xyz = np.column_stack(
        [
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles),
            np.full(n_segments, cz + z, dtype=np.float64),
        ]
    )
    return np.ascontiguousarray(xyz, dtype=np.float64)


def circular_loop_connectivity(n_segments: int = 100, weight: float = 1.0) -> np.ndarray:
    """Return [weight, start, stop] connectivity for a closed directed loop."""

    if n_segments < 3:
        raise ValueError("n_segments must be at least 3")
    starts = np.arange(1, n_segments + 1, dtype=np.float64)
    stops = np.roll(starts, -1)
    return np.ascontiguousarray(
        np.column_stack([np.full(n_segments, weight, dtype=np.float64), starts, stops]),
        dtype=np.float64,
    )


def circular_loop_m_spec(
    radius: float = 1.0,
    n_segments: int = 100,
    weight: float = 1.0,
    clockwise: bool = False,
) -> np.ndarray:
    """Return an M_spec table for a regular polygon approximation of a loop.

    Counter-clockwise paths, viewed from +z, produce a positive z field on the
    loop axis. Set clockwise=True to reverse the current direction.
    """

    xyz = circular_loop_points(radius=radius, n_segments=n_segments)
    if clockwise:
        xyz = np.ascontiguousarray(xyz[::-1])
    starts = xyz
    stops = np.roll(xyz, -1, axis=0)
    weights = np.full((n_segments, 1), weight, dtype=np.float64)
    return np.ascontiguousarray(np.hstack([weights, starts, stops]), dtype=np.float64)


def analytical_circular_loop_axis_field(radius: float, z: np.ndarray | float, weight: float = 1.0) -> np.ndarray:
    """Continuous-loop axis field in the same unitless convention as the kernel.

    The finite-segment kernel omits the physical mu0/(4*pi) prefactor, so the
    continuous-loop reference is 2*pi*I*R^2 / (R^2 + z^2)^(3/2).
    """

    z_arr = np.asarray(z, dtype=np.float64)
    return weight * 2.0 * np.pi * radius * radius / np.power(radius * radius + z_arr * z_arr, 1.5)
