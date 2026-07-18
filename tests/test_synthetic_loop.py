from __future__ import annotations

import numpy as np
import pytest

from rcm_python.examples import (
    analytical_circular_loop_axis_field,
    circular_loop_m_spec,
)
from rcm_python.field import field_for_points


def test_100_segment_loop_matches_continuous_axis_field_reference() -> None:
    radius = 2.0
    z_values = np.array([0.0, 0.5, 1.5, 4.0])
    points = np.column_stack([np.zeros_like(z_values), np.zeros_like(z_values), z_values])
    m_spec = circular_loop_m_spec(radius=radius, n_segments=100)

    field_z = field_for_points(m_spec, points, backend="python")[:, 2]
    expected = analytical_circular_loop_axis_field(radius, z_values)

    assert np.allclose(field_z, expected, rtol=8e-4, atol=1e-10)


def test_loop_axis_field_scales_linearly_with_weight() -> None:
    points = np.array([[0.0, 0.0, 0.75]])
    base = circular_loop_m_spec(radius=1.25, n_segments=150, weight=1.0)
    scaled = circular_loop_m_spec(radius=1.25, n_segments=150, weight=-2.5)

    base_field = field_for_points(base, points, backend="python")
    scaled_field = field_for_points(scaled, points, backend="python")

    assert np.allclose(scaled_field, -2.5 * base_field)


def test_clockwise_loop_flips_axis_field_sign() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    ccw = circular_loop_m_spec(radius=1.5, n_segments=120, clockwise=False)
    cw = circular_loop_m_spec(radius=1.5, n_segments=120, clockwise=True)

    ccw_field = field_for_points(ccw, points, backend="python")
    cw_field = field_for_points(cw, points, backend="python")

    assert np.all(ccw_field[:, 2] > 0)
    assert np.allclose(cw_field, -ccw_field, rtol=1e-12, atol=1e-12)


def test_loop_off_axis_symmetry_cancels_transverse_components_on_axis() -> None:
    z_values = np.linspace(-2.0, 2.0, 9)
    points = np.column_stack([np.zeros_like(z_values), np.zeros_like(z_values), z_values])
    m_spec = circular_loop_m_spec(radius=1.0, n_segments=100)

    field = field_for_points(m_spec, points, backend="python")

    assert np.max(np.abs(field[:, :2])) < 1e-12
    assert np.all(field[:, 2] > 0)
