from __future__ import annotations

import numpy as np
import pytest

from rcm_python.core import (
    adaptive_grid_points,
    area_finder,
    apply_sign_convention,
    fit_no_intercept,
    molecular_diameter,
    rcm_path_center,
    transform_connectivity,
)
from rcm_python.examples import circular_loop_connectivity, circular_loop_m_spec, circular_loop_points


def test_area_finder_returns_positive_normal_for_counter_clockwise_loop() -> None:
    xyz = circular_loop_points(radius=2.0, n_segments=32)
    conn = circular_loop_connectivity(n_segments=32)

    area = area_finder(xyz, conn)

    assert np.allclose(area, [0.0, 0.0, 1.0], atol=1e-12)


def test_reversing_row_order_reverses_area_normal() -> None:
    xyz = circular_loop_points(radius=2.0, n_segments=32)
    conn = circular_loop_connectivity(n_segments=32)

    area = area_finder(xyz, conn)
    reversed_area = area_finder(xyz, transform_connectivity(conn, "rows"))

    assert np.allclose(reversed_area, -area, atol=1e-12)


def test_swapping_segment_endpoints_does_not_change_area_start_polygon() -> None:
    xyz = circular_loop_points(radius=2.0, n_segments=32)
    conn = circular_loop_connectivity(n_segments=32)

    area = area_finder(xyz, conn)
    swapped_area = area_finder(xyz, transform_connectivity(conn, "swap"))

    assert np.allclose(swapped_area, area, atol=1e-12)


def test_transform_connectivity_variants_are_explicit() -> None:
    conn = np.array(
        [
            [1.0, 1.0, 2.0],
            [0.5, 2.0, 3.0],
            [1.0, 3.0, 1.0],
        ]
    )

    assert np.array_equal(transform_connectivity(conn, "none"), conn)
    assert np.array_equal(transform_connectivity(conn, "swap")[:, 1:], conn[:, [2, 1]])
    assert np.array_equal(transform_connectivity(conn, "rows"), conn[::-1])
    assert np.array_equal(transform_connectivity(conn, "swap_rows"), transform_connectivity(conn, "swap")[::-1])

    with pytest.raises(ValueError, match="Unknown connectivity transform"):
        transform_connectivity(conn, "inside_out")


def test_apply_sign_convention_flips_against_selected_interior_groups() -> None:
    rcgf_raw = np.array([-3.0, -2.0, 10.0])

    raw, raw_factor, _ = apply_sign_convention(rcgf_raw, "raw")
    fixed, fixed_factor, note = apply_sign_convention(rcgf_raw, "interior_positive", interior_groups=[0, 1])

    assert np.array_equal(raw, rcgf_raw)
    assert raw_factor == 1.0
    assert fixed_factor == -1.0
    assert np.array_equal(fixed, -rcgf_raw)
    assert "interior-positive" in note


def test_apply_sign_convention_can_use_internal_reference_value() -> None:
    rcgf_raw = np.array([-3.0, 2.0])

    fixed, sign_factor, note = apply_sign_convention(
        rcgf_raw,
        "interior_positive",
        reference_value=-0.25,
    )

    assert sign_factor == -1.0
    assert np.array_equal(fixed, -rcgf_raw)
    assert "internal RCM-path centre" in note


def test_rcm_path_center_uses_segment_lengths() -> None:
    m_spec = circular_loop_m_spec(radius=2.0, n_segments=64)

    center = rcm_path_center(m_spec)

    assert np.allclose(center, [0.0, 0.0, 0.0], atol=1e-12)


def test_fit_no_intercept_reports_exact_slope_and_r2() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 2.5 * x

    slope, r2, y_calc = fit_no_intercept(x, y)

    assert slope == pytest.approx(2.5)
    assert r2 == pytest.approx(1.0)
    assert np.allclose(y_calc, y)


def test_adaptive_grid_uses_cubic_box_scaled_by_molecular_diameter() -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    diameter = molecular_diameter(xyz)
    points, shape, (mins, maxs) = adaptive_grid_points(xyz, grid_size=5, span_factor=1.5)

    assert diameter == pytest.approx(np.sqrt(104.0))
    assert shape == (5, 5, 5)
    assert points.shape == (125, 3)
    assert np.allclose(maxs - mins, np.full(3, 1.5 * diameter))
    assert np.all(xyz >= mins)
    assert np.all(xyz <= maxs)
