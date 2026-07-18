from __future__ import annotations

import numpy as np
import pytest

from rcm_python.field import field_for_points, scalar_for_points


def test_single_segment_field_matches_closed_form_midpoint_value() -> None:
    m_spec = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    points = np.array([[0.5, 1.0, 0.0]])

    field = field_for_points(m_spec, points, backend="python")

    assert field.shape == (1, 3)
    assert field[0, 0] == pytest.approx(0.0, abs=1e-14)
    assert field[0, 1] == pytest.approx(0.0, abs=1e-14)
    assert field[0, 2] == pytest.approx(1.0 / np.sqrt(1.25), rel=0, abs=1e-14)


def test_reversing_single_segment_flips_field_direction() -> None:
    forward = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    reverse = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    points = np.array([[0.5, 1.0, 0.0], [0.5, -1.0, 0.0]])

    forward_field = field_for_points(forward, points, backend="python")
    reverse_field = field_for_points(reverse, points, backend="python")

    assert np.allclose(reverse_field, -forward_field)


def test_zero_length_segments_are_ignored() -> None:
    m_spec = np.array(
        [
            [999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    points = np.array([[0.5, 1.0, 0.0]])

    field = field_for_points(m_spec, points, backend="python")

    assert field[0, 2] == pytest.approx(1.0 / np.sqrt(1.25), rel=0, abs=1e-14)


def test_scalar_for_points_is_projected_field_divided_by_three() -> None:
    m_spec = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    points = np.array([[0.5, 1.0, 0.0], [0.5, -1.0, 0.0]])
    area = np.array([0.0, 0.0, 1.0])

    field = field_for_points(m_spec, points, backend="python")
    scalar = scalar_for_points(m_spec, points, area, backend="python")

    assert np.allclose(scalar, field @ area / 3.0)


def test_cpp_backend_is_not_available() -> None:
    m_spec = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    points = np.array([[0.5, 1.0, 0.0]])

    with pytest.raises(ValueError, match="only the Python backend"):
        field_for_points(m_spec, points, backend="cpp")
