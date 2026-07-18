from __future__ import annotations

from rcm_api import ModelRequest, health, model
from rcm_backend_core import CASE, GROUP_COLORS


def test_health() -> None:
    assert health() == {"ok": True}


def test_model_endpoint_returns_plotly_figures() -> None:
    payload = model(ModelRequest(layers=["atoms", "bonds", "connectivity", "offset"]))
    assert payload["model"]["backend"] == "python"
    assert payload["model"]["n_segments"] > 0
    assert "scene" in payload["figures"]
    assert payload["figures"]["scene"]["data"]
    assert payload["field_meta"] == ["field layer off"]


def test_probe_markers_and_labels_use_one_color_per_group() -> None:
    payload = model(ModelRequest(layers=["probes", "labels"]))
    traces = payload["figures"]["scene"]["data"]

    assert len(traces) == len(CASE.groups)
    for group_i, trace in enumerate(traces):
        color = GROUP_COLORS[group_i % len(GROUP_COLORS)]
        assert trace["name"] == f"group {group_i + 1}"
        assert trace["marker"]["color"] == color
        assert trace["textfont"]["color"] == color
