from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from plotly.utils import PlotlyJSONEncoder

from rcm_backend_core import (
    CASE,
    adaptive_iso_base,
    calculate_rcm,
    fit_figure,
    get_field,
    iso_factor,
    rcgf_figure,
    scene_figure,
    transform_connectivity,
)


def _cors_origins() -> list[str]:
    value = os.environ.get("RCM_CORS_ORIGINS", "")
    if value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://michaeljirasek.com",
        "https://www.michaeljirasek.com",
    ]


app = FastAPI(title="RCM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ModelRequest(BaseModel):
    mode: str = "legacy"
    transform: str = "none"
    separation: float = Field(0.7, ge=0.0)
    sign_convention: str = "raw"
    special_rotation: bool = False
    layers: list[str] = Field(default_factory=lambda: ["atoms", "bonds", "connectivity", "offset"])
    grid_size: int = Field(40, ge=20, le=150)
    field_span: float = Field(1.5, ge=1.0, le=5.0)
    iso_scale: float = Field(0.0, ge=-2.0, le=2.0)


def _plotly_json(value: Any) -> Any:
    return json.loads(json.dumps(value, cls=PlotlyJSONEncoder))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "case": "rung12",
        "n_atoms": int(CASE.xyz.shape[0]),
        "n_groups": len(CASE.groups),
        "defaults": ModelRequest().model_dump(),
        "options": {
            "mode": ["legacy", "chem_sc"],
            "transform": ["none", "swap", "rows", "swap_rows"],
            "sign_convention": ["raw", "interior_positive"],
            "layers": ["atoms", "labels", "bonds", "connectivity", "arrows", "offset", "probes", "field"],
        },
    }


@app.post("/api/model")
def model(payload: ModelRequest) -> dict[str, Any]:
    layer_set = set(payload.layers)
    conn = transform_connectivity(CASE.conn, payload.transform)

    t0 = time.perf_counter()
    result = calculate_rcm(
        CASE.xyz,
        conn,
        CASE.groups,
        separation=payload.separation,
        mode=payload.mode,
        special_rotation=payload.special_rotation,
        backend="python",
        nmr=CASE.nmr,
        sign_convention=payload.sign_convention,
    )
    model_elapsed = time.perf_counter() - t0

    field_data = None
    field_meta = ["field layer off"]
    field_elapsed_value = None
    field_cached = None
    if "field" in layer_set:
        points, values, shape, limits, field_elapsed, cached = get_field(
            payload.mode,
            payload.transform,
            payload.separation,
            payload.special_rotation,
            payload.grid_size,
            payload.field_span,
            "python",
        )
        field_data = (points, values)
        field_elapsed_value = field_elapsed
        field_cached = bool(cached)
        base_step = adaptive_iso_base(values)
        factor = iso_factor(payload.iso_scale)
        side = float(np.max(limits[1] - limits[0]))
        field_meta = [
            f"{shape[0]}^3 grid",
            f"box {side:.1f} A",
            "cached" if cached else f"computed {field_elapsed:.3f}s",
            f"base {base_step:.3g}",
            f"factor {factor:.3g}x",
        ]

    model_data = {
        "rcgf": result.rcgf.tolist(),
        "rcgf_raw": result.rcgf_raw.tolist(),
        "slope": result.slope,
        "r2": result.r2,
        "y_calc": None if result.y_calc is None else result.y_calc.tolist(),
        "sign_factor": result.sign_factor,
        "sign_note": result.sign_note,
        "area": result.area.tolist(),
        "n_segments": int(result.m_spec.shape[0]),
        "mode": payload.mode,
        "transform": payload.transform,
        "backend": "python",
        "separation": payload.separation,
        "special_rotation": payload.special_rotation,
        "model_elapsed": model_elapsed,
        "field_elapsed": field_elapsed_value,
        "field_cached": field_cached,
    }

    scene = scene_figure(conn, result.m_spec, layer_set, field_data=field_data, iso_log10=payload.iso_scale)
    fit = fit_figure(result.rcgf, CASE.nmr, result.slope, result.y_calc)
    rcgf = rcgf_figure(result.rcgf)
    return {
        "model": _plotly_json(model_data),
        "field_meta": field_meta,
        "figures": {
            "scene": _plotly_json(scene.to_plotly_json()),
            "fit": _plotly_json(fit.to_plotly_json()),
            "rcgf": _plotly_json(rcgf.to_plotly_json()),
        },
    }
