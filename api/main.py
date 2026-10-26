"""RainMap API: serves pipeline outputs to the React dashboard.

    uvicorn api.main:app --reload --port 8000

Reads only files written by ``python -m rainmap.pipeline`` and the ML scripts.
In production the built frontend (frontend/dist) is served from the same app.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="RainMap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["GET"],
                   allow_headers=["*"])


def _json(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found; run the pipeline first")
    return json.loads(path.read_text())


def _clean(v):
    """NaN/inf are not valid JSON."""
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def _area(name: str) -> Path:
    names = {e["name"] for e in _json(OUT / "index.json")}
    if name not in names:                              # also blocks path traversal
        raise HTTPException(404, f"unknown neighbourhood {name!r}")
    return OUT / name


@app.get("/api/neighbourhoods")
def neighbourhoods():
    return _json(OUT / "index.json")


@app.get("/api/neighbourhoods/{name}")
def summary(name: str):
    s = _json(_area(name) / "summary.json")
    s["designs"] = {k: {kk: _clean(vv) for kk, vv in d.items()} for k, d in s["designs"].items()}
    return s


@app.get("/api/neighbourhoods/{name}/buildings")
def buildings(name: str):
    return _json(_area(name) / "buildings.geojson")


@app.get("/api/neighbourhoods/{name}/routes")
def routes(name: str):
    return _json(_area(name) / "routes.geojson")


@app.get("/api/neighbourhoods/{name}/clusters")
def clusters(name: str):
    path = _area(name) / "clusters_ranked.csv"
    return _records(pd.read_csv(path)) if path.exists() else []


@app.get("/api/rainfall")
def rainfall_risk():
    return _json(OUT / "rainfall_risk.json")


@app.get("/api/ml")
def ml_status():
    def maybe(p: Path):
        return json.loads(p.read_text()) if p.exists() else None
    bench = OUT / "gnn_benchmark.csv"
    return dict(
        gnn=maybe(OUT / "gnn_summary.json"),
        benchmark=_records(pd.read_csv(bench)) if bench.exists() else [],
        roof_cnn=dict(trained=(OUT / "roof_cnn.pt").exists(),
                      predictions=(ROOT / "data" / "roof_predictions.csv").exists(),
                      validation=maybe(OUT / "roof_cnn_validation.json")),
        rainfall=maybe(OUT / "rainfall_risk.json"),
    )


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "unknown API path")
        return FileResponse(DIST / "index.html")
