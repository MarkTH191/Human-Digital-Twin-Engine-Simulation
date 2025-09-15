from __future__ import annotations

from typing import Dict, Any

from fastapi import FastAPI

from data.preprocess import build_minimal_knowledge_graph, serialize_graph_summary
from engine.body_simulator import BodySimulator


app = FastAPI(title="Human Digital Twin API")


_sim: BodySimulator | None = None


def _get_sim() -> BodySimulator:
    global _sim
    if _sim is None:
        _sim = BodySimulator(build_minimal_knowledge_graph())
    return _sim


@app.get("/health")
def health() -> Dict[str, Any]:
    sim = _get_sim()
    return {"status": "ok", "time": sim.get_time_readable()}


@app.post("/step")
def step(dt_hours: float = 1.0) -> Dict[str, Any]:
    sim = _get_sim()
    sim.step(dt_hours)
    return {"time": sim.get_time_readable(), "organs": sim.get_organ_health()}


@app.get("/graph/summary")
def graph_summary() -> Dict[str, Any]:
    sim = _get_sim()
    return serialize_graph_summary(sim.graph)

