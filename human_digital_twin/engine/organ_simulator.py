from __future__ import annotations

from typing import Dict
import networkx as nx


class OrganSimulator:
    """Aggregate tissue metrics to organ-level health indicators."""

    def __init__(self, knowledge_graph: nx.MultiDiGraph) -> None:
        self.graph = knowledge_graph
        self.organ_health: Dict[str, float] = {}

    def step(self, tissue_metrics: Dict[str, Dict[str, float]], dt: float) -> None:
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("level") == "organ":
                total_viability = 0.0
                count = 0
                for child in self.graph.successors(node):
                    metrics = tissue_metrics.get(child)
                    if metrics is not None:
                        total_viability += metrics.get("viability", 0.0)
                        count += 1
                if count > 0:
                    self.organ_health[node] = total_viability / count
                else:
                    self.organ_health[node] = 5.0

