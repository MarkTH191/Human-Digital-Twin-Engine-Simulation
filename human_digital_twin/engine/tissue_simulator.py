from __future__ import annotations

from typing import Dict
import networkx as nx


class TissueSimulator:
    """Aggregate cell states into tissue-level metrics."""

    def __init__(self, knowledge_graph: nx.MultiDiGraph) -> None:
        self.graph = knowledge_graph
        # tissue_metrics[tissue_node] = {"inflammation": float, "viability": float}
        self.tissue_metrics: Dict[str, Dict[str, float]] = {}

    def step(self, dt: float) -> None:
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("level") == "tissue":
                # Aggregate from child cells
                inflammation = 0.0
                viability = 0.0
                count = 0
                for child in self.graph.successors(node):
                    if self.graph.nodes[child].get("level") == "cell":
                        expr = self.graph.nodes[child].get("gene_expression", {})
                        inflammation += float(expr.get("CXCL8", 0.0))
                        viability += max(0.0, 10.0 - float(expr.get("TP53", 0.0)))
                        count += 1
                if count > 0:
                    inflammation /= count
                    viability /= count
                self.tissue_metrics[node] = {
                    "inflammation": inflammation,
                    "viability": viability,
                }

