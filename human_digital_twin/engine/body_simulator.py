from __future__ import annotations

from typing import Dict, Any, List
import time

import networkx as nx

from engine.cell_simulator import CellSimulator
from engine.tissue_simulator import TissueSimulator
from engine.organ_simulator import OrganSimulator
from engine.molecule_simulator import MoleculeSimulator
from data.preprocess import export_experiment_json


class BodySimulator:
    """Coordinates all simulators and provides a convenient control surface."""

    def __init__(self, knowledge_graph: nx.MultiDiGraph) -> None:
        self.graph = knowledge_graph
        self.time_hours: float = 0.0
        self.time_scale: float = 1.0  # 1.0 = real time, 87600 ~ 10 years per hour etc.
        self.cell_sim = CellSimulator(knowledge_graph)
        self.tissue_sim = TissueSimulator(knowledge_graph)
        self.organ_sim = OrganSimulator(knowledge_graph)
        self.molecule_sim = MoleculeSimulator()
        self.event_log: List[Dict[str, Any]] = []
        self._init_molecules_from_graph()

    def _init_molecules_from_graph(self) -> None:
        for node, attrs in self.graph.nodes(data=True):
            if str(node).startswith("Protein::"):
                # Initialize with small baseline, refined by cell expression mapping
                self.molecule_sim.set_initial_activity(node, baseline=1.0)
        # Seed with current expression
        self.cell_sim.update_proteins_from_expression(self.molecule_sim)

    # --- Controls ---
    def set_time_scale(self, scale: float) -> None:
        self.time_scale = max(0.0, float(scale))

    def step(self, dt_hours: float = 1.0) -> None:
        eff_dt = dt_hours * self.time_scale
        self.cell_sim.step(eff_dt)
        self.cell_sim.update_proteins_from_expression(self.molecule_sim)
        self.molecule_sim.step(eff_dt)
        self.tissue_sim.step(eff_dt)
        self.organ_sim.step(self.tissue_sim.tissue_metrics, eff_dt)
        self.time_hours += eff_dt

    def apply_gene_edit(self, organ: str, tissue: str, cell: str, gene: str, delta: float) -> Dict[str, Any]:
        cell_node = self.cell_sim.get_cell_node(organ, tissue, cell)
        result = self.cell_sim.apply_gene_edit(cell_node, gene, delta)
        event = {"type": "gene_edit", "ts": self.time_hours, **result}
        self.event_log.append(event)
        return event

    def apply_drug(self, target_uniprot_id: str, effect_strength: float) -> Dict[str, Any]:
        protein_node = f"Protein::{target_uniprot_id}"
        self.molecule_sim.apply_drug_influence(protein_node, effect_strength)
        event = {
            "type": "drug",
            "ts": self.time_hours,
            "target": protein_node,
            "effect_strength": effect_strength,
        }
        self.event_log.append(event)
        return event

    def export_experiment(self) -> str:
        return export_experiment_json(self.event_log, self.graph)

    # --- Read APIs for UI ---
    def get_time_readable(self) -> str:
        return f"{self.time_hours:.2f} h (x{self.time_scale:.1f})"

    def get_organ_health(self) -> Dict[str, float]:
        return {self.graph.nodes[k].get("label", k): v for k, v in self.organ_sim.organ_health.items()}

    def get_tissue_metrics(self) -> Dict[str, Dict[str, float]]:
        return self.tissue_sim.tissue_metrics

    def get_protein_activity(self) -> Dict[str, float]:
        return {k: v for k, v in self.molecule_sim.protein_activity.items()}

