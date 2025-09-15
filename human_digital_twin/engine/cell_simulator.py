from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
import random

import networkx as nx

from ai_models.genomics_ai import GenomicsAI
from ai_models.proteomics_ai import ProteomicsAI


class CellSimulator:
    """Operate on cell nodes in the knowledge graph.

    Each cell node has attribute `gene_expression: Dict[str, float]`.
    This simulator applies gene edits and predicts expression updates.
    """

    def __init__(self, knowledge_graph: nx.MultiDiGraph, random_seed: int = 42) -> None:
        self.graph = knowledge_graph
        self.random = random.Random(random_seed)
        self.genomics_ai = GenomicsAI(random_seed=random_seed)
        self.proteomics_ai = ProteomicsAI(random_seed=random_seed)

    def list_cells(self) -> Iterable[str]:
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("level") == "cell":
                yield node

    def list_organs(self) -> Iterable[str]:
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("level") == "organ":
                yield attrs.get("label", str(node))

    def list_tissues(self, organ_label: str) -> Iterable[str]:
        organ_node = f"Organ::{organ_label}"
        for neighbor in self.graph.successors(organ_node):
            if self.graph.nodes[neighbor].get("level") == "tissue":
                yield self.graph.nodes[neighbor].get("label", str(neighbor))

    def list_cell_types(self, organ_label: str, tissue_label: str) -> Iterable[Tuple[str, str, str]]:
        tissue_node = f"Tissue::{organ_label}::{tissue_label}"
        for neighbor in self.graph.successors(tissue_node):
            if self.graph.nodes[neighbor].get("level") == "cell":
                yield (
                    organ_label,
                    tissue_label,
                    self.graph.nodes[neighbor].get("label", str(neighbor)),
                )

    def get_cell_node(self, organ_label: str, tissue_label: str, cell_label: str) -> str:
        return f"Cell::{organ_label}::{tissue_label}::{cell_label}"

    def apply_gene_edit(self, cell_node: str, gene: str, delta: float) -> Dict[str, Any]:
        attrs = self.graph.nodes[cell_node]
        expression = dict(attrs.get("gene_expression", {}))
        before = float(expression.get(gene, 0.0))
        expression[gene] = max(0.0, before + delta)
        # Predict spillover effects with a small model
        expression = self.genomics_ai.predict_expression_update(expression, target_gene=gene, delta=delta)
        attrs["gene_expression"] = expression
        return {"cell": cell_node, "gene": gene, "before": before, "after": expression[gene]}

    def proteins_for_cell(self, cell_node: str) -> Iterable[str]:
        # Edges from cell -> protein with relation "expresses"
        for _, protein, edge_attrs in self.graph.out_edges(cell_node, data=True):
            if edge_attrs.get("relation") == "expresses":
                yield protein

    def update_proteins_from_expression(self, molecule_sim) -> None:
        for cell_node in self.list_cells():
            expr: Dict[str, float] = self.graph.nodes[cell_node].get("gene_expression", {})
            for protein_node in self.proteins_for_cell(cell_node):
                gene = self.graph.nodes[protein_node].get("gene")
                predicted = self.proteomics_ai.predict_abundance(expr.get(gene, 0.0))
                molecule_sim.update_from_gene_expression(protein_node, predicted)

    def step(self, dt: float) -> None:
        # Simple autoregressive drift on TP53 as a demo
        for cell_node in self.list_cells():
            expr: Dict[str, float] = dict(self.graph.nodes[cell_node].get("gene_expression", {}))
            current = expr.get("TP53", 0.0)
            expr["TP53"] = max(0.0, current + self.random.uniform(-0.02, 0.02) * dt)
            self.graph.nodes[cell_node]["gene_expression"] = expr

