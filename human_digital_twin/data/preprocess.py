from __future__ import annotations

import json
from typing import Dict, Any, List

import networkx as nx
import pandas as pd

from data.loaders.human_cell_atlas_loader import load_hca_mock
from data.loaders.uniprot_loader import load_uniprot_mock
from data.loaders.pdb_loader import load_pdb_mock


def build_minimal_knowledge_graph() -> nx.MultiDiGraph:
    """Construct a small but structured knowledge graph spanning body->organ->tissue->cell->molecule.

    Returns a NetworkX MultiDiGraph with node/edge attributes describing ontology relationships and data sources.
    """
    graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # Root body node
    graph.add_node(
        "Body",
        level="body",
        label="Human Body",
        sources=["Synthetic", "HCA", "UniProt", "PDB"],
    )

    # Load mock datasets
    hca_df: pd.DataFrame = load_hca_mock()
    uniprot_df: pd.DataFrame = load_uniprot_mock()
    pdb_df: pd.DataFrame = load_pdb_mock()

    # Create organ -> tissue -> cell hierarchy
    for organ_name, organ_group in hca_df.groupby("organ"):
        organ_node = f"Organ::{organ_name}"
        graph.add_node(organ_node, level="organ", label=organ_name)
        graph.add_edge("Body", organ_node, relation="part_of")

        for tissue_name, tissue_group in organ_group.groupby("tissue"):
            tissue_node = f"Tissue::{organ_name}::{tissue_name}"
            graph.add_node(tissue_node, level="tissue", label=tissue_name)
            graph.add_edge(organ_node, tissue_node, relation="part_of")

            for _, row in tissue_group.iterrows():
                cell_type = row["cell_type"]
                cell_node = f"Cell::{organ_name}::{tissue_name}::{cell_type}"
                graph.add_node(
                    cell_node,
                    level="cell",
                    label=cell_type,
                    gene_expression=row["gene_expression"],
                )
                graph.add_edge(tissue_node, cell_node, relation="contains")

    # Map proteins and structures
    for _, urow in uniprot_df.iterrows():
        protein_node = f"Protein::{urow['uniprot_id']}"
        graph.add_node(
            protein_node,
            level="molecule",
            label=urow["protein_name"],
            gene=urow["gene"],
            function=urow["function"],
        )
        # Link protein to cells expressing its gene
        for cell_node, attrs in graph.nodes(data=True):
            if attrs.get("level") == "cell":
                expr: Dict[str, float] = attrs.get("gene_expression", {})
                if urow["gene"] in expr:
                    graph.add_edge(cell_node, protein_node, relation="expresses")

    # Attach PDB where available
    pdb_map = {row["uniprot_id"]: row for _, row in pdb_df.iterrows()}
    for protein_node, attrs in list(graph.nodes(data=True)):
        if str(protein_node).startswith("Protein::"):
            uniprot_id = protein_node.split("::", 1)[1]
            pdb_row = pdb_map.get(uniprot_id)
            if pdb_row is not None:
                nx.set_node_attributes(
                    graph,
                    {protein_node: {"pdb_id": pdb_row["pdb_id"], "structure_quality": pdb_row["quality"]}},
                )

    return graph


def serialize_graph_summary(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Return a compact summary for UI and API responses."""
    nodes_by_level: Dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        level = attrs.get("level", "unknown")
        nodes_by_level[level] = nodes_by_level.get(level, 0) + 1

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "nodes_by_level": nodes_by_level,
    }


def export_experiment_json(event_log: List[Dict[str, Any]], graph: nx.MultiDiGraph) -> str:
    """Serialize an experiment log and a thin graph summary to JSON string."""
    payload = {
        "event_log": event_log,
        "graph_summary": serialize_graph_summary(graph),
    }
    return json.dumps(payload, indent=2)

