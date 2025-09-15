from __future__ import annotations

from typing import Dict, Any
import random


class MoleculeSimulator:
    """Simulate molecular-level activity such as protein abundance and simple interactions.

    This is a lightweight placeholder for complex biochemical simulations. It maintains a
    dictionary of protein activity scores that can be influenced by gene expression and
    drug interventions.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random = random.Random(random_seed)
        self.protein_activity: Dict[str, float] = {}

    def set_initial_activity(self, protein_node: str, baseline: float) -> None:
        self.protein_activity[protein_node] = float(baseline)

    def apply_drug_influence(self, protein_node: str, effect_strength: float) -> None:
        current = self.protein_activity.get(protein_node, 0.0)
        # Saturating update to keep values in a reasonable range
        updated = max(0.0, min(10.0, current + effect_strength))
        self.protein_activity[protein_node] = updated

    def update_from_gene_expression(self, protein_node: str, expression_level: float) -> None:
        # Map expression to activity with a simple monotonic transform
        baseline = 0.6 * expression_level
        current = self.protein_activity.get(protein_node, baseline)
        blended = 0.5 * current + 0.5 * baseline
        self.protein_activity[protein_node] = blended

    def step(self, dt: float) -> None:
        # Add small decay to avoid runaway values
        for protein, activity in list(self.protein_activity.items()):
            decayed = max(0.0, activity - 0.01 * dt)
            self.protein_activity[protein] = decayed

    def get_state(self) -> Dict[str, Any]:
        return {"protein_activity": dict(self.protein_activity)}

