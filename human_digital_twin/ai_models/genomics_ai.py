from __future__ import annotations

from typing import Dict
import random


class GenomicsAI:
    """A tiny probabilistic model for expression propagation.

    Given a gene edit, slightly nudges correlated genes. In real systems, this would
    use a trained transformer or regulatory network; here we craft a simple heuristic.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random = random.Random(random_seed)
        # Toy correlations
        self.corr = {
            "TP53": ["CDKN1A", "MDM2"],
            "ALB": ["TF", "HP"],
            "CYP3A4": ["POR", "CYP2D6"],
            "TNNT2": ["MYH7", "ACTC1"],
            "GFAP": ["SLC1A2", "ALDH1L1"],
        }

    def predict_expression_update(self, expression: Dict[str, float], target_gene: str, delta: float) -> Dict[str, float]:
        updated = dict(expression)
        # Main target already updated by caller
        # Apply small correlated nudges
        for neighbor in self.corr.get(target_gene, []):
            current = updated.get(neighbor, 0.0)
            updated[neighbor] = max(0.0, current + 0.25 * delta)
        return updated

