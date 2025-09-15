from __future__ import annotations

from typing import Dict


class MultiOmicsFusion:
    """Combine molecular, cellular, and tissue signals into a single score."""

    def __init__(self) -> None:
        pass

    def fuse(self, protein_activity: Dict[str, float], tissue_viability: float) -> float:
        if not protein_activity:
            return tissue_viability
        avg = sum(protein_activity.values()) / max(1, len(protein_activity))
        return 0.7 * tissue_viability + 0.3 * avg

