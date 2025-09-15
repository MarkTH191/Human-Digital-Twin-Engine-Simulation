from __future__ import annotations

import pandas as pd


def load_pdb_mock() -> pd.DataFrame:
    """Return a toy mapping from UniProt IDs to PDB IDs and quality flags."""
    records = [
        {"uniprot_id": "P02768", "pdb_id": "1AO6", "quality": "high"},
        {"uniprot_id": "P08684", "pdb_id": "5TE8", "quality": "medium"},
        {"uniprot_id": "P04637", "pdb_id": "4HJE", "quality": "high"},
    ]
    return pd.DataFrame.from_records(records)

