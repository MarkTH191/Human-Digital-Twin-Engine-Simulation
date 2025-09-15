from __future__ import annotations

import pandas as pd


def load_uniprot_mock() -> pd.DataFrame:
    """Return a tiny UniProt-like table with proteins and their genes."""
    records = [
        {"uniprot_id": "P02768", "gene": "ALB", "protein_name": "Serum albumin", "function": "Carrier protein"},
        {"uniprot_id": "P08684", "gene": "CYP3A4", "protein_name": "Cytochrome P450 3A4", "function": "Drug metabolism"},
        {"uniprot_id": "P04637", "gene": "TP53", "protein_name": "Cellular tumor antigen p53", "function": "Tumor suppressor"},
        {"uniprot_id": "Q13424", "gene": "GRIN1", "protein_name": "NMDA receptor subunit", "function": "Neurotransmission"},
        {"uniprot_id": "P14136", "gene": "GFAP", "protein_name": "Glial fibrillary acidic protein", "function": "Cytoskeleton"},
        {"uniprot_id": "Q07654", "gene": "TNNT2", "protein_name": "Troponin T, cardiac muscle", "function": "Muscle contraction"},
    ]
    return pd.DataFrame.from_records(records)

