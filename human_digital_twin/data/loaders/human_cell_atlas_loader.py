from __future__ import annotations

import pandas as pd


def load_hca_mock() -> pd.DataFrame:
    """Return a tiny mock of HCA-like data: organ, tissue, cell_type, gene_expression.

    gene_expression is a dict[str, float] per row to mimic averaged expression.
    """
    records = [
        {
            "organ": "Liver",
            "tissue": "Hepatic Tissue",
            "cell_type": "Hepatocyte",
            "gene_expression": {"ALB": 8.5, "CYP3A4": 7.2, "TP53": 2.1},
        },
        {
            "organ": "Liver",
            "tissue": "Hepatic Tissue",
            "cell_type": "Kupffer Cell",
            "gene_expression": {"CXCL8": 5.1, "TLR4": 6.4, "TP53": 2.7},
        },
        {
            "organ": "Heart",
            "tissue": "Myocardium",
            "cell_type": "Cardiomyocyte",
            "gene_expression": {"TNNT2": 9.1, "ACTC1": 8.7, "TP53": 2.0},
        },
        {
            "organ": "Brain",
            "tissue": "Cortex",
            "cell_type": "Neuron",
            "gene_expression": {"MAP2": 8.9, "GRIN1": 6.9, "TP53": 2.2},
        },
        {
            "organ": "Brain",
            "tissue": "Cortex",
            "cell_type": "Astrocyte",
            "gene_expression": {"GFAP": 8.2, "SLC1A2": 7.5, "TP53": 2.3},
        },
    ]
    return pd.DataFrame.from_records(records)

