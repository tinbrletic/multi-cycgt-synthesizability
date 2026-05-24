"""Convert peptide_baza.csv (linear peptides + descriptors) into Multi_CycGT input schema.

Reads:  data_process/peptide_baza.csv (semicolon-delimited, target_col = 0/1)
Writes: data_process/peptide_synthesis_adapted.csv (comma-delimited, matches
        Multi_CycGT/model/deep_learning/gcn_transformer_fc/data_pretreatment.py)

Generates SMILES via RDKit Chem.MolFromSequence, computes Sequence_LogP (Crippen) and
Sequence_TPSA (RDKit Descriptors). Drops hydrophobic_cornette (constant 0.0 across all
1771 rows -- zero-variance, no information).
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

INPUT_CSV = Path(__file__).parent / "peptide_baza.csv"
OUTPUT_CSV = Path(__file__).parent / "peptide_synthesis_adapted.csv"


def seq_to_smiles(seq: str) -> Optional[str]:
    """Generate canonical SMILES from an L-amino acid one-letter sequence."""
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def seq_to_logp(seq: str) -> Optional[float]:
    """Crippen MolLogP for the peptide molecule."""
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        return None
    return float(Crippen.MolLogP(mol))


def seq_to_tpsa(seq: str) -> Optional[float]:
    """Topological polar surface area for the peptide molecule."""
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        return None
    return float(Descriptors.TPSA(mol))


def main() -> None:
    df = pd.read_csv(INPUT_CSV, sep=";")
    print(f"Loaded {len(df)} rows x {len(df.columns)} cols from {INPUT_CSV.name}")

    df["SMILES"] = df["peptide_seq"].apply(seq_to_smiles)
    df["Sequence_LogP"] = df["peptide_seq"].apply(seq_to_logp)
    df["Sequence_TPSA"] = df["peptide_seq"].apply(seq_to_tpsa)

    df = df.rename(columns={"target_col": "label", "id": "CycPeptMPDB_ID"})
    df["Year"] = 0
    df["Structurally_Unique_ID"] = df["CycPeptMPDB_ID"]

    df = df.drop(columns=["peptide_seq", "synthesis_flag", "hydrophobic_cornette"])

    n_failed = df["SMILES"].isna().sum()
    if n_failed > 0:
        print(f"WARNING: dropping {n_failed} rows where SMILES generation failed")
        df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

    head_cols = [
        "Year", "CycPeptMPDB_ID", "Structurally_Unique_ID",
        "SMILES", "Sequence_LogP", "Sequence_TPSA", "label",
    ]
    other_cols = [c for c in df.columns if c not in set(head_cols)]
    df = df[head_cols + other_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {OUTPUT_CSV.name}")
    print(f"Class balance: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
