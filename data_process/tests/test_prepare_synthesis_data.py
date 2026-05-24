"""Unit tests for the peptide_baza -> Multi_CycGT adapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from prepare_synthesis_data import seq_to_smiles, seq_to_logp, seq_to_tpsa


def test_seq_to_smiles_standard_peptide():
    smi = seq_to_smiles("SELLTPLGIDLDEW")
    assert smi is not None
    assert isinstance(smi, str)
    assert "N" in smi and "O" in smi
    assert len(smi) > 30


def test_seq_to_smiles_diglycine():
    smi = seq_to_smiles("GG")
    assert smi is not None
    assert isinstance(smi, str)
    assert len(smi) >= 10


def test_seq_to_smiles_invalid_chars_returns_none():
    smi = seq_to_smiles("XZJ123")
    assert smi is None


def test_seq_to_logp_returns_float():
    val = seq_to_logp("SELLTPLGIDLDEW")
    assert isinstance(val, float)
    assert -50.0 < val < 50.0


def test_seq_to_logp_invalid_returns_none():
    val = seq_to_logp("???")
    assert val is None


def test_seq_to_tpsa_returns_positive_float():
    val = seq_to_tpsa("SELLTPLGIDLDEW")
    assert isinstance(val, float)
    assert val > 0.0


def test_seq_to_tpsa_invalid_returns_none():
    val = seq_to_tpsa("???")
    assert val is None
