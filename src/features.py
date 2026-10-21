"""
Feature engineering utilities for molecular property prediction.

Converts SMILES strings into numerical feature matrices using RDKit.
"""

from typing import Optional

import numpy as np
import pandas as pd


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule.
    radius : int
        Morgan fingerprint radius (default 2 ≈ ECFP4).
    n_bits : int
        Length of the folded fingerprint bit vector.

    Returns
    -------
    np.ndarray or None
        1-D array of shape (n_bits,) with 0/1 values, or None if the
        SMILES string cannot be parsed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8)


def smiles_to_descriptors(smiles: str) -> Optional[dict]:
    """Compute a standard set of RDKit 2-D molecular descriptors.

    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule.

    Returns
    -------
    dict or None
        Dictionary mapping descriptor names to values, or None if the
        SMILES cannot be parsed.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptor_names = [name for name, _ in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    values = calculator.CalcDescriptors(mol)
    return dict(zip(descriptor_names, values))


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str,
    mode: str = "fingerprint",
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Build a feature matrix from a DataFrame with a SMILES column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column of SMILES strings.
    smiles_col : str
        Name of the column holding SMILES strings.
    mode : {"fingerprint", "descriptors"}
        "fingerprint" uses Morgan fingerprints; "descriptors" uses RDKit
        2-D molecular descriptors.
    radius : int
        Morgan fingerprint radius (only used when mode="fingerprint").
    n_bits : int
        Fingerprint bit-vector length (only used when mode="fingerprint").

    Returns
    -------
    pd.DataFrame
        Feature matrix with the same index as *df*. Rows whose SMILES
        could not be parsed are filled with NaN.
    """
    if mode == "fingerprint":
        fps = df[smiles_col].apply(
            lambda s: smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits)
        )
        valid = fps.dropna()
        cols = [f"fp_{i}" for i in range(n_bits)]
        feat_df = pd.DataFrame(
            np.vstack(valid.values), index=valid.index, columns=cols
        )
        return feat_df.reindex(df.index)

    elif mode == "descriptors":
        descs = df[smiles_col].apply(smiles_to_descriptors)
        valid = descs.dropna()
        feat_df = pd.DataFrame(valid.tolist(), index=valid.index)
        return feat_df.reindex(df.index)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'fingerprint' or 'descriptors'.")
