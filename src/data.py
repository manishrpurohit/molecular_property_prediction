"""
Data loading and splitting utilities.

Loads sheets from the Excel workbook and provides scaffold-aware train/test
splits for realistic evaluation of molecular property models.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# ── Data loading ──────────────────────────────────────────────────────────────

WORKBOOK_NAME = "molecular_property_prediction_datasets.xlsx"

# Canonical SMILES column names per dataset
SMILES_COLUMNS = {
    "BBBP": "smiles",
    "ESOL": "smiles",
    "FreeSolv": "smiles",
    "Lipophilicity": "smiles",
    "ClinTox": "smiles",
    "Tox21": "smiles",
    "BACE": "mol",
}

# Primary target columns per dataset
TARGET_COLUMNS = {
    "BBBP": "p_np",
    "ESOL": "measured log solubility in mols per litre",
    "FreeSolv": "expt",
    "Lipophilicity": "exp",
    "ClinTox": "CT_TOX",
    "Tox21": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
        "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
        "SR-HSE", "SR-MMP", "SR-p53",
    ],
    "BACE": "Class",
}


def load_datasets(
    workbook_path: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    """Load all seven dataset sheets from the Excel workbook.

    Parameters
    ----------
    workbook_path : str or Path, optional
        Path to the workbook. Defaults to ``WORKBOOK_NAME`` in the current
        directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of sheet name → DataFrame for every dataset sheet
        (excludes 'Overview').
    """
    if workbook_path is None:
        workbook_path = Path(WORKBOOK_NAME)
    else:
        workbook_path = Path(workbook_path)

    if not workbook_path.exists():
        raise FileNotFoundError(
            f"Workbook not found at {workbook_path.resolve()}. "
            "Place the Excel file beside this script or pass the path explicitly."
        )

    xls = pd.ExcelFile(workbook_path)
    datasets = {}
    for sheet in xls.sheet_names:
        if sheet.lower() == "overview":
            continue
        datasets[sheet] = pd.read_excel(xls, sheet_name=sheet)
    return datasets


# ── Splitting ─────────────────────────────────────────────────────────────────

def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by Murcko scaffold to avoid data leakage.

    Molecules sharing the same core scaffold are placed in the same fold.
    Scaffolds are sorted by frequency (largest first) and greedily assigned
    to the test set until the target fraction is met.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a SMILES column.
    smiles_col : str
        Name of the SMILES column.
    test_size : float
        Approximate fraction of data for the test set.
    random_state : int
        Seed for reproducibility when shuffling scaffolds of equal size.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds: dict[str, list[int]] = {}
    for idx, smi in df[smiles_col].items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        else:
            scaffold = "INVALID"
        scaffolds.setdefault(scaffold, []).append(idx)

    # Sort scaffolds: largest groups first, then alphabetically for ties
    import random
    rng = random.Random(random_state)
    scaffold_groups = sorted(
        scaffolds.values(), key=lambda g: (-len(g), rng.random())
    )

    n_test = int(len(df) * test_size)
    test_indices: list[int] = []
    train_indices: list[int] = []

    for group in scaffold_groups:
        if len(test_indices) < n_test:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    return df.loc[train_indices].copy(), df.loc[test_indices].copy()


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple random train/test split (fallback when RDKit is unavailable).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    test_size : float
        Fraction for the test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df
