"""
Streamlit app — Molecular Property Prediction

Accepts a SMILES string and returns predicted molecular properties using
trained models (ESOL solubility, BBBP penetration, and optional extensions).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Molecular Property Predictor",
    page_icon="🧬",
    layout="wide",
)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(name: str):
    """Load a saved sklearn pipeline from the models directory."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def smiles_to_features(smiles: str):
    """Convert a SMILES string to a feature vector using Morgan fingerprints."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp, dtype=np.int8).reshape(1, -1)
    except ImportError:
        st.error("RDKit is required. Install it with: pip install rdkit")
        return None


def render_molecule(smiles: str):
    """Render a 2-D molecule image from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=(400, 300))
    except ImportError:
        return None


# ── UI ───────────────────────────────────────────────────────────────────────

st.title("🧬 Molecular Property Predictor")
st.markdown(
    "Enter a **SMILES** string to predict molecular properties using trained "
    "machine learning models."
)

smiles_input = st.text_input(
    "SMILES string",
    value="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    help="Simplified Molecular Input Line Entry System notation",
)

if st.button("Predict", type="primary"):
    if not smiles_input.strip():
        st.warning("Please enter a SMILES string.")
    else:
        features = smiles_to_features(smiles_input)
        if features is None:
            st.error(
                "Could not parse the SMILES string. Check the input and try again."
            )
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                mol_img = render_molecule(smiles_input)
                if mol_img is not None:
                    st.image(mol_img, caption="Molecular structure")

            with col2:
                # ESOL prediction
                esol_model = load_model("esol_pipeline")
                if esol_model is not None:
                    pred = esol_model.predict(features)[0]
                    st.metric(
                        label="💧 Predicted Log Solubility (ESOL)",
                        value=f"{pred:.2f} log mol/L",
                    )
                else:
                    st.info(
                        "ESOL model not found. Train it first using the "
                        "ESOL notebook and save as `models/esol_pipeline.joblib`."
                    )

                # BBBP prediction
                bbbp_model = load_model("bbbp_pipeline")
                if bbbp_model is not None:
                    prob = bbbp_model.predict_proba(features)[0, 1]
                    label = "Penetrates ✅" if prob >= 0.5 else "Does not penetrate ❌"
                    st.metric(
                        label="🧠 Blood-Brain Barrier Penetration (BBBP)",
                        value=label,
                        delta=f"{prob:.1%} probability",
                    )
                else:
                    st.info(
                        "BBBP model not found. Train it first using the "
                        "BBBP notebook and save as `models/bbbp_pipeline.joblib`."
                    )

                # ClinTox prediction (optional extension)
                clintox_model = load_model("clintox_pipeline")
                if clintox_model is not None:
                    prob = clintox_model.predict_proba(features)[0, 1]
                    label = "Toxic ⚠️" if prob >= 0.5 else "Non-toxic ✅"
                    st.metric(
                        label="☠️ Clinical Toxicity (ClinTox)",
                        value=label,
                        delta=f"{prob:.1%} probability",
                    )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** These predictions are for research purposes only. "
    "They do not constitute proof of clinical safety, drug effectiveness, or "
    "suitability for human use."
)
