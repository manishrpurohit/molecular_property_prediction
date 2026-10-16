# Molecular Property Prediction — EDA Capstone

An exploratory data analysis (EDA) project for seven MoleculeNet-style molecular property datasets. The notebook inspects the datasets and their schemas, missing values, target distributions and class balance, SMILES strings, duplicate records, and potential feature leakage. It also outlines suitable modelling experiments; it does not train or evaluate predictive models.

## Project files

| File | Description |
|---|---|
| `molecular_property_prediction_reviewed_v2.ipynb` | Main analysis notebook. |
| `molecular_property_prediction_datasets.xlsx` | Source workbook. The notebook reads its seven dataset sheets and excludes the `Overview` sheet from analysis. |

Keep the workbook in the same directory as the notebook before running it.

## Datasets

The workbook contains 18,848 rows across the seven dataset sheets. Rows across sheets may refer to the same molecule.

| Dataset | Rows | Main task / target |
|---|---:|---|
| BBBP | 2,050 | Binary classification — `p_np` |
| ESOL | 1,128 | Regression — measured log solubility |
| FreeSolv | 642 | Regression — `expt` |
| Lipophilicity | 4,200 | Regression — `exp` |
| ClinTox | 1,484 | Imbalanced classification — `CT_TOX` |
| Tox21 | 7,831 | Multi-label classification — assay labels |
| BACE | 1,513 | Classification — `Class` |

Exact column names and available features vary by sheet; consult the workbook and notebook before selecting predictors.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- `pandas`, `numpy`, `matplotlib`, `seaborn`, and `openpyxl`
- Optional: [RDKit](https://www.rdkit.org/) for molecular descriptor and chemistry-related checks

Install the Python dependencies in your environment:

```bash
python -m pip install pandas numpy matplotlib seaborn openpyxl jupyter
```

RDKit is optional. The notebook continues without it and reports that the optional checks are unavailable:

```bash
python -m pip install rdkit
```

## Run the analysis

1. Clone or download this project and place the Excel workbook beside the notebook.
2. Install the requirements above.
3. Start Jupyter from the project directory:

   ```bash
   jupyter notebook
   ```

4. Open `molecular_property_prediction_reviewed_v2.ipynb` and run the cells from top to bottom.

The notebook looks for `molecular_property_prediction_datasets.xlsx` in the current project directory. It raises a `FileNotFoundError` if the workbook is not found.

## Analysis and modelling notes

The notebook is an EDA and modelling-planning deliverable. Its suggested next step is to build an ESOL regression pipeline and a BBBP classification pipeline, then extend the work to ClinTox. Suggested evaluation includes MAE, RMSE, and R² for regression, and ROC-AUC, PR-AUC, recall, and F1 for classification, as appropriate.

For a realistic estimate of generalization to new chemical structures, prefer scaffold-based train/test splits where feasible. Handle missing Tox21 labels per assay rather than discarding a molecule because one assay label is missing. Treat notebook recommendations as proposals: report model performance only after running and validating the experiments.
