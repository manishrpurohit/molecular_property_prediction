"""
Model training, evaluation, and comparison utilities.

Provides helper functions for cross-validated training, metric reporting, and
side-by-side model comparison used by the modelling notebooks.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Compute regression metrics and return them as a dictionary.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Label for the model.

    Returns
    -------
    dict with keys: model, MAE, RMSE, R2
    """
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ── Classification metrics ───────────────────────────────────────────────────

def classification_report_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Compute classification metrics and return them as a dictionary.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    y_pred : array-like
        Predicted binary labels.
    model_name : str
        Label for the model.

    Returns
    -------
    dict with keys: model, ROC-AUC, PR-AUC, F1, Precision, Recall
    """
    return {
        "model": model_name,
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


# ── Model comparison ─────────────────────────────────────────────────────────

def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine a list of metric dictionaries into a comparison DataFrame.

    Parameters
    ----------
    results : list[dict]
        Each dict is a row from ``regression_report`` or
        ``classification_report_dict``.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by the first metric column descending.
    """
    df = pd.DataFrame(results)
    metric_cols = [c for c in df.columns if c != "model"]
    if metric_cols:
        df = df.sort_values(metric_cols[0], ascending=False)
    return df.reset_index(drop=True)


def cross_val_summary(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "neg_mean_absolute_error",
    cv: int = 5,
    model_name: str = "Model",
) -> dict[str, float]:
    """Run cross-validation and return mean ± std of the scoring metric.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Model to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        Target values.
    scoring : str
        Sklearn scoring string.
    cv : int
        Number of cross-validation folds.
    model_name : str
        Label for the model.

    Returns
    -------
    dict with keys: model, mean_score, std_score
    """
    scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    return {
        "model": model_name,
        "mean_score": scores.mean(),
        "std_score": scores.std(),
    }
