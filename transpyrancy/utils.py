"""
Internal utility functions for input validation, normalisation, and SHAP output
processing.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def validate_X(
    X: pd.DataFrame | np.ndarray,
    expected_n_features: int | None = None,
    expected_columns: list[str] | None = None,
) -> None:
    """
    Validate input data before SHAP computation.

    Raises ``TypeError`` for wrong input types, ``ValueError`` for shape or
    column mismatches, and emits ``UserWarning`` for NaN or infinite values
    (which may cause SHAP to fail or produce unreliable results).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Input data to validate.
    expected_n_features : int | None
        Expected number of features.  Checked when provided.
    expected_columns : list[str] | None
        Expected DataFrame column names (derived from background_data).
        Checked only when X is a DataFrame.
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"X must be a pd.DataFrame or np.ndarray, got {type(X).__name__}. "
            "Convert your data before calling explain()."
        )

    # Shape checks
    arr = np.asarray(X) if isinstance(X, np.ndarray) else None
    n_rows = len(X)
    if n_rows == 0:
        raise ValueError(
            "X is empty (0 rows). Pass at least one observation to explain()."
        )

    if expected_n_features is not None:
        actual_n = X.shape[1] if X.ndim > 1 else 1
        if actual_n != expected_n_features:
            raise ValueError(
                f"X has {actual_n} feature(s) but {expected_n_features} were expected "
                f"(inferred from background_data or feature_names). "
                "Ensure X has the same features the model was trained on."
            )

    # Column alignment for DataFrames
    if isinstance(X, pd.DataFrame) and expected_columns is not None:
        missing = set(expected_columns) - set(X.columns)
        extra = set(X.columns) - set(expected_columns)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected: {sorted(extra)}")
            raise ValueError(
                "X columns do not match background_data columns — "
                + "; ".join(parts) + ". "
                "Ensure X uses the same feature columns the model was trained on."
            )

    # Data quality — warn but don't block (SHAP handles some NaN cases)
    if isinstance(X, pd.DataFrame):
        numeric = X.select_dtypes(include=np.number)
        has_nan = X.isnull().any().any()
        has_inf = np.isinf(numeric.to_numpy()).any() if not numeric.empty else False
    else:
        X_arr = np.asarray(X, dtype=float)
        has_nan = bool(np.isnan(X_arr).any())
        has_inf = bool(np.isinf(X_arr).any())

    if has_nan:
        warnings.warn(
            "X contains NaN values. SHAP computation may fail or produce unreliable "
            "results. Consider imputing missing values before calling explain().",
            UserWarning,
            stacklevel=3,
        )
    if has_inf:
        warnings.warn(
            "X contains infinite values. SHAP computation may fail or produce "
            "unreliable results. Check your data for Inf / -Inf entries.",
            UserWarning,
            stacklevel=3,
        )


def detect_task(model: Any) -> str:
    """
    Infer whether a model is a classifier or regressor.

    Returns 'classification' if the model exposes `predict_proba`, otherwise
    'regression'.

    Parameters
    ----------
    model : Any
        A trained scikit-learn-compatible model.

    Returns
    -------
    str
        'classification' or 'regression'.
    """
    return "classification" if hasattr(model, "predict_proba") else "regression"


def normalize_input(
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None,
) -> tuple[pd.DataFrame | np.ndarray, list[str]]:
    """
    Normalise array-like input for model/SHAP consumption and resolve display
    feature names.

    The returned array or DataFrame is passed **as-is** to the model and SHAP
    explainer; column names are never changed.  The returned ``feature_names``
    list is applied only to the SHAP output columns afterwards.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Input data (single row or batch).
    feature_names : list[str] | None
        Explicit display names. If None, inferred from DataFrame columns or
        auto-generated as ['f0', 'f1', ...] for arrays.

    Returns
    -------
    tuple[pd.DataFrame | np.ndarray, list[str]]
        The original (possibly reshaped) input and the resolved feature names.
    """
    if isinstance(X, pd.DataFrame):
        names = feature_names if feature_names is not None else list(X.columns)
        return X, names
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_features = X.shape[1]
        names = feature_names if feature_names is not None else [f"f{i}" for i in range(n_features)]
        return X, names
    else:
        raise TypeError(
            f"X must be a pd.DataFrame or np.ndarray, got {type(X).__name__}."
        )


# ---------------------------------------------------------------------------
# SHAP output processing
# ---------------------------------------------------------------------------


def process_shap_output(
    shap_explanation: Any,
    feature_names: list[str],
    task: str,
    is_multiclass: bool,
) -> tuple[pd.DataFrame | list[pd.DataFrame], float | list[float]]:
    """
    Normalise raw SHAP output into a consistent DataFrame format.

    Handles the variety of shapes returned by different SHAP explainers:
    - 2-D array (n_samples, n_features): regression or binary classification
    - 3-D array (n_samples, n_features, n_classes): multiclass
    - list of 2-D arrays: old-style sklearn tree output

    Parameters
    ----------
    shap_explanation : shap.Explanation | np.ndarray | list
        Raw output from a SHAP explainer call.
    feature_names : list[str]
        Column names to apply to the resulting DataFrame(s).
    task : str
        'classification' or 'regression'.
    is_multiclass : bool
        True when more than two classes are present.

    Returns
    -------
    tuple[pd.DataFrame | list[pd.DataFrame], float | list[float]]
        Processed SHAP values and the expected (base) value(s).
    """
    if isinstance(shap_explanation, shap.Explanation):
        values = shap_explanation.values
        base = shap_explanation.base_values
    elif isinstance(shap_explanation, list):
        # Old sklearn API returns a list of arrays, one per class
        base = None
        values = np.stack(shap_explanation, axis=-1)  # → (n_samples, n_features, n_classes)
    else:
        values = np.asarray(shap_explanation)
        base = None

    # Resolve expected value
    expected_value: float | list[float]
    if base is None:
        expected_value = 0.0
    elif isinstance(base, np.ndarray):
        if base.ndim == 0:
            expected_value = float(base)
        elif base.ndim == 1:
            if is_multiclass:
                expected_value = base.tolist()
            else:
                expected_value = float(base[0]) if len(base) == 1 else float(base[-1])
        elif base.ndim == 2:
            # (n_samples, n_classes) — take the first row
            expected_value = base[0].tolist() if is_multiclass else float(base[0, -1])
        else:
            expected_value = float(np.mean(base))
    else:
        expected_value = float(base)

    # Resolve SHAP value array
    if values.ndim == 3:
        n_classes = values.shape[2]
        if is_multiclass:
            dfs = [
                pd.DataFrame(values[:, :, c], columns=feature_names)
                for c in range(n_classes)
            ]
            return dfs, expected_value
        elif n_classes == 1:
            # Single output (regression or single-node deep model)
            values = values[:, :, 0]
        else:
            # Binary: use positive-class column (index 1)
            values = values[:, :, 1]
    elif isinstance(values, list):
        # Should have been stacked above, but handle edge case
        values = np.asarray(values[1] if len(values) == 2 else values[-1])

    return pd.DataFrame(values, columns=feature_names), expected_value


def detect_multiclass(shap_explanation: Any) -> bool:
    """
    Return True if the SHAP output represents a multiclass problem (>2 classes).

    Parameters
    ----------
    shap_explanation : Any
        Raw SHAP explainer output.

    Returns
    -------
    bool
    """
    if isinstance(shap_explanation, shap.Explanation):
        values = shap_explanation.values
        if isinstance(values, list):
            return len(values) > 2
        if isinstance(values, np.ndarray) and values.ndim == 3:
            return values.shape[2] > 2
    elif isinstance(shap_explanation, list):
        # list of 1 → regression/single-output; 2 → binary; 3+ → multiclass
        return len(shap_explanation) > 2
    elif isinstance(shap_explanation, np.ndarray):
        return shap_explanation.ndim == 3 and shap_explanation.shape[2] > 2
    return False


# ---------------------------------------------------------------------------
# Aggregate / group helpers
# ---------------------------------------------------------------------------


def compute_aggregate(
    shap_values: pd.DataFrame | list[pd.DataFrame],
    is_multiclass: bool,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature (global feature importance).

    Parameters
    ----------
    shap_values : pd.DataFrame | list[pd.DataFrame]
        Per-observation SHAP values. For multiclass, a list of DataFrames.
    is_multiclass : bool
        True when the model is multiclass.

    Returns
    -------
    pd.DataFrame
        Sorted (descending) DataFrame of mean absolute SHAP values.
        Columns:
        - Binary/regression: ['mean_abs_shap', 'mean_shap']
        - Multiclass: one column per class label, plus 'mean_abs_shap_total'
    """
    if is_multiclass and isinstance(shap_values, list):
        class_means = {
            f"class_{i}": sv.abs().mean() for i, sv in enumerate(shap_values)
        }
        df = pd.DataFrame(class_means)
        df["mean_abs_shap_total"] = df.mean(axis=1)
        df = df.sort_values("mean_abs_shap_total", ascending=False)
    else:
        sv = shap_values if isinstance(shap_values, pd.DataFrame) else shap_values[0]
        df = pd.DataFrame(
            {
                "mean_abs_shap": sv.abs().mean(),
                "mean_shap": sv.mean(),
            }
        )
        df = df.sort_values("mean_abs_shap", ascending=False)

    return df


def compute_group_contributions(
    shap_values: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Aggregate SHAP values for user-defined feature groups.

    Each group's contribution is the sum of its member features' SHAP values
    for each observation.

    Parameters
    ----------
    shap_values : pd.DataFrame
        Per-observation SHAP values, shape (n_samples, n_features).
    feature_groups : dict[str, list[str]]
        Mapping of group name → list of feature names to aggregate.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, n_groups), one column per group.
    """
    available = set(shap_values.columns)
    result: dict[str, pd.Series] = {}
    for group_name, members in feature_groups.items():
        valid = [f for f in members if f in available]
        missing = set(members) - available
        if missing:
            import warnings
            warnings.warn(
                f"Feature group '{group_name}' contains features not found in "
                f"shap_values columns: {missing}. They will be ignored.",
                UserWarning,
                stacklevel=3,
            )
        if valid:
            result[group_name] = shap_values[valid].sum(axis=1)

    return pd.DataFrame(result)
