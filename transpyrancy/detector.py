"""
Model type detection and SHAP explainer selection logic.
"""
from __future__ import annotations

import warnings
from enum import Enum


class ExplainerType(str, Enum):
    """Enum of supported SHAP explainer backends."""

    TREE = "tree"
    LINEAR = "linear"
    KERNEL = "kernel"
    DEEP = "deep"


# Canonical mapping: model_type string → ExplainerType
MODEL_TYPE_MAP: dict[str, ExplainerType] = {
    # Tree-based
    "xgb": ExplainerType.TREE,
    "xgboost": ExplainerType.TREE,
    "lgb": ExplainerType.TREE,
    "lightgbm": ExplainerType.TREE,
    "catboost": ExplainerType.TREE,
    "rf": ExplainerType.TREE,
    "random_forest": ExplainerType.TREE,
    "gbm": ExplainerType.TREE,
    "gradient_boosting": ExplainerType.TREE,
    "gbc": ExplainerType.TREE,
    "gbr": ExplainerType.TREE,
    "et": ExplainerType.TREE,
    "extra_trees": ExplainerType.TREE,
    "extratrees": ExplainerType.TREE,
    "dt": ExplainerType.TREE,
    "decision_tree": ExplainerType.TREE,
    "decisiontree": ExplainerType.TREE,
    "histgbm": ExplainerType.TREE,
    "hist_gradient_boosting": ExplainerType.TREE,
    # Linear
    "linear": ExplainerType.LINEAR,
    "logistic": ExplainerType.LINEAR,
    "logistic_regression": ExplainerType.LINEAR,
    "logisticregression": ExplainerType.LINEAR,
    "linear_regression": ExplainerType.LINEAR,
    "linearregression": ExplainerType.LINEAR,
    "ridge": ExplainerType.LINEAR,
    "ridgeclassifier": ExplainerType.LINEAR,
    "lasso": ExplainerType.LINEAR,
    "elasticnet": ExplainerType.LINEAR,
    "sgd": ExplainerType.LINEAR,
    "sgdclassifier": ExplainerType.LINEAR,
    "sgdregressor": ExplainerType.LINEAR,
    "perceptron": ExplainerType.LINEAR,
    "svm_linear": ExplainerType.LINEAR,
    "linearsvc": ExplainerType.LINEAR,
    "linearsvr": ExplainerType.LINEAR,
    # Kernel (fallback)
    "kernel": ExplainerType.KERNEL,
    "svm": ExplainerType.KERNEL,
    "svc": ExplainerType.KERNEL,
    "svr": ExplainerType.KERNEL,
    "knn": ExplainerType.KERNEL,
    "mlp": ExplainerType.KERNEL,
    "naive_bayes": ExplainerType.KERNEL,
    "gaussiannb": ExplainerType.KERNEL,
    # Deep learning (PyTorch / TensorFlow / Keras)
    "torch": ExplainerType.DEEP,
    "pytorch": ExplainerType.DEEP,
    "tensorflow": ExplainerType.DEEP,
    "tf": ExplainerType.DEEP,
    "keras": ExplainerType.DEEP,
    "neural_network": ExplainerType.DEEP,
    "nn": ExplainerType.DEEP,
    "deep": ExplainerType.DEEP,
}

SUPPORTED_MODEL_TYPES: list[str] = sorted(MODEL_TYPE_MAP.keys())


def detect_explainer_type(model_type: str) -> ExplainerType:
    """
    Infer the appropriate SHAP explainer type from a model type string.

    Parameters
    ----------
    model_type : str
        The model type identifier (e.g., 'xgb', 'lightgbm', 'ridge').

    Returns
    -------
    ExplainerType
        The detected explainer type. Falls back to KERNEL with a warning
        if the model type is not recognised.
    """
    key = model_type.lower().strip()
    if key not in MODEL_TYPE_MAP:
        warnings.warn(
            f"Unknown model_type '{model_type}'. Falling back to KernelExplainer. "
            f"Supported model types: {SUPPORTED_MODEL_TYPES}",
            UserWarning,
            stacklevel=3,
        )
        return ExplainerType.KERNEL
    return MODEL_TYPE_MAP[key]


def validate_override(
    detected: ExplainerType,
    override: ExplainerType,
    model_type: str,
) -> None:
    """
    Warn the user when an explainer override conflicts with the detected type.

    Parameters
    ----------
    detected : ExplainerType
        The explainer type inferred from model_type.
    override : ExplainerType
        The explainer type explicitly requested by the user.
    model_type : str
        The original model_type string, used in the warning message.
    """
    if detected != override:
        warnings.warn(
            f"explainer_override='{override.value}' conflicts with the auto-detected "
            f"explainer type '{detected.value}' for model_type='{model_type}'. "
            f"Proceeding with '{override.value}'. This may produce unexpected or "
            f"incorrect SHAP values.",
            UserWarning,
            stacklevel=3,
        )
