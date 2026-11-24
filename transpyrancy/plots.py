"""
Visualisation helpers for ExplainResult.

All public functions accept an ExplainResult, generate a matplotlib Figure,
and optionally save it as a PNG and/or display it inline.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

if TYPE_CHECKING:
    from transpyrancy.result import ExplainResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle_output(
    fig: plt.Figure,
    save_path: str | Path | None,
    show: bool,
) -> plt.Figure:
    """Save and/or display a figure, then return it."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _get_shap_values_array(
    result: ExplainResult,
    class_index: int | None = None,
) -> np.ndarray:
    """Extract a 2-D SHAP value array from the result."""
    sv = result.shap_values
    if result.is_multiclass and isinstance(sv, list):
        idx = class_index if class_index is not None else 1
        return sv[idx].values
    if isinstance(sv, pd.DataFrame):
        return sv.values
    return np.asarray(sv)


def _get_shap_df(
    result: ExplainResult,
    class_index: int | None = None,
) -> pd.DataFrame:
    """Extract a SHAP DataFrame from the result."""
    sv = result.shap_values
    if result.is_multiclass and isinstance(sv, list):
        idx = class_index if class_index is not None else 1
        return sv[idx]
    if isinstance(sv, pd.DataFrame):
        return sv
    return pd.DataFrame(sv, columns=result.feature_names)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_summary(
    result: ExplainResult,
    max_display: int = 20,
    plot_type: str = "dot",
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    SHAP summary (beeswarm / dot) plot showing the distribution of SHAP values
    across all observations for each feature.

    Parameters
    ----------
    result : ExplainResult
        The explanation result to visualise.
    max_display : int
        Maximum number of features to display. Default 20.
    plot_type : str
        'dot' (beeswarm) or 'bar'. Default 'dot'.
    class_index : int | None
        For multiclass models, which class to plot. Defaults to class 1.
    save_path : str | Path | None
        File path to save the PNG. None skips saving.
    show : bool
        Whether to call plt.show(). Default True.

    Returns
    -------
    plt.Figure
    """
    sv = _get_shap_values_array(result, class_index)
    sv_df = _get_shap_df(result, class_index)

    fig, ax = plt.subplots(figsize=(10, max(4, min(max_display, sv.shape[1]) * 0.4 + 1)))
    plt.sca(ax)
    shap.summary_plot(
        sv,
        features=sv_df,
        feature_names=result.feature_names,
        max_display=max_display,
        plot_type=plot_type,
        show=False,
    )
    title_suffix = f" (class {class_index})" if result.is_multiclass else ""
    ax.set_title(f"SHAP Summary Plot{title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_beeswarm(
    result: ExplainResult,
    max_display: int = 20,
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    SHAP beeswarm plot — alias for plot_summary with plot_type='dot'.

    Parameters
    ----------
    result : ExplainResult
    max_display : int
    class_index : int | None
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    return plot_summary(
        result,
        max_display=max_display,
        plot_type="dot",
        class_index=class_index,
        save_path=save_path,
        show=show,
    )


def plot_feature_importance(
    result: ExplainResult,
    top_n: int = 20,
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of mean absolute SHAP values (global feature importance).

    Parameters
    ----------
    result : ExplainResult
    top_n : int
        Number of top features to display.
    class_index : int | None
        For multiclass: restrict to a single class. None aggregates all classes.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    agg = result.aggregate

    if result.is_multiclass and class_index is not None:
        col = f"class_{class_index}"
        if col not in agg.columns:
            raise ValueError(f"class_index {class_index} not found in aggregate columns.")
        importance = agg[col].sort_values(ascending=False).head(top_n)
        title_suffix = f" (class {class_index})"
    elif result.is_multiclass:
        importance = agg["mean_abs_shap_total"].sort_values(ascending=False).head(top_n)
        title_suffix = " (all classes)"
    else:
        importance = agg["mean_abs_shap"].sort_values(ascending=False).head(top_n)
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(9, max(4, len(importance) * 0.4 + 1)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance)))[::-1]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Global Feature Importance{title_suffix}", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_waterfall(
    result: ExplainResult,
    index: int = 0,
    max_display: int = 20,
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Waterfall plot for a single observation — shows how each feature pushes the
    prediction above or below the model's expected value.

    Parameters
    ----------
    result : ExplainResult
    index : int
        Row index of the observation to explain.
    max_display : int
        Maximum number of features to display.
    class_index : int | None
        For multiclass models, which class to plot.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    sv_df = _get_shap_df(result, class_index)
    sv_row = sv_df.iloc[index].values

    # Resolve expected value
    if result.is_multiclass and isinstance(result.expected_value, list):
        idx = class_index if class_index is not None else 1
        ev = result.expected_value[idx]
    else:
        ev = result.expected_value if isinstance(result.expected_value, (int, float)) else float(result.expected_value)  # type: ignore[arg-type]

    # Build an Explanation for the single row
    data_row = sv_df.iloc[index].rename(index=str).to_frame().T
    explanation = shap.Explanation(
        values=sv_row,
        base_values=ev,
        data=data_row.values[0],
        feature_names=result.feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, max(4, min(max_display, len(result.feature_names)) * 0.45 + 1)))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    title_suffix = f" (class {class_index})" if result.is_multiclass else ""
    ax.set_title(f"Waterfall — Observation {index}{title_suffix}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_force(
    result: ExplainResult,
    index: int = 0,
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Force plot for a single observation using matplotlib output.

    Displays how features push the prediction from the baseline (expected value)
    to the final output.

    Parameters
    ----------
    result : ExplainResult
    index : int
        Row index of the observation to explain.
    class_index : int | None
        For multiclass models, which class to plot.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    sv_df = _get_shap_df(result, class_index)
    sv_row = sv_df.iloc[index].values

    if result.is_multiclass and isinstance(result.expected_value, list):
        idx = class_index if class_index is not None else 1
        ev = result.expected_value[idx]
    else:
        ev = result.expected_value  # type: ignore[assignment]

    shap.force_plot(
        base_value=ev,
        shap_values=sv_row,
        features=sv_df.iloc[index],
        feature_names=result.feature_names,
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    title_suffix = f" — class {class_index}" if result.is_multiclass else ""
    fig.suptitle(f"Force Plot — Observation {index}{title_suffix}", fontsize=11, fontweight="bold", y=1.05)
    return _handle_output(fig, save_path, show)


def plot_dependence(
    result: ExplainResult,
    feature: str,
    interaction_feature: str | None = "auto",
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Dependence scatter plot for a single feature.

    Shows how that feature's SHAP value changes as its value changes, optionally
    coloured by an interaction feature.

    Parameters
    ----------
    result : ExplainResult
    feature : str
        Feature name to plot on the x-axis.
    interaction_feature : str | None
        Feature name to use for the colour axis. 'auto' lets SHAP choose.
        None disables colouring.
    class_index : int | None
        For multiclass models, which class to plot.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    sv = _get_shap_values_array(result, class_index)
    sv_df = _get_shap_df(result, class_index)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)
    shap.dependence_plot(
        ind=feature,
        shap_values=sv,
        features=sv_df,
        feature_names=result.feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    title_suffix = f" (class {class_index})" if result.is_multiclass else ""
    ax.set_title(f"Dependence: {feature}{title_suffix}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_heatmap(
    result: ExplainResult,
    max_display: int = 20,
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    SHAP heatmap showing feature contributions across all observations.

    Useful for spotting patterns: which observations are most influenced by which
    features, and in which direction.

    Parameters
    ----------
    result : ExplainResult
    max_display : int
        Number of top features to include.
    class_index : int | None
        For multiclass models, which class to plot.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    sv_df = _get_shap_df(result, class_index)

    # Select top features by mean |SHAP|
    top_features = sv_df.abs().mean().sort_values(ascending=False).head(max_display).index.tolist()
    sv_subset = sv_df[top_features]

    fig, ax = plt.subplots(figsize=(12, max(4, len(top_features) * 0.45 + 2)))
    cmap = plt.cm.RdBu_r
    vmax = np.abs(sv_subset.values).max()
    im = ax.imshow(
        sv_subset.T.values,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Observation index", fontsize=10)
    ax.set_title(
        "SHAP Value Heatmap" + (f" (class {class_index})" if result.is_multiclass else ""),
        fontsize=13,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="SHAP value")
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_decision(
    result: ExplainResult,
    max_display: int = 20,
    class_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    SHAP decision plot showing cumulative feature contributions per observation.

    Each line represents one observation's path from the expected value to the
    final prediction. Useful for comparing how different observations reach their
    predictions.

    Parameters
    ----------
    result : ExplainResult
    max_display : int
        Maximum features to display.
    class_index : int | None
        For multiclass models, which class to plot.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure
    """
    sv = _get_shap_values_array(result, class_index)

    if result.is_multiclass and isinstance(result.expected_value, list):
        idx = class_index if class_index is not None else 1
        ev = result.expected_value[idx]
    else:
        ev = result.expected_value  # type: ignore[assignment]

    fig, ax = plt.subplots(figsize=(9, max(5, min(max_display, sv.shape[1]) * 0.45 + 2)))
    plt.sca(ax)
    shap.decision_plot(
        base_value=ev,
        shap_values=sv,
        feature_names=result.feature_names,
        feature_display_range=slice(-1, -(max_display + 1), -1),
        show=False,
    )
    title_suffix = f" (class {class_index})" if result.is_multiclass else ""
    ax.set_title(f"Decision Plot{title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_group_contributions(
    result: ExplainResult,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart of mean absolute group-level SHAP contributions.

    Only available when feature_groups were defined at Explainer construction.

    Parameters
    ----------
    result : ExplainResult
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If group_contributions is None (no feature_groups were provided).
    """
    if result.group_contributions is None:
        raise ValueError(
            "No group contributions available. Pass `feature_groups` when constructing "
            "the Explainer."
        )
    gc = result.group_contributions.abs().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(3, len(gc) * 0.5 + 1)))
    colors = plt.cm.Paired(np.linspace(0, 1, len(gc)))
    ax.barh(gc.index[::-1], gc.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP contribution|", fontsize=11)
    ax.set_title("Feature Group Contributions", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_confidence_intervals(
    result: ExplainResult,
    top_n: int = 10,
    observation_index: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualise bootstrap confidence intervals for SHAP values.

    When `observation_index` is None, displays the mean SHAP value ± CI for
    each top feature (aggregated across all observations). When an index is
    given, displays the CI for that specific observation.

    Parameters
    ----------
    result : ExplainResult
    top_n : int
        Number of top features to display.
    observation_index : int | None
        Row to focus on. None shows aggregate CIs.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If confidence_intervals is None.
    """
    if result.confidence_intervals is None:
        raise ValueError(
            "No confidence intervals available. Pass `compute_confidence_intervals=True` "
            "to Explainer.explain()."
        )

    sv_df = _get_shap_df(result)
    lower = result.confidence_intervals["lower"]
    upper = result.confidence_intervals["upper"]

    if observation_index is not None:
        means = sv_df.iloc[observation_index]
        lo = lower.iloc[observation_index]
        hi = upper.iloc[observation_index]
        title = f"SHAP Confidence Intervals — Observation {observation_index}"
    else:
        means = sv_df.mean()
        lo = lower.mean()
        hi = upper.mean()
        title = "SHAP Confidence Intervals (mean across observations)"

    # Select top features by mean absolute SHAP
    top_features = sv_df.abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    means = means[top_features]
    lo = lo[top_features]
    hi = hi[top_features]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top_features) * 0.5 + 1)))
    y_pos = np.arange(len(top_features))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in means.values]

    xerr_left  = np.abs(means.values - lo.values)
    xerr_right = np.abs(hi.values - means.values)
    ax.barh(y_pos, means.values, xerr=[xerr_left, xerr_right],
            color=colors, alpha=0.8, capsize=4, error_kw={"elinewidth": 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1] if False else top_features, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _handle_output(fig, save_path, show)


def plot_interaction_heatmap(
    result: ExplainResult,
    feature: str,
    top_n: int = 10,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Heatmap of mean absolute SHAP interaction values for a given feature
    against all other features.

    Only available when compute_interactions=True was passed to explain().

    Parameters
    ----------
    result : ExplainResult
    feature : str
        The primary feature to examine interactions for.
    top_n : int
        Number of top interacting features to display.
    save_path : str | Path | None
    show : bool

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If interaction_effects is None or feature is not found.
    """
    if result.interaction_effects is None:
        raise ValueError(
            "No interaction effects available. Pass `compute_interactions=True` "
            "to Explainer.explain()."
        )
    if feature not in result.feature_names:
        raise ValueError(f"Feature '{feature}' not found in {result.feature_names}.")

    feat_idx = result.feature_names.index(feature)
    interactions = np.asarray(result.interaction_effects)
    # Ensure 3-D (n_samples, n_features, n_features); drop extra dims if present
    while interactions.ndim > 3:
        interactions = interactions[..., -1]

    # Mean absolute interaction between `feature` and all others
    slice_ = np.abs(interactions[:, feat_idx, :]).mean(axis=0)
    if slice_.ndim > 1:
        slice_ = slice_.mean(axis=-1)
    mean_abs: np.ndarray = np.asarray(slice_).flatten()
    if len(mean_abs) != len(result.feature_names):
        raise ValueError(
            f"Interaction shape mismatch: got {len(mean_abs)} values for "
            f"{len(result.feature_names)} features."
        )
    feat_interactions = pd.Series(mean_abs, index=result.feature_names).drop(feature)
    top = feat_interactions.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.5 + 1)))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top)))
    ax.barh(top.index[::-1], top.values[::-1], color=colors)
    ax.set_xlabel("Mean |SHAP interaction|", fontsize=11)
    ax.set_title(f"Interactions with '{feature}'", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _handle_output(fig, save_path, show)
