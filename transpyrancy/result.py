"""
ExplainResult — the rich container returned by Explainer.explain().

Holds all computed SHAP outputs and exposes every visualisation method
directly on the object, so results are self-documenting and easy to explore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExplainResult:
    """
    Container for all SHAP explanation outputs.

    Returned by :meth:`Explainer.explain`. Access data attributes directly
    or call any of the built-in visualisation methods.

    Attributes
    ----------
    shap_values : pd.DataFrame | list[pd.DataFrame]
        Per-observation SHAP values.

        - **Binary classification / regression**: ``DataFrame`` of shape
          ``(n_observations, n_features)``.  Positive values push the
          prediction higher; negative values push it lower.
        - **Multiclass classification**: ``list`` of DataFrames, one per
          class, each of shape ``(n_observations, n_features)``.

    aggregate : pd.DataFrame
        Global feature importance — mean absolute SHAP value per feature,
        sorted descending.

        - **Binary / regression** columns: ``mean_abs_shap``, ``mean_shap``
        - **Multiclass** columns: ``class_0``, ``class_1``, …,
          ``mean_abs_shap_total``

    expected_value : float | list[float]
        The model's baseline output — :math:`E[f(X)]` over the background
        dataset.  For multiclass, one value per class.

    predictions : np.ndarray
        Raw model predictions (class labels for classifiers, continuous
        values for regressors).  Shape ``(n_observations,)``.

    prediction_proba : np.ndarray | None
        Predicted probabilities for classification models.  Shape
        ``(n_observations, n_classes)``.  ``None`` for regressors.

    task : str
        ``'classification'`` or ``'regression'``.

    n_observations : int
        Number of observations explained.

    feature_names : list[str]
        Feature names corresponding to SHAP value columns.

    model_type : str
        The ``model_type`` string passed at construction (e.g. ``'xgb'``).

    explainer_type : str
        The SHAP explainer backend used: ``'tree'``, ``'linear'``, or
        ``'kernel'``.

    is_multiclass : bool
        ``True`` when the model has more than two output classes.

    n_classes : int | None
        Number of output classes.  ``None`` for regression.

    interaction_effects : np.ndarray | None
        SHAP interaction values of shape
        ``(n_observations, n_features, n_features)``.  ``None`` unless
        ``compute_interactions=True`` was passed to :meth:`Explainer.explain`.

    confidence_intervals : dict[str, pd.DataFrame] | None
        Bootstrap confidence intervals.  Dict with keys ``'lower'`` and
        ``'upper'``, each a DataFrame of shape
        ``(n_observations, n_features)``.  ``None`` unless
        ``compute_confidence_intervals=True`` was passed.

    group_contributions : pd.DataFrame | None
        Summed SHAP values per feature group, shape
        ``(n_observations, n_groups)``.  ``None`` unless ``feature_groups``
        was provided at :class:`Explainer` construction.
    """

    # ------------------------------------------------------------------ #
    # Core SHAP outputs
    # ------------------------------------------------------------------ #

    shap_values: pd.DataFrame | list[pd.DataFrame]
    """Per-observation SHAP values (see class docstring for shape details)."""

    aggregate: pd.DataFrame
    """Mean absolute SHAP per feature, sorted descending."""

    expected_value: float | list[float]
    """Baseline model output E[f(X)]."""

    predictions: np.ndarray
    """Model predictions, shape (n_observations,)."""

    task: str
    """'classification' or 'regression'."""

    n_observations: int
    """Number of rows explained."""

    feature_names: list[str]
    """Feature column names."""

    model_type: str
    """Model type identifier."""

    explainer_type: str
    """SHAP backend used: 'tree', 'linear', or 'kernel'."""

    is_multiclass: bool
    """True when the model produces more than two classes."""

    # ------------------------------------------------------------------ #
    # Optional outputs
    # ------------------------------------------------------------------ #

    prediction_proba: np.ndarray | None = field(default=None)
    """Predicted class probabilities (classification only)."""

    n_classes: int | None = field(default=None)
    """Number of output classes (None for regression)."""

    interaction_effects: np.ndarray | None = field(default=None)
    """SHAP interaction values (n_observations, n_features, n_features)."""

    confidence_intervals: dict[str, pd.DataFrame] | None = field(default=None)
    """Bootstrap CIs: {'lower': DataFrame, 'upper': DataFrame}."""

    group_contributions: pd.DataFrame | None = field(default=None)
    """Summed SHAP per feature group (n_observations, n_groups)."""

    # Internal — not displayed in repr
    _shap_explanation: Any = field(default=None, repr=False)
    """Raw shap.Explanation object for native SHAP plot compatibility."""

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def feature_importance(self) -> pd.Series:
        """
        Global feature importance as a Series sorted descending.

        For multiclass, returns the aggregate across all classes
        (``mean_abs_shap_total``); for binary/regression, returns
        ``mean_abs_shap``.
        """
        if self.is_multiclass:
            return self.aggregate["mean_abs_shap_total"].sort_values(ascending=False)
        return self.aggregate["mean_abs_shap"].sort_values(ascending=False)

    def shap_values_for_class(self, class_index: int) -> pd.DataFrame:
        """
        Return SHAP values for a specific class (multiclass models only).

        Parameters
        ----------
        class_index : int
            Zero-based class index.

        Returns
        -------
        pd.DataFrame
            Shape (n_observations, n_features).

        Raises
        ------
        ValueError
            If the model is not multiclass or the index is out of range.
        """
        if not self.is_multiclass:
            raise ValueError(
                "shap_values_for_class() is only available for multiclass models. "
                "Access shap_values directly for binary / regression."
            )
        if not isinstance(self.shap_values, list):
            raise ValueError("shap_values is not a list — unexpected internal state.")
        if class_index >= len(self.shap_values):
            raise ValueError(
                f"class_index {class_index} out of range "
                f"(model has {len(self.shap_values)} classes)."
            )
        return self.shap_values[class_index]

    def top_features(self, n: int = 10, class_index: int | None = None) -> pd.Series:
        """
        Return the top-n most important features by mean absolute SHAP.

        Parameters
        ----------
        n : int
            Number of features to return.
        class_index : int | None
            For multiclass: restrict ranking to a single class.

        Returns
        -------
        pd.Series
            Index = feature names, values = mean absolute SHAP, sorted desc.
        """
        if self.is_multiclass and class_index is not None:
            col = f"class_{class_index}"
            return self.aggregate[col].sort_values(ascending=False).head(n)
        return self.feature_importance.head(n)

    def summary(self) -> str:
        """
        Return a concise human-readable summary of the explanation.

        Returns
        -------
        str
        """
        lines = [
            "=" * 60,
            "ExplainResult Summary",
            "=" * 60,
            f"  Task          : {self.task}",
            f"  Model type    : {self.model_type}",
            f"  Explainer     : {self.explainer_type}",
            f"  Observations  : {self.n_observations}",
            f"  Features      : {len(self.feature_names)}",
            f"  Multiclass    : {self.is_multiclass}" + (f" ({self.n_classes} classes)" if self.n_classes else ""),
            f"  Expected value: {self.expected_value:.4f}" if isinstance(self.expected_value, float) else f"  Expected value: {self.expected_value}",
            "",
            "  Top 5 features (by mean |SHAP|):",
        ]
        top5 = self.top_features(5)
        for feat, val in top5.items():
            lines.append(f"    {feat:<30s} {val:.4f}")
        if self.interaction_effects is not None:
            lines.append("  Interaction effects : available")
        if self.confidence_intervals is not None:
            lines.append("  Confidence intervals: available")
        if self.group_contributions is not None:
            lines.append(f"  Group contributions : {list(self.group_contributions.columns)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def help(self, topic: str | None = None) -> None:
        """
        Display plain-English help for any attribute, plot, or concept.

        Call with no arguments for a full, context-aware overview of
        everything available on this result.  Pass a topic string to zoom
        in on a single item.

        Parameters
        ----------
        topic : str | None
            Name of any attribute or plot method, e.g.:

            - ``'shap_values'``       — what the SHAP values mean
            - ``'aggregate'``         — global feature importance
            - ``'expected_value'``    — the model baseline
            - ``'plot_waterfall'``    — how to read a waterfall chart
            - ``'plot_summary'``      — the beeswarm overview
            - ``'confidence_intervals'`` — bootstrap uncertainty
            - ``'interaction_effects'``  — pairwise SHAP interactions

            Short aliases also work: ``'waterfall'``, ``'ci'``, ``'groups'``.

        Examples
        --------
        Full overview (context-aware, shows which extras are enabled)::

            result.help()

        Help for one specific plot::

            result.help('plot_waterfall')

        Help for a data attribute::

            result.help('shap_values')
        """
        from transpyrancy.docs import show_help  # noqa: PLC0415
        show_help(topic=topic, result=self)

    # ------------------------------------------------------------------ #
    # Visualisation methods
    # ------------------------------------------------------------------ #

    def plot_summary(
        self,
        max_display: int = 20,
        plot_type: str = "dot",
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        SHAP summary (beeswarm / dot) plot.

        Shows the distribution of each feature's SHAP values across all
        observations, coloured by the feature's raw value.

        Parameters
        ----------
        max_display : int
            Maximum number of features to show. Default 20.
        plot_type : str
            ``'dot'`` (beeswarm) or ``'bar'``. Default ``'dot'``.
        class_index : int | None
            For multiclass: which class to display.
        save_path : str | Path | None
            File path to save as PNG.
        show : bool
            Whether to display the plot inline.

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_summary(self, max_display=max_display, plot_type=plot_type,
                                  class_index=class_index, save_path=save_path, show=show)

    def plot_beeswarm(
        self,
        max_display: int = 20,
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        SHAP beeswarm plot (alias for plot_summary with plot_type='dot').

        Parameters
        ----------
        max_display : int
        class_index : int | None
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_beeswarm(self, max_display=max_display, class_index=class_index,
                                   save_path=save_path, show=show)

    def plot_feature_importance(
        self,
        top_n: int = 20,
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Horizontal bar chart of global feature importance (mean |SHAP|).

        Parameters
        ----------
        top_n : int
            Number of features to display.
        class_index : int | None
            For multiclass: restrict to one class.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_feature_importance(self, top_n=top_n, class_index=class_index,
                                             save_path=save_path, show=show)

    def plot_waterfall(
        self,
        index: int = 0,
        max_display: int = 20,
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Waterfall plot for a single observation.

        Shows how each feature drives the prediction from the baseline
        expected value to the final output.

        Parameters
        ----------
        index : int
            Row index of the observation to explain. Default 0.
        max_display : int
            Max features shown. Default 20.
        class_index : int | None
            For multiclass: which class to display.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_waterfall(self, index=index, max_display=max_display,
                                    class_index=class_index, save_path=save_path, show=show)

    def plot_force(
        self,
        index: int = 0,
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Force plot for a single observation.

        Displays feature contributions as arrows pushing the prediction
        left (negative) or right (positive) from the baseline.

        Parameters
        ----------
        index : int
            Row index. Default 0.
        class_index : int | None
            For multiclass: which class.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_force(self, index=index, class_index=class_index,
                                save_path=save_path, show=show)

    def plot_dependence(
        self,
        feature: str,
        interaction_feature: str | None = "auto",
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Dependence scatter plot for a single feature.

        Shows how that feature's SHAP value changes with its raw value.

        Parameters
        ----------
        feature : str
            Feature name to plot on the x-axis.
        interaction_feature : str | None
            Feature to use for colour encoding. ``'auto'`` lets SHAP
            choose; ``None`` disables colouring.
        class_index : int | None
            For multiclass: which class.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_dependence(self, feature=feature,
                                     interaction_feature=interaction_feature,
                                     class_index=class_index, save_path=save_path, show=show)

    def plot_heatmap(
        self,
        max_display: int = 20,
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        SHAP heatmap across all observations and top features.

        Reveals patterns: which observations are driven by which features,
        and in which direction.

        Parameters
        ----------
        max_display : int
            Number of top features to include.
        class_index : int | None
            For multiclass: which class.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_heatmap(self, max_display=max_display, class_index=class_index,
                                  save_path=save_path, show=show)

    def plot_decision(
        self,
        max_display: int = 20,
        class_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Decision plot showing each observation's cumulative feature path.

        Each line traces one observation from the baseline through feature
        contributions to the final prediction. Useful for comparing cases.

        Parameters
        ----------
        max_display : int
            Max features. Default 20.
        class_index : int | None
            For multiclass: which class.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure
        """
        from transpyrancy import plots
        return plots.plot_decision(self, max_display=max_display, class_index=class_index,
                                   save_path=save_path, show=show)

    def plot_group_contributions(
        self,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Bar chart of mean absolute group-level SHAP contributions.

        Requires ``feature_groups`` to have been set at :class:`Explainer`
        construction.

        Parameters
        ----------
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure

        Raises
        ------
        ValueError
            If no group contributions are available.
        """
        from transpyrancy import plots
        return plots.plot_group_contributions(self, save_path=save_path, show=show)

    def plot_confidence_intervals(
        self,
        top_n: int = 10,
        observation_index: int | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Bootstrap confidence interval plot for SHAP values.

        Requires ``compute_confidence_intervals=True`` in
        :meth:`Explainer.explain`.

        Parameters
        ----------
        top_n : int
            Top features to show.
        observation_index : int | None
            Specific row to focus on, or None to show aggregate CIs.
        save_path : str | Path | None
        show : bool

        Returns
        -------
        plt.Figure

        Raises
        ------
        ValueError
            If confidence intervals were not computed.
        """
        from transpyrancy import plots
        return plots.plot_confidence_intervals(self, top_n=top_n,
                                               observation_index=observation_index,
                                               save_path=save_path, show=show)

    def plot_interaction_heatmap(
        self,
        feature: str,
        top_n: int = 10,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """
        Heatmap of SHAP interaction values for a given feature.

        Requires ``compute_interactions=True`` in :meth:`Explainer.explain`.

        Parameters
        ----------
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
            If interaction effects were not computed or feature not found.
        """
        from transpyrancy import plots
        return plots.plot_interaction_heatmap(self, feature=feature, top_n=top_n,
                                              save_path=save_path, show=show)

    def _visualize_all(
        self,
        save_dir: Path | None = None,
        show: bool = True,
    ) -> None:
        """
        Generate all available plots in sequence.

        Called internally when ``visualize=True`` or ``save_plots`` is set
        on :meth:`Explainer.explain`.

        Parameters
        ----------
        save_dir : Path | None
            Directory to save PNG files. None means no saving.
        show : bool
            Whether to display plots inline.
        """

        def _path(name: str) -> str | None:
            return str(save_dir / f"{name}.png") if save_dir else None

        self.plot_summary(save_path=_path("summary"), show=show)
        self.plot_feature_importance(save_path=_path("feature_importance"), show=show)
        self.plot_heatmap(save_path=_path("heatmap"), show=show)
        self.plot_decision(save_path=_path("decision"), show=show)

        if self.n_observations >= 1:
            self.plot_waterfall(index=0, save_path=_path("waterfall_0"), show=show)
            self.plot_force(index=0, save_path=_path("force_0"), show=show)

        if self.group_contributions is not None:
            self.plot_group_contributions(save_path=_path("group_contributions"), show=show)

        if self.confidence_intervals is not None:
            self.plot_confidence_intervals(save_path=_path("confidence_intervals"), show=show)
