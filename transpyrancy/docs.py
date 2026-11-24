"""
transpyrancy.docs — Plain-English help catalog for ExplainResult attributes and plots.

Usage::

    result.help()                    # full context-aware overview
    result.help('shap_values')       # one attribute
    result.help('plot_waterfall')    # one plot
    Explainer.help()                 # pre-explain overview of the library
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HelpEntry:
    """A single help entry in the catalog."""

    title: str
    category: str          # 'attribute' | 'plot' | 'concept'
    what: str              # What is this?
    when: str              # When should I use it?
    look_for: str          # What patterns / values matter?
    tip: str = ""          # Pro tip or common gotcha
    requires: str = ""     # e.g. "compute_interactions=True"
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


CATALOG: dict[str, HelpEntry] = {

    # ------------------------------------------------------------------ #
    # Attributes
    # ------------------------------------------------------------------ #

    "shap_values": HelpEntry(
        title="SHAP Values",
        category="attribute",
        what=(
            "A DataFrame of shape (n_observations, n_features) — one SHAP value "
            "per feature per row. Each value measures how much that feature "
            "shifted the model's prediction away from the baseline for that "
            "specific observation. Positive = pushed prediction up; negative = "
            "pushed it down."
        ),
        when=(
            "Use shap_values when you need per-row, per-feature attribution. "
            "Great for debugging individual predictions, building custom plots, "
            "or feeding into downstream analysis."
        ),
        look_for=(
            "Large absolute values flag the features doing the most work for "
            "that observation. A value near 0 means the feature had almost no "
            "influence on that specific prediction (even if it's globally "
            "important). For multiclass models, shap_values is a list of "
            "DataFrames — one per class."
        ),
        tip=(
            "Check: shap_values.sum(axis=1) + expected_value ≈ predictions. "
            "If this doesn't hold, something went wrong with the explainer setup."
        ),
    ),

    "aggregate": HelpEntry(
        title="Aggregate Feature Importance",
        category="attribute",
        what=(
            "A DataFrame summarising global feature importance via mean absolute "
            "SHAP across all explained observations. Columns: mean_abs_shap "
            "(overall importance) and mean_shap (average signed direction). "
            "Rows are sorted from most to least important."
        ),
        when=(
            "Use aggregate when you want a single ranked list of which features "
            "matter most to the model as a whole — not for any one prediction, "
            "but on average across the dataset you explained."
        ),
        look_for=(
            "A big drop-off after the top few features often reveals a small, "
            "dominant feature set. mean_shap tells you direction: a positive "
            "mean_shap means the feature tends to push predictions up. "
            "For multiclass models, you get one column per class plus "
            "mean_abs_shap_total for the cross-class ranking."
        ),
        tip=(
            "Aggregate importance depends on the dataset you passed to explain(). "
            "Explain a balanced sample to avoid skewing rankings toward the most "
            "common outcome."
        ),
    ),

    "expected_value": HelpEntry(
        title="Expected Value (Baseline)",
        category="attribute",
        what=(
            "The model's average output over the background dataset — written "
            "E[f(X)]. This is the starting point from which SHAP values are "
            "measured. Think of it as 'what would the model predict if it knew "
            "nothing specific about this observation?'"
        ),
        when=(
            "Reference expected_value when interpreting waterfall or force plots: "
            "the bars show how each feature moves the prediction from this "
            "baseline to the final value."
        ),
        look_for=(
            "For classification, expected_value is typically close to the "
            "class base-rate in your background data. For regression, it should "
            "be near the mean of your training targets. A very surprising "
            "expected_value is a sign your background_data is unrepresentative."
        ),
        tip=(
            "For multiclass models, expected_value is a list — one float per "
            "class. Use result.expected_value[class_index] to access a specific "
            "class baseline."
        ),
    ),

    "predictions": HelpEntry(
        title="Model Predictions",
        category="attribute",
        what=(
            "A NumPy array of raw model outputs, shape (n_observations,). "
            "For classifiers this is the predicted class label; for regressors "
            "it is the continuous output."
        ),
        when=(
            "Use predictions to compare the model's output against ground-truth "
            "labels, compute error metrics, or identify best/worst predictions "
            "before drilling into their SHAP explanations."
        ),
        look_for=(
            "Pair predictions with your ground-truth labels to find high-error "
            "cases, then use plot_waterfall(index=worst_case) to understand why "
            "the model got it wrong."
        ),
    ),

    "prediction_proba": HelpEntry(
        title="Prediction Probabilities",
        category="attribute",
        what=(
            "Array of shape (n_observations, n_classes) containing the model's "
            "estimated class probabilities (classification only). None for "
            "regression models."
        ),
        when=(
            "Use prediction_proba when calibration or decision-threshold analysis "
            "matters — e.g. you need to know not just the predicted class but how "
            "confident the model is."
        ),
        look_for=(
            "Low maximum probabilities (e.g. 0.51 vs 0.49) signal that the model "
            "is uncertain. These borderline cases are often the most interesting "
            "to explain."
        ),
    ),

    "feature_importance": HelpEntry(
        title="Feature Importance (property)",
        category="attribute",
        what=(
            "A pd.Series of mean absolute SHAP values, sorted descending. "
            "A convenience shortcut to the most useful column of aggregate — "
            "equivalent to result.aggregate['mean_abs_shap']."
        ),
        when="Use for a quick ranked list without slicing aggregate manually.",
        look_for=(
            "Compare feature_importance across different model versions or "
            "datasets to track which signals the model is relying on over time."
        ),
    ),

    "interaction_effects": HelpEntry(
        title="Interaction Effects",
        category="attribute",
        what=(
            "A 3-D NumPy array of shape (n_observations, n_features, n_features) "
            "containing pairwise SHAP interaction values. The [i, j, k] entry "
            "measures the joint effect of features j and k on observation i, "
            "above and beyond their individual contributions."
        ),
        when=(
            "Use interaction_effects when you suspect two features work together "
            "— e.g. age and income might jointly drive loan decisions in ways "
            "neither does alone. Only available for TreeExplainer."
        ),
        look_for=(
            "Large off-diagonal values in the interaction matrix indicate strong "
            "pairwise interactions. Use plot_interaction_heatmap('feature') to "
            "visualise this for any feature of interest."
        ),
        requires="compute_interactions=True passed to explain()",
        tip="Can be slow for large datasets or many features. Subsample first.",
    ),

    "confidence_intervals": HelpEntry(
        title="Bootstrap Confidence Intervals",
        category="attribute",
        what=(
            "A dict with keys 'lower' and 'upper', each a DataFrame matching "
            "shap_values in shape. Built by resampling the input data n_bootstrap "
            "times and computing SHAP values each time — the CI captures how "
            "stable the SHAP estimates are."
        ),
        when=(
            "Use confidence_intervals when you need to communicate uncertainty "
            "around SHAP estimates, especially with small datasets where individual "
            "SHAP values may be noisy."
        ),
        look_for=(
            "Wide CIs on a feature mean its SHAP estimate is unreliable — more "
            "data or a more stable model would help. Narrow CIs mean you can "
            "trust the attribution."
        ),
        requires="compute_confidence_intervals=True passed to explain()",
        tip=(
            "Use n_jobs=-1 to run bootstrap resamples in parallel — can be "
            "dramatically faster for large n_bootstrap values."
        ),
    ),

    "group_contributions": HelpEntry(
        title="Feature Group Contributions",
        category="attribute",
        what=(
            "A DataFrame of shape (n_observations, n_groups) where each column "
            "is the sum of SHAP values for all features in that group. Lets you "
            "reason about blocks of related features (e.g. 'location' or "
            "'demographics') rather than individual columns."
        ),
        when=(
            "Use group_contributions when your features have natural semantic "
            "groupings and you want a higher-level attribution story: 'location "
            "features drove 60% of this prediction'."
        ),
        look_for=(
            "Which groups dominate? Compare group contributions across correct "
            "vs. incorrect predictions to spot systematic biases."
        ),
        requires="feature_groups= set at Explainer construction",
    ),

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #

    "plot_summary": HelpEntry(
        title="Summary / Beeswarm Plot",
        category="plot",
        aliases=["plot_beeswarm"],
        what=(
            "A dot-plot where every dot is one observation × one feature. "
            "The x-axis shows the SHAP value (impact on prediction); colour "
            "shows the raw feature value (red = high, blue = low). Rows are "
            "sorted by mean absolute SHAP."
        ),
        when="Start here. This is the single best overview of global model behaviour.",
        look_for=(
            "Features near the top matter most. Colour gradients that run left-to-right "
            "(red = positive SHAP, blue = negative) tell a clean story: high feature "
            "value → higher prediction. Mixed colour means a non-linear or "
            "interaction-driven relationship."
        ),
        tip=(
            "Narrow dot clusters mean most observations behave similarly. "
            "Wide spread signals heterogeneity — the feature matters differently "
            "for different groups."
        ),
    ),

    "plot_beeswarm": HelpEntry(
        title="Beeswarm Plot",
        category="plot",
        aliases=["plot_summary"],
        what="Alias for plot_summary(plot_type='dot'). See plot_summary.",
        when="Use interchangeably with plot_summary.",
        look_for="See plot_summary.",
    ),

    "plot_feature_importance": HelpEntry(
        title="Feature Importance Bar Chart",
        category="plot",
        what=(
            "A horizontal bar chart of mean absolute SHAP per feature, sorted "
            "descending. The simplest possible summary of which features the "
            "model relies on most."
        ),
        when=(
            "Use when you need a clean, executive-friendly view of importance — "
            "no colours or distributions, just a ranked list."
        ),
        look_for=(
            "A sharp 'elbow' in the bars means a small number of features are "
            "carrying most of the prediction signal. Features below the elbow "
            "are candidates for removal."
        ),
    ),

    "plot_waterfall": HelpEntry(
        title="Waterfall Plot",
        category="plot",
        what=(
            "A step-by-step breakdown of a single observation's prediction. "
            "Starts at the model baseline (expected_value) and adds/subtracts "
            "each feature's SHAP value, walking up or down to the final "
            "prediction. Red bars push the prediction higher; blue bars push "
            "it lower."
        ),
        when=(
            "Use when you need to explain a specific individual prediction — "
            "e.g. 'why was this loan denied?' or 'why did the model flag this "
            "patient?'"
        ),
        look_for=(
            "The longest bars are the dominant drivers for this particular "
            "observation. Compare waterfall plots for correct vs. wrong "
            "predictions to understand where the model gets confused."
        ),
        tip=(
            "Use np.argmax/argmin on predictions or errors to find the best "
            "and worst predictions to explain first."
        ),
    ),

    "plot_force": HelpEntry(
        title="Force Plot",
        category="plot",
        what=(
            "A horizontal version of the waterfall concept. Features pushing "
            "the prediction above baseline are shown as red arrows from the "
            "left; features pushing it below are blue arrows from the right. "
            "The final prediction sits where the arrows balance."
        ),
        when="Use as a visually compact alternative to plot_waterfall for presentations.",
        look_for=(
            "The dominant colour tells you whether this observation was pushed "
            "up (red dominated) or down (blue dominated) from the average. "
            "The width of each segment is proportional to its SHAP value."
        ),
    ),

    "plot_dependence": HelpEntry(
        title="Dependence Plot",
        category="plot",
        what=(
            "A scatter plot with a feature's raw value on the x-axis and its "
            "SHAP value on the y-axis. Each dot is one observation. Points are "
            "coloured by a second feature (auto-selected by SHAP) to reveal "
            "interactions."
        ),
        when=(
            "Use when you want to understand how a single feature's influence "
            "changes across its value range — revealing non-linearities, "
            "threshold effects, or interactions."
        ),
        look_for=(
            "A flat line means the feature has a constant effect — typical for "
            "linear models. A curved or stepped line reveals non-linearity. "
            "If the colour gradient runs diagonally, the two features interact "
            "strongly."
        ),
        tip=(
            "Pass interaction_feature='feature_name' to explicitly colour by "
            "a feature you suspect interacts, rather than SHAP's auto-pick."
        ),
    ),

    "plot_heatmap": HelpEntry(
        title="SHAP Heatmap",
        category="plot",
        what=(
            "A colour-coded grid: rows are features (top features by importance), "
            "columns are observations. Cell colour encodes the SHAP value — "
            "red = positive contribution, blue = negative. Observations are "
            "clustered by similarity of their SHAP profiles."
        ),
        when=(
            "Use when you want to spot segments in your data — groups of "
            "observations that share the same feature-driving pattern."
        ),
        look_for=(
            "Vertical bands of similar colour reveal clusters of observations "
            "driven by the same features. Horizontal bands show a feature that "
            "consistently matters (or doesn't) across the dataset."
        ),
    ),

    "plot_decision": HelpEntry(
        title="Decision Plot",
        category="plot",
        what=(
            "Each observation is a line, starting at the model baseline and "
            "accumulating feature contributions from bottom to top until it "
            "reaches the final prediction. Lines that end further right indicate "
            "higher predictions."
        ),
        when=(
            "Use for comparing a set of observations — e.g. 'how do high-risk "
            "vs. low-risk cases diverge as features are added?' Great for "
            "communicating model logic to stakeholders."
        ),
        look_for=(
            "Lines that stay together until a specific feature then fan out "
            "indicate that feature is driving the between-group difference. "
            "Lines that cross frequently suggest complex interactions."
        ),
    ),

    "plot_group_contributions": HelpEntry(
        title="Group Contributions Bar Chart",
        category="plot",
        what=(
            "A bar chart of mean absolute SHAP summed within each feature group. "
            "Shows how much each semantic cluster of features (e.g. 'location', "
            "'demographics') contributes to the model overall."
        ),
        when=(
            "Use after defining feature_groups to communicate attribution at a "
            "higher level of abstraction than individual features."
        ),
        look_for=(
            "Which group dominates? If one group accounts for most of the "
            "attribution, the model may be over-relying on that signal."
        ),
        requires="feature_groups= set at Explainer construction",
    ),

    "plot_confidence_intervals": HelpEntry(
        title="Confidence Interval Plot",
        category="plot",
        what=(
            "A horizontal bar chart with error bars showing the bootstrap "
            "confidence interval around each feature's mean absolute SHAP "
            "value. The bar is the point estimate; the whiskers are the "
            "lower/upper CI bounds."
        ),
        when=(
            "Use when you need to communicate reliability of SHAP estimates — "
            "especially with small datasets or noisy models."
        ),
        look_for=(
            "Overlapping CIs between two features mean their relative importance "
            "ranking is not statistically distinguishable. Wide CIs are a signal "
            "to collect more data."
        ),
        requires="compute_confidence_intervals=True passed to explain()",
    ),

    "plot_interaction_heatmap": HelpEntry(
        title="Interaction Heatmap",
        category="plot",
        what=(
            "A square heatmap showing the mean absolute SHAP interaction value "
            "between a chosen feature and every other feature in the model. "
            "Bright cells indicate strong pairwise interactions."
        ),
        when=(
            "Use after computing interaction effects to understand which pairs "
            "of features work together. Start with your top feature and see "
            "what it interacts with most."
        ),
        look_for=(
            "The diagonal (self-interaction) is usually the largest — that's just "
            "the main effect. Large off-diagonal cells indicate genuine pairwise "
            "interactions worth investigating."
        ),
        requires="compute_interactions=True passed to explain()",
    ),
}

# Convenience alias keys (so result.help('beeswarm') works too)
CATALOG["beeswarm"] = CATALOG["plot_beeswarm"]
CATALOG["summary"] = CATALOG["plot_summary"]
CATALOG["waterfall"] = CATALOG["plot_waterfall"]
CATALOG["force"] = CATALOG["plot_force"]
CATALOG["dependence"] = CATALOG["plot_dependence"]
CATALOG["heatmap"] = CATALOG["plot_heatmap"]
CATALOG["decision"] = CATALOG["plot_decision"]
CATALOG["feature_importance"] = CATALOG["plot_feature_importance"]
CATALOG["interactions"] = CATALOG["interaction_effects"]
CATALOG["ci"] = CATALOG["confidence_intervals"]
CATALOG["groups"] = CATALOG["group_contributions"]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter/IPython environment."""
    try:
        from IPython import get_ipython  # noqa: PLC0415
        return get_ipython() is not None
    except ImportError:
        return False


def _render_entry_text(key: str, entry: HelpEntry) -> str:
    """Render a single HelpEntry as plain text."""
    lines = [
        "",
        f"{'─' * 60}",
        f"  {entry.title}  [{entry.category}]",
        f"{'─' * 60}",
        "",
        "  WHAT IT IS",
        _wrap(entry.what, indent=4),
        "",
        "  WHEN TO USE IT",
        _wrap(entry.when, indent=4),
        "",
        "  WHAT TO LOOK FOR",
        _wrap(entry.look_for, indent=4),
    ]
    if entry.requires:
        lines += ["", f"  REQUIRES  →  {entry.requires}"]
    if entry.tip:
        lines += ["", "  TIP", _wrap(entry.tip, indent=4)]
    lines.append("")
    return "\n".join(lines)


def _render_entry_md(key: str, entry: HelpEntry) -> str:
    """Render a single HelpEntry as Markdown (for Jupyter display)."""
    parts = [
        f"### {entry.title} &nbsp; <small>`{key}`</small>",
        f"**What it is:** {entry.what}",
        f"**When to use it:** {entry.when}",
        f"**What to look for:** {entry.look_for}",
    ]
    if entry.requires:
        parts.append(f"**Requires:** `{entry.requires}`")
    if entry.tip:
        parts.append(f"> **Tip:** {entry.tip}")
    return "\n\n".join(parts)


def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    """Simple word-wrap for plain-text output."""
    import textwrap  # noqa: PLC0415
    return textwrap.fill(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


def _overview_text(result: object | None = None) -> str:
    """Build the full overview string, optionally contextualised to a result."""
    attrs = [k for k, v in CATALOG.items() if v.category == "attribute" and k in {
        "shap_values", "aggregate", "expected_value", "predictions",
        "prediction_proba", "interaction_effects", "confidence_intervals",
        "group_contributions",
    }]
    plots = [k for k, v in CATALOG.items() if v.category == "plot" and not k.startswith("plot_") is False
             or (v.category == "plot" and k.startswith("plot_"))]
    plot_keys = [k for k, v in CATALOG.items()
                 if v.category == "plot" and k.startswith("plot_")]

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        "║           transpyrancy  —  Help & Reference              ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        "  Call result.help('topic') for a detailed explanation of",
        "  any attribute or plot listed below.",
        "",
        "  ── DATA ATTRIBUTES ─────────────────────────────────────",
    ]

    attr_rows = [
        ("shap_values",          "Per-obs SHAP values  (n_obs × n_features)"),
        ("aggregate",            "Global feature importance, sorted desc"),
        ("expected_value",       "Model baseline  E[f(X)]"),
        ("predictions",          "Raw model outputs"),
        ("prediction_proba",     "Class probabilities  (classification only)"),
        ("interaction_effects",  "Pairwise interactions  (TreeExplainer only)"),
        ("confidence_intervals", "Bootstrap CI on SHAP values"),
        ("group_contributions",  "Summed SHAP per feature group"),
    ]
    for name, desc in attr_rows:
        available = _attr_available(result, name) if result is not None else None
        marker = _marker(available)
        lines.append(f"  {marker}  result.{name:<26s} {desc}")

    lines += [
        "",
        "  ── PLOT METHODS ────────────────────────────────────────",
    ]

    plot_rows = [
        ("plot_summary()",              "Beeswarm / dot overview — start here"),
        ("plot_feature_importance()",   "Bar chart of mean |SHAP|"),
        ("plot_waterfall(index=0)",     "Single-observation step-by-step breakdown"),
        ("plot_force(index=0)",         "Horizontal force diagram"),
        ("plot_dependence('feature')",  "Feature value vs SHAP value scatter"),
        ("plot_heatmap()",              "Feature × observation colour grid"),
        ("plot_decision()",             "Cumulative prediction paths"),
        ("plot_group_contributions()",  "Group-level attribution bar chart"),
        ("plot_confidence_intervals()", "Bootstrap error bars"),
        ("plot_interaction_heatmap('feature')", "Pairwise interaction strengths"),
    ]
    for call, desc in plot_rows:
        lines.append(f"       result.{call:<36s} {desc}")

    lines += [
        "",
        "  ── QUICK INTERPRETATION ────────────────────────────────",
        "  prediction ≈ expected_value + sum(shap_values for that row)",
        "  SHAP > 0  →  feature pushed prediction ABOVE baseline",
        "  SHAP < 0  →  feature pushed prediction BELOW baseline",
        "  |SHAP|    →  magnitude of influence (direction-agnostic)",
        "",
    ]

    if result is not None:
        lines += _context_notes(result)

    return "\n".join(lines)


def _attr_available(result: object, name: str) -> bool | None:
    """Return True/False/None for whether an optional attribute has data."""
    optional = {"interaction_effects", "confidence_intervals", "group_contributions",
                "prediction_proba"}
    if name not in optional:
        return True
    val = getattr(result, name, None)
    return val is not None


def _marker(available: bool | None) -> str:
    if available is True:
        return "✓"
    if available is False:
        return "○"
    return " "


def _context_notes(result: object) -> list[str]:
    """Return context-specific notes based on the result's state."""
    notes = ["  ── YOUR RESULT ─────────────────────────────────────────"]
    task = getattr(result, "task", "")
    explainer_type = getattr(result, "explainer_type", "")
    is_multiclass = getattr(result, "is_multiclass", False)
    n_classes = getattr(result, "n_classes", None)
    has_ci = getattr(result, "confidence_intervals", None) is not None
    has_interactions = getattr(result, "interaction_effects", None) is not None
    has_groups = getattr(result, "group_contributions", None) is not None

    notes.append(f"  Task        : {task}")
    notes.append(f"  Backend     : {explainer_type}Explainer")

    if is_multiclass:
        notes.append(
            f"  Multiclass  : {n_classes} classes — shap_values is a list of "
            f"{n_classes} DataFrames. Use result.shap_values_for_class(i)."
        )

    if not has_ci:
        notes.append(
            "  ○ Confidence intervals not computed. "
            "Re-run with compute_confidence_intervals=True to enable."
        )
    if not has_interactions and explainer_type == "tree":
        notes.append(
            "  ○ Interaction effects not computed. "
            "Re-run with compute_interactions=True to enable."
        )
    if explainer_type not in ("tree",) and not has_interactions:
        notes.append(
            f"  ○ Interaction effects not supported for {explainer_type}Explainer "
            f"(TreeExplainer only)."
        )
    if not has_groups:
        notes.append(
            "  ○ Group contributions not available. "
            "Pass feature_groups= to Explainer() to enable."
        )
    notes.append("")
    return notes


def _overview_md(result: object | None = None) -> str:
    """Build the overview as Markdown (for Jupyter)."""
    parts = [
        "## transpyrancy — Help & Reference",
        "Call `result.help('topic')` for a detailed explanation of any item below.",
        "",
        "### Data Attributes",
        "| Attribute | Description | Available |",
        "|---|---|:---:|",
    ]

    attr_rows = [
        ("shap_values",          "Per-obs SHAP values (n_obs × n_features)"),
        ("aggregate",            "Global feature importance, sorted desc"),
        ("expected_value",       "Model baseline E[f(X)]"),
        ("predictions",          "Raw model outputs"),
        ("prediction_proba",     "Class probabilities (classification only)"),
        ("interaction_effects",  "Pairwise SHAP interactions (TreeExplainer only)"),
        ("confidence_intervals", "Bootstrap CI on SHAP values"),
        ("group_contributions",  "Summed SHAP per feature group"),
    ]
    for name, desc in attr_rows:
        available = _attr_available(result, name) if result is not None else None
        icon = "✓" if available else ("○" if available is False else "")
        parts.append(f"| `result.{name}` | {desc} | {icon} |")

    parts += [
        "",
        "### Plot Methods",
        "| Method | Description |",
        "|---|---|",
        "| `result.plot_summary()` | Beeswarm / dot overview — **start here** |",
        "| `result.plot_feature_importance()` | Bar chart of mean \\|SHAP\\| |",
        "| `result.plot_waterfall(index=0)` | Single-observation step-by-step breakdown |",
        "| `result.plot_force(index=0)` | Horizontal force diagram |",
        "| `result.plot_dependence('feature')` | Feature value vs SHAP scatter |",
        "| `result.plot_heatmap()` | Feature × observation colour grid |",
        "| `result.plot_decision()` | Cumulative prediction paths |",
        "| `result.plot_group_contributions()` | Group-level attribution bar chart |",
        "| `result.plot_confidence_intervals()` | Bootstrap error bars |",
        "| `result.plot_interaction_heatmap('feature')` | Pairwise interaction strengths |",
        "",
        "### Quick Interpretation",
        "```",
        "prediction ≈ expected_value + sum(shap_values for that row)",
        "SHAP > 0  →  feature pushed prediction ABOVE baseline",
        "SHAP < 0  →  feature pushed prediction BELOW baseline",
        "|SHAP|    →  magnitude of influence (direction-agnostic)",
        "```",
    ]

    if result is not None:
        context = _context_notes_md(result)
        if context:
            parts += ["", "### Your Result", context]

    return "\n".join(parts)


def _context_notes_md(result: object) -> str:
    """Markdown context notes for the current result."""
    lines = []
    task = getattr(result, "task", "")
    explainer_type = getattr(result, "explainer_type", "")
    is_multiclass = getattr(result, "is_multiclass", False)
    n_classes = getattr(result, "n_classes", None)
    has_ci = getattr(result, "confidence_intervals", None) is not None
    has_interactions = getattr(result, "interaction_effects", None) is not None
    has_groups = getattr(result, "group_contributions", None) is not None

    lines.append(f"**Task:** {task} &nbsp; **Backend:** {explainer_type}Explainer")

    if is_multiclass:
        lines.append(
            f"\n> **Multiclass ({n_classes} classes):** `shap_values` is a list of "
            f"{n_classes} DataFrames. Use `result.shap_values_for_class(i)` to access "
            "a specific class."
        )
    if not has_ci:
        lines.append(
            "\n> ○ **Confidence intervals** not computed. "
            "Re-run with `compute_confidence_intervals=True` to enable."
        )
    if not has_interactions and explainer_type == "tree":
        lines.append(
            "\n> ○ **Interaction effects** not computed. "
            "Re-run with `compute_interactions=True` to enable."
        )
    if not has_groups:
        lines.append(
            "\n> ○ **Group contributions** not available. "
            "Pass `feature_groups=` to `Explainer()` to enable."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def show_help(topic: str | None = None, result: object | None = None) -> None:
    """
    Display help for a specific topic or a full overview.

    Parameters
    ----------
    topic : str | None
        Name of an attribute or plot (e.g. ``'shap_values'``, ``'plot_waterfall'``).
        When ``None``, shows the full overview.
    result : ExplainResult | None
        When provided, contextualises the overview to the current result state
        (highlights which optional outputs are available).
    """
    jupyter = _in_jupyter()

    if topic is not None:
        key = topic.lower().strip()
        if key not in CATALOG:
            close = [k for k in CATALOG if key in k or k in key]
            msg = f"No help entry for '{topic}'."
            if close:
                msg += f" Did you mean: {', '.join(close[:4])}?"
            if jupyter:
                _display_md(f"**{msg}**")
            else:
                print(msg)
            return

        entry = CATALOG[key]
        if jupyter:
            _display_md(_render_entry_md(key, entry))
        else:
            print(_render_entry_text(key, entry))
        return

    # Full overview
    if jupyter:
        _display_md(_overview_md(result))
    else:
        print(_overview_text(result))


def _display_md(text: str) -> None:
    """Render Markdown in a Jupyter cell output."""
    from IPython.display import Markdown, display  # noqa: PLC0415
    display(Markdown(text))


EXPLAINER_OVERVIEW = """\

╔══════════════════════════════════════════════════════════╗
║           transpyrancy  —  Library Overview              ║
╚══════════════════════════════════════════════════════════╝

  transpyrancy wraps SHAP with a single, opinionated API.

  ── WORKFLOW ────────────────────────────────────────────

  1. Create an Explainer
     exp = Explainer('xgb', model, background_data=X_train)

  2. Call explain()
     result = exp.explain(X_test)

  3. Inspect the result
     print(result)           # summary of all outputs
     result.help()           # this help menu
     result.aggregate        # global feature importance
     result.shap_values      # per-row per-feature SHAP

  4. Visualise
     result.plot_summary()           # start here
     result.plot_waterfall(index=0)  # explain one row
     result.plot_dependence('age')   # feature relationships

  ── SUPPORTED MODEL TYPES ───────────────────────────────

  Tree-based  (TreeExplainer)  : 'xgb', 'lightgbm', 'rf', 'catboost', ...
  Linear      (LinearExplainer): 'ridge', 'lasso', 'logistic', ...
  Deep        (DeepExplainer)  : 'torch', 'keras', 'tensorflow', ...
  Kernel      (KernelExplainer): 'svm', 'knn', or any unknown string

  ── OPTIONAL EXTRAS ─────────────────────────────────────

  result = exp.explain(
      X_test,
      compute_interactions=True,         # pairwise SHAP interactions
      compute_confidence_intervals=True, # bootstrap uncertainty
      n_bootstrap=100,
      n_jobs=-1,                         # parallelise CI + chunking
      visualize=True,                    # show all plots inline
      save_plots='./plots/',             # save all plots as PNG
  )

  After calling explain(), use result.help() for a full
  guide to every attribute and plot.
"""

EXPLAINER_OVERVIEW_MD = """
## transpyrancy — Library Overview

`transpyrancy` wraps SHAP with a single, opinionated API.

### Workflow

```python
# 1. Create an Explainer
exp = Explainer('xgb', model, background_data=X_train)

# 2. Explain
result = exp.explain(X_test)

# 3. Inspect
print(result)          # summary
result.help()          # full help menu
result.aggregate       # global feature importance
result.shap_values     # per-row SHAP values

# 4. Visualise
result.plot_summary()           # start here
result.plot_waterfall(index=0)  # explain one row
result.plot_dependence('age')   # feature relationships
```

### Supported Model Types

| Backend | Strings |
|---|---|
| `TreeExplainer` | `'xgb'`, `'lightgbm'`, `'rf'`, `'catboost'`, `'gbm'`, ... |
| `LinearExplainer` | `'ridge'`, `'lasso'`, `'logistic'`, `'elasticnet'`, ... |
| `DeepExplainer` | `'torch'`, `'keras'`, `'tensorflow'`, `'nn'`, ... |
| `KernelExplainer` | `'svm'`, `'knn'`, or any unrecognised string |

### Optional Extras

```python
result = exp.explain(
    X_test,
    compute_interactions=True,          # pairwise SHAP interactions
    compute_confidence_intervals=True,  # bootstrap uncertainty bands
    n_bootstrap=100,
    n_jobs=-1,                          # parallelise CI + chunking
    visualize=True,                     # show all plots inline
    save_plots='./plots/',              # save all plots as PNG
)
```

After calling `explain()`, use `result.help()` for a full guide to every
attribute and plot method.
"""
