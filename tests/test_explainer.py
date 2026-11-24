"""
Comprehensive test suite for the explainer package.

Covers:
- Detector module (model type mapping, override validation)
- Utility functions (input normalisation, SHAP processing, aggregation)
- Explainer class (construction, explain(), all optional features)
- ExplainResult dataclass (attributes, convenience methods, repr)
- Visualisation methods (smoke tests; check Figure is returned)
- Edge cases (single row, missing background_data, bad inputs)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt

from transpyrancy import Explainer, ExplainResult, ExplainerType
from transpyrancy.detector import (
    MODEL_TYPE_MAP,
    SUPPORTED_MODEL_TYPES,
    detect_explainer_type,
    validate_override,
)
from transpyrancy.utils import (
    compute_aggregate,
    compute_group_contributions,
    detect_task,
    normalize_input,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_data():
    """Breast cancer binary classification dataset."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def regression_data():
    """Diabetes regression dataset."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def multiclass_data():
    """Iris multiclass dataset."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def rf_clf(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test


@pytest.fixture(scope="module")
def rf_reg(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test


@pytest.fixture(scope="module")
def linear_reg(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test


@pytest.fixture(scope="module")
def logistic_clf(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test


@pytest.fixture(scope="module")
def multiclass_rf(multiclass_data):
    X_train, X_test, y_train, y_test = multiclass_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test


# ---------------------------------------------------------------------------
# detector.py
# ---------------------------------------------------------------------------


class TestDetector:
    def test_known_tree_types(self):
        for key in ["xgb", "xgboost", "lgb", "lightgbm", "catboost", "rf",
                    "random_forest", "gbm", "gradient_boosting", "et", "extra_trees",
                    "dt", "decision_tree"]:
            assert detect_explainer_type(key) == ExplainerType.TREE

    def test_known_linear_types(self):
        for key in ["linear", "logistic", "logistic_regression", "ridge",
                    "lasso", "elasticnet", "sgd"]:
            assert detect_explainer_type(key) == ExplainerType.LINEAR

    def test_known_kernel_types(self):
        assert detect_explainer_type("kernel") == ExplainerType.KERNEL

    def test_case_insensitive(self):
        assert detect_explainer_type("XGB") == ExplainerType.TREE
        assert detect_explainer_type("LightGBM") == ExplainerType.TREE
        assert detect_explainer_type("RIDGE") == ExplainerType.LINEAR

    def test_whitespace_stripped(self):
        assert detect_explainer_type("  rf  ") == ExplainerType.TREE

    def test_unknown_type_warns_and_falls_back(self):
        with pytest.warns(UserWarning, match="Unknown model_type"):
            result = detect_explainer_type("magical_model")
        assert result == ExplainerType.KERNEL

    def test_validate_override_no_conflict_no_warning(self):
        # Should not raise
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_override(ExplainerType.TREE, ExplainerType.TREE, "xgb")

    def test_validate_override_conflict_warns(self):
        with pytest.warns(UserWarning, match="conflicts with the auto-detected"):
            validate_override(ExplainerType.TREE, ExplainerType.LINEAR, "xgb")

    def test_supported_model_types_is_sorted(self):
        assert SUPPORTED_MODEL_TYPES == sorted(SUPPORTED_MODEL_TYPES)

    def test_explainer_type_enum_values(self):
        assert ExplainerType.TREE.value == "tree"
        assert ExplainerType.LINEAR.value == "linear"
        assert ExplainerType.KERNEL.value == "kernel"


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestNormalizeInput:
    def test_dataframe_passthrough(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result_df, names = normalize_input(df, None)
        assert list(names) == ["a", "b"]
        assert result_df.shape == (2, 2)

    def test_dataframe_with_explicit_names(self):
        # Explicit feature_names override display names but do NOT rename DataFrame columns
        df = pd.DataFrame({"a": [1], "b": [2]})
        result_df, names = normalize_input(df, ["x", "y"])
        assert list(names) == ["x", "y"]
        # Original columns are preserved (no renaming to avoid sklearn compatibility issues)
        assert list(result_df.columns) == ["a", "b"]

    def test_numpy_1d_becomes_row(self):
        arr = np.array([1.0, 2.0, 3.0])
        result_df, names = normalize_input(arr, None)
        assert result_df.shape == (1, 3)
        assert names == ["f0", "f1", "f2"]

    def test_numpy_2d(self):
        arr = np.random.rand(5, 4)
        result_df, names = normalize_input(arr, None)
        assert result_df.shape == (5, 4)
        assert names == ["f0", "f1", "f2", "f3"]

    def test_numpy_with_feature_names(self):
        arr = np.random.rand(3, 2)
        result_arr, names = normalize_input(arr, ["alpha", "beta"])
        assert list(names) == ["alpha", "beta"]
        # numpy arrays are returned as-is (no DataFrame conversion)
        assert isinstance(result_arr, np.ndarray)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="pd.DataFrame or np.ndarray"):
            normalize_input([[1, 2], [3, 4]], None)  # type: ignore[arg-type]


class TestDetectTask:
    def test_classifier_detected(self):
        model = RandomForestClassifier()
        assert detect_task(model) == "classification"

    def test_regressor_detected(self):
        model = RandomForestRegressor()
        assert detect_task(model) == "regression"

    def test_plain_object_no_predict_proba(self):
        model = MagicMock(spec=[])
        assert detect_task(model) == "regression"


class TestComputeAggregate:
    def test_binary_returns_two_columns(self):
        shap_df = pd.DataFrame({"a": [1.0, -2.0], "b": [0.5, 0.5]})
        agg = compute_aggregate(shap_df, is_multiclass=False)
        assert "mean_abs_shap" in agg.columns
        assert "mean_shap" in agg.columns
        # Sorted descending by mean_abs_shap
        assert agg["mean_abs_shap"].iloc[0] >= agg["mean_abs_shap"].iloc[1]

    def test_multiclass_returns_class_columns(self):
        shap_dfs = [
            pd.DataFrame({"a": [1.0], "b": [0.5]}),
            pd.DataFrame({"a": [0.2], "b": [0.8]}),
            pd.DataFrame({"a": [0.3], "b": [0.1]}),
        ]
        agg = compute_aggregate(shap_dfs, is_multiclass=True)
        assert "class_0" in agg.columns
        assert "class_1" in agg.columns
        assert "class_2" in agg.columns
        assert "mean_abs_shap_total" in agg.columns


class TestComputeGroupContributions:
    def test_basic_grouping(self):
        shap_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        groups = {"g1": ["a", "b"], "g2": ["c"]}
        result = compute_group_contributions(shap_df, groups)
        assert list(result.columns) == ["g1", "g2"]
        assert result["g1"].iloc[0] == pytest.approx(4.0)
        assert result["g2"].iloc[1] == pytest.approx(6.0)

    def test_warns_on_missing_feature(self):
        shap_df = pd.DataFrame({"a": [1.0]})
        with pytest.warns(UserWarning, match="not found in shap_values columns"):
            compute_group_contributions(shap_df, {"g": ["a", "missing_feature"]})


# ---------------------------------------------------------------------------
# Explainer construction
# ---------------------------------------------------------------------------


class TestExplainerConstruction:
    def test_repr(self, rf_clf):
        model, X_train, _ = rf_clf
        exp = Explainer("rf", model)
        assert "rf" in repr(exp)
        assert "tree" in repr(exp)

    def test_task_auto_classification(self, rf_clf):
        model, X_train, _ = rf_clf
        exp = Explainer("rf", model)
        assert exp.task == "classification"

    def test_task_auto_regression(self, rf_reg):
        model, X_train, _ = rf_reg
        exp = Explainer("rf", model)
        assert exp.task == "regression"

    def test_task_explicit(self, rf_clf):
        model, X_train, _ = rf_clf
        exp = Explainer("rf", model, task="regression")
        assert exp.task == "regression"

    def test_invalid_task_raises(self, rf_clf):
        model, X_train, _ = rf_clf
        with pytest.raises(ValueError, match="task must be"):
            Explainer("rf", model, task="invalid")  # type: ignore[arg-type]

    def test_invalid_override_raises(self, rf_clf):
        model, X_train, _ = rf_clf
        with pytest.raises(ValueError, match="explainer_override must be one of"):
            Explainer("rf", model, explainer_override="banana")  # type: ignore[arg-type]

    def test_linear_without_background_raises(self, logistic_clf):
        model, X_train, _ = logistic_clf
        with pytest.raises(ValueError, match="background_data is required"):
            Explainer("logistic", model)

    def test_kernel_without_background_raises(self, rf_clf):
        model, X_train, _ = rf_clf
        with pytest.raises(ValueError, match="background_data is required"):
            Explainer("kernel", model)

    def test_override_conflict_warns(self, rf_clf):
        model, X_train, _ = rf_clf
        # 'xgb' would normally use tree; overriding to 'kernel' triggers the warning
        # but kernel explainer can still work with any model, so no build error
        with pytest.warns(UserWarning, match="conflicts with the auto-detected"):
            Explainer("xgb", model, background_data=X_train, explainer_override="kernel")

    def test_unknown_model_type_warns(self, rf_clf):
        model, X_train, _ = rf_clf
        with pytest.warns(UserWarning, match="Unknown model_type"):
            exp = Explainer("my_custom_model", model, background_data=X_train)
        assert exp._explainer_type == ExplainerType.KERNEL

    def test_feature_names_stored(self, rf_clf):
        model, X_train, _ = rf_clf
        exp = Explainer("rf", model, feature_names=["feat_a", "feat_b"])
        assert exp.feature_names == ["feat_a", "feat_b"]

    def test_feature_groups_stored(self, rf_clf):
        model, X_train, _ = rf_clf
        groups = {"g1": ["a"], "g2": ["b"]}
        exp = Explainer("rf", model, feature_groups=groups)
        assert exp.feature_groups == groups


# ---------------------------------------------------------------------------
# Explainer.explain() — core behaviour
# ---------------------------------------------------------------------------


class TestExplainCore:
    def test_returns_explain_result(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert isinstance(result, ExplainResult)

    def test_shap_values_shape_classification(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert isinstance(result.shap_values, pd.DataFrame)
        assert result.shap_values.shape == (len(X_test), X_test.shape[1])

    def test_shap_values_shape_regression(self, rf_reg):
        model, X_train, X_test = rf_reg
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.shap_values.shape == (len(X_test), X_test.shape[1])

    def test_aggregate_sorted_descending(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        vals = result.aggregate["mean_abs_shap"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_predictions_correct_length(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert len(result.predictions) == len(X_test)

    def test_prediction_proba_present_for_classifier(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.prediction_proba is not None
        assert result.prediction_proba.shape == (len(X_test), 2)

    def test_prediction_proba_none_for_regressor(self, rf_reg):
        model, X_train, X_test = rf_reg
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.prediction_proba is None

    def test_task_field(self, rf_clf, rf_reg):
        model_c, X_train_c, X_test_c = rf_clf
        model_r, X_train_r, X_test_r = rf_reg
        result_c = Explainer("rf", model_c).explain(X_test_c)
        result_r = Explainer("rf", model_r).explain(X_test_r)
        assert result_c.task == "classification"
        assert result_r.task == "regression"

    def test_n_observations(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test.iloc[:10])
        assert result.n_observations == 10

    def test_single_row_numpy(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        row = X_test.values[0]  # 1-D array
        result = exp.explain(row)
        assert result.n_observations == 1
        assert result.shap_values.shape == (1, X_test.shape[1])

    def test_feature_names_from_dataframe(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.feature_names == list(X_test.columns)

    def test_feature_names_from_numpy(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test.values)
        assert result.feature_names == [f"f{i}" for i in range(X_test.shape[1])]

    def test_explicit_feature_names_used(self, rf_clf):
        model, X_train, X_test = rf_clf
        names = [f"feat_{i}" for i in range(X_test.shape[1])]
        exp = Explainer("rf", model, feature_names=names)
        result = exp.explain(X_test.values)
        assert result.feature_names == names

    def test_explainer_type_in_result(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.explainer_type == "tree"

    def test_model_type_in_result(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.model_type == "rf"

    def test_expected_value_is_float(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        # For binary / regression, expected_value should be a scalar
        assert isinstance(result.expected_value, (int, float))


# ---------------------------------------------------------------------------
# LinearExplainer path
# ---------------------------------------------------------------------------


class TestLinearExplainer:
    def test_linear_regression(self, linear_reg):
        model, X_train, X_test = linear_reg
        exp = Explainer("linear", model, background_data=X_train, task="regression")
        result = exp.explain(X_test)
        assert result.shap_values.shape == (len(X_test), X_test.shape[1])
        assert result.task == "regression"
        assert result.explainer_type == "linear"

    def test_logistic_regression(self, logistic_clf):
        model, X_train, X_test = logistic_clf
        exp = Explainer("logistic", model, background_data=X_train)
        result = exp.explain(X_test)
        assert result.shap_values.shape[1] == X_test.shape[1]
        assert result.task == "classification"

    def test_ridge(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        model = Ridge().fit(X_train, y_train)
        exp = Explainer("ridge", model, background_data=X_train, task="regression")
        result = exp.explain(X_test)
        assert result.explainer_type == "linear"


# ---------------------------------------------------------------------------
# Optional features
# ---------------------------------------------------------------------------


class TestInteractionEffects:
    def test_interactions_available_for_tree(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test.iloc[:10], compute_interactions=True)
        assert result.interaction_effects is not None
        assert result.interaction_effects.ndim == 3
        n, f, f2 = result.interaction_effects.shape
        assert n == 10
        assert f == X_test.shape[1]
        assert f2 == X_test.shape[1]

    def test_interactions_warns_for_non_tree(self, logistic_clf):
        model, X_train, X_test = logistic_clf
        exp = Explainer("logistic", model, background_data=X_train)
        with pytest.warns(UserWarning, match="only supported for TreeExplainer"):
            result = exp.explain(X_test.iloc[:5], compute_interactions=True)
        assert result.interaction_effects is None

    def test_no_interactions_by_default(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.interaction_effects is None


class TestConfidenceIntervals:
    def test_ci_computed_when_requested(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(
            X_test.iloc[:20],
            compute_confidence_intervals=True,
            n_bootstrap=10,
        )
        assert result.confidence_intervals is not None
        assert "lower" in result.confidence_intervals
        assert "upper" in result.confidence_intervals
        assert result.confidence_intervals["lower"].shape == (20, X_test.shape[1])

    def test_lower_leq_upper(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(
            X_test.iloc[:20],
            compute_confidence_intervals=True,
            n_bootstrap=10,
        )
        lo = result.confidence_intervals["lower"]
        hi = result.confidence_intervals["upper"]
        assert (lo.values <= hi.values + 1e-9).all()

    def test_no_ci_by_default(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.confidence_intervals is None


class TestFeatureGroups:
    def test_group_contributions_computed(self, rf_clf):
        model, X_train, X_test = rf_clf
        feature_names = list(X_test.columns)
        groups = {
            "group_a": feature_names[:5],
            "group_b": feature_names[5:10],
        }
        exp = Explainer("rf", model, feature_groups=groups)
        result = exp.explain(X_test)
        assert result.group_contributions is not None
        assert set(result.group_contributions.columns) == {"group_a", "group_b"}
        assert result.group_contributions.shape[0] == len(X_test)

    def test_no_groups_by_default(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.group_contributions is None

    def test_missing_feature_in_group_warns(self, rf_clf):
        model, X_train, X_test = rf_clf
        feature_names = list(X_test.columns)
        groups = {"g": [feature_names[0], "nonexistent_feature"]}
        exp = Explainer("rf", model, feature_groups=groups)
        with pytest.warns(UserWarning, match="not found in shap_values columns"):
            result = exp.explain(X_test)
        assert result.group_contributions is not None


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------


class TestMulticlass:
    def test_is_multiclass_true(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.is_multiclass is True

    def test_shap_values_is_list_for_multiclass(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert isinstance(result.shap_values, list)
        assert len(result.shap_values) == 3  # iris has 3 classes

    def test_shap_values_for_class(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        for i in range(3):
            sv = result.shap_values_for_class(i)
            assert isinstance(sv, pd.DataFrame)
            assert sv.shape == (len(X_test), X_test.shape[1])

    def test_shap_values_for_class_out_of_range(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        with pytest.raises(ValueError, match="out of range"):
            result.shap_values_for_class(99)

    def test_aggregate_has_class_columns(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert "class_0" in result.aggregate.columns
        assert "mean_abs_shap_total" in result.aggregate.columns

    def test_n_classes_set(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        assert result.n_classes == 3

    def test_shap_values_for_class_raises_for_binary(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test)
        with pytest.raises(ValueError, match="only available for multiclass"):
            result.shap_values_for_class(0)


# ---------------------------------------------------------------------------
# ExplainResult convenience methods
# ---------------------------------------------------------------------------


class TestExplainResultMethods:
    @pytest.fixture
    def result(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        return exp.explain(X_test)

    def test_feature_importance_sorted(self, result):
        fi = result.feature_importance
        assert isinstance(fi, pd.Series)
        vals = fi.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_top_features_length(self, result):
        top = result.top_features(5)
        assert len(top) == 5

    def test_summary_is_string(self, result):
        s = result.summary()
        assert isinstance(s, str)
        assert "ExplainResult Summary" in s

    def test_repr_contains_summary(self, result):
        assert "ExplainResult Summary" in repr(result)

    def test_group_contributions_raises_when_none(self, result):
        with pytest.raises(ValueError, match="No group contributions"):
            result.plot_group_contributions(show=False)

    def test_confidence_intervals_raises_when_none(self, result):
        with pytest.raises(ValueError, match="No confidence intervals"):
            result.plot_confidence_intervals(show=False)

    def test_interaction_heatmap_raises_when_none(self, result):
        with pytest.raises(ValueError, match="No interaction effects"):
            result.plot_interaction_heatmap(result.feature_names[0], show=False)

    def test_interaction_heatmap_raises_bad_feature(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test.iloc[:10], compute_interactions=True)
        with pytest.raises(ValueError, match="not found"):
            result.plot_interaction_heatmap("nonexistent_feature", show=False)


# ---------------------------------------------------------------------------
# Visualisation smoke tests (just check Figure is returned)
# ---------------------------------------------------------------------------


class TestVisualisationSmoke:
    """
    These tests verify that each plot method runs without error and returns
    a matplotlib Figure.  They use the Agg backend so no display is required.
    """

    @pytest.fixture
    def result(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        return exp.explain(X_test)

    @pytest.fixture
    def result_with_all(self, rf_clf):
        model, X_train, X_test = rf_clf
        feature_names = list(X_test.columns)
        groups = {"g1": feature_names[:5], "g2": feature_names[5:]}
        exp = Explainer("rf", model, feature_groups=groups)
        return exp.explain(
            X_test.iloc[:30],
            compute_interactions=True,
            compute_confidence_intervals=True,
            n_bootstrap=5,
        )

    def test_plot_summary(self, result):
        fig = result.plot_summary(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_beeswarm(self, result):
        fig = result.plot_beeswarm(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_feature_importance(self, result):
        fig = result.plot_feature_importance(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_waterfall(self, result):
        fig = result.plot_waterfall(index=0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_force(self, result):
        fig = result.plot_force(index=0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_dependence(self, result):
        feature = result.feature_names[0]
        fig = result.plot_dependence(feature, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_heatmap(self, result):
        fig = result.plot_heatmap(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_decision(self, result):
        fig = result.plot_decision(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_group_contributions(self, result_with_all):
        fig = result_with_all.plot_group_contributions(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_confidence_intervals(self, result_with_all):
        fig = result_with_all.plot_confidence_intervals(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_confidence_intervals_single_obs(self, result_with_all):
        fig = result_with_all.plot_confidence_intervals(observation_index=0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_interaction_heatmap(self, result_with_all):
        feature = result_with_all.feature_names[0]
        fig = result_with_all.plot_interaction_heatmap(feature, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_summary_bar(self, result):
        fig = result.plot_summary(plot_type="bar", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# save_plots integration
# ---------------------------------------------------------------------------


class TestSavePlots:
    def test_plots_saved_to_directory(self, rf_clf, tmp_path):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        exp.explain(X_test.iloc[:20], save_plots=str(tmp_path), visualize=False)
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 4, f"Expected ≥4 PNGs, got {len(pngs)}"

    def test_directory_created_automatically(self, rf_clf, tmp_path):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        out_dir = tmp_path / "auto_created"
        exp.explain(X_test.iloc[:20], save_plots=str(out_dir), visualize=False)
        assert out_dir.exists()

    def test_save_path_on_individual_plot(self, rf_clf, tmp_path):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(X_test.iloc[:20])
        out = tmp_path / "summary.png"
        result.plot_summary(save_path=str(out), show=False)
        assert out.exists()
        plt.close("all")


# ---------------------------------------------------------------------------
# GradientBoosting sklearn (another tree model)
# ---------------------------------------------------------------------------


class TestGradientBoosting:
    def test_gbc(self, binary_data):
        X_train, X_test, y_train, _ = binary_data
        model = GradientBoostingClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        exp = Explainer("gbm", model)
        result = exp.explain(X_test.iloc[:20])
        assert result.shap_values.shape == (20, X_test.shape[1])

    def test_gbr(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        model = GradientBoostingRegressor(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        exp = Explainer("gbr", model)
        result = exp.explain(X_test.iloc[:20])
        assert result.task == "regression"


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


class TestDecisionTree:
    def test_dt_explainer(self, binary_data):
        X_train, X_test, y_train, _ = binary_data
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        exp = Explainer("dt", model)
        result = exp.explain(X_test.iloc[:20])
        assert result.explainer_type == "tree"
        assert result.shap_values.shape[0] == 20


# ---------------------------------------------------------------------------
# Lasso (LinearExplainer)
# ---------------------------------------------------------------------------


class TestLasso:
    def test_lasso_regression(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        model = Lasso(alpha=0.1).fit(X_train, y_train)
        exp = Explainer("lasso", model, background_data=X_train, task="regression")
        result = exp.explain(X_test.iloc[:20])
        assert result.explainer_type == "linear"
        assert result.task == "regression"


# ---------------------------------------------------------------------------
# Parallelism (n_jobs)
# ---------------------------------------------------------------------------


class TestParallelism:
    """
    Verify that n_jobs produces results consistent with n_jobs=1 and that
    all output shapes / types are preserved under parallel execution.
    """

    def test_n_jobs_minus1_same_shape(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        r1 = exp.explain(X_test)
        r_par = exp.explain(X_test, n_jobs=-1)
        assert r_par.shap_values.shape == r1.shap_values.shape
        assert r_par.n_observations == r1.n_observations

    def test_n_jobs_2_same_shape(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        r1 = exp.explain(X_test)
        r2 = exp.explain(X_test, n_jobs=2)
        assert r2.shap_values.shape == r1.shap_values.shape

    def test_n_jobs_parallel_ci(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        result = exp.explain(
            X_test.iloc[:30],
            compute_confidence_intervals=True,
            n_bootstrap=20,
            n_jobs=-1,
        )
        assert result.confidence_intervals is not None
        assert result.confidence_intervals["lower"].shape == (30, X_test.shape[1])

    def test_n_jobs_parallel_ci_matches_single(self, rf_clf):
        """Parallel and serial CI outputs should have the same shape and valid bounds."""
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        r1 = exp.explain(
            X_test.iloc[:20],
            compute_confidence_intervals=True,
            n_bootstrap=10,
            n_jobs=1,
        )
        r_par = exp.explain(
            X_test.iloc[:20],
            compute_confidence_intervals=True,
            n_bootstrap=10,
            n_jobs=-1,
        )
        # Shape must match
        assert r1.confidence_intervals["lower"].shape == r_par.confidence_intervals["lower"].shape
        # lower ≤ upper in both
        lo = r_par.confidence_intervals["lower"]
        hi = r_par.confidence_intervals["upper"]
        assert (lo.values <= hi.values + 1e-9).all()

    def test_n_jobs_1_is_default(self, rf_clf):
        """Explicit n_jobs=1 should be identical to not passing it."""
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        r_default = exp.explain(X_test.iloc[:10])
        r_explicit = exp.explain(X_test.iloc[:10], n_jobs=1)
        # SHAP values should be identical (same computation path)
        pd.testing.assert_frame_equal(r_default.shap_values, r_explicit.shap_values)

    def test_n_jobs_chunked_large_dataset(self, rf_clf):
        """Dataset large enough to trigger chunking produces correct shape."""
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model)
        # Use a dataset with >= MIN_CHUNK_ROWS * n_jobs rows to trigger chunking
        # MIN_CHUNK_ROWS = 50, so with n_jobs=2 we need >= 100 rows
        X_large = pd.concat([X_test] * 3, ignore_index=True)  # ~342 rows
        result = exp.explain(X_large, n_jobs=2)
        assert result.shap_values.shape == (len(X_large), X_test.shape[1])
        assert result.n_observations == len(X_large)

    def test_n_jobs_multiclass_parallel(self, multiclass_rf):
        model, X_train, X_test = multiclass_rf
        exp = Explainer("rf", model)
        r1 = exp.explain(X_test)
        r_par = exp.explain(X_test, n_jobs=-1)
        assert r_par.is_multiclass == r1.is_multiclass
        assert len(r_par.shap_values) == len(r1.shap_values)


# ---------------------------------------------------------------------------
# Deep Explainer (mock-based — no torch/tensorflow installation required)
# ---------------------------------------------------------------------------


class TestDeepExplainer:
    """
    Tests for the DeepExplainer backend.

    All tests here mock ``shap.DeepExplainer`` so that neither PyTorch nor
    TensorFlow needs to be installed.  Real-framework integration tests live
    in ``TestDeepExplainerPyTorch`` (skipped when torch is absent).
    """

    N_SAMPLES = 25
    N_FEATURES = 6
    RNG = np.random.default_rng(0)

    @pytest.fixture(scope="class")
    def arrays(self):
        X = self.RNG.standard_normal((self.N_SAMPLES, self.N_FEATURES)).astype(np.float32)
        background = X[:8]
        test_data = X[8:]
        return background, test_data

    # ---- detector / enum ----

    def test_deep_in_explainer_type_enum(self):
        assert ExplainerType.DEEP.value == "deep"

    def test_deep_model_type_strings_mapped(self):
        for key in ("torch", "pytorch", "tensorflow", "tf", "keras", "neural_network", "nn", "deep"):
            assert MODEL_TYPE_MAP[key] == ExplainerType.DEEP, f"Expected DEEP for '{key}'"

    def test_detect_explainer_type_returns_deep(self):
        assert detect_explainer_type("torch") == ExplainerType.DEEP
        assert detect_explainer_type("keras") == ExplainerType.DEEP
        assert detect_explainer_type("tensorflow") == ExplainerType.DEEP

    # ---- construction guards ----

    def test_deep_requires_background_data(self):
        with patch("shap.DeepExplainer"):
            with pytest.raises(ValueError, match="background_data is required"):
                Explainer("keras", MagicMock())

    # ---- binary classification (list of 2 arrays) ----

    def test_deep_binary_classification(self, arrays):
        background, test_data = arrays
        n = len(test_data)
        fake_sv = [self.RNG.standard_normal((n, self.N_FEATURES)),
                   self.RNG.standard_normal((n, self.N_FEATURES))]
        proba = np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])

        with patch("shap.DeepExplainer") as mock_cls:
            mock_exp = MagicMock()
            mock_cls.return_value = mock_exp
            mock_exp.shap_values.return_value = fake_sv
            mock_exp.expected_value = [0.3, 0.7]

            exp = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="classification",
                predict_fn=lambda x: proba[:len(x)],
            )
            result = exp.explain(test_data)

        assert result.explainer_type == "deep"
        assert isinstance(result.shap_values, pd.DataFrame)
        assert result.shap_values.shape == (n, self.N_FEATURES)
        assert result.predictions.shape == (n,)
        assert result.task == "classification"

    # ---- regression (list of 1 array) ----

    def test_deep_regression_single_output(self, arrays):
        background, test_data = arrays
        n = len(test_data)
        fake_sv = [self.RNG.standard_normal((n, self.N_FEATURES))]
        preds = self.RNG.standard_normal(n)

        with patch("shap.DeepExplainer") as mock_cls:
            mock_exp = MagicMock()
            mock_cls.return_value = mock_exp
            mock_exp.shap_values.return_value = fake_sv
            mock_exp.expected_value = [2.5]

            exp = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="regression",
                predict_fn=lambda x: preds[:len(x)],
            )
            result = exp.explain(test_data)

        assert result.task == "regression"
        assert result.shap_values.shape == (n, self.N_FEATURES)
        assert abs(result.expected_value - 2.5) < 1e-6

    # ---- multiclass (list of 3 arrays) ----

    def test_deep_multiclass(self, arrays):
        background, test_data = arrays
        n = len(test_data)
        fake_sv = [self.RNG.standard_normal((n, self.N_FEATURES)) for _ in range(3)]
        proba = np.tile([0.2, 0.5, 0.3], (n, 1))

        with patch("shap.DeepExplainer") as mock_cls:
            mock_exp = MagicMock()
            mock_cls.return_value = mock_exp
            mock_exp.shap_values.return_value = fake_sv
            mock_exp.expected_value = [0.2, 0.5, 0.3]

            exp = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="classification",
                predict_fn=lambda x: proba[:len(x)],
            )
            result = exp.explain(test_data)

        assert result.is_multiclass
        assert isinstance(result.shap_values, list)
        assert len(result.shap_values) == 3
        assert result.n_classes == 3

    # ---- interaction guard ----

    def test_deep_interactions_warns_and_skips(self, arrays):
        background, test_data = arrays
        n = len(test_data)
        fake_sv = [self.RNG.standard_normal((n, self.N_FEATURES)),
                   self.RNG.standard_normal((n, self.N_FEATURES))]

        with patch("shap.DeepExplainer") as mock_cls:
            mock_exp = MagicMock()
            mock_cls.return_value = mock_exp
            mock_exp.shap_values.return_value = fake_sv
            mock_exp.expected_value = [0.0, 0.0]

            exp = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="classification",
                predict_fn=lambda x: np.tile([0.4, 0.6], (len(x), 1)),
            )
            with pytest.warns(UserWarning, match="[Ii]nteraction"):
                result = exp.explain(test_data, compute_interactions=True)

        assert result.interaction_effects is None

    # ---- expected_value extraction ----

    def test_deep_expected_value_scalar_list(self, arrays):
        background, test_data = arrays
        n = len(test_data)
        fake_sv = [self.RNG.standard_normal((n, self.N_FEATURES))]

        with patch("shap.DeepExplainer") as mock_cls:
            mock_exp = MagicMock()
            mock_cls.return_value = mock_exp
            mock_exp.shap_values.return_value = fake_sv
            mock_exp.expected_value = [1.23]

            exp = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="regression",
                predict_fn=lambda x: np.zeros(len(x)),
            )
            result = exp.explain(test_data)

        assert abs(result.expected_value - 1.23) < 1e-6

    # ---- pytorch model type helper ----

    def test_is_pytorch_model_flag(self, arrays):
        pytest.importorskip("torch")
        background, _ = arrays
        with patch("shap.DeepExplainer"):
            exp_torch = Explainer(
                "torch", MagicMock(),
                background_data=background,
                task="regression",
                predict_fn=lambda x: np.zeros(len(x)),
            )
            exp_keras = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="regression",
                predict_fn=lambda x: np.zeros(len(x)),
            )
        assert exp_torch._is_pytorch_model() is True
        assert exp_keras._is_pytorch_model() is False

    # ---- predict_fn auto-detection (Keras path) ----

    def test_deep_keras_auto_predict_fn(self, arrays):
        background, test_data = arrays
        n = len(test_data)
        fake_sv = [self.RNG.standard_normal((n, self.N_FEATURES))]
        # Keras model exposes .predict(); no predict_fn needed
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(n)

        with patch("shap.DeepExplainer") as mock_cls:
            mock_exp = MagicMock()
            mock_cls.return_value = mock_exp
            mock_exp.shap_values.return_value = fake_sv
            mock_exp.expected_value = [0.0]

            exp = Explainer(
                "keras", mock_model,
                background_data=background,
                task="regression",
                # No predict_fn — should fall back to model.predict()
            )
            result = exp.explain(test_data)

        assert result.shap_values.shape == (n, self.N_FEATURES)
        mock_model.predict.assert_called()

    # ---- repr ----

    def test_deep_repr(self, arrays):
        background, _ = arrays
        with patch("shap.DeepExplainer"):
            exp = Explainer(
                "keras", MagicMock(),
                background_data=background,
                task="regression",
                predict_fn=lambda x: np.zeros(len(x)),
            )
        assert "deep" in repr(exp)


# ---------------------------------------------------------------------------
# Optional: real PyTorch integration (skipped if torch is not installed)
# ---------------------------------------------------------------------------


class TestDeepExplainerPyTorch:
    """Integration tests with a real torch.nn.Module — skipped if torch absent."""

    @pytest.fixture(scope="class")
    def torch_setup(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn  # noqa: PLC0415

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        model = TinyNet()
        model.eval()

        def predict_fn(X_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                out = model(torch.tensor(X_np, dtype=torch.float32)).numpy()
            return np.column_stack([1 - out, out])

        return model, X, predict_fn

    def test_pytorch_explainer_builds(self, torch_setup):
        model, X, predict_fn = torch_setup
        exp = Explainer(
            "torch", model,
            background_data=X[:10],
            task="classification",
            predict_fn=predict_fn,
        )
        assert exp._explainer_type == ExplainerType.DEEP

    def test_pytorch_explain_returns_result(self, torch_setup):
        model, X, predict_fn = torch_setup
        exp = Explainer(
            "torch", model,
            background_data=X[:10],
            task="classification",
            predict_fn=predict_fn,
        )
        result = exp.explain(X[10:30])
        assert isinstance(result, ExplainResult)
        assert result.shap_values.shape == (20, 4)
        assert result.explainer_type == "deep"
        assert result.predictions.shape == (20,)

    def test_pytorch_auto_predict_fn(self, torch_setup):
        """When predict_fn is omitted for a torch model, auto-wrap should work."""
        torch = pytest.importorskip("torch")
        model, X, _ = torch_setup
        exp = Explainer(
            "torch", model,
            background_data=X[:10],
            task="regression",
            # No predict_fn — should auto-wrap model(x)
        )
        result = exp.explain(X[10:20])
        assert result.shap_values.shape == (10, 4)


# ---------------------------------------------------------------------------
# Help system
# ---------------------------------------------------------------------------


class TestHelpSystem:
    """Tests for transpyrancy/docs.py and the help() entry points."""

    @pytest.fixture()
    def result(self, rf_clf):
        model, X_train, X_test = rf_clf
        exp = Explainer("rf", model, background_data=X_train)
        return exp.explain(X_test.iloc[:20])

    # ---- docs.py internals ----

    def test_catalog_covers_all_attributes(self):
        from transpyrancy.docs import CATALOG

        expected_attributes = {
            "shap_values", "aggregate", "expected_value", "predictions",
            "prediction_proba", "feature_importance", "interaction_effects",
            "confidence_intervals", "group_contributions",
        }
        assert expected_attributes.issubset(set(CATALOG.keys()))

    def test_catalog_covers_all_plots(self):
        from transpyrancy.docs import CATALOG

        expected_plots = {
            "plot_summary", "plot_beeswarm", "plot_feature_importance",
            "plot_waterfall", "plot_force", "plot_dependence",
            "plot_heatmap", "plot_decision", "plot_group_contributions",
            "plot_confidence_intervals", "plot_interaction_heatmap",
        }
        assert expected_plots.issubset(set(CATALOG.keys()))

    def test_aliases_resolve(self):
        from transpyrancy.docs import CATALOG

        for alias in ("ci", "waterfall", "groups", "summary", "beeswarm"):
            assert alias in CATALOG, f"alias '{alias}' missing from CATALOG"

    def test_help_entry_has_required_fields(self):
        from transpyrancy.docs import CATALOG, HelpEntry

        for key, entry in CATALOG.items():
            assert isinstance(entry, HelpEntry), f"{key} is not a HelpEntry"
            assert entry.title, f"{key}.title is empty"
            assert entry.what, f"{key}.what is empty"

    # ---- show_help() ----

    def test_show_help_overview_runs(self, result, capsys):
        from transpyrancy.docs import show_help

        show_help(topic=None, result=result)
        captured = capsys.readouterr()
        assert "shap_values" in captured.out.lower()

    def test_show_help_known_topic_runs(self, result, capsys):
        from transpyrancy.docs import show_help

        show_help(topic="shap_values", result=result)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_show_help_alias_runs(self, result, capsys):
        from transpyrancy.docs import show_help

        show_help(topic="ci", result=result)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_show_help_unknown_topic_prints(self, result, capsys):
        from transpyrancy.docs import show_help

        show_help(topic="nonexistent_topic", result=result)
        captured = capsys.readouterr()
        assert "nonexistent_topic" in captured.out

    def test_show_help_plot_topic_runs(self, result, capsys):
        from transpyrancy.docs import show_help

        show_help(topic="plot_waterfall", result=result)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    # ---- result.help() ----

    def test_result_help_no_args(self, result, capsys):
        result.help()
        captured = capsys.readouterr()
        assert "shap_values" in captured.out.lower()

    def test_result_help_with_topic(self, result, capsys):
        result.help("aggregate")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_result_help_unknown_prints(self, result, capsys):
        result.help("bogus")
        captured = capsys.readouterr()
        assert "bogus" in captured.out

    # ---- Explainer.help() classmethod ----

    def test_explainer_help_runs(self, capsys):
        Explainer.help()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_explainer_help_mentions_explain(self, capsys):
        Explainer.help()
        captured = capsys.readouterr()
        assert "explain" in captured.out.lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for validate_X() and Explainer constructor / explain() guards."""

    N_FEATURES = 10

    @pytest.fixture(scope="class")
    def rf_setup(self):
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(
            rng.standard_normal((100, self.N_FEATURES)),
            columns=[f"feat_{i}" for i in range(self.N_FEATURES)],
        )
        y_train = rng.integers(0, 2, size=100)
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
        X_test = pd.DataFrame(
            rng.standard_normal((20, self.N_FEATURES)),
            columns=X_train.columns,
        )
        return model, X_train, X_test

    # ---- validate_X standalone ----

    def test_validate_X_wrong_type(self):
        from transpyrancy.utils import validate_X
        with pytest.raises(TypeError, match="pd.DataFrame or np.ndarray"):
            validate_X([[1, 2], [3, 4]])

    def test_validate_X_empty_dataframe(self):
        from transpyrancy.utils import validate_X
        with pytest.raises(ValueError, match="empty"):
            validate_X(pd.DataFrame(columns=["a", "b"]))

    def test_validate_X_empty_array(self):
        from transpyrancy.utils import validate_X
        with pytest.raises(ValueError, match="empty"):
            validate_X(np.empty((0, 4)))

    def test_validate_X_feature_count_mismatch(self):
        from transpyrancy.utils import validate_X
        X = np.ones((5, 3))
        with pytest.raises(ValueError, match="3 feature"):
            validate_X(X, expected_n_features=5)

    def test_validate_X_column_mismatch_missing(self):
        from transpyrancy.utils import validate_X
        X = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing"):
            validate_X(X, expected_columns=["a", "b", "c"])

    def test_validate_X_column_mismatch_extra(self):
        from transpyrancy.utils import validate_X
        X = pd.DataFrame({"a": [1], "b": [2], "z": [3]})
        with pytest.raises(ValueError, match="unexpected"):
            validate_X(X, expected_columns=["a", "b"])

    def test_validate_X_nan_warning(self):
        from transpyrancy.utils import validate_X
        X = pd.DataFrame({"a": [1.0, float("nan")], "b": [2.0, 3.0]})
        with pytest.warns(UserWarning, match="NaN"):
            validate_X(X)

    def test_validate_X_inf_warning(self):
        from transpyrancy.utils import validate_X
        X = np.array([[1.0, float("inf")], [2.0, 3.0]])
        with pytest.warns(UserWarning, match="infinite"):
            validate_X(X)

    def test_validate_X_valid_passes(self):
        from transpyrancy.utils import validate_X
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        validate_X(X, expected_n_features=2, expected_columns=["a", "b"])  # no error

    # ---- Explainer.__init__ guards ----

    def test_init_bad_model_type(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(TypeError, match="model_type must be a non-empty string"):
            Explainer(123, model, background_data=X_train)

    def test_init_empty_model_type(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(TypeError, match="model_type must be a non-empty string"):
            Explainer("", model, background_data=X_train)

    def test_init_bad_model(self, rf_setup):
        _, X_train, _ = rf_setup
        with pytest.raises(TypeError, match="predict\\(\\) method"):
            Explainer("rf", "not_a_model", background_data=X_train)

    def test_init_bad_feature_names(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(TypeError, match="feature_names must be a list of strings"):
            Explainer("rf", model, background_data=X_train, feature_names="feat1,feat2")

    def test_init_bad_feature_groups(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(TypeError, match="feature_groups must be a dict"):
            Explainer("rf", model, background_data=X_train, feature_groups=["a", "b"])

    def test_init_feature_groups_bad_value(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(TypeError, match="must be a list"):
            Explainer("rf", model, background_data=X_train, feature_groups={"g": "feat_0"})

    def test_init_bad_background_type(self, rf_setup):
        model, _, _ = rf_setup
        with pytest.raises(TypeError, match="background_data must be a pd.DataFrame"):
            Explainer("rf", model, background_data=[[1, 2], [3, 4]])

    def test_init_feature_names_length_mismatch(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(ValueError, match="feature_names has"):
            Explainer("rf", model, background_data=X_train, feature_names=["a", "b"])

    def test_init_bad_task(self, rf_setup):
        model, X_train, _ = rf_setup
        with pytest.raises(ValueError, match="task must be"):
            Explainer("rf", model, background_data=X_train, task="unsupervised")

    # ---- explain() parameter guards ----

    def test_explain_bad_n_bootstrap(self, rf_setup):
        model, X_train, X_test = rf_setup
        exp = Explainer("rf", model, background_data=X_train)
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            exp.explain(X_test, compute_confidence_intervals=True, n_bootstrap=0)

    def test_explain_bad_ci_level(self, rf_setup):
        model, X_train, X_test = rf_setup
        exp = Explainer("rf", model, background_data=X_train)
        with pytest.raises(ValueError, match="ci_level must be a float"):
            exp.explain(X_test, compute_confidence_intervals=True, ci_level=1.5)

    def test_explain_save_plots_is_file_path(self, rf_setup, tmp_path):
        model, X_train, X_test = rf_setup
        exp = Explainer("rf", model, background_data=X_train)
        with pytest.raises(ValueError, match="directory path"):
            exp.explain(X_test, save_plots=str(tmp_path / "output.png"))

    def test_explain_wrong_feature_count(self, rf_setup):
        model, X_train, X_test = rf_setup
        exp = Explainer("rf", model, background_data=X_train)
        X_bad = X_test.iloc[:, :5]  # only 5 of 10 features
        with pytest.raises(ValueError, match="feature"):
            exp.explain(X_bad)

    def test_explain_column_mismatch(self, rf_setup):
        model, X_train, X_test = rf_setup
        exp = Explainer("rf", model, background_data=X_train)
        X_bad = X_test.rename(columns={"feat_0": "WRONG"})
        with pytest.raises(ValueError, match="columns do not match"):
            exp.explain(X_bad)

    def test_explain_empty_X(self, rf_setup):
        model, X_train, X_test = rf_setup
        exp = Explainer("rf", model, background_data=X_train)
        with pytest.raises(ValueError, match="empty"):
            exp.explain(X_test.iloc[:0])
