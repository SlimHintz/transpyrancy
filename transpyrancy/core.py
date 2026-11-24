"""
Core Explainer class — the primary entry point for the explainer package.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap

from transpyrancy.detector import ExplainerType, detect_explainer_type, validate_override
from transpyrancy.result import ExplainResult
from transpyrancy.utils import (
    compute_aggregate,
    compute_group_contributions,
    detect_multiclass,
    detect_task,
    normalize_input,
    process_shap_output,
    validate_X,
)


class Explainer:
    """
    A unified, feature-rich SHAP-based model explainer.

    Automatically selects the most appropriate SHAP backend (TreeExplainer,
    LinearExplainer, or KernelExplainer) based on ``model_type``, computes
    SHAP values, and returns a rich :class:`~transpyrancy.result.ExplainResult`
    object containing data and visualisation methods.

    Parameters
    ----------
    model_type : str
        The kind of model to explain.  Supported strings:

        **Tree-based** (uses ``shap.TreeExplainer``):
        ``'xgb'``, ``'xgboost'``, ``'lgb'``, ``'lightgbm'``, ``'catboost'``,
        ``'rf'``, ``'random_forest'``, ``'gbm'``, ``'gradient_boosting'``,
        ``'gbc'``, ``'gbr'``, ``'et'``, ``'extra_trees'``, ``'dt'``,
        ``'decision_tree'``, ``'histgbm'``, ``'hist_gradient_boosting'``

        **Linear** (uses ``shap.LinearExplainer``):
        ``'linear'``, ``'logistic'``, ``'logistic_regression'``,
        ``'linear_regression'``, ``'ridge'``, ``'ridgeclassifier'``,
        ``'lasso'``, ``'elasticnet'``, ``'sgd'``, ``'perceptron'``,
        ``'linearsvc'``, ``'linearsvr'``

        **Kernel fallback** (uses ``shap.KernelExplainer``):
        ``'kernel'``, ``'svm'``, ``'svc'``, ``'svr'``, ``'knn'``,
        ``'mlp'``, ``'naive_bayes'``, ``'gaussiannb'``

        **Deep learning** (uses ``shap.DeepExplainer``):
        ``'torch'``, ``'pytorch'``, ``'tensorflow'``, ``'tf'``,
        ``'keras'``, ``'neural_network'``, ``'nn'``, ``'deep'``

        Unrecognised strings fall back to ``KernelExplainer`` with a warning.

    model : Any
        Trained model.  Must expose a ``predict`` method.  For
        classification tasks, ``predict_proba`` should also be available.

    background_data : pd.DataFrame | np.ndarray | None
        Reference dataset used to marginalise over missing features.

        - **TreeExplainer**: optional but recommended (enables
          interventional perturbation explanations).
        - **LinearExplainer**: required.
        - **KernelExplainer**: required (keep to 50–200 rows for speed).
        - **DeepExplainer**: required (keep to 50–200 rows for speed).

        Typically a representative sample from the training set.

    explainer_override : {'tree', 'linear', 'kernel', 'deep'} | None
        Force a specific SHAP backend, overriding auto-detection.
        A :class:`UserWarning` is raised when this conflicts with the
        detected type for ``model_type``.

    predict_fn : callable | None
        Custom prediction function ``(X: np.ndarray) -> np.ndarray``.
        **Required for deep models** unless the model exposes a standard
        ``.predict()`` method (Keras) or is a ``torch.nn.Module`` (in
        which case an auto-wrapping predict function is used).

        For classification, return an ``(n_samples, n_classes)`` probability
        matrix and set ``task='classification'``; class predictions are
        derived via ``argmax``.  For regression, return a 1-D array.

    task : {'classification', 'regression', 'auto'}
        Prediction task.  ``'auto'`` detects by checking for
        ``predict_proba`` on the model.  Default ``'auto'``.

    feature_names : list[str] | None
        Explicit feature names.  Inferred from DataFrame columns or
        auto-assigned (``f0``, ``f1``, …) when not provided.

    feature_groups : dict[str, list[str]] | None
        Optional feature grouping for aggregate group-level contributions.
        Example::

            feature_groups = {
                'demographics': ['age', 'gender', 'education'],
                'financial':    ['income', 'debt', 'credit_score'],
            }

    Examples
    --------
    **XGBoost binary classification**::

        import xgboost as xgb
        from transpyrancy import Explainer

        model = xgb.XGBClassifier().fit(X_train, y_train)
        exp = Explainer('xgb', model, background_data=X_train)

        result = exp.explain(X_test)
        print(result.aggregate)
        result.plot_summary()
        result.plot_waterfall(index=0)

    **LightGBM regression with all extras**::

        import lightgbm as lgb
        from transpyrancy import Explainer

        model = lgb.LGBMRegressor().fit(X_train, y_train)
        exp = Explainer('lightgbm', model)

        result = exp.explain(
            X_test,
            compute_interactions=True,
            compute_confidence_intervals=True,
            visualize=True,
            save_plots='./plots/',
        )

    **Ridge regression (LinearExplainer)**::

        from sklearn.linear_model import Ridge
        from transpyrancy import Explainer

        model = Ridge().fit(X_train, y_train)
        exp = Explainer('ridge', model, background_data=X_train, task='regression')
        result = exp.explain(X_test)

    **Feature groups**::

        exp = Explainer(
            'rf', model,
            feature_groups={'location': ['lat', 'lon', 'zip'],
                             'time':     ['hour', 'day', 'month']},
        )
        result = exp.explain(X_test)
        result.plot_group_contributions()
    """

    def __init__(
        self,
        model_type: str,
        model: Any,
        background_data: pd.DataFrame | np.ndarray | None = None,
        explainer_override: Literal["tree", "linear", "kernel", "deep"] | None = None,
        task: Literal["classification", "regression", "auto"] = "auto",
        feature_names: list[str] | None = None,
        feature_groups: dict[str, list[str]] | None = None,
        predict_fn: Callable | None = None,
    ) -> None:
        # ---- Validate constructor arguments ----
        if not isinstance(model_type, str) or not model_type.strip():
            raise TypeError(
                f"model_type must be a non-empty string (e.g. 'xgb', 'rf', 'ridge'), "
                f"got {type(model_type).__name__!r}."
            )
        if not (callable(model) or hasattr(model, "predict")):
            raise TypeError(
                "model must be a trained model with a predict() method or a callable. "
                f"Got {type(model).__name__!r}."
            )
        if feature_names is not None:
            if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names):
                raise TypeError(
                    "feature_names must be a list of strings, "
                    f"got {type(feature_names).__name__!r}."
                )
        if feature_groups is not None:
            if not isinstance(feature_groups, dict):
                raise TypeError(
                    f"feature_groups must be a dict mapping group names to lists of feature names, "
                    f"got {type(feature_groups).__name__!r}."
                )
            for k, v in feature_groups.items():
                if not isinstance(v, list):
                    raise TypeError(
                        f"feature_groups['{k}'] must be a list of feature name strings, "
                        f"got {type(v).__name__!r}."
                    )
        if background_data is not None and not isinstance(background_data, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                f"background_data must be a pd.DataFrame or np.ndarray, "
                f"got {type(background_data).__name__!r}."
            )

        self.model_type = model_type.lower().strip()
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.feature_groups = feature_groups
        self._predict_fn = predict_fn

        # Derive feature metadata from background_data for explain()-time validation
        if background_data is not None and hasattr(background_data, "shape") and background_data.ndim > 1:
            self._n_features: int | None = background_data.shape[1]
            self._training_columns: list[str] | None = (
                list(background_data.columns)
                if isinstance(background_data, pd.DataFrame)
                else None
            )
        elif feature_names is not None:
            self._n_features = len(feature_names)
            self._training_columns = None
        else:
            self._n_features = None
            self._training_columns = None

        # Validate feature_names length against background_data
        if (
            feature_names is not None
            and background_data is not None
            and hasattr(background_data, "shape")
            and background_data.ndim > 1
            and len(feature_names) != background_data.shape[1]
        ):
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but background_data has "
                f"{background_data.shape[1]} columns. They must match."
            )

        # Resolve task — deep models rarely expose predict_proba, so default to regression
        if task == "auto":
            self.task: str = detect_task(model)
        elif task in ("classification", "regression"):
            self.task = task
        else:
            raise ValueError(f"task must be 'classification', 'regression', or 'auto', got '{task!r}'.")

        # Detect / validate explainer type
        detected_type = detect_explainer_type(self.model_type)
        if explainer_override is not None:
            try:
                override_type = ExplainerType(explainer_override)
            except ValueError:
                raise ValueError(
                    f"explainer_override must be one of 'tree', 'linear', 'kernel', 'deep', "
                    f"got '{explainer_override}'."
                )
            validate_override(detected_type, override_type, self.model_type)
            self._explainer_type: ExplainerType = override_type
        else:
            self._explainer_type = detected_type

        # Build the underlying SHAP explainer
        self._shap_explainer: Any = self._build_explainer()

    # ------------------------------------------------------------------ #
    # Private construction helpers
    # ------------------------------------------------------------------ #

    def _build_explainer(self) -> Any:
        """Instantiate the appropriate shap explainer."""
        if self._explainer_type == ExplainerType.TREE:
            return self._build_tree_explainer()
        elif self._explainer_type == ExplainerType.LINEAR:
            return self._build_linear_explainer()
        elif self._explainer_type == ExplainerType.DEEP:
            return self._build_deep_explainer()
        else:
            return self._build_kernel_explainer()

    def _build_tree_explainer(self) -> shap.TreeExplainer:
        kwargs: dict[str, Any] = {}
        if self.background_data is not None:
            kwargs["data"] = self.background_data
        try:
            return shap.TreeExplainer(self.model, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build shap.TreeExplainer for model_type='{self.model_type}'. "
                f"Ensure the model is a fitted tree-based estimator.\nOriginal error: {exc}"
            ) from exc

    def _build_linear_explainer(self) -> shap.LinearExplainer:
        if self.background_data is None:
            raise ValueError(
                "background_data is required for LinearExplainer. "
                "Provide a sample from the training data (e.g., X_train or a subset)."
            )
        try:
            return shap.LinearExplainer(self.model, masker=self.background_data)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build shap.LinearExplainer for model_type='{self.model_type}'. "
                f"Ensure the model is a fitted linear estimator.\nOriginal error: {exc}"
            ) from exc

    def _build_kernel_explainer(self) -> shap.KernelExplainer:
        if self.background_data is None:
            raise ValueError(
                "background_data is required for KernelExplainer. "
                "Provide a representative sample (50–200 rows) from the training data."
            )
        # Choose prediction function
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        try:
            return shap.KernelExplainer(predict_fn, self.background_data)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build shap.KernelExplainer.\nOriginal error: {exc}"
            ) from exc

    def _build_deep_explainer(self) -> Any:
        if self.background_data is None:
            raise ValueError(
                "background_data is required for DeepExplainer. "
                "Provide a representative sample (50–200 rows) from the training data."
            )
        background = self._to_deep_input(np.asarray(self.background_data))
        # Put PyTorch models in eval mode before building
        if self._is_pytorch_model() and hasattr(self.model, "eval"):
            self.model.eval()
        try:
            return shap.DeepExplainer(self.model, background)
        except Exception as exc:
            raise RuntimeError(
                "Failed to build shap.DeepExplainer. "
                "Ensure the model is a PyTorch (torch.nn.Module) or "
                "TensorFlow/Keras neural network.\n"
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Deep model helpers
    # ------------------------------------------------------------------ #

    def _is_pytorch_model(self) -> bool:
        return self.model_type in ("torch", "pytorch")

    def _is_tensorflow_model(self) -> bool:
        return self.model_type in ("tensorflow", "tf", "keras")

    def _to_deep_input(self, X_np: np.ndarray) -> Any:
        """Convert a numpy array to the tensor type expected by the deep model."""
        if self._is_pytorch_model():
            try:
                import torch  # noqa: PLC0415
                return torch.tensor(X_np, dtype=torch.float32)
            except ImportError as exc:
                raise ImportError(
                    "PyTorch is not installed. Install it with:\n"
                    "  pip install torch\n"
                    "or use the transpyrancy[torch] extra."
                ) from exc
        # Keras/TF and generic deep models accept numpy directly
        return X_np.astype(np.float32)

    def _get_deep_predict_fn(self) -> Callable:
        """Return the prediction callable for deep models."""
        if self._predict_fn is not None:
            return self._predict_fn

        if self._is_pytorch_model():
            try:
                import torch  # noqa: PLC0415
                model = self.model

                def _pytorch_predict(X_np: np.ndarray) -> np.ndarray:
                    with torch.no_grad():
                        out = model(torch.tensor(X_np, dtype=torch.float32))
                        return out.numpy() if isinstance(out, torch.Tensor) else np.asarray(out)

                return _pytorch_predict
            except ImportError as exc:
                raise ImportError("PyTorch is not installed.") from exc

        # Keras/TF — .predict() is standard
        if hasattr(self.model, "predict"):
            return lambda X: self.model.predict(X, verbose=0)

        # Generic callable (e.g. a wrapped function)
        if callable(self.model):
            return lambda X: np.asarray(self.model(X))

        raise ValueError(
            "Cannot auto-detect a prediction function for this deep model. "
            "Pass predict_fn= to Explainer()."
        )

    def _call_shap_explainer(self, X: Any) -> Any:
        """
        Invoke the SHAP explainer, handling the different APIs across backends.

        - Tree/Linear/Kernel: use ``__call__`` (returns ``shap.Explanation``).
        - Deep: use ``.shap_values()`` (returns a list of arrays).
        """
        if self._explainer_type == ExplainerType.DEEP:
            X_input = self._to_deep_input(np.asarray(X))
            return self._shap_explainer.shap_values(X_input)
        return self._shap_explainer(X)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def explain(
        self,
        X: pd.DataFrame | np.ndarray,
        visualize: bool = False,
        save_plots: str | Path | None = None,
        compute_interactions: bool = False,
        compute_confidence_intervals: bool = False,
        n_bootstrap: int = 100,
        ci_level: float = 0.95,
        n_jobs: int = 1,
    ) -> ExplainResult:
        """
        Compute SHAP explanations for one or more observations.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Observations to explain.  A single row or a batch.
        visualize : bool
            If ``True``, display all available plots inline after computing
            explanations.  Default ``False``.
        save_plots : str | Path | None
            Directory path.  When provided, every plot is saved as a PNG
            inside that directory (created automatically if needed).
            Implies plot generation even when ``visualize=False``.
        compute_interactions : bool
            Compute pairwise SHAP interaction values.  Only supported for
            ``TreeExplainer``.  Can be slow for many features.
            Default ``False``.
        compute_confidence_intervals : bool
            Estimate bootstrap confidence intervals for SHAP values.
            Adds ``n_bootstrap`` model calls.  Default ``False``.
        n_bootstrap : int
            Number of bootstrap resamples for CI estimation.  Default 100.
        ci_level : float
            Confidence level (e.g., ``0.95`` for 95% CI).  Default 0.95.
        n_jobs : int
            Number of parallel workers.  ``1`` disables parallelism (default).
            ``-1`` uses all available CPUs.  Any positive integer caps the
            number of workers.

            Parallelism is applied to two independently-parallelisable
            operations:

            * **Bootstrap CI** — each resample is computed on a separate
              thread.  With ``n_bootstrap=100`` and ``n_jobs=-1`` you can
              expect near-linear speedup on all available cores.
            * **Batch SHAP** — for datasets with more than
              ``chunk_threshold`` rows (default 500) the input is split into
              ``n_jobs`` chunks and SHAP values are computed in parallel
              threads.  This helps most for ``LinearExplainer`` and
              ``KernelExplainer``; ``TreeExplainer`` already parallelises
              internally via XGBoost / LightGBM thread pools, so gains are
              model-dependent.

            .. note::
                Threading (not multiprocessing) is used so that the model
                and explainer objects are shared without pickling overhead.
                This works well because SHAP's C++ backends and NumPy
                release the GIL during heavy computation.

        Returns
        -------
        ExplainResult
            Rich result object with data attributes and visualisation methods:

            - ``result.shap_values`` — per-observation SHAP DataFrame(s)
            - ``result.aggregate`` — global feature importance
            - ``result.expected_value`` — model baseline
            - ``result.predictions`` — model outputs
            - ``result.prediction_proba`` — class probabilities (classification)
            - ``result.feature_importance`` — ranked importance Series
            - ``result.interaction_effects`` — pairwise interactions (if requested)
            - ``result.confidence_intervals`` — bootstrap CIs (if requested)
            - ``result.group_contributions`` — group sums (if groups defined)
            - ``result.plot_*()`` — all visualisation methods

        Examples
        --------
        Basic explanation::

            result = exp.explain(X_test)
            print(result)

        With inline visualisation::

            result = exp.explain(X_test, visualize=True)

        Save all plots to disk::

            result = exp.explain(X_test, save_plots='./output/plots/')

        With interaction effects and confidence intervals::

            result = exp.explain(
                X_test,
                compute_interactions=True,
                compute_confidence_intervals=True,
                n_bootstrap=200,
            )
            result.plot_interaction_heatmap('age')
            result.plot_confidence_intervals(top_n=10)
        """
        # ---- Validate explain() parameters ----
        if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
            raise ValueError(
                f"n_bootstrap must be a positive integer, got {n_bootstrap!r}."
            )
        if not isinstance(ci_level, float) or not (0.0 < ci_level < 1.0):
            raise ValueError(
                f"ci_level must be a float strictly between 0 and 1 (e.g. 0.95), "
                f"got {ci_level!r}."
            )
        if save_plots is not None and Path(save_plots).suffix:
            raise ValueError(
                f"save_plots must be a directory path, not a file path. "
                f"Got {str(save_plots)!r}. Example: save_plots='./my_plots/'"
            )

        # ---- Validate input data ----
        validate_X(X, self._n_features, self._training_columns)

        # Normalise input
        X_df, feature_names = normalize_input(X, self.feature_names)

        # --- Compute SHAP values (parallel chunks when n_jobs != 1) ---
        shap_values_df, expected_value, is_multiclass, shap_explanation = (
            self._compute_shap(X_df, feature_names, n_jobs)
        )

        # --- Predictions ---
        prediction_proba: np.ndarray | None = None
        if self._explainer_type == ExplainerType.DEEP:
            raw = np.asarray(self._get_deep_predict_fn()(np.asarray(X_df)))
            if self.task == "classification" and raw.ndim == 2 and raw.shape[1] > 1:
                prediction_proba = raw
                predictions = raw.argmax(axis=1)
            else:
                predictions = raw.ravel() if raw.ndim > 1 else raw
        else:
            predictions = self.model.predict(X_df)
            if self.task == "classification" and hasattr(self.model, "predict_proba"):
                prediction_proba = self.model.predict_proba(X_df)

        # --- Aggregate importance ---
        aggregate = compute_aggregate(shap_values_df, is_multiclass)

        # --- Interaction effects ---
        interaction_effects: np.ndarray | None = None
        if compute_interactions:
            if self._explainer_type not in (ExplainerType.TREE,):
                warnings.warn(
                    "Interaction effects are only supported for TreeExplainer. "
                    f"Current explainer is '{self._explainer_type.value}'. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                try:
                    raw_interactions = self._shap_explainer.shap_interaction_values(X_df)
                    # Normalise to 3-D (n_samples, n_features, n_features)
                    if isinstance(raw_interactions, list):
                        raw_interactions = np.stack(raw_interactions, axis=-1)
                    interaction_effects = np.asarray(raw_interactions)
                    # (n, f, f, n_classes) → take last class (positive for binary)
                    while interaction_effects.ndim > 3:
                        interaction_effects = interaction_effects[..., -1]
                except Exception as exc:
                    warnings.warn(
                        f"Could not compute interaction effects: {exc}. Skipping.",
                        UserWarning,
                        stacklevel=2,
                    )

        # --- Confidence intervals ---
        confidence_intervals: dict[str, pd.DataFrame] | None = None
        if compute_confidence_intervals:
            confidence_intervals = self._bootstrap_ci(
                X_df, feature_names, n_bootstrap, ci_level, n_jobs
            )

        # --- Group contributions ---
        group_contributions: pd.DataFrame | None = None
        if self.feature_groups is not None:
            sv_for_groups = (
                shap_values_df
                if isinstance(shap_values_df, pd.DataFrame)
                else shap_values_df[0]
            )
            group_contributions = compute_group_contributions(sv_for_groups, self.feature_groups)

        # --- Resolve class count ---
        n_classes: int | None = None
        if is_multiclass and isinstance(shap_values_df, list):
            n_classes = len(shap_values_df)
        elif self.task == "classification" and prediction_proba is not None:
            n_classes = prediction_proba.shape[1]

        # --- Assemble result ---
        result = ExplainResult(
            shap_values=shap_values_df,
            aggregate=aggregate,
            expected_value=expected_value,
            predictions=predictions,
            prediction_proba=prediction_proba,
            task=self.task,
            n_observations=len(X_df),
            feature_names=feature_names,
            model_type=self.model_type,
            explainer_type=self._explainer_type.value,
            is_multiclass=is_multiclass,
            n_classes=n_classes,
            interaction_effects=interaction_effects,
            confidence_intervals=confidence_intervals,
            group_contributions=group_contributions,
            _shap_explanation=shap_explanation,
        )

        # --- Optional visualisation ---
        if visualize or save_plots is not None:
            save_dir = Path(save_plots) if save_plots is not None else None
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
            result._visualize_all(save_dir=save_dir, show=visualize)

        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    # Minimum rows per chunk — smaller splits add overhead without benefit.
    _MIN_CHUNK_ROWS: int = 50

    def _compute_shap(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: list[str],
        n_jobs: int,
    ) -> tuple[
        pd.DataFrame | list[pd.DataFrame],
        float | list[float],
        bool,
        Any,
    ]:
        """
        Compute SHAP values, optionally in parallel chunks.

        Returns (shap_values_df, expected_value, is_multiclass, raw_explanation).
        ``raw_explanation`` is ``None`` when chunked (per-chunk explanations
        are discarded after processing to save memory).
        """
        from joblib import Parallel, delayed, effective_n_jobs

        n_rows = len(X) if hasattr(X, "__len__") else 1
        actual_jobs = effective_n_jobs(n_jobs)

        # Only chunk when it makes sense
        use_chunks = (
            n_jobs != 1
            and actual_jobs > 1
            and n_rows >= actual_jobs * self._MIN_CHUNK_ROWS
        )

        if not use_chunks:
            # Single-shot computation — preserves raw explanation object
            try:
                explanation = self._call_shap_explainer(X)
            except Exception as exc:
                raise RuntimeError(
                    f"SHAP computation failed.\nOriginal error: {exc}"
                ) from exc
            is_mc = detect_multiclass(explanation)
            sv_df, ev = process_shap_output(explanation, feature_names, self.task, is_mc)
            # DeepExplainer stores expected_value as an attribute; override 0.0 default
            if self._explainer_type == ExplainerType.DEEP:
                ev = self._extract_deep_expected_value(is_mc)
            return sv_df, ev, is_mc, explanation

        # ---- Chunked parallel computation ----
        chunk_size = max(self._MIN_CHUNK_ROWS, n_rows // actual_jobs)
        if isinstance(X, pd.DataFrame):
            chunks = [X.iloc[i: i + chunk_size] for i in range(0, n_rows, chunk_size)]
        else:
            chunks = [X[i: i + chunk_size] for i in range(0, n_rows, chunk_size)]

        def _run_chunk(
            chunk: pd.DataFrame | np.ndarray,
        ) -> tuple[pd.DataFrame | list[pd.DataFrame], float | list[float], bool]:
            try:
                sv = self._call_shap_explainer(chunk)
            except Exception as exc:
                raise RuntimeError(f"SHAP chunk computation failed: {exc}") from exc
            is_mc = detect_multiclass(sv)
            sv_df, ev = process_shap_output(sv, feature_names, self.task, is_mc)
            return sv_df, ev, is_mc

        chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_chunk)(chunk) for chunk in chunks
        )

        # Determine multiclass from first chunk
        is_multiclass = chunk_results[0][2]
        expected_value = chunk_results[0][1]

        # Concatenate SHAP DataFrames
        if is_multiclass:
            n_classes = len(chunk_results[0][0])
            shap_values_df: pd.DataFrame | list[pd.DataFrame] = [
                pd.concat(
                    [r[0][c] for r in chunk_results],  # type: ignore[index]
                    ignore_index=True,
                )
                for c in range(n_classes)
            ]
        else:
            shap_values_df = pd.concat(
                [r[0] for r in chunk_results],  # type: ignore[arg-type]
                ignore_index=True,
            )

        return shap_values_df, expected_value, is_multiclass, None

    def _extract_deep_expected_value(self, is_multiclass: bool) -> float | list[float]:
        """Extract and normalise the expected_value from a DeepExplainer."""
        ev_raw = getattr(self._shap_explainer, "expected_value", 0.0)
        if isinstance(ev_raw, (list, np.ndarray)):
            ev_arr = np.asarray(ev_raw, dtype=float)
            if is_multiclass:
                return ev_arr.tolist()
            return float(ev_arr[-1]) if ev_arr.size > 0 else 0.0
        return float(ev_raw)

    def _bootstrap_ci(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: list[str],
        n_bootstrap: int,
        ci_level: float,
        n_jobs: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """
        Estimate bootstrap confidence intervals for SHAP values.

        Resamples ``X`` with replacement ``n_bootstrap`` times, computes
        SHAP values for each resample, then reports percentile-based CIs.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        feature_names : list[str]
            Feature names.
        n_bootstrap : int
            Number of bootstrap iterations.
        ci_level : float
            Confidence level.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{'lower': DataFrame, 'upper': DataFrame}``.
        """
        from joblib import Parallel, delayed

        n_samples = len(X) if hasattr(X, "__len__") else 1

        def _one_bootstrap(seed: int) -> np.ndarray | None:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n_samples, size=n_samples)
            if isinstance(X, pd.DataFrame):
                X_boot = X.iloc[idx].reset_index(drop=True)
            else:
                X_boot = X[idx]
            try:
                sv_boot = self._call_shap_explainer(X_boot)
            except Exception as exc:
                warnings.warn(
                    f"Bootstrap iteration {seed} failed: {exc}. Skipping.",
                    UserWarning,
                    stacklevel=5,
                )
                return None

            if isinstance(sv_boot, shap.Explanation):
                values = sv_boot.values
            else:
                values = np.asarray(sv_boot)

            # Normalise to 2-D (n_samples, n_features)
            if isinstance(values, list):
                values = values[1] if len(values) == 2 else values[0]
            values = np.asarray(values)
            while values.ndim > 2:
                values = values[..., -1]

            return values

        raw_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_bootstrap)(seed) for seed in range(n_bootstrap)
        )
        bootstrap_shap = [r for r in raw_results if r is not None]

        if not bootstrap_shap:
            raise RuntimeError("All bootstrap iterations failed — cannot compute CIs.")

        # Stack: (n_bootstrap, n_samples_orig_size, n_features)
        # Since bootstrap resamples may differ in effective distribution,
        # we compute mean SHAP across observations per bootstrap, then CI over those means.
        boot_means = np.stack([b.mean(axis=0) for b in bootstrap_shap], axis=0)  # (n_bootstrap, n_features)

        alpha = (1 - ci_level) / 2
        lower_means = np.percentile(boot_means, alpha * 100, axis=0)
        upper_means = np.percentile(boot_means, (1 - alpha) * 100, axis=0)

        # Broadcast back to (n_samples, n_features) for consistency with shap_values shape
        lower = pd.DataFrame(
            np.tile(lower_means, (n_samples, 1)),
            columns=feature_names,
        )
        upper = pd.DataFrame(
            np.tile(upper_means, (n_samples, 1)),
            columns=feature_names,
        )

        return {"lower": lower, "upper": upper}

    # ------------------------------------------------------------------ #
    # Help
    # ------------------------------------------------------------------ #

    @classmethod
    def help(cls) -> None:
        """
        Display a plain-English overview of the transpyrancy library.

        Shows the standard workflow, all supported model type strings,
        and every optional parameter available on ``explain()``.

        For help on a specific result attribute or plot, call
        ``result.help()`` or ``result.help('topic')`` after running
        ``explain()``.

        Examples
        --------
        ::

            from transpyrancy import Explainer

            Explainer.help()          # library overview
        """
        from transpyrancy.docs import (  # noqa: PLC0415
            EXPLAINER_OVERVIEW,
            EXPLAINER_OVERVIEW_MD,
            _display_md,
            _in_jupyter,
        )
        if _in_jupyter():
            _display_md(EXPLAINER_OVERVIEW_MD)
        else:
            print(EXPLAINER_OVERVIEW)

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Explainer("
            f"model_type='{self.model_type}', "
            f"explainer='{self._explainer_type.value}', "
            f"task='{self.task}')"
        )
