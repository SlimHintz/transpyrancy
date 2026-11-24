# transpyrancy

**Transparent, SHAP-based ML explainability — batteries included.**

`transpyrancy` wraps the [SHAP](https://shap.readthedocs.io/) library with a clean,
opinionated API that auto-detects the right explainer backend, returns a rich result
object with both per-observation and aggregate data, and exposes every visualisation
method directly on that result.

---

## Features

| Feature | Details |
|---|---|
| **Auto-backend detection** | Selects `TreeExplainer`, `LinearExplainer`, or `KernelExplainer` based on `model_type` — with a manual override |
| **Conflict warnings** | Raises `UserWarning` when an override conflicts with auto-detection |
| **Rich result object** | `ExplainResult` dataclass with SHAP values, aggregate importance, predictions, probabilities, and more |
| **10 plot types** | Summary, beeswarm, waterfall, force, feature importance, dependence, heatmap, decision, group contributions, confidence intervals |
| **Save to PNG** | Pass `save_plots='./plots/'` to persist every chart |
| **Interaction effects** | Pairwise SHAP interaction values (TreeExplainer only) |
| **Bootstrap CIs** | Confidence intervals on SHAP values via bootstrap resampling |
| **Feature groups** | Aggregate SHAP contributions for user-defined feature groups |
| **Multiclass support** | Full handling of 2+ class classification problems |
| **Classification & regression** | Auto-detects task from `predict_proba` |

---

## Installation

### From GitHub

```bash
pip install "transpyrancy @ git+https://github.com/SlimHintz/transpyrancy"

# With optional model support: xgboost, lightgbm, catboost, torch, tensorflow, or all
pip install "transpyrancy[xgboost] @ git+https://github.com/SlimHintz/transpyrancy"
pip install "transpyrancy[all] @ git+https://github.com/SlimHintz/transpyrancy"
```

### Editable install (development)

```bash
# Install in editable mode — changes to source take effect immediately
pip install -e .

# With optional extras
pip install -e ".[xgboost]"
pip install -e ".[all]"

# Full development environment (includes test + notebook dependencies)
pip install -e ".[dev]"
```

---

## Quick Start

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from transpyrancy import Explainer

# Train a model
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create an Explainer
exp = Explainer('xgb', model, background_data=X_train)

# Explain the test set
result = exp.explain(X_test)

# Inspect the result
print(result)                    # rich text summary
print(result.aggregate)          # mean |SHAP| per feature, ranked
print(result.shap_values.head()) # per-row SHAP values
print(result.top_features(5))    # top-5 most important features
```

---

## Supported Model Types

### Tree-based → `shap.TreeExplainer`

| String(s) | Model |
|---|---|
| `'xgb'`, `'xgboost'` | XGBoost |
| `'lgb'`, `'lightgbm'` | LightGBM |
| `'catboost'` | CatBoost |
| `'rf'`, `'random_forest'` | `RandomForestClassifier/Regressor` |
| `'gbm'`, `'gradient_boosting'`, `'gbc'`, `'gbr'` | `GradientBoostingClassifier/Regressor` |
| `'et'`, `'extra_trees'` | `ExtraTreesClassifier/Regressor` |
| `'dt'`, `'decision_tree'` | `DecisionTreeClassifier/Regressor` |
| `'histgbm'`, `'hist_gradient_boosting'` | `HistGradientBoosting*` |

### Linear → `shap.LinearExplainer`

| String(s) | Model |
|---|---|
| `'linear'`, `'linear_regression'` | `LinearRegression` |
| `'logistic'`, `'logistic_regression'` | `LogisticRegression` |
| `'ridge'`, `'ridgeclassifier'` | `Ridge`, `RidgeClassifier` |
| `'lasso'` | `Lasso` |
| `'elasticnet'` | `ElasticNet` |
| `'sgd'`, `'sgdclassifier'`, `'sgdregressor'` | `SGDClassifier/Regressor` |
| `'linearsvc'`, `'linearsvr'` | `LinearSVC`, `LinearSVR` |

### Kernel fallback → `shap.KernelExplainer`

Any model not covered above, including `'svm'`, `'svc'`, `'svr'`, `'knn'`, `'mlp'`, `'naive_bayes'`, or any unrecognised string.

### Deep learning → `shap.DeepExplainer`

| String(s) | Framework |
|---|---|
| `'torch'`, `'pytorch'` | PyTorch (`torch.nn.Module`) |
| `'tensorflow'`, `'tf'`, `'keras'` | TensorFlow / Keras |
| `'neural_network'`, `'nn'`, `'deep'` | Generic deep model |

**Requirements:**
- `background_data` is required (50–200 rows recommended).
- `task` should be set explicitly — deep models rarely expose `predict_proba`.
- `predict_fn` is required unless the model has a standard `.predict()` method (Keras) or is a `torch.nn.Module` (auto-wrapped).

**Limitations:**
- Interaction effects (`compute_interactions=True`) are not supported and will be silently skipped with a warning.
- Bootstrap confidence intervals work but can be slow without `n_jobs`.

**Installation:**

```bash
# PyTorch
pip install "transpyrancy[torch]"

# TensorFlow / Keras
pip install "transpyrancy[tensorflow]"
```

**PyTorch example:**

```python
import torch
import torch.nn as nn
import numpy as np
from transpyrancy import Explainer

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

model = MyNet()
model.eval()

def predict_fn(X_np):
    with torch.no_grad():
        out = model(torch.tensor(X_np, dtype=torch.float32)).numpy()
    return np.column_stack([1 - out, out])  # (n, 2) probability matrix

exp = Explainer(
    'torch', model,
    background_data=X_train,
    task='classification',
    predict_fn=predict_fn,
)
result = exp.explain(X_test)
result.plot_summary()
```

**Keras example:**

```python
from transpyrancy import Explainer

# Keras model with .predict() — no predict_fn needed
exp = Explainer(
    'keras', keras_model,
    background_data=X_train,
    task='classification',
    predict_fn=lambda X: keras_model.predict(X, verbose=0),
)
result = exp.explain(X_test)
```

---

## The `ExplainResult` Object

`exp.explain(X)` returns an `ExplainResult` dataclass.

### Data Attributes

| Attribute | Type | Description |
|---|---|---|
| `shap_values` | `DataFrame` or `list[DataFrame]` | Per-observation SHAP values.  List for multiclass. |
| `aggregate` | `DataFrame` | Mean absolute SHAP per feature, sorted descending. |
| `expected_value` | `float` or `list[float]` | Model baseline `E[f(X)]`. |
| `predictions` | `np.ndarray` | Model predictions. |
| `prediction_proba` | `np.ndarray` or `None` | Class probabilities (classification only). |
| `feature_importance` | `pd.Series` | Ranked features by mean \|SHAP\|. |
| `task` | `str` | `'classification'` or `'regression'`. |
| `n_observations` | `int` | Number of rows explained. |
| `feature_names` | `list[str]` | Feature column names. |
| `model_type` | `str` | `model_type` as passed at construction. |
| `explainer_type` | `str` | Backend used: `'tree'`, `'linear'`, `'kernel'`, `'deep'`. |
| `is_multiclass` | `bool` | True when >2 output classes. |
| `n_classes` | `int` or `None` | Number of classes. |
| `interaction_effects` | `np.ndarray` or `None` | Pairwise SHAP interactions `(n, f, f)`. |
| `confidence_intervals` | `dict` or `None` | `{'lower': DataFrame, 'upper': DataFrame}`. |
| `group_contributions` | `DataFrame` or `None` | Summed SHAP per feature group. |

### Convenience Methods

```python
result.summary()                     # str — human-readable summary
result.top_features(n=10)            # pd.Series — top-n features by importance
result.shap_values_for_class(idx)    # DataFrame — multiclass only
```

---

## Built-in Help System

Every `ExplainResult` carries a `help()` method that renders plain-English
explanations of every attribute and plot — context-aware and Jupyter-friendly.

```python
# Full overview: what's available, what each thing means, interpretation tips
result.help()

# Zoom in on one attribute
result.help('shap_values')
result.help('expected_value')
result.help('confidence_intervals')

# Zoom in on one plot
result.help('plot_waterfall')
result.help('plot_summary')
result.help('plot_dependence')

# Short aliases work too
result.help('waterfall')
result.help('ci')
result.help('groups')
```

The overview is **context-aware**: it shows which optional outputs (confidence
intervals, interaction effects, group contributions) are present in your specific
result, and tells you what to pass to `explain()` to enable the missing ones.

In **Jupyter**, output is rendered as formatted Markdown. In a terminal, it
prints clean plain text.

Before calling `explain()`, you can get a library overview from the class itself:

```python
from transpyrancy import Explainer

Explainer.help()    # workflow, supported model types, all explain() options
```

---

## Visualisation

All plot methods return a `plt.Figure` and accept `save_path` and `show` arguments.

```python
result.plot_summary()                    # SHAP beeswarm / dot plot
result.plot_beeswarm()                   # alias for plot_summary(plot_type='dot')
result.plot_feature_importance(top_n=15) # horizontal bar chart
result.plot_waterfall(index=0)           # single-observation waterfall
result.plot_force(index=0)              # force plot (matplotlib)
result.plot_dependence('age')           # scatter with auto interaction colouring
result.plot_heatmap()                   # feature × observation SHAP heatmap
result.plot_decision()                  # cumulative contribution lines
result.plot_group_contributions()       # group-level bar chart
result.plot_confidence_intervals()      # bootstrap CI error bars
result.plot_interaction_heatmap('age')  # pairwise interactions for one feature
```

### Save All Plots

```python
result = exp.explain(X_test, save_plots='./my_plots/')
# Produces: summary.png, feature_importance.png, heatmap.png, decision.png,
#           waterfall_0.png, force_0.png, ...
```

### Visualise Inline + Save

```python
result = exp.explain(X_test, visualize=True, save_plots='./my_plots/')
```

### Multiclass Plots

```python
result.plot_summary(class_index=2)
result.plot_waterfall(index=0, class_index=1)
result.shap_values_for_class(0)
```

---

## `explain()` Parameters

```python
result = exp.explain(
    X,                                  # pd.DataFrame or np.ndarray
    visualize=False,                    # show all plots inline
    save_plots=None,                    # directory to save PNG files
    compute_interactions=False,         # pairwise SHAP interactions (tree only)
    compute_confidence_intervals=False, # bootstrap CIs
    n_bootstrap=100,                    # CI bootstrap iterations
    ci_level=0.95,                      # confidence level
)
```

---

## Advanced Usage

### Interaction Effects

```python
exp = Explainer('xgb', model)
result = exp.explain(X_test, compute_interactions=True)

# Heatmap of interactions with 'mean radius'
result.plot_interaction_heatmap('mean radius', top_n=8)

# Raw interaction array: (n_observations, n_features, n_features)
print(result.interaction_effects.shape)
```

### Confidence Intervals

```python
result = exp.explain(X_test, compute_confidence_intervals=True, n_bootstrap=200)

result.plot_confidence_intervals(top_n=10)
# Or for a single observation:
result.plot_confidence_intervals(observation_index=5)
```

### Feature Groups

```python
exp = Explainer(
    'rf', model,
    feature_groups={
        'size':     ['mean radius', 'mean area', 'mean perimeter'],
        'texture':  ['mean texture', 'mean smoothness', 'mean symmetry'],
        'severity': ['mean concavity', 'mean concave points', 'worst radius'],
    },
)
result = exp.explain(X_test)

print(result.group_contributions)  # DataFrame (n_obs, n_groups)
result.plot_group_contributions()
```

### Override Explainer Backend

```python
# Force KernelExplainer even for a tree model (raises a UserWarning)
exp = Explainer('xgb', model, background_data=X_train, explainer_override='kernel')
```

### Linear Models

```python
from sklearn.linear_model import Ridge
from transpyrancy import Explainer

model = Ridge().fit(X_train, y_train)
# background_data is required for LinearExplainer
exp = Explainer('ridge', model, background_data=X_train, task='regression')
result = exp.explain(X_test)
```

---

## Running the Example Notebooks

```bash
pip install -e ".[dev]"
jupyter notebook notebooks/
```

- `classification_example.ipynb` — XGBoost on breast cancer; all features shown.
- `regression_example.ipynb` — LightGBM on California Housing; all features shown.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
# With coverage:
pytest --cov=explainer --cov-report=term-missing
```

---

## License

MIT
