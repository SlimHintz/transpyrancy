"""
transpyrancy — Transparent, SHAP-based ML explainability, batteries included.

Quick start::

    from transpyrancy import Explainer

    exp = Explainer('xgb', trained_model, background_data=X_train)
    result = exp.explain(X_test)

    print(result)               # rich text summary
    print(result.aggregate)     # global feature importance
    result.plot_summary()       # beeswarm plot
    result.plot_waterfall(0)    # single-observation waterfall
"""
from transpyrancy.core import Explainer
from transpyrancy.detector import ExplainerType
from transpyrancy.result import ExplainResult

__all__ = ["Explainer", "ExplainResult", "ExplainerType"]
__version__ = "0.1.0"
