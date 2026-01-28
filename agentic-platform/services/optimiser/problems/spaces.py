"""
Search spaces for different ML models.

Defines hyperparameter search spaces for various model types.
"""

from search_space import Dimension, SearchSpace

# XGBoost hyperparameter search space for classification
XGBOOST_SPACE = SearchSpace(
    [
        Dimension("learning_rate", "continuous", 0.01, 0.3, log_scale=True),
        Dimension("n_estimators", "integer", 50, 300),
        Dimension("max_depth", "integer", 3, 10),
        Dimension("min_child_weight", "integer", 1, 7),
        Dimension("subsample", "continuous", 0.6, 1.0),
        Dimension("colsample_bytree", "continuous", 0.6, 1.0),
        Dimension("gamma", "continuous", 0.0, 0.5),
    ]
)

# XGBoost hyperparameter search space for regression (includes regularization)
XGBOOST_REGRESSOR_SPACE = SearchSpace(
    [
        Dimension("learning_rate", "continuous", 0.01, 0.3, log_scale=True),
        Dimension("n_estimators", "integer", 50, 300),
        Dimension("max_depth", "integer", 3, 10),
        Dimension("min_child_weight", "integer", 1, 7),
        Dimension("subsample", "continuous", 0.6, 1.0),
        Dimension("colsample_bytree", "continuous", 0.6, 1.0),
        Dimension("gamma", "continuous", 0.0, 0.5),
        Dimension("reg_alpha", "continuous", 0.0, 1.0),
        Dimension("reg_lambda", "continuous", 0.1, 10.0, log_scale=True),
    ]
)
