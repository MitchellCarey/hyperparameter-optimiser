"""
Objective functions for ML problems.

Creates objective functions that train models and return scores
suitable for the minimization-based optimization framework.
"""

from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from problems.loaders import load_fraud, load_housing, load_titanic


def create_xgboost_objective(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Callable[[dict[str, Any]], float]:
    """
    Create an objective function for XGBoost classification.

    Returns NEGATIVE accuracy for minimization compatibility.
    (Lower internal score = better model)

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets

    Returns:
        Callable that takes config dict and returns -accuracy
    """

    def objective(config: dict[str, Any]) -> float:
        model = XGBClassifier(
            learning_rate=config["learning_rate"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_child_weight=config["min_child_weight"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            gamma=config["gamma"],
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Return negative accuracy for minimization
        # (lower is better in our framework)
        return -accuracy

    return objective


def create_titanic_objective_factory() -> Callable[[], Callable[[dict[str, Any]], float]]:
    """
    Factory that returns an objective factory for Titanic.

    This lazy-loads the data when the factory is called,
    allowing the ProblemConfig to be defined without loading data.

    Returns:
        A callable that, when called, returns the objective function
    """

    def factory() -> Callable[[dict[str, Any]], float]:
        X_train, X_val, y_train, y_val = load_titanic()
        return create_xgboost_objective(X_train, X_val, y_train, y_val)

    return factory


def create_fraud_objective(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Callable[[dict[str, Any]], float]:
    """
    Create an objective function for Credit Card Fraud detection.

    Returns NEGATIVE F1 score for minimization compatibility.
    F1 score is used instead of accuracy because the dataset is imbalanced.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets

    Returns:
        Callable that takes config dict and returns -F1_score
    """

    def objective(config: dict[str, Any]) -> float:
        model = XGBClassifier(
            learning_rate=config["learning_rate"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_child_weight=config["min_child_weight"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            gamma=config["gamma"],
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        # Return negative F1 for minimization
        return -f1

    return objective


def create_fraud_objective_factory() -> Callable[[], Callable[[dict[str, Any]], float]]:
    """
    Factory that returns an objective factory for Credit Card Fraud.

    Returns:
        A callable that, when called, returns the objective function
    """

    def factory() -> Callable[[dict[str, Any]], float]:
        X_train, X_val, y_train, y_val = load_fraud()
        return create_fraud_objective(X_train, X_val, y_train, y_val)

    return factory


def create_xgboost_regressor_objective(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Callable[[dict[str, Any]], float]:
    """
    Create an objective function for XGBoost regression.

    Returns RMSE (Root Mean Squared Error) which is naturally minimization-friendly.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets

    Returns:
        Callable that takes config dict and returns RMSE
    """

    def objective(config: dict[str, Any]) -> float:
        model = XGBRegressor(
            learning_rate=config["learning_rate"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_child_weight=config["min_child_weight"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            gamma=config["gamma"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        # RMSE is already minimization-friendly (lower is better)
        return rmse

    return objective


def create_housing_objective_factory() -> Callable[[], Callable[[dict[str, Any]], float]]:
    """
    Factory that returns an objective factory for California Housing.

    Returns:
        A callable that, when called, returns the objective function
    """

    def factory() -> Callable[[dict[str, Any]], float]:
        X_train, X_val, y_train, y_val = load_housing()
        return create_xgboost_regressor_objective(X_train, X_val, y_train, y_val)

    return factory
