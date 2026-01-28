"""
Problem registration module.

Registers all built-in problems. Import this module to populate the registry.
"""

from evaluation.objectives import create_rastrigin_space, rastrigin
from problems.objectives import (
    create_fraud_objective_factory,
    create_housing_objective_factory,
    create_titanic_objective_factory,
)
from problems.registry import ProblemConfig, ProblemType, register_problem
from problems.spaces import XGBOOST_REGRESSOR_SPACE, XGBOOST_SPACE


def register_all_problems() -> None:
    """Register all built-in problems."""

    # Register Rastrigin (existing test function)
    register_problem(
        ProblemConfig(
            id="rastrigin",
            name="Rastrigin Function",
            emoji="📈",
            problem_type=ProblemType.REGRESSION,
            search_space=create_rastrigin_space(),
            objective_factory=lambda: rastrigin,
            metric_name="Score",
            minimize=True,
        )
    )

    # Register Titanic
    register_problem(
        ProblemConfig(
            id="titanic",
            name="Titanic Survival",
            emoji="🚢",
            problem_type=ProblemType.CLASSIFICATION,
            search_space=XGBOOST_SPACE,
            objective_factory=create_titanic_objective_factory(),
            metric_name="Accuracy",
            minimize=False,
        )
    )

    # Register Credit Card Fraud
    register_problem(
        ProblemConfig(
            id="fraud",
            name="Credit Card Fraud",
            emoji="💳",
            problem_type=ProblemType.CLASSIFICATION,
            search_space=XGBOOST_SPACE,
            objective_factory=create_fraud_objective_factory(),
            metric_name="F1 Score",
            minimize=False,
        )
    )

    # Register California Housing
    register_problem(
        ProblemConfig(
            id="housing",
            name="California Housing",
            emoji="🏠",
            problem_type=ProblemType.REGRESSION,
            search_space=XGBOOST_REGRESSOR_SPACE,
            objective_factory=create_housing_objective_factory(),
            metric_name="RMSE",
            minimize=True,
        )
    )


# Auto-register when this module is imported
register_all_problems()
