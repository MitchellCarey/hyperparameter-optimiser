"""
Problems module for hyperparameter optimization.

Provides problem configurations, dataset loaders, search spaces,
and objective functions for various ML problems.
"""

from problems.registry import (
    PROBLEMS,
    ProblemConfig,
    ProblemType,
    get_problem,
    register_problem,
)

# Import to trigger registration of built-in problems
from problems import _register  # noqa: F401

__all__ = [
    "ProblemConfig",
    "ProblemType",
    "PROBLEMS",
    "get_problem",
    "register_problem",
]
