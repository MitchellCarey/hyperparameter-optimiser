"""
Problem registry for hyperparameter optimization.

Defines ProblemConfig dataclass and maintains a registry of available problems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from search_space import SearchSpace


class ProblemType(Enum):
    """Type of ML problem."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ProblemConfig:
    """
    Configuration for an optimization problem.

    Attributes:
        id: Unique identifier for the problem
        name: Human-readable name
        emoji: Emoji for display
        problem_type: Classification or regression
        search_space: The hyperparameter search space
        objective_factory: Callable that returns the objective function
        metric_name: Name of the metric (e.g., "Accuracy", "RMSE")
        minimize: If True, lower scores are better (for display purposes)
                  Note: Internally, objectives always return values where lower is better.
                  This flag controls how scores are displayed to users.
    """

    id: str
    name: str
    emoji: str
    problem_type: ProblemType
    search_space: "SearchSpace"
    objective_factory: Callable[[], Callable[[dict[str, Any]], float]]
    metric_name: str
    minimize: bool = True

    def display_score(self, internal_score: float) -> float:
        """
        Convert internal score to display score.

        For minimization problems, display = internal.
        For maximization problems (like accuracy), display = -internal.

        Args:
            internal_score: The score returned by the objective function

        Returns:
            Score suitable for display to users
        """
        return internal_score if self.minimize else -internal_score

    def format_score(self, internal_score: float) -> str:
        """
        Format score for display.

        Handles percentage formatting for accuracy metrics and
        appropriate precision for other metrics.

        Args:
            internal_score: The score returned by the objective function

        Returns:
            Formatted string for display
        """
        display = self.display_score(internal_score)
        if self.metric_name == "Accuracy":
            return f"{display * 100:.1f}%"
        elif self.metric_name == "F1 Score":
            return f"{display:.3f}"
        return f"{display:.4f}"


# Problem registry - populated by register_problems()
PROBLEMS: dict[str, ProblemConfig] = {}


def get_problem(problem_id: str) -> ProblemConfig:
    """
    Get a problem configuration by ID.

    Args:
        problem_id: The problem identifier (e.g., "titanic", "rastrigin")

    Returns:
        The ProblemConfig for the requested problem

    Raises:
        ValueError: If the problem ID is not found
    """
    if problem_id not in PROBLEMS:
        available = ", ".join(PROBLEMS.keys()) if PROBLEMS else "none"
        raise ValueError(f"Unknown problem: {problem_id}. Available: {available}")
    return PROBLEMS[problem_id]


def register_problem(config: ProblemConfig) -> None:
    """
    Register a problem in the registry.

    Args:
        config: The problem configuration to register
    """
    PROBLEMS[config.id] = config
