"""
Objective functions for testing hyperparameter optimization.

Provides well-known test functions with known global optima.
"""

import numpy as np

from search_space import Dimension, SearchSpace


def rastrigin(config: dict[str, float]) -> float:
    """
    2D Rastrigin function - a common optimization test function.

    The Rastrigin function is highly multimodal with many local minima
    arranged in a regular lattice. This makes it challenging for
    optimization algorithms that can get stuck in local optima.

    f(x, y) = 20 + x^2 + y^2 - 10(cos(2*pi*x) + cos(2*pi*y))

    Properties:
        - Domain: x, y in [-5.12, 5.12]
        - Global minimum: f(0, 0) = 0
        - Many local minima at integer coordinates
        - MINIMIZE (lower is better)

    Args:
        config: Dict with keys "x" and "y" containing float values

    Returns:
        Objective value (lower is better, minimum is 0 at origin)

    Raises:
        KeyError: If "x" or "y" not in config

    Example:
        >>> rastrigin({"x": 0.0, "y": 0.0})
        0.0
        >>> rastrigin({"x": 1.0, "y": 1.0})  # Local minimum
        2.0
    """
    x = config["x"]
    y = config["y"]

    A = 10
    return float(
        2 * A
        + (x**2 - A * np.cos(2 * np.pi * x))
        + (y**2 - A * np.cos(2 * np.pi * y))
    )


def create_rastrigin_space() -> SearchSpace:
    """
    Create the standard search space for the 2D Rastrigin function.

    Returns:
        SearchSpace with x and y dimensions in [-5.12, 5.12]
    """
    return SearchSpace(
        [
            Dimension("x", "continuous", -5.12, 5.12),
            Dimension("y", "continuous", -5.12, 5.12),
        ]
    )
