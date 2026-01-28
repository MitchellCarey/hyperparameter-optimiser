"""
Search space definition for hyperparameter optimization.

Provides:
- Dimension: A single hyperparameter with bounds and type
- SearchSpace: Collection of dimensions with sampling and normalization
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Dimension:
    """
    A single dimension (hyperparameter) in the search space.

    Attributes:
        name: Unique identifier for this dimension
        dim_type: Either "continuous" (float) or "integer" (int)
        low: Lower bound (inclusive)
        high: Upper bound (inclusive for integer, exclusive for continuous)
        log_scale: If True, sample uniformly in log space (useful for learning rates)

    Examples:
        >>> Dimension("learning_rate", "continuous", 0.001, 0.1, log_scale=True)
        >>> Dimension("n_estimators", "integer", 50, 500)
    """

    name: str
    dim_type: Literal["continuous", "integer"]
    low: float
    high: float
    log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate dimension parameters."""
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.log_scale and self.low <= 0:
            raise ValueError("log_scale requires positive bounds")
        if self.dim_type == "integer" and (
            not float(self.low).is_integer() or not float(self.high).is_integer()
        ):
            raise ValueError("integer dimensions require integer bounds")

    def sample(self, rng: np.random.Generator | None = None) -> float | int:
        """
        Sample a random value from this dimension.

        Args:
            rng: NumPy random generator (uses default if None)

        Returns:
            Sampled value of appropriate type
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.dim_type == "continuous":
            if self.log_scale:
                log_val = rng.uniform(np.log(self.low), np.log(self.high))
                return float(np.exp(log_val))
            return float(rng.uniform(self.low, self.high))
        else:  # integer
            return int(rng.integers(int(self.low), int(self.high) + 1))

    def normalize(self, value: float | int) -> float:
        """
        Normalize a value to [0, 1] range.

        Args:
            value: A value within this dimension's bounds

        Returns:
            Normalized value in [0, 1]
        """
        if self.log_scale:
            log_val = np.log(value)
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return float((log_val - log_low) / (log_high - log_low))
        return float((value - self.low) / (self.high - self.low))

    def denormalize(self, normalized: float) -> float | int:
        """
        Convert a [0, 1] normalized value back to dimension's range.

        Args:
            normalized: A value in [0, 1]

        Returns:
            Value in the dimension's original range
        """
        if self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            log_val = normalized * (log_high - log_low) + log_low
            result = float(np.exp(log_val))
        else:
            result = normalized * (self.high - self.low) + self.low

        if self.dim_type == "integer":
            return int(round(result))
        return result


class SearchSpace:
    """
    A collection of dimensions defining the hyperparameter search space.

    Provides methods for:
    - Random sampling of configurations
    - Validation of configurations
    - Normalization to/from unit hypercube [0,1]^n

    Attributes:
        dimensions: Dict mapping dimension names to Dimension objects
        dim_names: List of dimension names in consistent order
        n_dims: Number of dimensions

    Example:
        >>> space = SearchSpace([
        ...     Dimension("x", "continuous", -5.12, 5.12),
        ...     Dimension("y", "continuous", -5.12, 5.12),
        ... ])
        >>> config = space.sample_random()
        >>> space.validate(config)
        True
    """

    def __init__(self, dimensions: list[Dimension]) -> None:
        """
        Initialize search space from a list of dimensions.

        Args:
            dimensions: List of Dimension objects (must have unique names)
        """
        if not dimensions:
            raise ValueError("SearchSpace requires at least one dimension")

        self.dimensions: dict[str, Dimension] = {}
        for dim in dimensions:
            if dim.name in self.dimensions:
                raise ValueError(f"Duplicate dimension name: {dim.name}")
            self.dimensions[dim.name] = dim

        self.dim_names: list[str] = [d.name for d in dimensions]
        self.n_dims: int = len(dimensions)

    def sample_random(
        self, rng: np.random.Generator | None = None
    ) -> dict[str, float | int]:
        """
        Sample a random configuration from the search space.

        Args:
            rng: NumPy random generator (uses default if None)

        Returns:
            Dict mapping dimension names to sampled values
        """
        if rng is None:
            rng = np.random.default_rng()
        return {name: dim.sample(rng) for name, dim in self.dimensions.items()}

    def validate(self, config: dict[str, float | int]) -> bool:
        """
        Check if a configuration is valid (all dims present and in bounds).

        Args:
            config: Configuration dict to validate

        Returns:
            True if valid, False otherwise
        """
        if set(config.keys()) != set(self.dim_names):
            return False

        for name, value in config.items():
            dim = self.dimensions[name]
            if value < dim.low or value > dim.high:
                return False
            if dim.dim_type == "integer" and not isinstance(value, int):
                return False
        return True

    def normalize(self, config: dict[str, float | int]) -> np.ndarray:
        """
        Normalize a configuration to [0, 1]^n unit hypercube.

        Useful for visualization and distance calculations.

        Args:
            config: Configuration dict

        Returns:
            NumPy array of shape (n_dims,) with values in [0, 1]
        """
        return np.array(
            [self.dimensions[name].normalize(config[name]) for name in self.dim_names]
        )

    def denormalize(self, normalized: np.ndarray) -> dict[str, float | int]:
        """
        Convert normalized [0, 1]^n array back to configuration dict.

        Args:
            normalized: NumPy array of shape (n_dims,) with values in [0, 1]

        Returns:
            Configuration dict
        """
        return {
            name: self.dimensions[name].denormalize(normalized[i])
            for i, name in enumerate(self.dim_names)
        }

    def __repr__(self) -> str:
        dims_str = ", ".join(
            f"{d.name}[{d.low}, {d.high}]" for d in self.dimensions.values()
        )
        return f"SearchSpace({dims_str})"
