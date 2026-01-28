"""
Blackboard shared memory system for agent coordination.

The Blackboard is the central hub where agents share discoveries:
- Evaluated configurations and scores
- Best results found so far
- Events for visualization and debugging
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from search_space import SearchSpace


@dataclass
class EvaluatedConfig:
    """
    A single evaluated configuration with metadata.

    Attributes:
        config: The hyperparameter configuration dict
        score: Objective value (lower is better for minimization)
        agent_id: ID of the agent that evaluated this config
        timestamp: Unix timestamp when evaluation was recorded
        position: Normalized [0-1] coordinates for visualization
    """

    config: dict[str, float | int]
    score: float
    agent_id: str
    timestamp: float
    position: list[float]


@dataclass
class Event:
    """
    An event in the optimization process.

    Used for visualization and debugging.

    Attributes:
        type: Event type ("spawn", "death", "evaluation", "new_best")
        agent_id: ID of the agent involved, or None for system events
        data: Event-specific data dict
        timestamp: Unix timestamp when event occurred
    """

    type: str
    agent_id: str | None
    data: dict[str, Any]
    timestamp: float


class Blackboard:
    """
    Central shared memory for agent coordination.

    The Blackboard stores all evaluated configurations, tracks the best
    result, and logs events for visualization. It also provides deduplication
    to avoid re-evaluating the same (or very similar) configurations.

    Attributes:
        evaluated_configs: All configurations that have been evaluated
        best_config: The best configuration found so far
        best_score: The best (lowest) score found so far
        events: Log of all events that have occurred

    Example:
        >>> from search_space import SearchSpace, Dimension
        >>> space = SearchSpace([Dimension("x", "continuous", 0, 1)])
        >>> bb = Blackboard(space)
        >>> bb.post_evaluation({"x": 0.5}, 0.25, "agent-1")
        True
        >>> bb.get_best().score
        0.25
    """

    def __init__(self, search_space: "SearchSpace") -> None:
        """
        Initialize the blackboard.

        Args:
            search_space: The search space for normalization
        """
        self._search_space = search_space
        self.evaluated_configs: list[EvaluatedConfig] = []
        self.best_config: EvaluatedConfig | None = None
        self.best_score: float | None = None
        self.events: list[Event] = []
        self._config_hashes: set[tuple[float, ...]] = set()

        # Analyst insights
        self.promising_regions: list[dict[str, Any]] = []
        self.robustness_warnings: list[dict[str, Any]] = []
        self.convergence_signal: dict[str, Any] = {}
        self.best_score_history: list[float] = []
        self.surrogate_r2: float | None = None

    def _config_to_hash(self, config: dict[str, float | int]) -> tuple[float, ...]:
        """
        Convert a config to a hashable tuple for deduplication.

        Uses rounded normalized values (3 decimal places) for approximate matching.

        Args:
            config: The configuration dict

        Returns:
            Tuple of rounded normalized values
        """
        normalized = self._search_space.normalize(config)
        return tuple(round(float(v), 3) for v in normalized)

    def is_evaluated(
        self, config: dict[str, float | int], tolerance: float = 0.001
    ) -> bool:
        """
        Check if a configuration (or one very close) has been evaluated.

        Uses hash-based lookup first (fast path), then distance check if needed.

        Args:
            config: The configuration to check
            tolerance: Maximum normalized Euclidean distance to consider as duplicate

        Returns:
            True if config (or similar) has been evaluated
        """
        # Fast path: exact hash match
        config_hash = self._config_to_hash(config)
        if config_hash in self._config_hashes:
            return True

        # Slow path: check distance to all evaluated points
        if not self.evaluated_configs:
            return False

        normalized = self._search_space.normalize(config)
        for evaluated in self.evaluated_configs:
            existing_normalized = np.array(evaluated.position)
            distance = float(np.linalg.norm(normalized - existing_normalized))
            if distance < tolerance:
                return True

        return False

    def post_evaluation(
        self, config: dict[str, float | int], score: float, agent_id: str
    ) -> bool:
        """
        Record a new evaluation result.

        Args:
            config: The evaluated configuration
            score: The objective value
            agent_id: ID of the evaluating agent

        Returns:
            True if this was a new evaluation, False if duplicate
        """
        # Check for duplicates
        if self.is_evaluated(config):
            return False

        # Create the evaluated config record
        timestamp = time.time()
        normalized = self._search_space.normalize(config)
        position = [float(v) for v in normalized]

        evaluated_config = EvaluatedConfig(
            config=config,
            score=score,
            agent_id=agent_id,
            timestamp=timestamp,
            position=position,
        )

        # Store it
        self.evaluated_configs.append(evaluated_config)
        self._config_hashes.add(self._config_to_hash(config))

        # Log evaluation event
        self.log_event(
            "evaluation",
            agent_id,
            {"config": config, "score": score, "position": position},
        )

        # Check if this is a new best
        if self.best_score is None or score < self.best_score:
            self.best_config = evaluated_config
            self.best_score = score
            self.log_event(
                "new_best",
                agent_id,
                {"config": config, "score": score, "position": position},
            )

        return True

    def get_best(self) -> EvaluatedConfig | None:
        """
        Get the best configuration found so far.

        Returns:
            The EvaluatedConfig with the lowest score, or None if no evaluations
        """
        return self.best_config

    def get_evaluated_positions(self) -> np.ndarray:
        """
        Get all evaluated positions in normalized [0,1] space.

        Useful for visualization and distance calculations.

        Returns:
            NumPy array of shape (n_evals, n_dims) with normalized positions
        """
        if not self.evaluated_configs:
            return np.array([]).reshape(0, self._search_space.n_dims)

        return np.array([ec.position for ec in self.evaluated_configs])

    def score_percentile(self, p: float) -> float:
        """
        Get the p-th percentile of all evaluated scores.

        Useful for determining if a score is "good" relative to history.

        Args:
            p: Percentile (0-100)

        Returns:
            The p-th percentile score

        Raises:
            ValueError: If no evaluations have been recorded
        """
        if not self.evaluated_configs:
            raise ValueError("No evaluations recorded yet")

        scores = [ec.score for ec in self.evaluated_configs]
        return float(np.percentile(scores, p))

    def log_event(
        self, event_type: str, agent_id: str | None, data: dict[str, Any]
    ) -> None:
        """
        Log an event.

        Args:
            event_type: Type of event (e.g., "spawn", "death", "evaluation", "new_best")
            agent_id: ID of the agent involved, or None for system events
            data: Event-specific data
        """
        event = Event(
            type=event_type,
            agent_id=agent_id,
            data=data,
            timestamp=time.time(),
        )
        self.events.append(event)

    def update_analyst_insights(
        self,
        promising_regions: list[dict[str, Any]],
        robustness_warnings: list[dict[str, Any]],
        convergence_signal: dict[str, Any],
        surrogate_r2: float | None = None,
    ) -> None:
        """
        Update blackboard with analyst insights.

        Args:
            promising_regions: List of high-potential regions
            robustness_warnings: List of potential robustness issues
            convergence_signal: Convergence analysis results
            surrogate_r2: R-squared score of surrogate model fit
        """
        self.promising_regions = promising_regions
        self.robustness_warnings = robustness_warnings
        self.convergence_signal = convergence_signal
        self.surrogate_r2 = surrogate_r2

        self.log_event(
            "analyst_update",
            None,
            {
                "n_promising_regions": len(promising_regions),
                "n_warnings": len(robustness_warnings),
                "convergence": convergence_signal,
                "surrogate_r2": surrogate_r2,
            },
        )

    def __repr__(self) -> str:
        n_evals = len(self.evaluated_configs)
        best_str = f", best={self.best_score:.4f}" if self.best_score is not None else ""
        return f"Blackboard(evals={n_evals}{best_str})"
