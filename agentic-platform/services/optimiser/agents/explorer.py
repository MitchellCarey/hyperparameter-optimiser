"""
Explorer agents that map unknown territory in the search space.

Explorers prioritize coverage over exploitation, using various
sampling strategies to discover the landscape.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agents.base import Agent

if TYPE_CHECKING:
    from orchestration import Blackboard
    from search_space import SearchSpace


@dataclass
class RandomExplorer(Agent):
    """
    Simplest explorer: samples uniformly at random from the search space.

    This agent has no memory-based strategy - each proposal is independent.
    Useful as a baseline and for initial space coverage.

    When the agent finds a score in the top 20% of evaluated configurations,
    it can spawn a HillClimbExploiter child to refine that region.

    The agent dies after max_stagnation iterations without personal improvement.

    Attributes:
        id: Unique identifier (inherited)
        position: Last evaluated config (inherited)
        history: All (config, score) pairs (inherited)
        alive: Whether agent is still active (inherited)
        generation: Generation number in genealogy (inherited)
        parent_id: ID of parent agent (inherited)
        _seed: Optional seed for reproducible random sampling
        stagnation_counter: Iterations without personal improvement
        max_stagnation: Maximum stagnation before death

    Example:
        >>> from search_space import SearchSpace, Dimension
        >>> space = SearchSpace([Dimension("x", "continuous", 0, 1)])
        >>> agent = RandomExplorer(id="explorer-1")
        >>> config = agent.propose_next(space)
        >>> agent.update(config, 0.5)
        >>> len(agent.history)
        1
    """

    _seed: int | None = field(default=None, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    stagnation_counter: int = field(default=0, repr=False)
    max_stagnation: int = field(default=10, repr=False)
    _best_score: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random generator."""
        self._rng = np.random.default_rng(self._seed)

    def propose_next(
        self,
        search_space: "SearchSpace",
        blackboard: "Blackboard | None" = None,
    ) -> dict[str, float | int]:
        """
        Sample a random configuration from the search space.

        If a blackboard is provided, resamples if the config was already evaluated.

        Args:
            search_space: The space to sample from
            blackboard: Optional blackboard for duplicate checking

        Returns:
            A randomly sampled configuration
        """
        max_retries = 100
        for _ in range(max_retries):
            config = search_space.sample_random(self._rng)
            if blackboard is None or not blackboard.is_evaluated(config):
                return config
        # Return last sample if all retries exhausted (unlikely in continuous space)
        return config

    def update(
        self,
        config: dict[str, float | int],
        score: float,
        blackboard: "Blackboard | None" = None,
    ) -> None:
        """
        Record the evaluation result and update stagnation counter.

        Stagnation resets when agent finds a personal best.

        Args:
            config: The evaluated configuration
            score: The objective value
            blackboard: Optional blackboard to post the evaluation to
        """
        self.history.append((config, score))
        self.position = config

        # Track stagnation based on personal improvement
        if self._best_score is None or score < self._best_score:
            self._best_score = score
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if blackboard is not None:
            blackboard.post_evaluation(config, score, self.id)

    def should_die(self) -> bool:
        """
        RandomExplorer dies after max_stagnation iterations without improvement.

        Returns:
            True if stagnation_counter >= max_stagnation
        """
        return self.stagnation_counter >= self.max_stagnation

    def should_spawn(self, blackboard: "Blackboard") -> bool:
        """
        Spawn a child if the most recent evaluation is in the top 20% of scores.

        Requires at least 10 evaluations in the blackboard for meaningful
        percentile calculation.

        Args:
            blackboard: Blackboard to check score percentiles

        Returns:
            True if last evaluation score is in top 20%
        """
        if not self.history:
            return False

        # Need at least 10 evaluations for meaningful percentiles
        if len(blackboard.evaluated_configs) < 10:
            return False

        _, last_score = self.history[-1]

        # For minimization: top 20% means below 20th percentile
        threshold = blackboard.score_percentile(20)
        return last_score <= threshold

    def create_child(
        self,
        search_space: "SearchSpace",
        child_id: str,
    ) -> "Agent | None":
        """
        Create a HillClimbExploiter at the current position.

        Args:
            search_space: Search space for the child
            child_id: Unique identifier for the child

        Returns:
            HillClimbExploiter positioned at this agent's current location
        """
        from agents.exploiter import HillClimbExploiter

        if self.position is None:
            return None

        return HillClimbExploiter(
            id=child_id,
            initial_position=self.position.copy(),
            generation=self.generation + 1,
            parent_id=self.id,
        )
