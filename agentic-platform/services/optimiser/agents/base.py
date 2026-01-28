"""
Base agent class for hyperparameter optimization.

All agents inherit from Agent and implement the propose_next and update methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration import Blackboard
    from search_space import SearchSpace


@dataclass
class Agent(ABC):
    """
    Abstract base class for optimization agents.

    An agent explores the hyperparameter search space by proposing
    configurations to evaluate and updating its internal state based
    on evaluation results.

    Attributes:
        id: Unique identifier for this agent
        position: Current position in search space (last evaluated config),
                  or None if no evaluations yet
        history: List of (config, score) tuples for all evaluations
        alive: Whether the agent is still active
        generation: Generation number in genealogy (0 for root agents)
        parent_id: ID of parent agent, or None for root agents

    Subclasses must implement:
        - propose_next(): Generate the next configuration to evaluate
        - update(): Update internal state after an evaluation

    Example:
        class MyAgent(Agent):
            def propose_next(self, search_space):
                return search_space.sample_random()

            def update(self, config, score):
                self.history.append((config, score))
                self.position = config
    """

    id: str
    position: dict[str, float | int] | None = None
    history: list[tuple[dict[str, float | int], float]] = field(default_factory=list)
    alive: bool = True
    generation: int = 0
    parent_id: str | None = None

    @abstractmethod
    def propose_next(
        self,
        search_space: "SearchSpace",
        blackboard: "Blackboard | None" = None,
    ) -> dict[str, float | int]:
        """
        Propose the next configuration to evaluate.

        Args:
            search_space: The search space to sample from
            blackboard: Optional blackboard for checking already-evaluated configs

        Returns:
            A configuration dict to evaluate
        """
        pass

    @abstractmethod
    def update(
        self,
        config: dict[str, float | int],
        score: float,
        blackboard: "Blackboard | None" = None,
    ) -> None:
        """
        Update agent state after evaluating a configuration.

        Args:
            config: The configuration that was evaluated
            score: The objective value (interpretation depends on objective)
            blackboard: Optional blackboard to post the evaluation to
        """
        pass

    def should_die(self) -> bool:
        """
        Determine if this agent should be terminated.

        Default implementation: agents never die.
        Subclasses can override with stagnation logic.

        Returns:
            True if agent should be terminated
        """
        return False

    def should_spawn(self, blackboard: "Blackboard") -> bool:
        """
        Determine if this agent should spawn a child agent.

        Default implementation: agents never spawn.
        Subclasses can override with score-based logic.

        Args:
            blackboard: Blackboard to check score percentiles

        Returns:
            True if agent should spawn a child
        """
        return False

    def create_child(
        self,
        search_space: "SearchSpace",
        child_id: str,
    ) -> "Agent | None":
        """
        Create a child agent to exploit a promising region.

        Default implementation: return None (no child).
        Subclasses override to return appropriate child agent type.

        Args:
            search_space: Search space for the child to operate in
            child_id: Unique identifier for the child agent

        Returns:
            A new Agent instance, or None if spawning not supported
        """
        return None

    def get_best(self) -> tuple[dict[str, float | int], float] | None:
        """
        Get the best configuration found by this agent.

        For minimization problems, "best" means lowest score.

        Returns:
            Tuple of (config, score) for best result, or None if no history
        """
        if not self.history:
            return None
        return min(self.history, key=lambda x: x[1])

    def __repr__(self) -> str:
        n_evals = len(self.history)
        status = "alive" if self.alive else "dead"
        gen_str = f", gen={self.generation}" if self.generation > 0 else ""
        return f"{self.__class__.__name__}(id={self.id!r}, evals={n_evals}, {status}{gen_str})"
