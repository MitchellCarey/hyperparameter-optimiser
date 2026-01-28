"""
SwarmState TypedDict for LangGraph orchestration.

Defines the state that flows through the swarm optimization workflow,
including configuration, agent pool, shared memory, and accumulated events.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from agents.base import Agent
from orchestration.blackboard import Blackboard
from search_space import SearchSpace


class PendingProposal(TypedDict):
    """A configuration proposed by an agent, awaiting evaluation."""

    agent_id: str
    config: dict[str, float | int]


class EvalResult(TypedDict):
    """Result of evaluating a proposed configuration."""

    agent_id: str
    config: dict[str, float | int]
    score: float


class SwarmState(TypedDict):
    """
    State flowing through the LangGraph swarm workflow.

    Immutable fields (set at initialization):
        search_space: The hyperparameter search space
        objective: Function mapping config -> score (lower is better)
        max_iterations: Maximum iteration count before stopping
        min_explorers: Minimum explorer population to maintain

    Mutable fields (updated by nodes):
        agents: Dict of agent_id -> Agent instance
        blackboard: Shared memory for coordination
        iteration: Current iteration number
        converged: Whether stopping criteria met
        agent_counter: Counter for generating unique agent IDs

    Transient fields (used between nodes within an iteration):
        pending_proposals: Configs awaiting evaluation
        eval_results: Scores from batch evaluation

    Accumulating fields (grow across iterations):
        events: List of dicts for visualization, uses operator.add reducer
    """

    # Immutable configuration
    search_space: SearchSpace
    objective: Callable[[dict[str, Any]], float]
    max_iterations: int
    min_explorers: int

    # Core mutable state
    agents: dict[str, Agent]
    blackboard: Blackboard
    iteration: int
    converged: bool
    agent_counter: int

    # Inter-node communication
    pending_proposals: list[PendingProposal]
    eval_results: list[EvalResult]

    # Accumulating events (uses operator.add reducer)
    events: Annotated[list[dict[str, Any]], operator.add]
