"""
JSON serialization helpers for WebSocket streaming.

Converts SwarmState, Agent, Blackboard and related objects to JSON-serializable dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.base import Agent
    from orchestration.blackboard import Blackboard, EvaluatedConfig
    from orchestration.state import SwarmState
    from problems.registry import ProblemConfig
    from search_space.space import SearchSpace


def serialize_agent(agent: Agent) -> dict[str, Any]:
    """
    Convert an Agent to a JSON-serializable dict.

    Handles AnalystAgents specially since they don't have position/history.

    Args:
        agent: The agent to serialize

    Returns:
        Dict with agent data suitable for JSON encoding
    """
    from agents import AnalystAgent

    # Analysts don't have position or history from evaluation
    if isinstance(agent, AnalystAgent):
        return {
            "id": agent.id,
            "type": agent.__class__.__name__,
            "alive": agent.alive,
            "generation": agent.generation,
            "parent_id": agent.parent_id,
            "position": None,
            "history_length": 0,
            "best_score": None,
            "analysis_interval": agent.analysis_interval,
        }

    best = agent.get_best()
    return {
        "id": agent.id,
        "type": agent.__class__.__name__,
        "alive": agent.alive,
        "generation": agent.generation,
        "parent_id": agent.parent_id,
        "position": agent.position,
        "history_length": len(agent.history),
        "best_score": best[1] if best else None,
    }


def serialize_evaluated_config(ec: EvaluatedConfig) -> dict[str, Any]:
    """
    Convert an EvaluatedConfig to a JSON-serializable dict.

    Args:
        ec: The evaluated config to serialize

    Returns:
        Dict with evaluation data suitable for JSON encoding
    """
    return {
        "config": ec.config,
        "score": ec.score,
        "agent_id": ec.agent_id,
        "timestamp": ec.timestamp,
        "position": ec.position,
    }


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to native Python types
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def serialize_analyst_insights(bb: Blackboard) -> dict[str, Any]:
    """
    Serialize analyst insights from the blackboard.

    Args:
        bb: The blackboard containing analyst insights

    Returns:
        Dict with analyst insights suitable for JSON encoding
    """
    return _convert_numpy_types({
        "promising_regions": bb.promising_regions,
        "robustness_warnings": bb.robustness_warnings,
        "convergence_signal": bb.convergence_signal,
        "best_score_history": bb.best_score_history,
        "surrogate_r2": bb.surrogate_r2,
    })


def serialize_blackboard(bb: Blackboard) -> dict[str, Any]:
    """
    Convert a Blackboard to a JSON-serializable dict.

    Args:
        bb: The blackboard to serialize

    Returns:
        Dict with blackboard data suitable for JSON encoding
    """
    return {
        "evaluated_configs": [
            serialize_evaluated_config(ec) for ec in bb.evaluated_configs
        ],
        "best_score": bb.best_score,
        "best_config": bb.best_config.config if bb.best_config else None,
        "best_position": bb.best_config.position if bb.best_config else None,
        "analyst_insights": serialize_analyst_insights(bb),
    }


def serialize_search_space(ss: SearchSpace) -> dict[str, Any]:
    """
    Serialize SearchSpace metadata for frontend configuration.

    Args:
        ss: The search space to serialize

    Returns:
        Dict with search space metadata suitable for JSON encoding
    """
    return {
        "dimensions": [
            {
                "name": dim.name,
                "type": dim.dim_type,
                "low": dim.low,
                "high": dim.high,
                "log_scale": dim.log_scale,
            }
            for dim in ss.dimensions.values()
        ],
        "dim_names": ss.dim_names,
        "n_dims": ss.n_dims,
    }


def serialize_problem(problem: ProblemConfig) -> dict[str, Any]:
    """
    Serialize a ProblemConfig for the frontend.

    Args:
        problem: The problem configuration to serialize

    Returns:
        Dict with problem metadata suitable for JSON encoding
    """
    return {
        "id": problem.id,
        "name": problem.name,
        "emoji": problem.emoji,
        "metric_name": problem.metric_name,
        "minimize": problem.minimize,
        "problem_type": problem.problem_type.value,
    }


def serialize_state_snapshot(
    state: SwarmState,
    problem: ProblemConfig,
) -> dict[str, Any]:
    """
    Create a full state snapshot for streaming to the frontend.

    This is the main serialization function called after each iteration.

    Args:
        state: The current SwarmState
        problem: The problem configuration for display formatting

    Returns:
        Dict representing a complete state snapshot suitable for JSON encoding
    """
    return {
        "type": "state_snapshot",
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"],
        "converged": state["converged"],
        "agents": {
            agent_id: serialize_agent(agent)
            for agent_id, agent in state["agents"].items()
        },
        "blackboard": serialize_blackboard(state["blackboard"]),
        "search_space": serialize_search_space(state["search_space"]),
        "problem": serialize_problem(problem),
    }
