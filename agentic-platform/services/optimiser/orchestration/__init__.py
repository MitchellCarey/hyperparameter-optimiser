"""
Orchestration module for agent coordination.

Provides:
- Blackboard shared memory system
- LangGraph workflow for multi-agent optimization
- State definitions for swarm orchestration
"""

from orchestration.blackboard import Blackboard, EvaluatedConfig, Event
from orchestration.graph import build_swarm_graph, create_initial_state
from orchestration.state import EvalResult, PendingProposal, SwarmState

__all__ = [
    "Blackboard",
    "EvaluatedConfig",
    "Event",
    "SwarmState",
    "PendingProposal",
    "EvalResult",
    "build_swarm_graph",
    "create_initial_state",
]
