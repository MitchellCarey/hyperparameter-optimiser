"""
LangGraph workflow for multi-agent hyperparameter optimization.

Provides the graph construction and helper functions for running
the swarm optimization workflow.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from langgraph.graph import END, StateGraph

from orchestration.nodes import (
    agent_propose,
    analyst_step,
    check_convergence,
    evaluate_batch,
    initialize_swarm,
    log_iteration,
    manage_lifecycle,
    update_agents,
)
from orchestration.state import SwarmState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from orchestration.blackboard import Blackboard
    from search_space import SearchSpace


def should_continue(state: SwarmState) -> str:
    """
    Conditional edge: continue iteration loop or end.

    Returns:
        "continue" if not converged
        "end" if converged
    """
    if state["converged"]:
        return "end"
    return "continue"


def build_swarm_graph() -> CompiledStateGraph:
    """
    Construct the LangGraph workflow for swarm optimization.

    Graph structure:
        initialize_swarm
              |
              v
        agent_propose <----------------+
              |                        |
              v                        |
        evaluate_batch                 |
              |                        |
              v                        |
        update_agents                  |
              |                        |
              v                        |
        analyst_step                   |
              |                        |
              v                        |
        manage_lifecycle               |
              |                        |
              v                        |
        check_convergence              |
              |                        |
              v                        |
        log_iteration                  |
              |                        |
              v                        |
        (conditional) --- continue ----+
              |
              +-- end --> END

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with our state schema
    workflow = StateGraph(SwarmState)

    # Add all nodes
    workflow.add_node("initialize_swarm", initialize_swarm)
    workflow.add_node("agent_propose", agent_propose)
    workflow.add_node("evaluate_batch", evaluate_batch)
    workflow.add_node("update_agents", update_agents)
    workflow.add_node("analyst_step", analyst_step)
    workflow.add_node("manage_lifecycle", manage_lifecycle)
    workflow.add_node("check_convergence", check_convergence)
    workflow.add_node("log_iteration", log_iteration)

    # Set entry point
    workflow.set_entry_point("initialize_swarm")

    # Linear edges through the main pipeline
    workflow.add_edge("initialize_swarm", "agent_propose")
    workflow.add_edge("agent_propose", "evaluate_batch")
    workflow.add_edge("evaluate_batch", "update_agents")
    workflow.add_edge("update_agents", "analyst_step")
    workflow.add_edge("analyst_step", "manage_lifecycle")
    workflow.add_edge("manage_lifecycle", "check_convergence")
    workflow.add_edge("check_convergence", "log_iteration")

    # Conditional edge: loop back or end
    workflow.add_conditional_edges(
        "log_iteration",
        should_continue,
        {
            "continue": "agent_propose",
            "end": END,
        },
    )

    return workflow.compile()


def create_initial_state(
    search_space: SearchSpace,
    objective: Callable[[dict[str, Any]], float],
    blackboard: Blackboard,
    max_iterations: int = 100,
    min_explorers: int = 2,
) -> SwarmState:
    """
    Create the initial state dict for running the graph.

    This is separate from the graph so callers can customize.

    Args:
        search_space: SearchSpace instance
        objective: Callable[[dict], float] objective function
        blackboard: Blackboard instance
        max_iterations: Maximum iterations before stopping
        min_explorers: Minimum explorer population to maintain

    Returns:
        SwarmState dict ready for graph.invoke() or graph.stream()
    """
    return {
        "search_space": search_space,
        "objective": objective,
        "blackboard": blackboard,
        "max_iterations": max_iterations,
        "min_explorers": min_explorers,
        # These will be populated by initialize_swarm
        "agents": {},
        "iteration": 0,
        "converged": False,
        "agent_counter": 0,
        "pending_proposals": [],
        "eval_results": [],
        "events": [],
    }


def run_with_visualization(
    initial_state: SwarmState,
    visualizer: Any = None,
    delay: float = 0.1,
) -> SwarmState:
    """
    Run the swarm graph with optional console visualization.

    This function streams the graph execution and updates the visualizer
    in real-time using Rich's Live display.

    Args:
        initial_state: The initial SwarmState from create_initial_state()
        visualizer: Optional ConsoleVisualizer instance for real-time display
        delay: Seconds to pause between iterations for human readability

    Returns:
        The final SwarmState after optimization completes

    Example:
        >>> from visualization.console import ConsoleVisualizer
        >>> visualizer = ConsoleVisualizer(search_space)
        >>> final_state = run_with_visualization(initial_state, visualizer)
    """
    graph = build_swarm_graph()

    if visualizer is None:
        # No visualization - just run normally
        return graph.invoke(initial_state)

    # Import Rich's Live display
    from rich.live import Live

    # Track accumulated state (graph.stream yields partial updates)
    accumulated_state: SwarmState = dict(initial_state)  # type: ignore

    with Live(visualizer.build_layout(), refresh_per_second=4, screen=True) as live:
        last_iteration = -1

        for chunk in graph.stream(initial_state):
            # Merge updates from each node into accumulated state
            for node_name, node_output in chunk.items():
                for key, value in node_output.items():
                    if key == "events":
                        # Events accumulate (handled by reducer, but we track manually)
                        accumulated_state["events"] = (
                            accumulated_state.get("events", []) + value
                        )
                    else:
                        accumulated_state[key] = value  # type: ignore

            # Update visualization
            visualizer.update(accumulated_state)
            live.update(visualizer.build_layout())

            # Add delay between iterations (when iteration completes)
            current_iteration = accumulated_state.get("iteration", 0)
            if current_iteration > last_iteration:
                last_iteration = current_iteration
                time.sleep(delay)

    return accumulated_state
