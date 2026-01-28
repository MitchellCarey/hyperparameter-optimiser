"""
LangGraph node functions for swarm orchestration.

Each node is a function that takes SwarmState and returns a partial state update.
Nodes are executed in sequence by the workflow graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents import AnalystAgent, HillClimbExploiter, RandomExplorer
from orchestration.state import EvalResult, PendingProposal

if TYPE_CHECKING:
    from orchestration.state import SwarmState


def initialize_swarm(state: SwarmState) -> dict[str, Any]:
    """
    Create initial agent population.

    Creates min_explorers RandomExplorer agents with sequential IDs,
    plus one AnalystAgent to observe and provide insights.
    Sets iteration=0 and converged=False.

    Returns:
        Partial state with agents, iteration, converged, agent_counter, events
    """
    agents: dict[str, RandomExplorer | HillClimbExploiter | AnalystAgent] = {}
    events: list[dict[str, Any]] = []
    agent_counter = 0

    n_initial = state.get("min_explorers", 2)

    # Create initial explorers
    for i in range(n_initial):
        agent_counter += 1
        agent_id = f"explorer-{agent_counter}"
        agent = RandomExplorer(
            id=agent_id,
            _seed=42 + i,
            max_stagnation=10,
        )
        agents[agent_id] = agent
        events.append(
            {
                "type": "spawn",
                "agent_id": agent_id,
                "agent_type": "RandomExplorer",
                "generation": 0,
                "message": f"[initialize] Created explorer {agent_id}",
            }
        )

    # Create one analyst to observe and provide insights
    agent_counter += 1
    analyst_id = f"analyst-{agent_counter}"
    analyst = AnalystAgent(
        id=analyst_id,
        analysis_interval=10,
        min_samples_for_analysis=15,
    )
    agents[analyst_id] = analyst
    events.append(
        {
            "type": "spawn",
            "agent_id": analyst_id,
            "agent_type": "AnalystAgent",
            "generation": 0,
            "message": f"[initialize] Created analyst {analyst_id}",
        }
    )

    events.append(
        {
            "type": "system",
            "message": f"[initialize] Created {n_initial} explorers + 1 analyst",
        }
    )

    return {
        "agents": agents,
        "iteration": 0,
        "converged": False,
        "agent_counter": agent_counter,
        "events": events,
    }


def agent_propose(state: SwarmState) -> dict[str, Any]:
    """
    Each alive agent proposes its next configuration.

    Iterates through alive agents, calls propose_next(), and collects
    proposals into pending_proposals list.

    Note: AnalystAgents are skipped as they do not propose configurations.

    Returns:
        Partial state with pending_proposals, events
    """
    pending_proposals: list[PendingProposal] = []
    events: list[dict[str, Any]] = []

    search_space = state["search_space"]
    blackboard = state["blackboard"]
    agents = state["agents"]

    # Skip AnalystAgents - they do not propose configurations
    alive_agents = {
        aid: a
        for aid, a in agents.items()
        if a.alive and not isinstance(a, AnalystAgent)
    }

    for agent_id, agent in alive_agents.items():
        config = agent.propose_next(search_space, blackboard)
        pending_proposals.append(
            {
                "agent_id": agent_id,
                "config": config,
            }
        )

    events.append(
        {
            "type": "system",
            "message": f"[propose] {len(pending_proposals)} agents proposing configs",
        }
    )

    return {
        "pending_proposals": pending_proposals,
        "events": events,
    }


def evaluate_batch(state: SwarmState) -> dict[str, Any]:
    """
    Evaluate all pending proposals.

    Calls the objective function for each proposed config.

    Returns:
        Partial state with eval_results, events
    """
    objective = state["objective"]
    pending_proposals = state["pending_proposals"]

    eval_results: list[EvalResult] = []
    events: list[dict[str, Any]] = []

    for proposal in pending_proposals:
        score = objective(proposal["config"])
        eval_results.append(
            {
                "agent_id": proposal["agent_id"],
                "config": proposal["config"],
                "score": score,
            }
        )

    events.append(
        {
            "type": "system",
            "message": f"[evaluate] Batch of {len(eval_results)} evaluations complete",
        }
    )

    return {
        "eval_results": eval_results,
        "events": events,
    }


def update_agents(state: SwarmState) -> dict[str, Any]:
    """
    Update each agent with its evaluation result.

    Calls agent.update() which:
    - Appends to agent's history
    - Posts evaluation to blackboard
    - Updates stagnation counter

    Note: Agents are mutated in-place. We return the same agents dict
    to signal the state has changed.

    Returns:
        Partial state with agents, blackboard, events
    """
    agents = state["agents"]
    blackboard = state["blackboard"]
    eval_results = state["eval_results"]

    events: list[dict[str, Any]] = []

    for result in eval_results:
        agent = agents[result["agent_id"]]
        agent.update(result["config"], result["score"], blackboard)

    events.append(
        {
            "type": "system",
            "message": "[update] Agents updated",
        }
    )

    return {
        "agents": agents,
        "blackboard": blackboard,
        "events": events,
    }


def manage_lifecycle(state: SwarmState) -> dict[str, Any]:
    """
    Handle agent spawning and death.

    Process:
    1. Check each agent for spawn condition (found top 20% score)
    2. Check each agent for death condition (stagnation)
    3. Ensure minimum explorer population

    Spawning is processed before death so a dying agent can still spawn.

    Returns:
        Partial state with agents, agent_counter, events
    """
    agents = state["agents"].copy()  # Copy since we're adding new agents
    blackboard = state["blackboard"]
    search_space = state["search_space"]
    agent_counter = state["agent_counter"]
    min_explorers = state.get("min_explorers", 2)

    events: list[dict[str, Any]] = []
    lifecycle_changes: list[str] = []

    # Process spawns first (collect then execute to avoid iteration issues)
    pending_spawns: list[tuple[str, dict[str, float | int]]] = []

    for agent_id, agent in list(agents.items()):
        if not agent.alive:
            continue

        if agent.should_spawn(blackboard):
            if agent.position:
                pending_spawns.append((agent_id, agent.position.copy()))

    # Execute spawns
    for parent_id, position in pending_spawns:
        parent = agents[parent_id]
        agent_counter += 1
        child_id = f"exploiter-{agent_counter}"
        child = parent.create_child(search_space, child_id)

        if child is not None:
            agents[child_id] = child
            pos_str = ", ".join(f"{k}={v:.3f}" for k, v in position.items())
            lifecycle_changes.append(f"{parent_id} spawned {child_id}")
            events.append(
                {
                    "type": "spawn",
                    "agent_id": child_id,
                    "parent_id": parent_id,
                    "agent_type": child.__class__.__name__,
                    "generation": child.generation,
                    "position": position,
                    "message": f"[lifecycle] {parent_id} spawned {child_id} at ({pos_str})",
                }
            )

    # Process deaths
    for agent_id, agent in list(agents.items()):
        if not agent.alive:
            continue

        if agent.should_die():
            agent.alive = False
            best = agent.get_best()
            best_score = best[1] if best else None

            # Determine death reason
            death_reason = "stagnation"
            if hasattr(agent, "step_size") and hasattr(agent, "min_step_size"):
                if agent.step_size < agent.min_step_size:
                    death_reason = "converged"

            lifecycle_changes.append(f"{agent_id} died ({death_reason})")
            events.append(
                {
                    "type": "death",
                    "agent_id": agent_id,
                    "reason": death_reason,
                    "final_best": best_score,
                    "evaluations": len(agent.history),
                    "message": f"[lifecycle] {agent_id} died ({death_reason})",
                }
            )

    # Maintain minimum explorer population
    alive_explorers = sum(
        1 for a in agents.values() if a.alive and isinstance(a, RandomExplorer)
    )

    while alive_explorers < min_explorers:
        agent_counter += 1
        agent_id = f"explorer-{agent_counter}"
        agent = RandomExplorer(
            id=agent_id,
            _seed=42 + agent_counter,
            max_stagnation=10,
        )
        agents[agent_id] = agent
        alive_explorers += 1
        lifecycle_changes.append(f"spawned {agent_id} (population maintenance)")
        events.append(
            {
                "type": "spawn",
                "agent_id": agent_id,
                "agent_type": "RandomExplorer",
                "generation": 0,
                "reason": "population_maintenance",
                "message": f"[lifecycle] Spawned {agent_id} (population maintenance)",
            }
        )

    # Summary message
    if lifecycle_changes:
        summary = "; ".join(lifecycle_changes)
    else:
        summary = "No changes"
    events.append(
        {
            "type": "system",
            "message": f"[lifecycle] {summary}",
        }
    )

    return {
        "agents": agents,
        "agent_counter": agent_counter,
        "events": events,
    }


def check_convergence(state: SwarmState) -> dict[str, Any]:
    """
    Check if optimization should stop.

    Convergence conditions:
    1. Max iterations reached
    2. All agents dead (and min_explorers respawn failed somehow)
    3. Analyst detected convergence (diminishing returns)

    Returns:
        Partial state with converged, events
    """
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    agents = state["agents"]
    blackboard = state["blackboard"]

    events: list[dict[str, Any]] = []
    converged = False
    convergence_reason = None

    # Check max iterations (iteration is 0-indexed, incremented after this)
    if iteration + 1 >= max_iterations:
        converged = True
        convergence_reason = "max_iterations"

    # Check if all agents dead (shouldn't happen with population maintenance)
    alive_count = sum(1 for a in agents.values() if a.alive)
    if alive_count == 0:
        converged = True
        convergence_reason = "all_agents_dead"

    # Check analyst convergence signal (optional early stopping)
    if not converged and blackboard.convergence_signal.get("suggested_stop"):
        converged = True
        convergence_reason = "analyst_convergence"

    if converged:
        events.append(
            {
                "type": "convergence",
                "reason": convergence_reason,
                "message": f"[converged] {convergence_reason.replace('_', ' ').title()}",
            }
        )

    return {
        "converged": converged,
        "events": events,
    }


def log_iteration(state: SwarmState) -> dict[str, Any]:
    """
    Emit iteration summary and increment counter.

    This is the final node in each iteration loop.

    Returns:
        Partial state with iteration (incremented), events
    """
    iteration = state["iteration"]
    agents = state["agents"]
    blackboard = state["blackboard"]

    best = blackboard.get_best()
    best_score = best.score if best else None
    alive_count = sum(1 for a in agents.values() if a.alive)

    best_str = f"{best_score:.4f}" if best_score is not None else "N/A"

    events: list[dict[str, Any]] = [
        {
            "type": "iteration_complete",
            "iteration": iteration + 1,
            "best_score": best_score,
            "alive_agents": alive_count,
            "total_evaluations": len(blackboard.evaluated_configs),
            "message": f"[iteration] Iteration {iteration + 1} complete, best={best_str}, agents={alive_count}",
        }
    ]

    return {
        "iteration": iteration + 1,
        "events": events,
    }


def analyst_step(state: SwarmState) -> dict[str, Any]:
    """
    Run analyst agents to analyze the optimization landscape.

    Analysts:
    - Build surrogate models from evaluated configurations
    - Identify promising unexplored regions
    - Detect robustness issues (isolated peaks, unstable regions)
    - Monitor convergence (diminishing returns)

    Insights are stored in the blackboard for use by visualization
    and the convergence checker.

    Returns:
        Partial state with blackboard (updated with insights), events
    """
    agents = state["agents"]
    blackboard = state["blackboard"]
    search_space = state["search_space"]
    events: list[dict[str, Any]] = []

    # Find all analyst agents
    analysts = {
        aid: a
        for aid, a in agents.items()
        if isinstance(a, AnalystAgent) and a.alive
    }

    if not analysts:
        return {"events": events}

    # Run analysis for each analyst that's ready
    for agent_id, analyst in analysts.items():
        if not analyst.should_analyze(blackboard):
            continue

        result = analyst.analyze(blackboard, search_space)

        # Update blackboard with insights
        blackboard.update_analyst_insights(
            promising_regions=result.promising_regions,
            robustness_warnings=result.robustness_warnings,
            convergence_signal=result.convergence_signal,
            surrogate_r2=result.surrogate_r2,
        )

        # Log analysis event
        r2_str = f"{result.surrogate_r2:.3f}" if result.surrogate_r2 else "N/A"
        events.append(
            {
                "type": "analyst_insight",
                "agent_id": agent_id,
                "n_promising_regions": len(result.promising_regions),
                "n_warnings": len(result.robustness_warnings),
                "convergence": result.convergence_signal,
                "surrogate_r2": result.surrogate_r2,
                "message": f"[analyst] {agent_id}: {len(result.promising_regions)} promising regions, R\u00b2={r2_str}",
            }
        )

        # Add warning events
        for warning in result.robustness_warnings:
            severity = warning.get("severity", "medium")
            events.append(
                {
                    "type": "robustness_warning",
                    "agent_id": agent_id,
                    "warning_type": warning["type"],
                    "severity": severity,
                    "message": f"[analyst] \u26a0 {warning['message']}",
                }
            )

    return {
        "blackboard": blackboard,
        "events": events,
    }
