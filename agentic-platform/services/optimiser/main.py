"""
FastAPI WebSocket server for streaming hyperparameter optimization.

Provides:
- WebSocket endpoint for real-time streaming of optimization progress
- REST endpoint for listing available problems
- SwarmServer class for managing connections and running optimizations
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from orchestration.blackboard import Blackboard
from orchestration.graph import build_swarm_graph, create_initial_state
from problems import PROBLEMS, get_problem
from problems._register import register_all_problems
from serializers import (
    _convert_numpy_types,
    serialize_problem,
    serialize_state_snapshot,
)

if TYPE_CHECKING:
    from problems.registry import ProblemConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Hyperopt API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    print("Optimiser service")


@app.get("/health")
async def health():
    return {"status": "healthy"}


class SwarmServer:
    """
    Manages WebSocket connections and optimization runs.

    Handles:
    - Connection management (connect/disconnect)
    - Broadcasting messages to all connected clients
    - Running optimization and streaming progress
    - Stop requests
    """

    def __init__(self) -> None:
        self.connections: list[WebSocket] = []
        self.running: bool = False
        self._stop_requested: bool = False
        self._current_problem: ProblemConfig | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if websocket in self.connections:
            self.connections.remove(websocket)
            logger.info(
                f"Client disconnected. Total connections: {len(self.connections)}"
            )

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all connected clients."""
        if not self.connections:
            return

        text = json.dumps(message)
        disconnected: list[WebSocket] = []

        for ws in self.connections:
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    async def send_to(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)

    async def send_problems_list(self, websocket: WebSocket) -> None:
        """Send the list of available problems to a client."""
        register_all_problems()
        problems_data = [serialize_problem(p) for p in PROBLEMS.values()]
        await self.send_to(
            websocket,
            {
                "type": "problems_list",
                "problems": problems_data,
            },
        )

    async def run_optimization(self, problem_id: str) -> None:
        """
        Run an optimization and stream results to all connected clients.

        Args:
            problem_id: ID of the problem to optimize
        """
        if self.running:
            await self.broadcast(
                {"type": "error", "message": "Optimization already running"}
            )
            return

        self.running = True
        self._stop_requested = False

        try:
            register_all_problems()
            problem = get_problem(problem_id)
            self._current_problem = problem

            await self.broadcast(
                {
                    "type": "optimization_started",
                    "problem_id": problem_id,
                    "problem_name": problem.name,
                }
            )

            objective = problem.objective_factory()
            blackboard = Blackboard(problem.search_space)

            initial_state = create_initial_state(
                search_space=problem.search_space,
                objective=objective,
                blackboard=blackboard,
                max_iterations=100,
                min_explorers=2,
            )

            graph = build_swarm_graph()

            accumulated_state: dict[str, Any] = dict(initial_state)

            for chunk in graph.stream(initial_state):
                if self._stop_requested:
                    await self.broadcast({"type": "stopped", "reason": "user_requested"})
                    break

                for node_name, node_output in chunk.items():
                    for key, value in node_output.items():
                        if key == "events":
                            accumulated_state["events"] = (
                                accumulated_state.get("events", []) + value
                            )
                        else:
                            accumulated_state[key] = value

                for node_name, node_output in chunk.items():
                    for event in node_output.get("events", []):
                        # Convert numpy types to native Python types for JSON serialization
                        await self.broadcast(
                            _convert_numpy_types({
                                "type": "event",
                                **event,
                            })
                        )

                if "log_iteration" in chunk:
                    snapshot = serialize_state_snapshot(accumulated_state, problem)
                    await self.broadcast(snapshot)

                await asyncio.sleep(0.05)

            if not self._stop_requested:
                best = accumulated_state["blackboard"].get_best()
                await self.broadcast(
                    {
                        "type": "optimization_complete",
                        "problem_id": problem_id,
                        "best_score": best.score if best else None,
                        "best_config": best.config if best else None,
                        "total_evaluations": len(
                            accumulated_state["blackboard"].evaluated_configs
                        ),
                    }
                )

        except Exception as e:
            logger.exception("Error during optimization")
            await self.broadcast({"type": "error", "message": str(e)})
        finally:
            self.running = False
            self._current_problem = None

    def stop(self) -> None:
        """Request the current optimization to stop."""
        self._stop_requested = True


server = SwarmServer()


@app.get("/api/problems")
async def get_problems_endpoint() -> list[dict[str, Any]]:
    """REST endpoint to get available problems."""
    register_all_problems()
    return [serialize_problem(p) for p in PROBLEMS.values()]


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Get current server status."""
    return {
        "running": server.running,
        "connections": len(server.connections),
        "current_problem": (
            server._current_problem.id if server._current_problem else None
        ),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time optimization streaming.

    Commands:
        {"action": "start", "problem_id": "titanic"} - Start optimization
        {"action": "stop"} - Stop current optimization
        {"action": "get_problems"} - Request problems list
    """
    await server.connect(websocket)
    await server.send_problems_list(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
            except json.JSONDecodeError:
                await server.send_to(
                    websocket, {"type": "error", "message": "Invalid JSON"}
                )
                continue

            action = command.get("action")

            if action == "start":
                problem_id = command.get("problem_id", "titanic")
                asyncio.create_task(server.run_optimization(problem_id))

            elif action == "stop":
                server.stop()

            elif action == "get_problems":
                await server.send_problems_list(websocket)

            else:
                await server.send_to(
                    websocket,
                    {"type": "error", "message": f"Unknown action: {action}"},
                )

    except WebSocketDisconnect:
        server.disconnect(websocket)


def start_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
