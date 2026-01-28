# Optimiser Service

FastAPI backend service for hyperparameter optimization using a multi-agent swarm architecture.

## Features

- Multi-agent optimization with Explorer, Exploiter, and Analyst agents
- Real-time WebSocket updates
- Blackboard shared memory pattern
- LangGraph workflow orchestration

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run locally
uvicorn src.main:app --reload --port 8001

# Run tests
pytest
```

## API

- `GET /health` - Health check endpoint
- `WS /ws` - WebSocket for real-time updates
