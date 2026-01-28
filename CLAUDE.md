# Agentic Hyperparameter Optimizer

A multi-agent swarm system for hyperparameter optimization with real-time visualization. Agents explore hyperparameter space using different strategies, share discoveries via a central blackboard, and converge on optimal configurations.

**One-liner:** Pick a prediction problem - fraud, housing prices, survival odds - and watch AI agents race to build the best model, live.

## Project Overview

**Goal:** Build a visual hyperparameter optimization system where AI agents compete to find the best model configuration for ML problems. Demonstrated on ML (Titanic, Fraud, Housing) but designed for trading strategy parameter optimization.

**Why it exists:** The ML demo is the proof of concept. The real target is optimizing trading strategy parameters (stop-loss, entry thresholds, position sizing) with robustness checks to avoid overfitting.

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Optimiser Service** | Python 3.11+, FastAPI, LangGraph, WebSocket |
| **Agent Orchestration** | LangGraph (state management, workflow) |
| **ML** | scikit-learn, XGBoost, numpy, pandas |
| **UI Service** | React 18, TypeScript, Recharts, D3.js, Tailwind |
| **Infrastructure** | Docker, docker-compose, nginx |
| **Testing** | pytest, React Testing Library |

## Project Structure

```
agentic-platform/
├── docker-compose.yml           # Runs all services
├── docker-compose.dev.yml       # Development overrides
├── .env.example
├── README.md
├── Makefile
│
├── services/
│   ├── optimiser/               # Core optimization engine (port 8001)
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── src/
│   │       ├── main.py          # FastAPI entrypoint
│   │       ├── api/
│   │       │   ├── routes.py    # REST endpoints
│   │       │   ├── websocket.py # WebSocket streaming
│   │       │   └── models.py    # Pydantic schemas
│   │       ├── core/
│   │       │   ├── config.py    # Environment settings
│   │       │   └── job_manager.py # Job lifecycle
│   │       ├── agents/
│   │       │   ├── base.py
│   │       │   ├── explorer.py
│   │       │   ├── exploiter.py
│   │       │   └── analyst.py
│   │       ├── orchestration/
│   │       │   ├── graph.py     # LangGraph workflow
│   │       │   ├── state.py     # Typed state
│   │       │   ├── blackboard.py
│   │       │   └── nodes.py
│   │       ├── search_space/
│   │       │   └── space.py
│   │       ├── problems/
│   │       │   ├── registry.py
│   │       │   ├── loaders.py
│   │       │   └── objectives.py
│   │       └── data/
│   │           ├── titanic.csv
│   │           └── creditcard_sampled.csv
│   │
│   ├── ui/                      # React dashboard (port 3000)
│   │   ├── Dockerfile
│   │   ├── package.json
│   │   ├── nginx.conf
│   │   └── src/
│   │       ├── App.tsx
│   │       ├── config.ts
│   │       ├── types/
│   │       │   └── swarm.ts
│   │       ├── hooks/
│   │       │   ├── useSwarmApi.ts
│   │       │   └── useSwarmSocket.ts
│   │       ├── components/
│   │       │   ├── Layout.tsx
│   │       │   ├── ProblemSelector.tsx
│   │       │   ├── Controls.tsx
│   │       │   ├── SearchSpaceViz.tsx
│   │       │   ├── ScoreChart.tsx
│   │       │   ├── AgentTable.tsx
│   │       │   ├── ActivityFeed.tsx
│   │       │   ├── StatsPanel.tsx
│   │       │   └── AnalystPanel.tsx
│   │       └── styles/
│   │
│   └── backtester/              # Future: trading backtester (port 8002)
│       └── ...
│
└── shared/
    └── schemas/                 # API contracts (reference only)
        ├── optimiser-api.json
        └── websocket-events.json
```

## Architecture

### Microservices

```
┌─────────────────┐      ┌─────────────────┐
│   Optimiser     │      │       UI        │
│   (FastAPI)     │◄────►│    (React)      │
│    :8001        │  WS  │    :3000        │
│                 │      │                 │
│ - Agents        │      │ - Dashboard     │
│ - Blackboard    │      │ - Controls      │
│ - LangGraph     │      │ - Charts        │
│ - Evaluator     │      │                 │
└─────────────────┘      └─────────────────┘
```

**Key rule:** Services communicate via REST/WebSocket only. No shared code.

### Optimiser API

```
POST /api/start          # Start optimization job
POST /api/stop           # Stop job
GET  /api/status/{id}    # Job status
GET  /api/problems       # Available problems
GET  /api/results/{id}   # Final results
WS   /ws/{id}            # Stream events
GET  /health             # Health check
```

### Agent Types

| Type | Role | Strategies |
|------|------|------------|
| **Explorer** | Map unknown territory | random, centroid_void, boundary |
| **Exploiter** | Refine promising regions | hill_climb, nelder_mead |
| **Analyst** | Build surrogate models, detect robustness | gaussian_process |

### Agent Lifecycle

```
Explorer spawns → Explores randomly → Finds good region → Spawns Exploiter
                                                              ↓
                                      Exploiter hill-climbs → Converges or dies
                                                              ↓
                                                        Best config recorded
```

### Blackboard (Shared Memory)

- `evaluated_configs` - All (config, score) pairs
- `best_config` - Current champion
- `promising_regions` - Analyst-identified areas
- `robustness_warnings` - Overfit alerts
- `events` - Full event log for visualization

## Development Commands

```bash
# Run everything (production)
docker-compose up

# Run everything (development with hot reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run single service
docker-compose up optimiser
docker-compose up ui

# Local development without Docker
cd services/optimiser && pip install -e . && uvicorn src.main:app --reload --port 8001
cd services/ui && npm install && npm start

# Tests
cd services/optimiser && pytest
cd services/ui && npm test

# Build images
docker-compose build

# Clean up
docker-compose down -v
```

## Code Style

- **Python:** Type hints, Pydantic models, async where appropriate
- **TypeScript:** Strict mode, interfaces matching backend models
- **General:** Atomic commits, tests for new functionality
- **Services:** No code sharing - REST/WebSocket contracts only

## Current Problems

| ID | Name | Type | Metric |
|----|------|------|--------|
| `titanic` | 🚢 Titanic Survival | Classification | Accuracy |
| `fraud` | 💳 Credit Card Fraud | Classification | F1 Score |
| `housing` | 🏠 California Housing | Regression | RMSE |

## Adding a New Service

1. Create `services/new-service/` with Dockerfile
2. Add to `docker-compose.yml`
3. Document API in `shared/schemas/`
4. No imports from other services

## Current Status

**Completed:**
- ✅ Core agent framework (Explorer, Exploiter, Analyst)
- ✅ LangGraph orchestration
- ✅ Blackboard shared memory
- ✅ Console visualization (Rich)
- ✅ Multi-dataset support (Titanic, Fraud, Housing)
- ✅ Robustness detection + convergence signals

**In Progress:**
- 🔄 Microservices refactor (Step 1: Scaffolding)

**Planned:**
- ⬚ FastAPI REST + WebSocket API
- ⬚ React dashboard
- ⬚ Docker deployment
- ⬚ Trading backtester integration
- ⬚ BYOD (Bring Your Own Dataset)

## Future Services

| Service | Purpose | Port |
|---------|---------|------|
| `backtester` | Trading strategy backtesting | 8002 |
| `data` | Market data API (Polygon wrapper) | 8003 |
| `scanner` | Minervini trend template scanner | 8004 |
| `alerts` | Notify on new setups | 8005 |