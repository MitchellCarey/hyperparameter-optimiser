# Agentic Hyperparameter Optimizer

A multi-agent swarm system for hyperparameter optimization with real-time visualization. Agents explore hyperparameter space using different strategies, share discoveries via a central blackboard, and converge on optimal configurations.

**Pick a prediction problem - fraud, housing prices, survival odds - and watch AI agents race to build the best model, live.**

## How It Works

Multiple AI agents collaborate to find optimal model configurations:

```
Explorer spawns → Explores randomly → Finds good region → Spawns Exploiter
                                                              ↓
                                      Exploiter hill-climbs → Converges or dies
                                                              ↓
                                                        Best config recorded
```

| Agent | Role | Strategies |
|-------|------|------------|
| **Explorer** | Map unknown territory | random, centroid_void, boundary |
| **Exploiter** | Refine promising regions | hill_climb, nelder_mead |
| **Analyst** | Build surrogate models, detect overfitting | gaussian_process |

Agents communicate through a shared **blackboard** - posting discoveries, warnings about overfitting, and promising regions to explore.

## Quick Start

```bash
# Clone and enter directory
git clone https://github.com/mitchellcarey/agentic-hyperparameter-optimiser.git
cd agentic-hyperparameter-optimiser/agentic-platform

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up
```

Then open http://localhost:3000 to watch the agents optimize.

## Demo Problems

| Problem | Type | Metric | Description |
|---------|------|--------|-------------|
| Titanic Survival | Classification | Accuracy | Predict passenger survival |
| Credit Card Fraud | Classification | F1 Score | Detect fraudulent transactions |
| California Housing | Regression | RMSE | Predict house prices |

## Architecture

```
┌─────────────────┐      ┌─────────────────┐
│   Optimiser     │      │       UI        │
│   (FastAPI)     │◄────►│    (React)      │
│    :8001        │  WS  │    :3000        │
│                 │      │                 │
│ - Agents        │      │ - Dashboard     │
│ - Blackboard    │      │ - Controls      │
│ - LangGraph     │      │ - Charts        │
└─────────────────┘      └─────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11+, FastAPI, LangGraph, WebSocket |
| **ML** | scikit-learn, XGBoost, numpy, pandas |
| **Frontend** | React 18, TypeScript, Recharts, D3.js, Tailwind |
| **Infrastructure** | Docker, docker-compose, nginx |

## Development

```bash
# Run with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or run services locally
cd agentic-platform/services/optimiser && pip install -e . && python main.py
cd agentic-platform/services/ui && npm install && npm run dev

# Tests
cd agentic-platform/services/optimiser && pytest
cd agentic-platform/services/ui && npm test
```

## Project Structure

```
agentic-platform/
├── docker-compose.yml
├── services/
│   ├── optimiser/          # Python backend (port 8001)
│   │   ├── agents/         # Explorer, Exploiter, Analyst
│   │   ├── orchestration/  # LangGraph workflow, blackboard
│   │   ├── problems/       # Dataset loaders, objectives
│   │   └── data/           # Sample datasets
│   └── ui/                 # React frontend (port 3000)
│       ├── components/     # Visualization components
│       └── hooks/          # WebSocket & API hooks
└── shared/schemas/         # API contracts
```

## Status

**Working:**
- Core agent framework (Explorer, Exploiter, Analyst)
- LangGraph orchestration with blackboard communication
- Console visualization with Rich
- Multi-dataset support

**In Progress:**
- Microservices refactor
- React dashboard

## License

MIT
