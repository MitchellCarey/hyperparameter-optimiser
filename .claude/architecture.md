# Agentic Hyperparameter Optimizer

**One-liner:** Pick a prediction problem - fraud, housing prices, survival odds - and watch AI agents race to build the best model, live.

A multi-agent swarm that explores hyperparameter space using different search strategies, shares discoveries, and converges on optimal configurations. Real-time visualization shows agents exploring, communicating, and the landscape being mapped.

## Live Demo Flow

1. User lands on the dashboard
2. Picks a problem from the dropdown:
   - 🚢 **Titanic Survival** - Predict who survives (classification)
   - 💳 **Credit Card Fraud** - Detect fraudulent transactions (classification)  
   - 🏠 **California Housing** - Predict house prices (regression)
3. Hits "Start Swarm"
4. Watches agents optimise in real-time
5. Gets the best model config + performance metrics

## Concept

Instead of traditional grid search, random search, or Bayesian optimization run as a single process, this system deploys **autonomous agents** with different exploration strategies. Agents:

- Explore regions of hyperparameter space independently
- Share promising findings to a central blackboard
- Adapt their strategies based on collective knowledge
- Spawn child agents in promising regions
- Die off when exploring barren territory

The visualization shows this as a living system - agents moving through parameter space, trails showing their paths, heat maps emerging as the landscape is mapped.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VISUALIZATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  2D/3D      │  │  Agent      │  │  Score      │  │  Agent     │ │
│  │  Search     │  │  Activity   │  │  Timeline   │  │  Genealogy │ │
│  │  Space      │  │  Feed       │  │  Chart      │  │  Tree      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Spawns initial agent population                          │   │
│  │  • Monitors convergence criteria                            │   │
│  │  • Manages agent lifecycle (spawn/kill)                     │   │
│  │  • Broadcasts important discoveries                         │   │
│  │  • Decides when to stop                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BLACKBOARD (Shared Memory)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Evaluated   │  │  Promising   │  │  Landscape Model         │  │
│  │  Configs     │  │  Regions     │  │  (Surrogate/Heatmap)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   EXPLORER    │       │   EXPLOITER   │       │   ANALYST     │
│    AGENTS     │       │    AGENTS     │       │    AGENTS     │
│               │       │               │       │               │
│ • Random      │       │ • Local       │       │ • Pattern     │
│ • Latin Hyper │       │   refinement  │       │   detection   │
│ • Boundary    │       │ • Gradient    │       │ • Surrogate   │
│   probing     │       │   approxim.   │       │   modeling    │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EVALUATION ENGINE                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Trains model with given hyperparameters                  │   │
│  │  • Returns validation score                                 │   │
│  │  • Caches results                                           │   │
│  │  • Supports parallel evaluation                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TARGET MODEL                                 │
│           (XGBoost / Random Forest / Neural Net / etc.)             │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Types & Strategies

### Problem Configurations

The framework ships with three bundled problems. Each defines a dataset, objective metric, and appropriate model.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable
import numpy as np

class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

@dataclass
class ProblemConfig:
    id: str
    name: str
    description: str
    emoji: str
    problem_type: ProblemType
    metric_name: str
    metric_direction: str  # "maximize" or "minimize"
    dataset_loader: Callable
    
PROBLEM_REGISTRY = {
    "titanic": ProblemConfig(
        id="titanic",
        name="Titanic Survival",
        description="Predict passenger survival based on class, age, fare, etc.",
        emoji="🚢",
        problem_type=ProblemType.CLASSIFICATION,
        metric_name="Accuracy",
        metric_direction="maximize",
        dataset_loader=load_titanic,
    ),
    "fraud": ProblemConfig(
        id="fraud", 
        name="Credit Card Fraud",
        description="Detect fraudulent transactions from anonymized features.",
        emoji="💳",
        problem_type=ProblemType.CLASSIFICATION,
        metric_name="F1 Score",  # Better for imbalanced
        metric_direction="maximize",
        dataset_loader=load_fraud,
    ),
    "housing": ProblemConfig(
        id="housing",
        name="California Housing",
        description="Predict median house prices from location and demographics.",
        emoji="🏠",
        problem_type=ProblemType.REGRESSION,
        metric_name="RMSE",
        metric_direction="minimize",
        dataset_loader=load_housing,
    ),
}
```

### Dataset Loaders

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_titanic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Titanic dataset from bundled CSV.
    Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    Target: Survived (0/1)
    """
    df = pd.read_csv("data/titanic.csv")
    
    # Basic preprocessing
    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
    
    X = df.drop("Survived", axis=1).values
    y = df["Survived"].values
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_fraud() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Credit Card Fraud dataset (sampled for speed).
    Features: V1-V28 (PCA transformed), Amount
    Target: Class (0=legit, 1=fraud)
    
    Note: Dataset is heavily imbalanced (~0.17% fraud).
    We stratify split and use F1 as metric.
    """
    df = pd.read_csv("data/creditcard.csv")
    
    # Sample for faster demo (keep all fraud, sample non-fraud)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=5000, random_state=42)
    df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    
    X = df.drop(["Class", "Time"], axis=1).values
    y = df["Class"].values
    
    # Scale Amount
    X[:, -1] = StandardScaler().fit_transform(X[:, -1].reshape(-1, 1)).flatten()
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_housing() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load California Housing dataset (built into sklearn).
    Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long
    Target: Median house value (in $100k)
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Scale features
    X = StandardScaler().fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### Objective Factory

```python
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import numpy as np

def create_objective(problem_config: ProblemConfig) -> Callable[[dict], float]:
    """
    Factory that creates the objective function for a given problem.
    Returns a function that takes hyperparameters and returns a score.
    """
    X_train, X_val, y_train, y_val = problem_config.dataset_loader()
    
    def objective(config: dict) -> float:
        # Build model based on problem type
        if problem_config.problem_type == ProblemType.CLASSIFICATION:
            model = XGBClassifier(
                **config,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if problem_config.metric_name == "Accuracy":
                score = accuracy_score(y_val, y_pred)
            elif problem_config.metric_name == "F1 Score":
                score = f1_score(y_val, y_pred)
            
        else:  # Regression
            model = XGBRegressor(
                **config,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            # Invert RMSE so higher is always better (for agent logic)
            score = -rmse if problem_config.metric_direction == "minimize" else rmse
        
        return score
    
    return objective, (X_train, X_val, y_train, y_val)

# Search space is the same for XGBoost regardless of problem
DEFAULT_XGBOOST_SPACE = SearchSpace([
    Dimension("learning_rate", "continuous", 0.001, 0.3, log_scale=True),
    Dimension("n_estimators", "integer", 50, 500),
    Dimension("max_depth", "integer", 3, 12),
    Dimension("subsample", "continuous", 0.5, 1.0),
    Dimension("colsample_bytree", "continuous", 0.5, 1.0),
    Dimension("min_child_weight", "integer", 1, 10),
    Dimension("gamma", "continuous", 0, 5),
    Dimension("reg_alpha", "continuous", 0, 1, log_scale=False),
    Dimension("reg_lambda", "continuous", 0.1, 10, log_scale=True),
])
```

### 1. Explorer Agents
**Goal:** Map unknown territory, maximize coverage

```python
class ExplorerAgent:
    strategies = [
        "random",           # Pure random sampling
        "latin_hypercube",  # Space-filling design
        "boundary",         # Probe edges of valid space
        "diagonal",         # Traverse across dimensions
        "centroid_void",    # Target largest unexplored regions
    ]
    
    behavior:
        - Start in unexplored region
        - Sample points, log results to blackboard
        - If promising region found → signal for Exploiter spawn
        - If barren for N iterations → move to new region or die
        - Occasionally check blackboard for "avoid" zones
```

### 2. Exploiter Agents
**Goal:** Refine promising regions, find local optima

```python
class ExploiterAgent:
    strategies = [
        "hill_climb",       # Greedy local search
        "nelder_mead",      # Simplex method
        "gradient_approx",  # Finite difference gradients
        "binary_refine",    # Bisection in each dimension
    ]
    
    behavior:
        - Spawned at promising coordinates
        - Local search to find optimum
        - Shrinking step size over iterations
        - Report best found to blackboard
        - Die when improvement < threshold for N steps
```

### 3. Analyst Agents
**Goal:** Build understanding, guide other agents

```python
class AnalystAgent:
    strategies = [
        "surrogate_model",  # Fit GP/RF to predict scores
        "pattern_detect",   # Find correlations (e.g., "high LR + low batch = bad")
        "region_ranker",    # Rank unexplored regions by predicted promise
    ]
    
    behavior:
        - Periodically read all results from blackboard
        - Build/update surrogate model
        - Identify promising unexplored regions
        - Post predictions to guide Explorers
        - Detect diminishing returns → signal convergence
```

## Blackboard (Shared Memory)

The blackboard is the central nervous system - all agents read from and write to it.

```python
class Blackboard:
    """Shared memory for agent communication."""
    
    # Raw data
    evaluated_configs: list[EvaluatedConfig]  # All (config, score) pairs
    
    # Derived insights
    promising_regions: list[Region]           # High-score areas
    avoid_regions: list[Region]               # Known barren areas
    best_config: EvaluatedConfig              # Current champion
    best_score_history: list[float]           # Score over time
    
    # Surrogate model
    surrogate: GaussianProcessRegressor | None
    acquisition_surface: np.ndarray | None    # For visualization
    
    # Agent coordination
    active_agents: dict[str, AgentState]
    pending_evaluations: set[ConfigHash]      # Avoid duplicates
    
    # Events (for visualization)
    event_log: list[Event]  # Agent spawns, deaths, discoveries
```

## State Machine: Agent Lifecycle

```
                    ┌─────────┐
                    │  SPAWN  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
         ┌─────────│  INIT   │
         │         └────┬────┘
         │              │
         │              ▼
         │         ┌─────────┐
         │    ┌───▶│ EXPLORE │◀───┐
         │    │    └────┬────┘    │
         │    │         │         │
         │    │         ▼         │
         │    │    ┌─────────┐    │
         │    │    │EVALUATE │    │
         │    │    └────┬────┘    │
         │    │         │         │
         │    │         ▼         │
         │    │    ┌─────────┐    │
         │    └────│ DECIDE  │────┘
         │         └────┬────┘
         │              │
         │         ┌────┴────┐
         │         ▼         ▼
         │    ┌─────────┐ ┌─────────┐
         │    │  SPAWN  │ │  SHARE  │
         │    │  CHILD  │ │ FINDING │
         │    └────┬────┘ └────┬────┘
         │         │           │
         │         ▼           │
         │    ┌─────────┐      │
         └───▶│  DEATH  │◀─────┘
              └─────────┘    (if stagnant)
```

## LangGraph Implementation

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    agent_id: str
    agent_type: Literal["explorer", "exploiter", "analyst"]
    strategy: str
    position: list[float]           # Current position in hyperparameter space
    velocity: list[float]           # Direction of movement
    step_size: float
    history: list[dict]             # This agent's evaluations
    stagnation_counter: int
    generation: int                 # For genealogy tracking
    parent_id: str | None

class SwarmState(TypedDict):
    # Search space definition
    search_space: dict[str, tuple[float, float]]
    
    # Agents
    agents: dict[str, AgentState]
    pending_spawns: list[AgentState]
    pending_deaths: list[str]
    
    # Blackboard
    blackboard: dict
    
    # Orchestration
    iteration: int
    max_iterations: int
    converged: bool
    convergence_reason: str | None
    
    # For visualization
    events: Annotated[list[dict], operator.add]

def build_swarm_graph():
    workflow = StateGraph(SwarmState)
    
    # Nodes
    workflow.add_node("initialize", initialize_swarm)
    workflow.add_node("agent_step", execute_agent_steps)
    workflow.add_node("evaluate_batch", run_evaluations)
    workflow.add_node("update_blackboard", sync_blackboard)
    workflow.add_node("analyst_pass", run_analyst_agents)
    workflow.add_node("lifecycle", manage_agent_lifecycle)
    workflow.add_node("check_convergence", check_convergence)
    workflow.add_node("emit_viz_state", emit_visualization_state)
    
    # Flow
    workflow.set_entry_point("initialize")
    
    workflow.add_edge("initialize", "agent_step")
    workflow.add_edge("agent_step", "evaluate_batch")
    workflow.add_edge("evaluate_batch", "update_blackboard")
    workflow.add_edge("update_blackboard", "analyst_pass")
    workflow.add_edge("analyst_pass", "lifecycle")
    workflow.add_edge("lifecycle", "check_convergence")
    workflow.add_edge("check_convergence", "emit_viz_state")
    
    # Conditional: continue or end
    workflow.add_conditional_edges(
        "emit_viz_state",
        lambda s: "end" if s["converged"] else "continue",
        {
            "continue": "agent_step",
            "end": END
        }
    )
    
    return workflow.compile()
```

## Visualization Components

### 0. Problem Selector & Info Panel
Top bar showing selected problem and live metrics.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Problem: [🚢 Titanic Survival ▼]     [▶ Start Swarm]  [⏹ Stop]        │
├─────────────────────────────────────────────────────────────────────────┤
│  📊 Accuracy: 0.847 (+0.012)  │  🔄 Iteration: 34/100  │  👥 Agents: 7 │
│  📈 Evaluations: 156          │  ⏱️ Elapsed: 2m 34s    │  🎯 Best: 0.847│
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Search Space View (Main Visual)
2D or 3D scatter plot showing:
- **Evaluated points** - colored by score (red=bad, green=good)
- **Agent positions** - icons showing current location
- **Agent trails** - fading paths showing exploration history
- **Promising regions** - highlighted zones
- **Surrogate surface** - contour/heatmap of predicted scores

```
┌─────────────────────────────────────────┐
│ Learning Rate                           │
│ ▲                                       │
│ │      ░░░░░▒▒▒▓▓▓███                  │
│ │    ░░░░░▒▒▒▒▓▓▓▓███ ←Agent 3        │
│ │   ░░░░▒▒▒▒▒▓▓▓███★                   │
│ │  ░░░▒▒▒▒▒▓▓▓▓███   ← Best found     │
│ │ ░░▒▒▒▒▒▓▓▓▓████                      │
│ │░░▒▒▒▒▓▓▓▓█████  Agent 1→ ●──────    │
│ │▒▒▒▒▓▓▓▓█████                         │
│ │▒▒▓▓▓▓██████    ●←Agent 2            │
│ └──────────────────────────────────▶   │
│                          Batch Size    │
└─────────────────────────────────────────┘
```

### 2. Score Timeline
Line chart showing best score over iterations, with markers for discoveries.

```
┌─────────────────────────────────────────┐
│ Score                                   │
│ ▲                                       │
│ │                          ●━━━━━━━━━  │
│ │                    ●━━━━━┛           │
│ │              ●━━━━━┛                  │
│ │        ●━━━━━┛                        │
│ │  ●━━━━━┛                              │
│ │━━┛                                    │
│ └──────────────────────────────────▶   │
│                            Iteration   │
└─────────────────────────────────────────┘
```

### 3. Agent Activity Feed
Real-time log of agent actions:

```
┌─────────────────────────────────────────┐
│ 🟢 Agent-7 spawned (exploiter) at       │
│    [lr=0.01, batch=64]                  │
│ 📊 Agent-3 evaluated: score=0.847       │
│ 🎯 Agent-3 found new best! 0.847        │
│ 💀 Agent-2 died (stagnation)            │
│ 🔍 Analyst: Promising region detected   │
│    at [lr=0.005-0.02, batch=32-128]    │
│ 🟢 Agent-8 spawned (explorer) targeting │
│    unexplored region                    │
└─────────────────────────────────────────┘
```

### 4. Agent Genealogy Tree
Shows parent-child relationships, which discoveries led to spawns:

```
         Agent-0 (explorer)
         ├── Agent-2 (exploiter) → died
         └── Agent-3 (exploiter) ★ best
             ├── Agent-5 (exploiter)
             └── Agent-7 (exploiter)
```

### 5. Parameter Importance (Live)
Bar chart showing which hyperparameters most affect score:

```
learning_rate  ████████████████░░░░  82%
batch_size     ████████████░░░░░░░░  61%
n_estimators   ████████░░░░░░░░░░░░  43%
max_depth      ████░░░░░░░░░░░░░░░░  22%
```

## Tech Stack

```
├── LangGraph              # Agent orchestration
├── Python 3.11+
├── scikit-learn           # Target models, surrogate models
├── XGBoost / LightGBM     # Example target model
├── numpy / pandas
├── FastAPI                # WebSocket server for viz
├── React + D3.js          # Frontend visualization
│   ├── react-force-graph  # Agent network view
│   ├── plotly.js          # 3D scatter, contours
│   └── framer-motion      # Smooth animations
├── Redis                  # Blackboard (optional, for scale)
└── pytest
```

## Project Structure

```
agentic-hyperopt/
├── README.md
├── pyproject.toml
├── .env.example
│
├── data/                          # Bundled datasets
│   ├── titanic.csv
│   ├── creditcard.csv             # Sampled version (~6k rows)
│   └── README.md                  # Data sources & licenses
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── problems/                  # Problem configurations
│   │   ├── __init__.py
│   │   ├── registry.py            # PROBLEM_REGISTRY
│   │   ├── loaders.py             # Dataset loaders
│   │   └── objectives.py          # Objective factory
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseAgent class
│   │   ├── explorer.py            # Exploration strategies
│   │   ├── exploiter.py           # Local optimization
│   │   └── analyst.py             # Surrogate modeling
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── swarm.py               # LangGraph workflow
│   │   ├── blackboard.py          # Shared state
│   │   ├── lifecycle.py           # Spawn/death logic
│   │   └── convergence.py         # Stopping criteria
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── engine.py              # Parallel evaluation
│   │   └── cache.py               # Result caching
│   │
│   ├── search_space/
│   │   ├── __init__.py
│   │   ├── space.py               # SearchSpace definition
│   │   └── sampling.py            # Sampling strategies
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── server.py              # FastAPI + WebSocket
│   │   └── events.py              # Event formatting
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ProblemSelector.tsx    # Dropdown for problem selection
│   │   │   ├── SearchSpaceView.tsx
│   │   │   ├── ScoreTimeline.tsx
│   │   │   ├── ActivityFeed.tsx
│   │   │   ├── GenealogyTree.tsx
│   │   │   ├── MetricsPanel.tsx       # Problem-specific metrics
│   │   │   └── ParameterImportance.tsx
│   │   ├── hooks/
│   │   │   └── useSwarmSocket.ts
│   │   └── types/
│   │       └── swarm.ts
│   └── public/
│
├── tests/
│   ├── test_agents/
│   ├── test_problems/
│   ├── test_orchestration/
│   └── test_evaluation/
│
└── examples/
    ├── run_titanic.py
    ├── run_fraud.py
    ├── run_housing.py
    └── custom_problem.py          # How to add your own
```

## Core Implementation

### Search Space Definition

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass
class Dimension:
    name: str
    type: Literal["continuous", "integer", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: list | None = None
    log_scale: bool = False
    
    def sample(self) -> float | int | str:
        if self.type == "continuous":
            if self.log_scale:
                return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
            return np.random.uniform(self.low, self.high)
        elif self.type == "integer":
            return np.random.randint(self.low, self.high + 1)
        else:
            return np.random.choice(self.choices)

class SearchSpace:
    def __init__(self, dimensions: list[Dimension]):
        self.dimensions = {d.name: d for d in dimensions}
        self.dim_names = [d.name for d in dimensions]
        self.n_dims = len(dimensions)
    
    def sample_random(self) -> dict:
        return {name: dim.sample() for name, dim in self.dimensions.items()}
    
    def to_unit_cube(self, config: dict) -> np.ndarray:
        """Normalize config to [0, 1]^n for visualization."""
        result = []
        for name in self.dim_names:
            dim = self.dimensions[name]
            if dim.type == "categorical":
                result.append(dim.choices.index(config[name]) / len(dim.choices))
            elif dim.log_scale:
                log_val = np.log(config[name])
                result.append((log_val - np.log(dim.low)) / (np.log(dim.high) - np.log(dim.low)))
            else:
                result.append((config[name] - dim.low) / (dim.high - dim.low))
        return np.array(result)
```

### Agent Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import uuid

@dataclass
class AgentConfig:
    agent_type: str
    strategy: str
    initial_position: dict | None = None
    step_size: float = 0.1
    max_stagnation: int = 5

@dataclass 
class Agent(ABC):
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: AgentConfig
    position: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    stagnation_counter: int = 0
    generation: int = 0
    parent_id: str | None = None
    alive: bool = True
    
    @abstractmethod
    def propose_next(self, blackboard: "Blackboard") -> dict:
        """Propose next hyperparameter config to evaluate."""
        pass
    
    @abstractmethod
    def update(self, config: dict, score: float, blackboard: "Blackboard") -> None:
        """Update internal state after evaluation."""
        pass
    
    def should_die(self) -> bool:
        return self.stagnation_counter >= self.config.max_stagnation
    
    def should_spawn(self, score: float, blackboard: "Blackboard") -> bool:
        """Decide if this discovery warrants spawning a child."""
        if not blackboard.best_score:
            return False
        # Spawn if we found something in top 10%
        return score > blackboard.score_percentile(90)
```

### Explorer Agent

```python
class ExplorerAgent(Agent):
    """Explores unknown regions of the search space."""
    
    def propose_next(self, blackboard: Blackboard) -> dict:
        if self.config.strategy == "random":
            return blackboard.search_space.sample_random()
        
        elif self.config.strategy == "centroid_void":
            # Find largest unexplored region
            evaluated_points = blackboard.get_evaluated_positions()
            if len(evaluated_points) < 5:
                return blackboard.search_space.sample_random()
            
            # Sample candidates, pick one furthest from evaluated points
            candidates = [blackboard.search_space.sample_random() for _ in range(50)]
            candidate_positions = [blackboard.search_space.to_unit_cube(c) for c in candidates]
            
            best_candidate = None
            best_min_dist = -1
            
            for candidate, pos in zip(candidates, candidate_positions):
                min_dist = min(np.linalg.norm(pos - ep) for ep in evaluated_points)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
            
            return best_candidate
        
        elif self.config.strategy == "boundary":
            # Probe edges of the space
            config = {}
            for name, dim in blackboard.search_space.dimensions.items():
                if np.random.random() < 0.3:  # 30% chance to be at boundary
                    config[name] = dim.low if np.random.random() < 0.5 else dim.high
                else:
                    config[name] = dim.sample()
            return config
    
    def update(self, config: dict, score: float, blackboard: Blackboard) -> None:
        self.history.append({"config": config, "score": score})
        self.position = config
        
        # Check if we're in a promising region
        if blackboard.best_score and score > blackboard.score_percentile(70):
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
```

### Exploiter Agent

```python
class ExploiterAgent(Agent):
    """Refines promising regions with local search."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_local_score = float("-inf")
        self.best_local_config = None
    
    def propose_next(self, blackboard: Blackboard) -> dict:
        if self.config.strategy == "hill_climb":
            if not self.position:
                return self.config.initial_position or blackboard.search_space.sample_random()
            
            # Perturb current position
            new_config = {}
            for name, dim in blackboard.search_space.dimensions.items():
                if dim.type == "continuous":
                    delta = np.random.normal(0, self.config.step_size * (dim.high - dim.low))
                    new_val = np.clip(self.position[name] + delta, dim.low, dim.high)
                    new_config[name] = new_val
                elif dim.type == "integer":
                    delta = np.random.choice([-1, 0, 1])
                    new_val = np.clip(self.position[name] + delta, dim.low, dim.high)
                    new_config[name] = int(new_val)
                else:
                    # Small chance to try different category
                    if np.random.random() < 0.1:
                        new_config[name] = np.random.choice(dim.choices)
                    else:
                        new_config[name] = self.position[name]
            
            return new_config
    
    def update(self, config: dict, score: float, blackboard: Blackboard) -> None:
        self.history.append({"config": config, "score": score})
        
        if score > self.best_local_score:
            self.best_local_score = score
            self.best_local_config = config
            self.position = config
            self.stagnation_counter = 0
            # Reduce step size on improvement (annealing)
            self.config.step_size *= 0.9
        else:
            self.stagnation_counter += 1
```

### Analyst Agent

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class AnalystAgent(Agent):
    """Builds surrogate models and identifies promising regions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surrogate = None
    
    def propose_next(self, blackboard: Blackboard) -> dict:
        # Analyst doesn't propose configs, it updates the surrogate
        return None
    
    def update(self, config: dict, score: float, blackboard: Blackboard) -> None:
        # Rebuild surrogate periodically
        pass
    
    def analyze(self, blackboard: Blackboard) -> dict:
        """Run analysis and update blackboard with insights."""
        evaluated = blackboard.evaluated_configs
        
        if len(evaluated) < 10:
            return {"status": "insufficient_data"}
        
        # Build surrogate model
        X = np.array([blackboard.search_space.to_unit_cube(e["config"]) for e in evaluated])
        y = np.array([e["score"] for e in evaluated])
        
        self.surrogate = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=5,
            random_state=42
        )
        self.surrogate.fit(X, y)
        
        # Identify promising unexplored regions (high uncertainty + predicted good)
        # Sample grid and compute acquisition function
        grid = self._generate_grid(blackboard.search_space, resolution=20)
        mu, sigma = self.surrogate.predict(grid, return_std=True)
        
        # Upper Confidence Bound acquisition
        ucb = mu + 1.96 * sigma
        
        # Find top unexplored regions
        top_indices = np.argsort(ucb)[-5:]
        promising_regions = [self._grid_to_config(grid[i], blackboard.search_space) 
                           for i in top_indices]
        
        # Update blackboard
        blackboard.surrogate = self.surrogate
        blackboard.acquisition_surface = ucb.reshape((20,) * blackboard.search_space.n_dims)
        blackboard.promising_regions = promising_regions
        
        # Check for convergence signals
        recent_improvement = self._compute_recent_improvement(blackboard)
        
        return {
            "status": "analyzed",
            "promising_regions": promising_regions,
            "surrogate_r2": self.surrogate.score(X, y),
            "recent_improvement": recent_improvement,
            "converging": recent_improvement < 0.001
        }
```

### Orchestrator (LangGraph Nodes)

```python
def initialize_swarm(state: SwarmState) -> SwarmState:
    """Spawn initial agent population."""
    agents = {}
    events = []
    
    # Spawn explorers with different strategies
    for strategy in ["random", "centroid_void", "boundary"]:
        agent = ExplorerAgent(
            config=AgentConfig(agent_type="explorer", strategy=strategy)
        )
        agents[agent.id] = agent
        events.append({
            "type": "spawn",
            "agent_id": agent.id,
            "agent_type": "explorer",
            "strategy": strategy,
            "timestamp": time.time()
        })
    
    # Spawn one analyst
    analyst = AnalystAgent(
        config=AgentConfig(agent_type="analyst", strategy="surrogate_model")
    )
    agents[analyst.id] = analyst
    events.append({
        "type": "spawn",
        "agent_id": analyst.id,
        "agent_type": "analyst",
        "timestamp": time.time()
    })
    
    return {
        **state,
        "agents": agents,
        "events": events,
        "iteration": 0,
        "converged": False
    }

def execute_agent_steps(state: SwarmState) -> SwarmState:
    """Each agent proposes its next config."""
    pending_evals = []
    events = []
    
    for agent_id, agent in state["agents"].items():
        if not agent.alive or agent.config.agent_type == "analyst":
            continue
        
        proposed = agent.propose_next(state["blackboard"])
        if proposed:
            pending_evals.append({
                "agent_id": agent_id,
                "config": proposed
            })
            events.append({
                "type": "propose",
                "agent_id": agent_id,
                "config": proposed,
                "position": state["blackboard"].search_space.to_unit_cube(proposed).tolist()
            })
    
    return {
        **state,
        "pending_evals": pending_evals,
        "events": events
    }

def run_evaluations(state: SwarmState) -> SwarmState:
    """Evaluate proposed configs in parallel."""
    results = []
    events = []
    
    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(state["objective"], eval["config"]): eval
            for eval in state["pending_evals"]
        }
        
        for future in as_completed(futures):
            eval_info = futures[future]
            score = future.result()
            results.append({
                "agent_id": eval_info["agent_id"],
                "config": eval_info["config"],
                "score": score
            })
            events.append({
                "type": "evaluation",
                "agent_id": eval_info["agent_id"],
                "config": eval_info["config"],
                "score": score,
                "timestamp": time.time()
            })
    
    return {**state, "eval_results": results, "events": events}

def manage_agent_lifecycle(state: SwarmState) -> SwarmState:
    """Spawn children, kill stagnant agents."""
    agents = state["agents"].copy()
    events = []
    
    for result in state["eval_results"]:
        agent = agents[result["agent_id"]]
        agent.update(result["config"], result["score"], state["blackboard"])
        
        # Check for death
        if agent.should_die():
            agent.alive = False
            events.append({
                "type": "death",
                "agent_id": agent.id,
                "reason": "stagnation",
                "final_best": max(e["score"] for e in agent.history) if agent.history else None
            })
        
        # Check for spawn
        elif agent.should_spawn(result["score"], state["blackboard"]):
            child = ExploiterAgent(
                config=AgentConfig(
                    agent_type="exploiter",
                    strategy="hill_climb",
                    initial_position=result["config"],
                    step_size=0.05
                ),
                parent_id=agent.id,
                generation=agent.generation + 1
            )
            agents[child.id] = child
            events.append({
                "type": "spawn",
                "agent_id": child.id,
                "agent_type": "exploiter",
                "parent_id": agent.id,
                "position": result["config"],
                "trigger_score": result["score"]
            })
    
    # Ensure minimum explorer population
    alive_explorers = sum(1 for a in agents.values() 
                         if a.alive and a.config.agent_type == "explorer")
    if alive_explorers < 2:
        new_explorer = ExplorerAgent(
            config=AgentConfig(agent_type="explorer", strategy="centroid_void")
        )
        agents[new_explorer.id] = new_explorer
        events.append({
            "type": "spawn",
            "agent_id": new_explorer.id,
            "agent_type": "explorer",
            "reason": "population_maintenance"
        })
    
    return {**state, "agents": agents, "events": events}
```

### WebSocket Server for Visualization

```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

from .problems import PROBLEM_REGISTRY, create_objective, DEFAULT_XGBOOST_SPACE

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Serve React frontend
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

class SwarmVisualizer:
    def __init__(self):
        self.connections: list[WebSocket] = []
        self.swarm_graph = None
        self.running = False
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        
        # Send available problems on connect
        await websocket.send_text(json.dumps({
            "type": "problems_list",
            "problems": [
                {
                    "id": p.id,
                    "name": p.name,
                    "emoji": p.emoji,
                    "description": p.description,
                    "metric_name": p.metric_name,
                    "problem_type": p.problem_type.value,
                }
                for p in PROBLEM_REGISTRY.values()
            ]
        }))
    
    async def broadcast(self, event: dict):
        message = json.dumps(event)
        for connection in self.connections:
            await connection.send_text(message)
    
    async def run_optimization(self, problem_id: str):
        if problem_id not in PROBLEM_REGISTRY:
            await self.broadcast({"type": "error", "message": f"Unknown problem: {problem_id}"})
            return
        
        self.running = True
        problem = PROBLEM_REGISTRY[problem_id]
        
        # Notify clients we're starting
        await self.broadcast({
            "type": "optimization_started",
            "problem": {
                "id": problem.id,
                "name": problem.name,
                "emoji": problem.emoji,
                "metric_name": problem.metric_name,
            }
        })
        
        # Create objective for this problem
        objective, _ = create_objective(problem)
        
        # Build the swarm graph
        self.swarm_graph = build_swarm_graph()
        
        state = {
            "search_space": DEFAULT_XGBOOST_SPACE,
            "objective": objective,
            "problem": problem,
            "blackboard": Blackboard(DEFAULT_XGBOOST_SPACE),
            "agents": {},
            "iteration": 0,
            "max_iterations": 100,
            "converged": False,
            "events": []
        }
        
        # Stream with visualization
        async for chunk in self.swarm_graph.astream(state):
            if not self.running:
                break
                
            # Broadcast each event to connected clients
            for event in chunk.get("events", []):
                event["problem_id"] = problem_id
                await self.broadcast(event)
            
            # Send state snapshot every iteration
            await self.broadcast({
                "type": "state_snapshot",
                "problem_id": problem_id,
                "iteration": chunk.get("iteration", 0),
                "best_score": chunk.get("blackboard", {}).get("best_score"),
                "best_config": chunk.get("blackboard", {}).get("best_config"),
                "active_agents": len([a for a in chunk.get("agents", {}).values() if a.alive]),
                "total_evaluations": len(chunk.get("blackboard", {}).get("evaluated_configs", [])),
                "metric_name": problem.metric_name,
            })
            
            await asyncio.sleep(0.1)  # Rate limit for smooth viz
        
        # Send completion
        await self.broadcast({
            "type": "optimization_complete",
            "problem_id": problem_id,
            "best_score": state["blackboard"].best_score,
            "best_config": state["blackboard"].best_config,
            "total_evaluations": len(state["blackboard"].evaluated_configs),
        })
        
        self.running = False
    
    def stop(self):
        self.running = False

visualizer = SwarmVisualizer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await visualizer.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command["action"] == "start":
                problem_id = command.get("problem_id", "titanic")
                await visualizer.run_optimization(problem_id)
            
            elif command["action"] == "stop":
                visualizer.stop()
                await websocket.send_text(json.dumps({"type": "stopped"}))
    
    except Exception as e:
        visualizer.connections.remove(websocket)

# REST endpoint for quick status
@app.get("/api/problems")
async def get_problems():
    return [
        {
            "id": p.id,
            "name": p.name,
            "emoji": p.emoji,
            "description": p.description,
        }
        for p in PROBLEM_REGISTRY.values()
    ]
```

## Example Usage

### CLI Usage

```bash
# Run with Titanic dataset
python -m agentic_hyperopt --problem titanic --visualize

# Run with Credit Card Fraud dataset
python -m agentic_hyperopt --problem fraud --visualize

# Run with California Housing dataset
python -m agentic_hyperopt --problem housing --visualize

# Run headless (no visualization)
python -m agentic_hyperopt --problem titanic --max-iterations 100
```

### Python API

```python
from agentic_hyperopt import SwarmOptimizer
from agentic_hyperopt.problems import PROBLEM_REGISTRY, create_objective

# Pick a problem
problem = PROBLEM_REGISTRY["titanic"]  # or "fraud" or "housing"

# Create objective function for this problem
objective, data = create_objective(problem)

# Run optimization with visualization
optimizer = SwarmOptimizer(
    problem=problem,
    objective=objective,
    n_initial_explorers=3,
    max_iterations=50,
    visualize=True,  # Opens browser with live dashboard
)

result = optimizer.run()

print(f"Problem: {problem.emoji} {problem.name}")
print(f"Best {problem.metric_name}: {result.best_score:.4f}")
print(f"Best config: {result.best_config}")
print(f"Total evaluations: {result.total_evaluations}")
print(f"Agents spawned: {result.agents_spawned}")
```

### Custom Problem

```python
from agentic_hyperopt import SwarmOptimizer, SearchSpace, Dimension, ProblemConfig, ProblemType

# Define your own problem
my_problem = ProblemConfig(
    id="churn",
    name="Customer Churn",
    description="Predict which customers will leave",
    emoji="📉",
    problem_type=ProblemType.CLASSIFICATION,
    metric_name="F1 Score",
    metric_direction="maximize",
    dataset_loader=my_custom_loader,  # Your function that returns (X_train, X_val, y_train, y_val)
)

# Optional: custom search space
custom_space = SearchSpace([
    Dimension("learning_rate", "continuous", 0.0001, 0.5, log_scale=True),
    Dimension("n_estimators", "integer", 100, 1000),
    # ... etc
])

optimizer = SwarmOptimizer(
    problem=my_problem,
    search_space=custom_space,  # Optional, uses default XGBoost space if not provided
    visualize=True,
)

result = optimizer.run()
```

### Web Dashboard API

```python
# Start the visualization server programmatically
from agentic_hyperopt.visualization import start_server

# This opens the browser and exposes WebSocket endpoint
server = start_server(port=8000)

# Problems are selectable in the UI
# User picks from dropdown, hits "Start Swarm", watches magic happen
```

## Why This Is Portfolio-Worthy

1. **Visual Impact** - Live dashboard showing agents exploring is immediately impressive
2. **Multi-Agent Orchestration** - Clear demonstration of LangGraph patterns at scale
3. **ML Relevance** - Hyperparameter optimization is universally understood
4. **Novel Approach** - Swarm-based HPO is differentiated from standard Optuna/Ray Tune
5. **Full Stack** - Python backend + React frontend shows breadth
6. **Extensible** - Easy to add new agent types, strategies, objectives
7. **Production Patterns** - WebSocket streaming, parallel evaluation, state management

## Stretch Goals

- [ ] **Bring Your Own CSV** - User uploads dataset, selects target column, agents figure out the rest
- [ ] **Distributed Execution** - Celery/Ray workers for parallel model training
- [ ] **Transfer Learning** - Agents share knowledge across similar optimization tasks
- [ ] **Auto-Configuration** - LLM agent that designs the search space from dataset inspection
- [ ] **Comparative Benchmarking** - Side-by-side vs Optuna/Hyperopt on standard benchmarks
- [ ] **Early Stopping** - Agents can terminate unpromising training runs mid-flight
- [ ] **Model Zoo** - Support for different model types (LightGBM, CatBoost, sklearn, neural nets)
- [ ] **Export to Production** - One-click export of best model as pickle/ONNX
