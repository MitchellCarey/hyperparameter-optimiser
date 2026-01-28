# Agentic Hyperparameter Optimiser Platform

A multi-agent swarm system for hyperparameter optimization with real-time visualization, deployed as microservices.

## Services

- **optimiser** - Python FastAPI backend with LangGraph agent orchestration
- **ui** - React TypeScript frontend with real-time visualization

## Quick Start

```bash
# Copy environment file
cp .env.example .env

# Start all services
docker-compose up

# Or build and start
docker-compose up --build
```

## Ports

| Service   | Port |
|-----------|------|
| optimiser | 8001 |
| ui        | 3000 |

## Development

See individual service READMEs for development instructions:
- [Optimiser Service](./services/optimiser/README.md)
- [UI Service](./services/ui/README.md)
