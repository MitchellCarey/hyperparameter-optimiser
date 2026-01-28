import { Activity, Target, Users, Zap } from "lucide-react";
import type { Agent, Problem } from "../types/swarm";

interface StatsPanelProps {
  iteration: number;
  maxIterations: number;
  bestScore: number | null;
  agents: Record<string, Agent>;
  evaluationsCount: number;
  currentProblem: Problem | null;
  converged: boolean;
}

export function StatsPanel({
  iteration,
  maxIterations,
  bestScore,
  agents,
  evaluationsCount,
  currentProblem,
  converged,
}: StatsPanelProps) {
  const agentsList = Object.values(agents);
  const aliveAgents = agentsList.filter((a) => a.alive);
  const explorers = aliveAgents.filter((a) => a.type === "RandomExplorer");
  const exploiters = aliveAgents.filter((a) => a.type === "HillClimbExploiter");

  const formatScore = (score: number | null): string => {
    if (score === null) return "-";
    const metricName = currentProblem?.metric_name;
    const minimize = currentProblem?.minimize ?? true;

    // Handle each metric type appropriately
    if (metricName === "Accuracy") {
      return `${(-score * 100).toFixed(1)}%`;
    } else if (metricName === "F1 Score") {
      return (-score).toFixed(3);
    } else if (metricName === "RMSE") {
      return score.toFixed(4);
    }

    // Default: maximize problems show as percentage, minimize show raw score
    return minimize ? score.toFixed(4) : `${(-score * 100).toFixed(1)}%`;
  };

  return (
    <div className="panel">
      <h3 className="panel-title">Statistics</h3>
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-icon">
            <Activity size={16} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Iteration</div>
            <div className="stat-value">
              {iteration} / {maxIterations}
              {converged && <span className="badge badge-success">Done</span>}
            </div>
          </div>
        </div>

        <div className="stat-item">
          <div className="stat-icon">
            <Target size={16} />
          </div>
          <div className="stat-content">
            <div className="stat-label">
              Best {currentProblem?.metric_name || "Score"}
            </div>
            <div className="stat-value stat-value-highlight">
              {formatScore(bestScore)}
            </div>
          </div>
        </div>

        <div className="stat-item">
          <div className="stat-icon">
            <Users size={16} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Agents</div>
            <div className="stat-value">
              <span className="agent-count explorer">{explorers.length} E</span>
              <span className="agent-count exploiter">
                {exploiters.length} X
              </span>
              <span className="agent-count dead">
                {agentsList.length - aliveAgents.length} dead
              </span>
            </div>
          </div>
        </div>

        <div className="stat-item">
          <div className="stat-icon">
            <Zap size={16} />
          </div>
          <div className="stat-content">
            <div className="stat-label">Evaluations</div>
            <div className="stat-value">{evaluationsCount}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
