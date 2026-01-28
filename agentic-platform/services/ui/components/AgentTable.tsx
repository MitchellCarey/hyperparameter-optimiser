import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { Agent, Problem } from "../types/swarm";

interface AgentTableProps {
  agents: Record<string, Agent>;
  currentProblem: Problem | null;
}

type SortKey = "id" | "type" | "alive" | "generation" | "best_score" | "history_length";
type SortDirection = "asc" | "desc";

export function AgentTable({ agents, currentProblem }: AgentTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  const agentsList = Object.values(agents);

  const sortedAgents = [...agentsList].sort((a, b) => {
    let aVal: string | number | boolean;
    let bVal: string | number | boolean;

    switch (sortKey) {
      case "id":
        aVal = a.id;
        bVal = b.id;
        break;
      case "type":
        aVal = a.type;
        bVal = b.type;
        break;
      case "alive":
        aVal = a.alive ? 1 : 0;
        bVal = b.alive ? 1 : 0;
        break;
      case "generation":
        aVal = a.generation;
        bVal = b.generation;
        break;
      case "best_score":
        aVal = a.best_score ?? Infinity;
        bVal = b.best_score ?? Infinity;
        break;
      case "history_length":
        aVal = a.history_length;
        bVal = b.history_length;
        break;
      default:
        return 0;
    }

    if (aVal < bVal) return sortDirection === "asc" ? -1 : 1;
    if (aVal > bVal) return sortDirection === "asc" ? 1 : -1;
    return 0;
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDirection("asc");
    }
  };

  const formatScore = (score: number | null): string => {
    if (score === null) return "-";
    if (currentProblem?.minimize === false) {
      return `${(-score * 100).toFixed(1)}%`;
    }
    return score.toFixed(4);
  };

  const SortIcon = ({ column }: { column: SortKey }) => {
    if (sortKey !== column) return null;
    return sortDirection === "asc" ? (
      <ChevronUp size={14} />
    ) : (
      <ChevronDown size={14} />
    );
  };

  return (
    <div className="panel agent-table-panel">
      <h3 className="panel-title">Agents</h3>
      <div className="table-container">
        <table className="agent-table">
          <thead>
            <tr>
              <th onClick={() => handleSort("id")} className="sortable">
                ID <SortIcon column="id" />
              </th>
              <th onClick={() => handleSort("type")} className="sortable">
                Type <SortIcon column="type" />
              </th>
              <th onClick={() => handleSort("alive")} className="sortable">
                Status <SortIcon column="alive" />
              </th>
              <th onClick={() => handleSort("generation")} className="sortable">
                Gen <SortIcon column="generation" />
              </th>
              <th onClick={() => handleSort("best_score")} className="sortable">
                Best <SortIcon column="best_score" />
              </th>
              <th onClick={() => handleSort("history_length")} className="sortable">
                Evals <SortIcon column="history_length" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedAgents.length === 0 ? (
              <tr>
                <td colSpan={6} className="table-empty">
                  No agents yet
                </td>
              </tr>
            ) : (
              sortedAgents.map((agent) => (
                <tr
                  key={agent.id}
                  className={`${!agent.alive ? "agent-dead" : ""} ${agent.type === "RandomExplorer" ? "agent-explorer" : "agent-exploiter"}`}
                >
                  <td className="agent-id">{agent.id}</td>
                  <td>
                    <span
                      className={`agent-type-badge ${agent.type === "RandomExplorer" ? "explorer" : "exploiter"}`}
                    >
                      {agent.type === "RandomExplorer" ? "E" : "X"}
                    </span>
                  </td>
                  <td>
                    <span
                      className={`status-badge ${agent.alive ? "alive" : "dead"}`}
                    >
                      {agent.alive ? "Alive" : "Dead"}
                    </span>
                  </td>
                  <td>{agent.generation}</td>
                  <td className="score-cell">{formatScore(agent.best_score)}</td>
                  <td>{agent.history_length}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
