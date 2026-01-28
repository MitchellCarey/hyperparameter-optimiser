import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceDot,
} from "recharts";
import type { Agent, EvaluatedConfig, SearchSpace, Problem } from "../types/swarm";

interface SearchSpaceVizProps {
  evaluatedConfigs: EvaluatedConfig[];
  agents: Record<string, Agent>;
  searchSpace: SearchSpace | null;
  bestPosition: number[] | null;
  currentProblem: Problem | null;
}

function interpolateColor(t: number): string {
  const r = Math.round(239 * (1 - t) + 34 * t);
  const g = Math.round(68 * (1 - t) + 197 * t);
  const b = Math.round(68 * (1 - t) + 94 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function SearchSpaceViz({
  evaluatedConfigs,
  agents,
  searchSpace,
  bestPosition,
  currentProblem,
}: SearchSpaceVizProps) {
  const dimNames = searchSpace?.dim_names || ["x", "y"];
  const [xDim, setXDim] = useState(0);
  const [yDim, setYDim] = useState(Math.min(1, dimNames.length - 1));

  const { pointsData, minScore, maxScore } = useMemo(() => {
    if (evaluatedConfigs.length === 0) {
      return { pointsData: [], minScore: 0, maxScore: 1 };
    }

    const scores = evaluatedConfigs.map((c) => c.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);

    const points = evaluatedConfigs.map((config) => ({
      x: config.position[xDim] ?? 0,
      y: config.position[yDim] ?? 0,
      score: config.score,
      agentId: config.agent_id,
      normalizedScore: max > min ? (config.score - min) / (max - min) : 0.5,
    }));

    return { pointsData: points, minScore: min, maxScore: max };
  }, [evaluatedConfigs, xDim, yDim]);

  const agentPositions = useMemo(() => {
    return Object.values(agents)
      .filter((a) => a.alive && a.position)
      .map((agent) => {
        const normalized = searchSpace
          ? Object.entries(agent.position!).map(([name, value]) => {
              const dim = searchSpace.dimensions.find((d) => d.name === name);
              if (!dim) return 0;
              if (dim.log_scale) {
                const logVal = Math.log(value);
                const logLow = Math.log(dim.low);
                const logHigh = Math.log(dim.high);
                return (logVal - logLow) / (logHigh - logLow);
              }
              return (value - dim.low) / (dim.high - dim.low);
            })
          : [0, 0];

        return {
          x: normalized[xDim] ?? 0,
          y: normalized[yDim] ?? 0,
          id: agent.id,
          type: agent.type,
        };
      });
  }, [agents, searchSpace, xDim, yDim]);

  const formatScore = (score: number): string => {
    if (currentProblem?.minimize === false) {
      return `${(-score * 100).toFixed(1)}%`;
    }
    return score.toFixed(4);
  };

  return (
    <div className="panel search-space-panel">
      <div className="panel-header">
        <h3 className="panel-title">Search Space</h3>
        {searchSpace && searchSpace.n_dims > 2 && (
          <div className="dim-selectors">
            <select
              value={xDim}
              onChange={(e) => setXDim(Number(e.target.value))}
              className="dim-select"
            >
              {dimNames.map((name, i) => (
                <option key={i} value={i}>
                  X: {name}
                </option>
              ))}
            </select>
            <select
              value={yDim}
              onChange={(e) => setYDim(Number(e.target.value))}
              className="dim-select"
            >
              {dimNames.map((name, i) => (
                <option key={i} value={i}>
                  Y: {name}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      <div className="chart-container search-space-chart">
        {pointsData.length === 0 ? (
          <div className="chart-empty">Waiting for evaluations...</div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 30, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                tickLine={{ stroke: "#475569" }}
                label={{
                  value: dimNames[xDim] || "x",
                  position: "bottom",
                  fill: "#94a3b8",
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[0, 1]}
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                tickLine={{ stroke: "#475569" }}
                label={{
                  value: dimNames[yDim] || "y",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#94a3b8",
                  fontSize: 12,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: "6px",
                  color: "#f1f5f9",
                }}
                formatter={(value: number, name: string) => {
                  if (name === "score") return [formatScore(value), "Score"];
                  return [value.toFixed(3), name];
                }}
              />
              <Scatter data={pointsData} fill="#8884d8">
                {pointsData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={interpolateColor(1 - entry.normalizedScore)}
                    opacity={0.7}
                  />
                ))}
              </Scatter>
              {bestPosition && (
                <ReferenceDot
                  x={bestPosition[xDim]}
                  y={bestPosition[yDim]}
                  r={8}
                  fill="#eab308"
                  stroke="#fef08a"
                  strokeWidth={2}
                />
              )}
              {agentPositions.map((agent) => (
                <ReferenceDot
                  key={agent.id}
                  x={agent.x}
                  y={agent.y}
                  r={6}
                  fill={agent.type === "RandomExplorer" ? "#3b82f6" : "#a855f7"}
                  stroke="#f1f5f9"
                  strokeWidth={1}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="legend">
        <div className="legend-item">
          <span
            className="legend-dot"
            style={{ backgroundColor: "#3b82f6" }}
          ></span>
          <span>Explorer</span>
        </div>
        <div className="legend-item">
          <span
            className="legend-dot"
            style={{ backgroundColor: "#a855f7", borderRadius: "2px", transform: "rotate(45deg)" }}
          ></span>
          <span>Exploiter</span>
        </div>
        <div className="legend-item">
          <span
            className="legend-dot"
            style={{ backgroundColor: "#eab308" }}
          ></span>
          <span>Best</span>
        </div>
        <div className="legend-gradient">
          <span className="legend-label">Score:</span>
          <div className="gradient-bar"></div>
          <span className="legend-label-range">
            {formatScore(maxScore)} - {formatScore(minScore)}
          </span>
        </div>
      </div>
    </div>
  );
}
