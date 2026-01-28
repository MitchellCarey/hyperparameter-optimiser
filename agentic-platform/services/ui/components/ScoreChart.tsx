import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { Problem } from "../types/swarm";

interface ScoreChartProps {
  data: { iteration: number; score: number }[];
  currentProblem: Problem | null;
}

export function ScoreChart({ data, currentProblem }: ScoreChartProps) {
  const isMaximize = currentProblem?.minimize === false;
  const metricName = currentProblem?.metric_name;

  // Format display score based on metric type
  const formatDisplayScore = (score: number): string => {
    if (metricName === "Accuracy") {
      return `${(-score * 100).toFixed(1)}%`;
    } else if (metricName === "F1 Score") {
      return (-score).toFixed(3);
    } else if (metricName === "RMSE") {
      return score.toFixed(4);
    }
    return isMaximize ? `${(-score * 100).toFixed(1)}%` : score.toFixed(4);
  };

  const chartData = data.map((d) => ({
    iteration: d.iteration,
    score: isMaximize ? -d.score : d.score,
    displayScore: formatDisplayScore(d.score),
  }));

  const formatYAxis = (value: number): string => {
    if (metricName === "Accuracy") {
      return `${(value * 100).toFixed(0)}%`;
    } else if (metricName === "F1 Score") {
      return value.toFixed(2);
    } else if (metricName === "RMSE") {
      return value.toFixed(2);
    }
    return isMaximize ? `${(value * 100).toFixed(0)}%` : value.toFixed(2);
  };

  // Determine Y-axis domain based on metric type
  const getYDomain = (): [number | "auto", number | "auto"] => {
    if (metricName === "Accuracy" || metricName === "F1 Score") {
      return [0, 1];
    }
    return ["auto", "auto"];
  };

  return (
    <div className="panel score-chart-panel">
      <h3 className="panel-title">
        {currentProblem?.metric_name || "Score"} Over Time
      </h3>
      <div className="chart-container">
        {chartData.length === 0 ? (
          <div className="chart-empty">Waiting for data...</div>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="iteration"
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                tickLine={{ stroke: "#475569" }}
              />
              <YAxis
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                tickLine={{ stroke: "#475569" }}
                tickFormatter={formatYAxis}
                domain={getYDomain()}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: "6px",
                  color: "#f1f5f9",
                }}
                formatter={(value: number) => {
                  let formattedValue: string;
                  if (metricName === "Accuracy") {
                    formattedValue = `${(value * 100).toFixed(1)}%`;
                  } else if (metricName === "F1 Score") {
                    formattedValue = value.toFixed(3);
                  } else if (metricName === "RMSE") {
                    formattedValue = value.toFixed(4);
                  } else {
                    formattedValue = isMaximize ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
                  }
                  return [formattedValue, metricName || "Score"];
                }}
                labelFormatter={(label) => `Iteration ${label}`}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                animationDuration={300}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
