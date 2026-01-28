import { Brain, AlertTriangle, TrendingDown, Target } from "lucide-react";
import type { AnalystInsights as AnalystInsightsType } from "../types/swarm";

interface AnalystInsightsPanelProps {
  insights: AnalystInsightsType | null;
}

export function AnalystInsightsPanel({ insights }: AnalystInsightsPanelProps) {
  if (!insights) {
    return (
      <div className="panel analyst-panel">
        <h3 className="panel-title">
          <Brain size={16} /> Analyst Insights
        </h3>
        <div className="analyst-waiting">Waiting for analysis...</div>
      </div>
    );
  }

  const { convergence_signal, promising_regions, robustness_warnings, surrogate_r2 } =
    insights;

  const getStatusClass = () => {
    if (convergence_signal?.suggested_stop) return "converged";
    if (convergence_signal?.diminishing_returns) return "diminishing";
    return "improving";
  };

  const getStatusLabel = () => {
    if (convergence_signal?.suggested_stop) return "Converged";
    if (convergence_signal?.diminishing_returns) return "Diminishing Returns";
    return "Improving";
  };

  return (
    <div className="panel analyst-panel">
      <h3 className="panel-title">
        <Brain size={16} /> Analyst Insights
      </h3>

      {/* Convergence Status */}
      <div className="analyst-section">
        <div className="section-header">
          <TrendingDown size={14} /> Status
        </div>
        <div className="convergence-status">
          <span className={`status-badge ${getStatusClass()}`}>
            {getStatusLabel()}
          </span>
          {surrogate_r2 !== null && (
            <span className="surrogate-r2">R&sup2;: {surrogate_r2.toFixed(3)}</span>
          )}
        </div>
        {convergence_signal?.improvement_rate !== null &&
          convergence_signal?.improvement_rate !== undefined && (
            <div className="improvement-rate">
              Rate: {convergence_signal.improvement_rate.toFixed(6)}
            </div>
          )}
      </div>

      {/* Promising Regions */}
      <div className="analyst-section">
        <div className="section-header">
          <Target size={14} /> Promising ({promising_regions.length})
        </div>
        <div className="promising-regions">
          {promising_regions.slice(0, 3).map((region, idx) => (
            <div key={idx} className="region-item">
              <span className="region-rank">#{idx + 1}</span>
              <span className="region-score">
                {region.predicted_score.toFixed(4)}
              </span>
              <span className="region-uncertainty">
                &plusmn;{region.uncertainty.toFixed(3)}
              </span>
            </div>
          ))}
          {promising_regions.length === 0 && (
            <div className="no-data">None identified yet</div>
          )}
        </div>
      </div>

      {/* Robustness Warnings */}
      {robustness_warnings.length > 0 && (
        <div className="analyst-section warnings">
          <div className="section-header warning-header">
            <AlertTriangle size={14} /> Warnings
          </div>
          <div className="warning-list">
            {robustness_warnings.slice(0, 2).map((warning, idx) => (
              <div
                key={idx}
                className={`warning-item severity-${warning.severity}`}
              >
                <AlertTriangle size={12} className="warning-icon" />
                <span className="warning-message">
                  {warning.message.length > 50
                    ? warning.message.slice(0, 47) + "..."
                    : warning.message}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
