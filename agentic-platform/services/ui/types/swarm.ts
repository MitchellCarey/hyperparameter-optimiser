/**
 * TypeScript interfaces for the swarm hyperparameter optimization system.
 */

export interface Dimension {
  name: string;
  type: "continuous" | "integer";
  low: number;
  high: number;
  log_scale: boolean;
}

export interface SearchSpace {
  dimensions: Dimension[];
  dim_names: string[];
  n_dims: number;
}

export interface EvaluatedConfig {
  config: Record<string, number>;
  score: number;
  agent_id: string;
  timestamp: number;
  position: number[];
}

export interface Agent {
  id: string;
  type: "RandomExplorer" | "HillClimbExploiter" | "AnalystAgent";
  alive: boolean;
  generation: number;
  parent_id: string | null;
  position: Record<string, number> | null;
  history_length: number;
  best_score: number | null;
  // AnalystAgent specific
  analysis_interval?: number;
}

export interface PromisingRegion {
  config: Record<string, number>;
  center_normalized: number[];
  predicted_score: number;
  uncertainty: number;
  acquisition_value: number;
}

export interface RobustnessWarning {
  type: "isolated_peak" | "unstable_region" | "insufficient_neighbours";
  position: number[];
  message: string;
  severity: "high" | "medium" | "low";
}

export interface ConvergenceSignal {
  diminishing_returns: boolean;
  improvement_rate: number | null;
  suggested_stop: boolean;
  message: string;
}

export interface AnalystInsights {
  promising_regions: PromisingRegion[];
  robustness_warnings: RobustnessWarning[];
  convergence_signal: ConvergenceSignal;
  best_score_history: number[];
  surrogate_r2: number | null;
}

export interface Blackboard {
  evaluated_configs: EvaluatedConfig[];
  best_score: number | null;
  best_config: Record<string, number> | null;
  best_position: number[] | null;
  analyst_insights: AnalystInsights;
}

export interface Problem {
  id: string;
  name: string;
  emoji: string;
  metric_name: string;
  minimize: boolean;
  problem_type: "classification" | "regression";
}

export interface StateSnapshot {
  type: "state_snapshot";
  iteration: number;
  max_iterations: number;
  converged: boolean;
  agents: Record<string, Agent>;
  blackboard: Blackboard;
  search_space: SearchSpace;
  problem: Problem;
}

export interface SwarmEvent {
  type: string;
  agent_id?: string;
  message?: string;
  score?: number;
  config?: Record<string, number>;
  position?: number[];
  reason?: string;
  parent_id?: string;
  agent_type?: string;
  generation?: number;
  [key: string]: unknown;
}

export interface ProblemsListMessage {
  type: "problems_list";
  problems: Problem[];
}

export interface OptimizationStartedMessage {
  type: "optimization_started";
  problem_id: string;
  problem_name: string;
}

export interface EventMessage extends SwarmEvent {
  type: "event";
}

export interface OptimizationCompleteMessage {
  type: "optimization_complete";
  problem_id: string;
  best_score: number | null;
  best_config: Record<string, number> | null;
  total_evaluations: number;
}

export interface StoppedMessage {
  type: "stopped";
  reason?: string;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type WebSocketMessage =
  | ProblemsListMessage
  | OptimizationStartedMessage
  | EventMessage
  | StateSnapshot
  | OptimizationCompleteMessage
  | StoppedMessage
  | ErrorMessage;

export interface SwarmState {
  connected: boolean;
  error: string | null;
  problems: Problem[];
  running: boolean;
  currentProblem: Problem | null;
  iteration: number;
  maxIterations: number;
  converged: boolean;
  agents: Record<string, Agent>;
  evaluatedConfigs: EvaluatedConfig[];
  bestScore: number | null;
  bestConfig: Record<string, number> | null;
  bestPosition: number[] | null;
  searchSpace: SearchSpace | null;
  events: SwarmEvent[];
  bestScoreHistory: { iteration: number; score: number }[];
  analystInsights: AnalystInsights | null;
}
