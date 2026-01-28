import { Play, Square, Wifi, WifiOff } from "lucide-react";
import type { Problem } from "../types/swarm";

interface ProblemSelectorProps {
  problems: Problem[];
  selectedProblem: string;
  onSelectProblem: (id: string) => void;
  connected: boolean;
  running: boolean;
  onStart: () => void;
  onStop: () => void;
}

export function ProblemSelector({
  problems,
  selectedProblem,
  onSelectProblem,
  connected,
  running,
  onStart,
  onStop,
}: ProblemSelectorProps) {
  return (
    <div className="problem-selector">
      <div className="selector-left">
        <select
          value={selectedProblem}
          onChange={(e) => onSelectProblem(e.target.value)}
          disabled={running}
          className="problem-dropdown"
        >
          {problems.map((problem) => (
            <option key={problem.id} value={problem.id}>
              {problem.emoji} {problem.name}
            </option>
          ))}
        </select>

        {!running ? (
          <button
            onClick={onStart}
            disabled={!connected || problems.length === 0}
            className="btn btn-primary"
          >
            <Play size={16} />
            Start
          </button>
        ) : (
          <button onClick={onStop} className="btn btn-danger">
            <Square size={16} />
            Stop
          </button>
        )}
      </div>

      <div className="selector-right">
        <div className={`connection-status ${connected ? "connected" : "disconnected"}`}>
          {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span>{connected ? "Connected" : "Disconnected"}</span>
        </div>
      </div>
    </div>
  );
}
