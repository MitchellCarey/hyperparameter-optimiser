import { useState } from "react";
import { useSwarmSocket } from "./hooks/useSwarmSocket";
import { ProblemSelector } from "./components/ProblemSelector";
import { SearchSpaceViz } from "./components/SearchSpaceViz";
import { ScoreChart } from "./components/ScoreChart";
import { StatsPanel } from "./components/StatsPanel";
import { AgentTable } from "./components/AgentTable";
import { ActivityFeed } from "./components/ActivityFeed";
import { AnalystInsightsPanel } from "./components/AnalystInsights";
import "./index.css";

function App() {
  const {
    connected,
    error,
    problems,
    running,
    currentProblem,
    iteration,
    maxIterations,
    converged,
    agents,
    evaluatedConfigs,
    bestScore,
    bestPosition,
    searchSpace,
    events,
    bestScoreHistory,
    analystInsights,
    start,
    stop,
  } = useSwarmSocket();

  const [selectedProblem, setSelectedProblem] = useState<string>("titanic");

  const handleStart = () => {
    start(selectedProblem);
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">Agentic Hyperparameter Optimizer</h1>
        {error && <div className="error-banner">{error}</div>}
      </header>

      <ProblemSelector
        problems={problems}
        selectedProblem={selectedProblem}
        onSelectProblem={setSelectedProblem}
        connected={connected}
        running={running}
        onStart={handleStart}
        onStop={stop}
      />

      <main className="main-content">
        <div className="left-column">
          <SearchSpaceViz
            evaluatedConfigs={evaluatedConfigs}
            agents={agents}
            searchSpace={searchSpace}
            bestPosition={bestPosition}
            currentProblem={currentProblem}
          />
          <AgentTable agents={agents} currentProblem={currentProblem} />
        </div>

        <div className="right-column">
          <StatsPanel
            iteration={iteration}
            maxIterations={maxIterations}
            bestScore={bestScore}
            agents={agents}
            evaluationsCount={evaluatedConfigs.length}
            currentProblem={currentProblem}
            converged={converged}
          />
          <AnalystInsightsPanel insights={analystInsights} />
          <ScoreChart data={bestScoreHistory} currentProblem={currentProblem} />
          <ActivityFeed events={events} />
        </div>
      </main>
    </div>
  );
}

export default App;
