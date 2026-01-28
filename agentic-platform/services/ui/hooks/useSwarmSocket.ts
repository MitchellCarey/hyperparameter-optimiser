import { useState, useEffect, useCallback, useRef } from "react";
import type {
  StateSnapshot,
  SwarmEvent,
  WebSocketMessage,
  SwarmState,
} from "../types/swarm";

const initialState: SwarmState = {
  connected: false,
  error: null,
  problems: [],
  running: false,
  currentProblem: null,
  iteration: 0,
  maxIterations: 100,
  converged: false,
  agents: {},
  evaluatedConfigs: [],
  bestScore: null,
  bestConfig: null,
  bestPosition: null,
  searchSpace: null,
  events: [],
  bestScoreHistory: [],
  analystInsights: null,
};

export function useSwarmSocket(url: string = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`) {
  const [state, setState] = useState<SwarmState>(initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case "problems_list":
        setState((prev) => ({ ...prev, problems: message.problems }));
        break;

      case "optimization_started":
        setState((prev) => ({
          ...prev,
          running: true,
          currentProblem:
            prev.problems.find((p) => p.id === message.problem_id) || null,
          iteration: 0,
          agents: {},
          evaluatedConfigs: [],
          bestScore: null,
          bestConfig: null,
          bestPosition: null,
          searchSpace: null,
          events: [],
          bestScoreHistory: [],
          converged: false,
          error: null,
        }));
        break;

      case "event":
        setState((prev) => ({
          ...prev,
          events: [...prev.events.slice(-199), message as SwarmEvent],
        }));
        break;

      case "state_snapshot": {
        const snapshot = message as StateSnapshot;
        setState((prev) => {
          const newHistory = [...prev.bestScoreHistory];
          if (snapshot.blackboard.best_score !== null) {
            const lastEntry = newHistory[newHistory.length - 1];
            if (
              !lastEntry ||
              lastEntry.iteration !== snapshot.iteration ||
              lastEntry.score !== snapshot.blackboard.best_score
            ) {
              newHistory.push({
                iteration: snapshot.iteration,
                score: snapshot.blackboard.best_score,
              });
            }
          }
          return {
            ...prev,
            iteration: snapshot.iteration,
            maxIterations: snapshot.max_iterations,
            converged: snapshot.converged,
            agents: snapshot.agents,
            evaluatedConfigs: snapshot.blackboard.evaluated_configs,
            bestScore: snapshot.blackboard.best_score,
            bestConfig: snapshot.blackboard.best_config,
            bestPosition: snapshot.blackboard.best_position,
            searchSpace: snapshot.search_space,
            bestScoreHistory: newHistory,
            analystInsights: snapshot.blackboard.analyst_insights || null,
          };
        });
        break;
      }

      case "optimization_complete":
        setState((prev) => ({
          ...prev,
          running: false,
          converged: true,
        }));
        break;

      case "stopped":
        setState((prev) => ({ ...prev, running: false }));
        break;

      case "error":
        setState((prev) => ({
          ...prev,
          error: message.message,
          running: false,
        }));
        break;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setState((prev) => ({ ...prev, connected: true, error: null }));
    };

    ws.onclose = () => {
      setState((prev) => ({ ...prev, connected: false }));
      reconnectTimeoutRef.current = window.setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      setState((prev) => ({
        ...prev,
        error: "WebSocket connection error",
      }));
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleMessage(message);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    wsRef.current = ws;
  }, [url, handleMessage]);

  const start = useCallback((problemId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ action: "start", problem_id: problemId })
      );
    }
  }, []);

  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "stop" }));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  return { ...state, start, stop };
}
