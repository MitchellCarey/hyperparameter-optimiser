import { useEffect, useRef } from "react";
import type { SwarmEvent } from "../types/swarm";

interface ActivityFeedProps {
  events: SwarmEvent[];
}

function getEventColor(eventType: string): string {
  switch (eventType) {
    case "spawn":
      return "event-spawn";
    case "death":
      return "event-death";
    case "new_best":
      return "event-best";
    case "evaluation":
      return "event-eval";
    case "iteration_complete":
      return "event-iteration";
    case "convergence":
      return "event-convergence";
    default:
      return "event-default";
  }
}

function formatEventMessage(event: SwarmEvent): string {
  if (event.message) {
    return event.message;
  }

  switch (event.type) {
    case "spawn":
      return `${event.agent_type || "Agent"} ${event.agent_id} spawned${event.parent_id ? ` from ${event.parent_id}` : ""}`;
    case "death":
      return `Agent ${event.agent_id} died (${event.reason || "unknown"})`;
    case "new_best":
      return `New best score: ${event.score?.toFixed(4)} by ${event.agent_id}`;
    case "evaluation":
      return `${event.agent_id} evaluated: ${event.score?.toFixed(4)}`;
    case "iteration_complete":
      return `Iteration complete`;
    case "convergence":
      return `Optimization converged: ${event.reason || "complete"}`;
    default:
      return `${event.type}`;
  }
}

export function ActivityFeed({ events }: ActivityFeedProps) {
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [events]);

  const displayEvents = events
    .filter((e) => e.type !== "evaluation")
    .slice(-50);

  return (
    <div className="panel activity-feed-panel">
      <h3 className="panel-title">Activity Feed</h3>
      <div className="activity-feed" ref={feedRef}>
        {displayEvents.length === 0 ? (
          <div className="activity-empty">Waiting for events...</div>
        ) : (
          displayEvents.map((event, index) => (
            <div
              key={index}
              className={`activity-item ${getEventColor(event.type)}`}
            >
              <span className="activity-type">{event.type}</span>
              <span className="activity-message">
                {formatEventMessage(event)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
