"""
Rich console visualization for hyperparameter optimization.

Provides a real-time dashboard showing:
- Search space heatmap with agent positions
- Best score chart over time
- Agent table with status
- Activity feed of recent events
- Summary statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agents import AnalystAgent, HillClimbExploiter, RandomExplorer

if TYPE_CHECKING:
    from orchestration.blackboard import Blackboard
    from orchestration.state import SwarmState
    from search_space import SearchSpace


class ConsoleVisualizer:
    """
    Real-time console visualization for swarm optimization.

    Uses Rich's Live display to show an updating dashboard with:
    - ASCII heatmap of evaluated configurations
    - Agent positions and the best point found
    - Score chart showing convergence
    - Agent status table
    - Activity feed of recent events

    Example:
        >>> from visualization.console import ConsoleVisualizer
        >>> visualizer = ConsoleVisualizer(search_space, width=60, height=20)
        >>> visualizer.update(state)
        >>> layout = visualizer.build_layout()
    """

    # Characters for score intensity (lower score = better = more filled)
    INTENSITY_CHARS = " ░▒▓█"

    def __init__(
        self, search_space: SearchSpace, width: int = 60, height: int = 20
    ) -> None:
        """
        Initialize the console visualizer.

        Args:
            search_space: The search space being optimized
            width: Width of the heatmap in characters
            height: Height of the heatmap in characters
        """
        self._search_space = search_space
        self._width = width
        self._height = height
        self._console = Console()

        # State tracking for visualization
        self._best_score_history: list[float] = []
        self._current_state: SwarmState | None = None

        # Get dimension names for axis labels
        self._x_name = search_space.dim_names[0] if search_space.n_dims > 0 else "x"
        self._y_name = search_space.dim_names[1] if search_space.n_dims > 1 else "y"

    def update(self, state: SwarmState) -> None:
        """
        Update the visualizer with new state.

        Args:
            state: The current SwarmState from the graph
        """
        self._current_state = state

        # Track best score history
        best_score = state["blackboard"].best_score
        if best_score is not None:
            if not self._best_score_history or best_score < self._best_score_history[-1]:
                self._best_score_history.append(best_score)
            elif self._best_score_history:
                # Repeat last best if no improvement
                self._best_score_history.append(self._best_score_history[-1])

    def render_search_space(
        self, blackboard: Blackboard, agents: dict[str, Any]
    ) -> Panel:
        """
        Render ASCII heatmap of the search space.

        Shows evaluated configurations as intensity based on score,
        with agent positions marked.

        Args:
            blackboard: The shared blackboard with evaluations
            agents: Dict of agent_id -> Agent

        Returns:
            Rich Panel containing the heatmap
        """
        # Create empty grid
        grid = [[" " for _ in range(self._width)] for _ in range(self._height)]
        colors = [[None for _ in range(self._width)] for _ in range(self._height)]

        # Get score range for normalization
        if blackboard.evaluated_configs:
            scores = [ec.score for ec in blackboard.evaluated_configs]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0

            # Plot evaluated configurations
            for ec in blackboard.evaluated_configs:
                # Use first two dimensions for 2D plot
                x_norm = ec.position[0] if len(ec.position) > 0 else 0.5
                y_norm = ec.position[1] if len(ec.position) > 1 else 0.5

                # Convert to grid coordinates
                gx = int(x_norm * (self._width - 1))
                gy = int((1 - y_norm) * (self._height - 1))  # Flip y for display
                gx = max(0, min(self._width - 1, gx))
                gy = max(0, min(self._height - 1, gy))

                # Calculate intensity (lower score = better = more intense)
                normalized_score = (ec.score - min_score) / score_range
                # Invert: low score -> high intensity
                intensity = 1.0 - normalized_score
                char_idx = int(intensity * (len(self.INTENSITY_CHARS) - 1))
                char_idx = max(0, min(len(self.INTENSITY_CHARS) - 1, char_idx))

                grid[gy][gx] = self.INTENSITY_CHARS[char_idx]

                # Color by score (good = green, bad = red)
                if intensity > 0.7:
                    colors[gy][gx] = "green"
                elif intensity > 0.4:
                    colors[gy][gx] = "yellow"
                else:
                    colors[gy][gx] = "red"

        # Mark agent positions
        for agent_id, agent in agents.items():
            if agent.position is None:
                continue

            # Normalize position
            try:
                normalized = self._search_space.normalize(agent.position)
                x_norm = normalized[0] if len(normalized) > 0 else 0.5
                y_norm = normalized[1] if len(normalized) > 1 else 0.5
            except (KeyError, IndexError):
                continue

            gx = int(x_norm * (self._width - 1))
            gy = int((1 - y_norm) * (self._height - 1))
            gx = max(0, min(self._width - 1, gx))
            gy = max(0, min(self._height - 1, gy))

            # Choose marker based on agent type
            if isinstance(agent, RandomExplorer):
                grid[gy][gx] = "E"
                colors[gy][gx] = "blue"
            elif isinstance(agent, HillClimbExploiter):
                grid[gy][gx] = "X"
                colors[gy][gx] = "magenta"

        # Mark best position with star
        best = blackboard.get_best()
        if best:
            x_norm = best.position[0] if len(best.position) > 0 else 0.5
            y_norm = best.position[1] if len(best.position) > 1 else 0.5
            gx = int(x_norm * (self._width - 1))
            gy = int((1 - y_norm) * (self._height - 1))
            gx = max(0, min(self._width - 1, gx))
            gy = max(0, min(self._height - 1, gy))
            grid[gy][gx] = "★"
            colors[gy][gx] = "yellow"

        # Build rich text with colors
        text = Text()
        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                color = colors[y][x]
                if color:
                    text.append(char, style=color)
                else:
                    text.append(char, style="dim")
            if y < len(grid) - 1:
                text.append("\n")

        title = f"Search Space (x: {self._x_name}, y: {self._y_name})"
        return Panel(text, title=title, border_style="blue")

    def render_score_chart(self, blackboard: Blackboard) -> Panel:
        """
        Render ASCII chart of best score over time.

        Args:
            blackboard: The shared blackboard

        Returns:
            Rich Panel containing the score chart
        """
        chart_width = 20
        chart_height = 8

        if not self._best_score_history:
            text = Text("No data yet", style="dim")
            return Panel(text, title="Best Score", border_style="green")

        scores = self._best_score_history
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        # Resample scores to fit chart width
        if len(scores) > chart_width:
            step = len(scores) / chart_width
            resampled = [scores[int(i * step)] for i in range(chart_width)]
        else:
            resampled = scores + [scores[-1]] * (chart_width - len(scores))

        # Build chart
        lines = []
        for row in range(chart_height):
            line = ""
            threshold = max_score - (row / (chart_height - 1)) * score_range
            for score in resampled:
                if score <= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)

        # Add axis labels
        text = Text()
        text.append(f"{max_score:>6.1f}│", style="dim")
        text.append(lines[0] + "\n", style="green")

        for i, line in enumerate(lines[1:-1], 1):
            text.append("      │", style="dim")
            text.append(line + "\n", style="green")

        text.append(f"{min_score:>6.1f}│", style="dim")
        text.append(lines[-1] + "\n", style="green")
        text.append("      └" + "─" * chart_width, style="dim")

        return Panel(text, title="Best Score", border_style="green")

    def render_agent_table(self, agents: dict[str, Any]) -> Panel:
        """
        Render table of agent status.

        Args:
            agents: Dict of agent_id -> Agent

        Returns:
            Rich Panel containing the agent table
        """
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("ID", style="dim", width=14)
        table.add_column("Type", width=10)
        table.add_column("Status", width=6)
        table.add_column("Best", justify="right", width=8)
        table.add_column("Position", width=16)

        for agent_id, agent in sorted(agents.items()):
            # Determine type and color
            if isinstance(agent, RandomExplorer):
                agent_type = "Explorer"
                type_style = "blue"
            elif isinstance(agent, HillClimbExploiter):
                agent_type = "Exploiter"
                type_style = "magenta"
            else:
                agent_type = "Unknown"
                type_style = "white"

            # Status
            status = "alive" if agent.alive else "dead"
            status_style = "green" if agent.alive else "red"

            # Best score
            best = agent.get_best()
            best_str = f"{best[1]:.4f}" if best else "-"

            # Position (truncated)
            if agent.position:
                pos_vals = [f"{v:.2f}" for v in list(agent.position.values())[:2]]
                pos_str = f"({', '.join(pos_vals)})"
            else:
                pos_str = "-"

            table.add_row(
                agent_id,
                Text(agent_type, style=type_style),
                Text(status, style=status_style),
                best_str,
                pos_str,
            )

        return Panel(table, title="Agents", border_style="cyan")

    def render_activity_feed(self, events: list[dict[str, Any]], n: int = 8) -> Panel:
        """
        Render recent activity events.

        Args:
            events: List of event dicts
            n: Number of events to show

        Returns:
            Rich Panel containing the activity feed
        """
        text = Text()
        recent = events[-n:] if len(events) > n else events

        for event in reversed(recent):  # Most recent first
            event_type = event.get("type", "unknown")
            message = event.get("message", "")

            # Truncate message
            if len(message) > 35:
                message = message[:32] + "..."

            # Color by event type
            if event_type == "spawn":
                text.append(f"[spawn] ", style="green")
            elif event_type == "death":
                text.append(f"[death] ", style="red")
            elif event_type == "new_best":
                text.append(f"[best]  ", style="yellow bold")
            elif event_type == "evaluation":
                text.append(f"[eval]  ", style="dim")
            elif event_type == "iteration_complete":
                text.append(f"[iter]  ", style="cyan")
            else:
                text.append(f"[{event_type[:5]}] ", style="dim")

            text.append(f"{message}\n", style="white")

        if not recent:
            text.append("No events yet", style="dim")

        return Panel(text, title="Activity", border_style="yellow")

    def render_analyst_insights(self, blackboard: Blackboard) -> Panel:
        """
        Render analyst insights panel.

        Shows:
        - Convergence status
        - Promising regions identified
        - Robustness warnings

        Args:
            blackboard: The shared blackboard with analyst insights

        Returns:
            Rich Panel containing analyst insights
        """
        text = Text()

        # Convergence status
        conv = blackboard.convergence_signal
        if conv:
            if conv.get("suggested_stop"):
                text.append("Status: ", style="dim")
                text.append("Converged\n", style="green bold")
            elif conv.get("diminishing_returns"):
                text.append("Status: ", style="dim")
                text.append("Diminishing\n", style="yellow")
            else:
                text.append("Status: ", style="dim")
                text.append("Improving\n", style="cyan")

            rate = conv.get("improvement_rate")
            if rate is not None:
                text.append(f"Rate: {rate:.6f}\n", style="dim")
        else:
            text.append("Status: ", style="dim")
            text.append("Waiting...\n", style="dim")

        # Surrogate model fit
        if blackboard.surrogate_r2 is not None:
            text.append(f"R\u00b2: {blackboard.surrogate_r2:.3f}\n", style="dim")

        text.append("\n")

        # Promising regions
        text.append("Promising:\n", style="bold")
        if blackboard.promising_regions:
            for i, region in enumerate(blackboard.promising_regions[:2], 1):
                pred = region.get("predicted_score", 0)
                unc = region.get("uncertainty", 0)
                text.append(f" {i}. ", style="dim")
                text.append(f"{pred:.4f}", style="green")
                text.append(f"\u00b1{unc:.3f}\n", style="yellow")
        else:
            text.append(" None yet\n", style="dim")

        # Robustness warnings
        if blackboard.robustness_warnings:
            text.append("\n")
            text.append("Warnings:\n", style="bold yellow")
            for warning in blackboard.robustness_warnings[:2]:
                severity = warning.get("severity", "medium")
                msg = warning.get("message", "")[:30]
                if severity == "high":
                    text.append(f" \u26a0 {msg}\n", style="red")
                else:
                    text.append(f" \u26a0 {msg}\n", style="yellow")

        return Panel(text, title="Analyst", border_style="magenta")

    def render_stats(self, state: SwarmState) -> Panel:
        """
        Render summary statistics.

        Args:
            state: The current SwarmState

        Returns:
            Rich Panel containing stats
        """
        blackboard = state["blackboard"]
        agents = state["agents"]

        # Count agent types
        n_explorers = sum(
            1 for a in agents.values() if isinstance(a, RandomExplorer) and a.alive
        )
        n_exploiters = sum(
            1 for a in agents.values() if isinstance(a, HillClimbExploiter) and a.alive
        )
        n_analysts = sum(
            1 for a in agents.values() if isinstance(a, AnalystAgent) and a.alive
        )
        total_alive = n_explorers + n_exploiters + n_analysts

        text = Text()
        text.append(f"Iteration: ", style="dim")
        text.append(f"{state['iteration']}/{state['max_iterations']}\n", style="bold")

        text.append(f"Evaluations: ", style="dim")
        text.append(f"{len(blackboard.evaluated_configs)}\n", style="bold")

        text.append(f"Best: ", style="dim")
        if blackboard.best_score is not None:
            text.append(f"{blackboard.best_score:.4f}\n", style="bold green")
        else:
            text.append("-\n", style="dim")

        text.append(f"Agents: ", style="dim")
        text.append(f"{total_alive}\n", style="bold")

        text.append(f"  Explorers: ", style="dim")
        text.append(f"{n_explorers}\n", style="blue")

        text.append(f"  Exploiters: ", style="dim")
        text.append(f"{n_exploiters}\n", style="magenta")

        text.append(f"  Analysts: ", style="dim")
        text.append(f"{n_analysts}", style="magenta")

        return Panel(text, title="Stats", border_style="white")

    def build_layout(self) -> Layout:
        """
        Build the full dashboard layout.

        Returns:
            Rich Layout with all panels arranged
        """
        layout = Layout()

        # Main vertical split
        layout.split_column(
            Layout(name="top", ratio=3),
            Layout(name="bottom", ratio=2),
        )

        # Top row: search space + right column
        layout["top"].split_row(
            Layout(name="search_space", ratio=3),
            Layout(name="right_col", ratio=1),
        )

        # Right column: stats + score chart + analyst
        layout["top"]["right_col"].split_column(
            Layout(name="stats", ratio=2),
            Layout(name="analyst", ratio=2),
            Layout(name="score_chart", ratio=1),
        )

        # Bottom row: agents + activity
        layout["bottom"].split_row(
            Layout(name="agents", ratio=3),
            Layout(name="activity", ratio=1),
        )

        # Render panels if we have state
        if self._current_state:
            state = self._current_state
            blackboard = state["blackboard"]
            agents = state["agents"]
            events = state["events"]

            layout["top"]["search_space"].update(
                self.render_search_space(blackboard, agents)
            )
            layout["top"]["right_col"]["stats"].update(self.render_stats(state))
            layout["top"]["right_col"]["analyst"].update(
                self.render_analyst_insights(blackboard)
            )
            layout["top"]["right_col"]["score_chart"].update(
                self.render_score_chart(blackboard)
            )
            layout["bottom"]["agents"].update(self.render_agent_table(agents))
            layout["bottom"]["activity"].update(self.render_activity_feed(events))
        else:
            # Placeholder panels
            layout["top"]["search_space"].update(
                Panel("Waiting for data...", title="Search Space")
            )
            layout["top"]["right_col"]["stats"].update(Panel("...", title="Stats"))
            layout["top"]["right_col"]["analyst"].update(Panel("...", title="Analyst"))
            layout["top"]["right_col"]["score_chart"].update(
                Panel("...", title="Best Score")
            )
            layout["bottom"]["agents"].update(Panel("...", title="Agents"))
            layout["bottom"]["activity"].update(Panel("...", title="Activity"))

        return layout
