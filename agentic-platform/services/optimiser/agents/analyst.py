"""
Analyst agents that build surrogate models and provide optimization insights.

Analysts observe the optimization landscape built by Explorer and Exploiter
evaluations. They do not propose configurations - instead they provide
insights about promising regions, robustness issues, and convergence signals.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from agents.base import Agent

if TYPE_CHECKING:
    from orchestration.blackboard import Blackboard
    from search_space import SearchSpace


@dataclass
class AnalysisResult:
    """
    Result of analyst analysis.

    Attributes:
        promising_regions: Regions with high predicted performance
        robustness_warnings: Potential issues like isolated peaks
        convergence_signal: Whether optimization should stop
        surrogate_r2: R-squared score of the surrogate model fit
    """

    promising_regions: list[dict[str, Any]]
    robustness_warnings: list[dict[str, Any]]
    convergence_signal: dict[str, Any]
    surrogate_r2: float | None


@dataclass
class AnalystAgent(Agent):
    """
    Observes the optimization landscape and provides insights.

    Key behaviors:
    - Never dies (observe until the end)
    - Does not propose configurations for evaluation
    - Builds Gaussian Process surrogate model
    - Detects robustness issues (isolated peaks, unstable regions)
    - Monitors convergence (diminishing returns)
    - Identifies promising unexplored regions

    Attributes:
        analysis_interval: Analyze every N evaluations
        min_samples_for_analysis: Minimum evaluations before analyzing
    """

    analysis_interval: int = 10
    min_samples_for_analysis: int = 15
    _last_analysis_count: int = field(default=0, init=False, repr=False)
    _surrogate: GaussianProcessRegressor | None = field(
        default=None, init=False, repr=False
    )
    _y_mean: float = field(default=0.0, init=False, repr=False)
    _y_std: float = field(default=1.0, init=False, repr=False)

    def propose_next(
        self,
        search_space: "SearchSpace",
        blackboard: "Blackboard | None" = None,
    ) -> dict[str, float | int]:
        """Analysts do not propose configurations."""
        raise NotImplementedError("AnalystAgent does not propose configurations")

    def update(
        self,
        config: dict[str, float | int],
        score: float,
        blackboard: "Blackboard | None" = None,
    ) -> None:
        """Analysts do not receive individual updates."""
        raise NotImplementedError("AnalystAgent does not receive updates")

    def should_die(self) -> bool:
        """Analysts never die - they observe until the end."""
        return False

    def should_spawn(self, blackboard: "Blackboard") -> bool:
        """Analysts do not spawn children."""
        return False

    def should_analyze(self, blackboard: "Blackboard") -> bool:
        """
        Determine if analysis should run this iteration.

        Returns True if:
        - Enough evaluations exist (min_samples_for_analysis)
        - Enough new evaluations since last analysis (analysis_interval)
        """
        n_evals = len(blackboard.evaluated_configs)

        if n_evals < self.min_samples_for_analysis:
            return False

        new_evals = n_evals - self._last_analysis_count
        return new_evals >= self.analysis_interval

    def analyze(
        self,
        blackboard: "Blackboard",
        search_space: "SearchSpace",
    ) -> AnalysisResult:
        """
        Run full analysis and return insights.

        Steps:
        1. Fit Gaussian Process surrogate model
        2. Identify promising unexplored regions (Lower Confidence Bound)
        3. Detect robustness issues
        4. Check convergence signals

        Args:
            blackboard: Shared blackboard with evaluated configurations
            search_space: The search space being optimized

        Returns:
            AnalysisResult with all insights
        """
        self._last_analysis_count = len(blackboard.evaluated_configs)

        # Extract training data
        X = blackboard.get_evaluated_positions()  # Already normalized [0,1]
        y = np.array([ec.score for ec in blackboard.evaluated_configs])

        # Normalize y for GP stability
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8
        y_normalized = (y - self._y_mean) / self._y_std

        # Fit surrogate (GP with Matern kernel)
        kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(
            noise_level=0.1, noise_level_bounds=(1e-5, 1.0)
        )
        self._surrogate = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=42,
            alpha=1e-6,
        )

        try:
            self._surrogate.fit(X, y_normalized)
            r2 = float(self._surrogate.score(X, y_normalized))
        except Exception:
            # GP fitting can fail with certain data configurations
            r2 = None

        # Find promising regions via LCB acquisition
        promising = self._find_promising_regions(search_space)

        # Detect robustness issues
        robustness = self._detect_robustness_issues(X, y)

        # Check convergence
        convergence = self._check_convergence(blackboard)

        return AnalysisResult(
            promising_regions=promising,
            robustness_warnings=robustness,
            convergence_signal=convergence,
            surrogate_r2=r2,
        )

    def _find_promising_regions(
        self,
        search_space: "SearchSpace",
        n_candidates: int = 500,
        n_top: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find regions with high acquisition value (Lower Confidence Bound).

        Uses LCB = mean - kappa * std for minimization (lower is better).

        Args:
            search_space: SearchSpace for denormalization
            n_candidates: Number of random candidates to evaluate
            n_top: Number of top regions to return

        Returns:
            List of promising region dicts with predicted scores
        """
        if self._surrogate is None:
            return []

        # Sample random candidate points in unit cube
        rng = np.random.default_rng(42)
        candidates = rng.uniform(0, 1, size=(n_candidates, search_space.n_dims))

        # Predict mean and std (normalized)
        mu_norm, sigma_norm = self._surrogate.predict(candidates, return_std=True)

        # Convert back to original scale
        mu = mu_norm * self._y_std + self._y_mean
        sigma = sigma_norm * self._y_std

        # Lower Confidence Bound (for minimization: lower is better)
        kappa = 1.96  # 95% confidence
        lcb = mu - kappa * sigma

        # Find top regions (lowest LCB)
        top_indices = np.argsort(lcb)[:n_top]

        regions = []
        for idx in top_indices:
            center_normalized = candidates[idx]
            center_config = search_space.denormalize(center_normalized)
            regions.append(
                {
                    "config": center_config,
                    "center_normalized": center_normalized.tolist(),
                    "predicted_score": float(mu[idx]),
                    "uncertainty": float(sigma[idx]),
                    "acquisition_value": float(lcb[idx]),
                }
            )

        return regions

    def _detect_robustness_issues(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Detect potential robustness issues.

        Checks for:
        - Isolated peaks: best point with no nearby good points
        - Unstable regions: high prediction variance near best

        Args:
            X: Normalized positions of evaluated configs
            y: Scores of evaluated configs

        Returns:
            List of warning dicts with type and message
        """
        warnings: list[dict[str, Any]] = []

        if len(y) < 5:
            return warnings

        # Find best point
        best_idx = int(np.argmin(y))
        best_pos = X[best_idx]
        best_score = float(y[best_idx])

        # Find neighbors within radius (20% of unit cube)
        radius = 0.2
        distances = np.linalg.norm(X - best_pos, axis=1)
        nearby_mask = (distances > 0) & (distances < radius)
        n_nearby = int(nearby_mask.sum())

        if n_nearby < 3:
            warnings.append(
                {
                    "type": "insufficient_neighbours",
                    "position": best_pos.tolist(),
                    "message": f"Best point has only {n_nearby} nearby evaluations - robustness uncertain",
                    "severity": "medium",
                }
            )
        else:
            nearby_scores = y[nearby_mask]
            nearby_mean = float(nearby_scores.mean())
            nearby_std = float(nearby_scores.std())

            # Is best score an outlier vs neighbours?
            score_diff = abs(best_score - nearby_mean)
            if nearby_std > 0 and score_diff > 2 * nearby_std:
                warnings.append(
                    {
                        "type": "isolated_peak",
                        "position": best_pos.tolist(),
                        "message": f"Best score is {score_diff/nearby_std:.1f}\u03c3 away from neighbours - possible overfit",
                        "severity": "high",
                        "best_score": best_score,
                        "neighbour_mean": nearby_mean,
                        "neighbour_std": nearby_std,
                    }
                )

            # High variance in neighbourhood = unstable region
            global_std = float(y.std())
            if global_std > 0:
                relative_std = nearby_std / global_std
                if relative_std > 0.5:
                    warnings.append(
                        {
                            "type": "unstable_region",
                            "position": best_pos.tolist(),
                            "message": f"High variance ({relative_std:.0%} of global) near best point - sensitive region",
                            "severity": "medium",
                        }
                    )

        return warnings

    def _check_convergence(self, blackboard: "Blackboard") -> dict[str, Any]:
        """
        Analyze improvement trends to detect convergence.

        Returns convergence signal with:
        - diminishing_returns: True if improvement is slowing
        - improvement_rate: Recent rate of improvement
        - suggested_stop: True if optimization should consider stopping

        Args:
            blackboard: Blackboard to analyze

        Returns:
            Convergence signal dict
        """
        scores = [ec.score for ec in blackboard.evaluated_configs]

        if len(scores) < 20:
            return {
                "diminishing_returns": False,
                "improvement_rate": None,
                "suggested_stop": False,
                "message": "Insufficient data for convergence analysis",
            }

        # Build running best history
        running_best: list[float] = []
        current_best = float("inf")
        for score in scores:
            if score < current_best:
                current_best = score
            running_best.append(current_best)

        # Update blackboard history for visualization
        blackboard.best_score_history = running_best

        # Calculate improvement over windows
        window = min(20, len(running_best) // 2)
        recent = running_best[-window:]
        earlier = (
            running_best[-2 * window : -window]
            if len(running_best) >= 2 * window
            else running_best[:window]
        )

        # Improvement is reduction in score (for minimization)
        recent_improvement = earlier[-1] - recent[-1] if earlier else 0
        total_improvement = running_best[0] - running_best[-1]

        # Diminishing returns if recent improvement is <5% of total
        diminishing = (
            recent_improvement < 0.05 * abs(total_improvement)
            if total_improvement != 0
            else True
        )

        # No recent improvement if last 10 evaluations haven't improved
        no_recent = recent[-1] == recent[0] if len(recent) > 1 else False

        # Suggest stop if diminishing returns AND no recent improvement AND enough evaluations
        suggested_stop = diminishing and no_recent and len(scores) > 50

        improvement_rate = float(recent_improvement / window) if window > 0 else 0.0

        if suggested_stop:
            message = "Convergence detected - diminishing returns"
        elif diminishing:
            message = "Diminishing returns - consider stopping soon"
        else:
            message = "Still improving"

        return {
            "diminishing_returns": diminishing,
            "improvement_rate": improvement_rate,
            "suggested_stop": suggested_stop,
            "message": message,
        }
