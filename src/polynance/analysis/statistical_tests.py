"""Statistical tests for market prediction analysis."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Result from a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool  # p < 0.05
    interpretation: str
    sample_size: int
    confidence_level: float = 0.95


@dataclass
class MarketPredictivenessSummary:
    """Summary of market predictiveness analysis."""

    asset: str
    total_windows: int

    # Basic directional stats
    up_windows: int
    down_windows: int
    up_rate: float

    # Prediction analysis
    predictions_made: int  # Windows with strong signal (>0.6 or <0.4)
    predictions_correct: int
    accuracy: float

    # Statistical tests
    binomial_test: Optional[StatisticalResult] = None
    auc_score: Optional[float] = None
    optimal_threshold: Optional[float] = None

    # Timing analysis
    best_timepoint: Optional[str] = None  # "t5", "t10", etc.

    def is_statistically_significant(self) -> bool:
        """Check if results are statistically significant."""
        if self.binomial_test:
            return self.binomial_test.significant
        return False

    def can_analyze(self) -> bool:
        """Check if we have enough data for meaningful analysis."""
        return self.total_windows >= 100


def binomial_prediction_test(
    predictions_up: List[bool],
    actual_up: List[bool],
    p_null: float = 0.5,
) -> StatisticalResult:
    """Test if prediction accuracy is significantly different from random.

    Uses binomial test to check if market predictions are better than chance.

    Args:
        predictions_up: List of market predictions (True = predicted up)
        actual_up: List of actual outcomes (True = actually went up)
        p_null: Null hypothesis probability (0.5 = random chance)

    Returns:
        StatisticalResult with test outcome
    """
    try:
        from scipy.stats import binomtest
    except ImportError:
        logger.error("scipy not installed, cannot run statistical tests")
        return StatisticalResult(
            test_name="binomial_test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="scipy not available",
            sample_size=0,
        )

    # Count correct predictions
    n_total = len(predictions_up)
    n_correct = sum(pred == actual for pred, actual in zip(predictions_up, actual_up))

    # Run binomial test
    result = binomtest(n_correct, n_total, p=p_null, alternative='two-sided')

    # Interpret results
    accuracy = n_correct / n_total if n_total > 0 else 0
    if result.pvalue < 0.05:
        if accuracy > p_null:
            interpretation = f"Market is significantly predictive ({accuracy:.1%} accuracy, p={result.pvalue:.4f})"
        else:
            interpretation = f"Market is significantly worse than random ({accuracy:.1%} accuracy, p={result.pvalue:.4f})"
    else:
        interpretation = f"Market is not significantly different from random ({accuracy:.1%} accuracy, p={result.pvalue:.4f})"

    return StatisticalResult(
        test_name="binomial_test",
        statistic=n_correct / n_total,
        p_value=result.pvalue,
        significant=result.pvalue < 0.05,
        interpretation=interpretation,
        sample_size=n_total,
    )


def calculate_roc_auc(
    yes_prices: List[float],
    actual_up: List[bool],
) -> Tuple[float, float]:
    """Calculate ROC AUC score and optimal threshold.

    Args:
        yes_prices: Market YES prices (0-1)
        actual_up: Actual outcomes (True = went up)

    Returns:
        Tuple of (auc_score, optimal_threshold)
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        logger.error("scikit-learn not installed, cannot calculate ROC AUC")
        return 0.5, 0.5

    # Convert to numpy arrays
    y_true = np.array([1 if up else 0 for up in actual_up])
    y_score = np.array(yes_prices)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate AUC
    auc_score = auc(fpr, tpr)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return auc_score, optimal_threshold


def mann_whitney_u_test(
    group1: List[float],
    group2: List[float],
) -> StatisticalResult:
    """Non-parametric test for comparing two independent groups.

    Useful for comparing market prices when outcome was up vs down.

    Args:
        group1: Values from first group
        group2: Values from second group

    Returns:
        StatisticalResult with test outcome
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        logger.error("scipy not installed, cannot run Mann-Whitney U test")
        return StatisticalResult(
            test_name="mann_whitney_u",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="scipy not available",
            sample_size=0,
        )

    result = mannwhitneyu(group1, group2, alternative='two-sided')

    median1 = np.median(group1)
    median2 = np.median(group2)

    interpretation = (
        f"Group 1 median: {median1:.3f}, Group 2 median: {median2:.3f}. "
        f"{'Significantly different' if result.pvalue < 0.05 else 'Not significantly different'} (p={result.pvalue:.4f})"
    )

    return StatisticalResult(
        test_name="mann_whitney_u",
        statistic=result.statistic,
        p_value=result.pvalue,
        significant=result.pvalue < 0.05,
        interpretation=interpretation,
        sample_size=len(group1) + len(group2),
    )


def analyze_market_predictiveness(
    windows: List,
    asset: str,
    threshold: float = 0.5,
) -> MarketPredictivenessSummary:
    """Comprehensive analysis of market predictiveness.

    Args:
        windows: List of Window objects
        asset: Asset symbol
        threshold: Price threshold for making predictions (default 0.5)

    Returns:
        MarketPredictivenessSummary with all analysis results
    """
    # Basic stats
    total = len(windows)
    up_windows = sum(1 for w in windows if w.outcome == "up")
    down_windows = total - up_windows
    up_rate = up_windows / total if total > 0 else 0.0

    # Filter windows with t5 prices
    windows_with_t5 = [w for w in windows if w.pm_yes_t5 is not None]

    # Make predictions using threshold
    predictions_up = [w.pm_yes_t5 > threshold for w in windows_with_t5]
    actual_up = [w.outcome == "up" for w in windows_with_t5]

    # Count strong predictions (>0.6 or <0.4)
    strong_threshold_high = 0.6
    strong_threshold_low = 0.4
    strong_predictions = [
        w for w in windows_with_t5
        if w.pm_yes_t5 > strong_threshold_high or w.pm_yes_t5 < strong_threshold_low
    ]

    strong_pred_up = [
        w.pm_yes_t5 > 0.5 for w in strong_predictions
    ]
    strong_actual_up = [
        w.outcome == "up" for w in strong_predictions
    ]

    predictions_made = len(strong_predictions)
    predictions_correct = sum(
        pred == actual for pred, actual in zip(strong_pred_up, strong_actual_up)
    )
    accuracy = predictions_correct / predictions_made if predictions_made > 0 else 0.0

    summary = MarketPredictivenessSummary(
        asset=asset,
        total_windows=total,
        up_windows=up_windows,
        down_windows=down_windows,
        up_rate=up_rate,
        predictions_made=predictions_made,
        predictions_correct=predictions_correct,
        accuracy=accuracy,
    )

    # Only run statistical tests if we have enough data
    if total < 30:
        logger.info(f"[{asset}] Only {total} windows, skipping statistical tests (need 30+)")
        return summary

    # Binomial test on strong predictions
    if predictions_made >= 10:
        summary.binomial_test = binomial_prediction_test(
            strong_pred_up,
            strong_actual_up,
            p_null=0.5,
        )

    # ROC AUC analysis (requires 50+ samples for reliability)
    if len(windows_with_t5) >= 50:
        try:
            yes_prices = [w.pm_yes_t5 for w in windows_with_t5]
            auc_score, optimal_threshold = calculate_roc_auc(yes_prices, actual_up)
            summary.auc_score = auc_score
            summary.optimal_threshold = optimal_threshold
        except Exception as e:
            logger.warning(f"Error calculating ROC AUC: {e}")

    # Compare different timepoints (if we have 100+ samples)
    if total >= 100:
        # Test which timepoint is best predictor
        timepoints = {
            't2.5': [w.pm_yes_t2_5 for w in windows if w.pm_yes_t2_5 is not None],
            't5': [w.pm_yes_t5 for w in windows if w.pm_yes_t5 is not None],
            't7.5': [w.pm_yes_t7_5 for w in windows if w.pm_yes_t7_5 is not None],
            't10': [w.pm_yes_t10 for w in windows if w.pm_yes_t10 is not None],
        }

        # Find timepoint with highest correlation to outcome (simple heuristic)
        best_time = 't5'  # Default
        best_score = 0.0

        for time_name, prices in timepoints.items():
            if len(prices) >= 50:
                try:
                    actual = [w.outcome == "up" for w in windows if getattr(w, f'pm_yes_{time_name.replace(".", "_")}') is not None]
                    auc_score, _ = calculate_roc_auc(prices, actual)
                    if auc_score > best_score:
                        best_score = auc_score
                        best_time = time_name
                except Exception:
                    pass

        summary.best_timepoint = best_time

    return summary


def format_analysis_report(summary: MarketPredictivenessSummary) -> str:
    """Format analysis summary as readable report.

    Args:
        summary: MarketPredictivenessSummary object

    Returns:
        Formatted string report
    """
    lines = []
    lines.append(f"=" * 70)
    lines.append(f"MARKET PREDICTIVENESS ANALYSIS: {summary.asset}")
    lines.append(f"=" * 70)
    lines.append("")

    # Sample size warning
    if summary.total_windows < 100:
        lines.append(f"⚠️  WARNING: Only {summary.total_windows} windows collected")
        lines.append(f"   Need at least 100 windows for reliable statistical analysis")
        lines.append(f"   Current results are DESCRIPTIVE ONLY, not statistically valid")
        lines.append("")

    # Basic stats
    lines.append(f"Sample Size: {summary.total_windows} windows")
    lines.append(f"  Up: {summary.up_windows} ({summary.up_rate:.1%})")
    lines.append(f"  Down: {summary.down_windows} ({(1-summary.up_rate):.1%})")
    lines.append("")

    # Prediction stats
    lines.append(f"Strong Predictions (YES > 0.6 or < 0.4): {summary.predictions_made}")
    if summary.predictions_made > 0:
        lines.append(f"  Correct: {summary.predictions_correct}")
        lines.append(f"  Accuracy: {summary.accuracy:.1%}")
    lines.append("")

    # Statistical tests
    if summary.binomial_test:
        lines.append("Binomial Test (Is market better than random?):")
        lines.append(f"  {summary.binomial_test.interpretation}")
        if summary.binomial_test.significant:
            lines.append(f"  ✓ SIGNIFICANT at p < 0.05")
        else:
            lines.append(f"  ✗ Not significant (p = {summary.binomial_test.p_value:.4f})")
        lines.append("")

    # ROC AUC
    if summary.auc_score is not None:
        lines.append(f"ROC AUC Score: {summary.auc_score:.3f}")
        lines.append(f"  (0.5 = random, 1.0 = perfect)")
        if summary.optimal_threshold:
            lines.append(f"  Optimal Threshold: {summary.optimal_threshold:.3f}")
        lines.append("")

    # Best timepoint
    if summary.best_timepoint:
        lines.append(f"Best Prediction Timepoint: {summary.best_timepoint}")
        lines.append("")

    # Conclusion
    lines.append("CONCLUSION:")
    if summary.total_windows < 100:
        lines.append("  Insufficient data for conclusions. Keep collecting.")
    elif summary.is_statistically_significant():
        lines.append(f"  ✓ Market shows statistically significant predictive power")
        lines.append(f"  Accuracy: {summary.accuracy:.1%} (p < 0.05)")
    else:
        lines.append(f"  ✗ Market does not show significant predictive power")
        lines.append(f"  Cannot reject null hypothesis of random predictions")

    lines.append("=" * 70)

    return "\n".join(lines)
