"""
Util functions to compute/append/log metrics such as:
- Mean Pinball Loss for lower and upper bounds
- Coverage probability
- Average, max, and min interval widths
"""

# package imports
import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_absolute_error
import logging


def append_metrics(
    list_mean_minball_loss_lower_perc: list,
    list_mean_minball_loss_upper_perc: list,
    list_coverage: list,
    list_avg_width: list,
    y: np.array,
    y_pred_upper: np.array,
    y_pred_lower: np.array,
    alpha_lower: float = 0.1,
    alpha_upper: float = 0.9,
) -> tuple:
    """
    Appends various metrics to the provided lists.

    Args:
        list_mean_minball_loss_lower_perc (list): list of mean pinball loss for lower bounds.
        list_mean_minball_loss_upper_perc (list): list of mean pinball loss for upper bounds.
        list_coverage (list): list of coverage probabilities.
        list_avg_width (list): list of average interval widths.
        y (np.array): true values of the target variable.
        y_pred_upper (np.array): predicted upper bounds of the intervals.
        y_pred_lower (np.array): predicted lower bounds of the intervals.
        alpha_lower (float, optional): lower percentile. Defaults to 0.1.
        alpha_upper (float, optional): upper percentile. Defaults to 0.9.

    Returns:
        tuple: lists containing mean pinball loss for lower and upper bounds, coverage probabilities, and average widths.
    """
    list_mean_minball_loss_lower_perc.append(
        mean_pinball_loss(y, y_pred_lower, alpha=alpha_lower)
    )
    list_mean_minball_loss_upper_perc.append(
        mean_pinball_loss(y, y_pred_upper, alpha=alpha_upper)
    )
    list_coverage.append(((y >= y_pred_lower) & (y <= y_pred_upper)).mean())
    list_avg_width.append(np.mean(y_pred_upper - y_pred_lower))

    return (
        list_mean_minball_loss_lower_perc,
        list_mean_minball_loss_upper_perc,
        list_coverage,
        list_avg_width,
    )


def compute_metrics(
    y: np.array,
    y_pred_upper: np.array,
    y_pred_lower: np.array,
    alpha_lower: float = 0.1,
    alpha_upper: float = 0.9,
):
    """
    Computes and logs various metrics for evaluating prediction intervals.

    Args:
        y (np.array): true values of the target variable.
        y_pred_upper (np.array): predicted upper bounds of the intervals.
        y_pred_lower (np.array): predicted lower bounds of the intervals.
        alpha_lower (float, optional): lower percentile. Defaults to 0.1.
        alpha_upper (float, optional): upper percentile. Defaults to 0.9.
    """

    mean_pinball_loss_lower_perc = mean_pinball_loss(y, y_pred_lower, alpha=alpha_lower)
    logging.info(
        f"Mean Pinball Loss for Lower Bound: {mean_pinball_loss_lower_perc:.4f}"
    )
    mean_pinball_loss_upper_perc = mean_pinball_loss(y, y_pred_upper, alpha=alpha_upper)
    logging.info(
        f"Mean Pinball Loss for Upper Bound: {mean_pinball_loss_upper_perc:.4f}"
    )

    coverage = ((y >= y_pred_lower) & (y <= y_pred_upper)).mean()
    logging.info(f"Coverage probability: {coverage:.2%}")

    avg_width = np.mean(y_pred_upper - y_pred_lower)
    logging.info(f"Average Interval Width: {avg_width:.2f}")

    max_width = np.max(y_pred_upper - y_pred_lower)
    logging.info(f"Max Interval Width: {max_width:.2f}")

    min_width = np.min(y_pred_upper - y_pred_lower)
    logging.info(f"Min Interval Width: {min_width:.2f}")

    # predictions where both bounds are below 8k
    both_below_8k = (y_pred_lower < 8000) & (y_pred_upper < 8000)
    logging.info(
        f"Percentage of instances where both ends or range are below 8k: {np.mean(both_below_8k) * 100:.2f}%"
    )

    # predictions where lower bound is below 8k
    lower_below_8k = y_pred_lower < 8000
    logging.info(
        f"Percentage of instances where lower bound is below 8k: {np.mean(lower_below_8k) * 100:.2f}%"
    )

    # average width of intervals for which lower bound is below 8k (but not the upper end) after replacing lower bound with 8k
    lower_below_8k_upper_above = lower_below_8k & (y_pred_upper > 8000)
    adjusted_lower_bound = np.where(lower_below_8k_upper_above, 8000, y_pred_lower)
    adjusted_width = np.mean(
        y_pred_upper - adjusted_lower_bound, where=lower_below_8k_upper_above
    )
    logging.info(
        f"Average width of intervals where lower bound is below 8k after adjusting to 8k: {adjusted_width:.2f}"
    )

    # smallest interval after readjusting lower bound
    smallest_interval = np.min(y_pred_upper - adjusted_lower_bound)
    logging.info(
        f"Smallest interval after adjusting lower bound to 8k: {smallest_interval:.2f}"
    )

    # largest interval after readjusting lower bound
    largest_interval = np.max(y_pred_upper - adjusted_lower_bound)
    logging.info(
        f"Largest interval after adjusting lower bound to 8k: {largest_interval:.2f}"
    )

    logging.info(
        "**Although not very meaningful, calculating mean absolute error for both bounds**"
    )

    mae_upper = mean_absolute_error(y, y_pred_upper)
    mae_lower = mean_absolute_error(y, y_pred_lower)

    logging.info(f"MAE for Upper Bound: {mae_upper:.4f}")
    logging.info(f"MAE for Lower Bound: {mae_lower:.4f}")
