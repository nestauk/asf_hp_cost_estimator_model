"""
Util functions to compute/append/log metrics such as:
- Mean Pinball Loss for lower and upper bounds
- Coverage probability
- Average, max, and min interval widths
- Proportion of samples where median is below lower bound, above upper bound or outside bounds
- Proportion of samples where median is closer to upper bound than lower bound
"""

# package imports
import numpy as np
import os
from sklearn.metrics import mean_pinball_loss
import logging
import matplotlib.pyplot as plt

# local imports
from asf_hp_cost_estimator_model import PROJECT_DIR


def compute_metrics(
    dataset_name: str,
    y: np.array,
    y_pred_upper: np.array,
    y_pred_lower: np.array,
    y_pred_median: np.array,
    alpha_lower: float = 0.1,
    alpha_upper: float = 0.9,
    log_metrics: bool = True,
    save_histogram: bool = False,
) -> dict:
    """
    Computes and logs various metrics for evaluating prediction intervals.

    Args:
        dataset_name (str): name of the dataset (e.g., 'train', 'validation', 'test').
        y (np.array): true values of the target variable.
        y_pred_upper (np.array): predicted upper bounds of the intervals.
        y_pred_lower (np.array): predicted lower bounds of the intervals.
        y_pred_median (np.array): predicted median
        alpha_lower (float, optional): lower percentile. Defaults to 0.1.
        alpha_upper (float, optional): upper percentile. Defaults to 0.9.
        save_histogram (bool, optional): whether to save histograms of interval widths and ratio of distances. Defaults to False.

    Returns:
        dict: dictionary containing main metrics.
    """
    logging.info(f"----- MODEL EVALUATION RESULTS ON {dataset_name.upper()} -----")

    main_metrics = {}

    # Size of set
    n = len(y)

    # Our main metric to minimise: mean pinball loss
    mean_pinball_loss_lower_perc = mean_pinball_loss(y, y_pred_lower, alpha=alpha_lower)
    mean_pinball_loss_median = mean_pinball_loss(y, y_pred_median, alpha=0.5)
    mean_pinball_loss_upper_perc = mean_pinball_loss(y, y_pred_upper, alpha=alpha_upper)

    # Coverage probability: the proportion of true values that fall within the predicted intervals
    coverage = ((y >= y_pred_lower) & (y <= y_pred_upper)).mean()

    # Width of the prediction intervals
    avg_width = np.mean(y_pred_upper - y_pred_lower)
    max_width = np.max(y_pred_upper - y_pred_lower)
    min_width = np.min(y_pred_upper - y_pred_lower)

    if save_histogram:
        plt.hist(y_pred_upper - y_pred_lower, bins=30, edgecolor="black")
        plt.title("Histogram of interval widths")
        plt.xlabel("Interval width")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.join(
                PROJECT_DIR, "outputs/figures", "histogram_interval_widths.png"
            )
        )
        plt.close()

    # Order of the predictions
    median_below_lower = (y_pred_median < y_pred_lower).mean()
    median_above_upper = (y_pred_median > y_pred_upper).mean()
    median_outside_bounds = (
        (y_pred_median < y_pred_lower) | (y_pred_median > y_pred_upper)
    ).mean()

    # Average distance between the median prediction and the lower bound
    distance_median_lower = y_pred_median - y_pred_lower
    avg_distance_median_lower = np.mean(distance_median_lower)

    # Average distance between the median prediction and the upper bound
    distance_median_upper = y_pred_upper - y_pred_median
    avg_distance_median_upper = np.mean(distance_median_upper)
    ratio = distance_median_lower / (distance_median_lower + distance_median_upper)

    if save_histogram:
        plt.hist(ratio, bins=30, edgecolor="black")
        plt.title("Histogram of ratio of distances")
        plt.xlabel(
            "Distance(median, lower bound) / Distance(lower bound, upper bound)\n>0.5 means median is closer to upper bound"
        )
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.join(
                PROJECT_DIR, "outputs/figures", "histogram_ratio_distances.png"
            )
        )
        plt.close()

    if log_metrics:
        logging.info(f"Number of samples: {n}")
        logging.info(
            f"Mean pinball loss for lower bound: {mean_pinball_loss_lower_perc:.4f}"
        )
        logging.info(f"Mean pinball loss for median: {mean_pinball_loss_median:.4f}")
        logging.info(
            f"Mean pinball loss for upper bound: {mean_pinball_loss_upper_perc:.4f}"
        )
        logging.info(f"Coverage probability: {coverage:.2%}")
        logging.info(f"Average interval width: {avg_width:.2f}")
        logging.info(f"Max interval width: {max_width:.2f}")
        logging.info(f"Min interval width: {min_width:.2f}")
        logging.warning(
            f"Proportion of samples where median is below lower bound: {median_below_lower:.2%}"
        )
        logging.warning(
            f"Proportion of samples where median is above upper bound: {median_above_upper:.2%}"
        )
        logging.warning(
            f"Proportion of samples where median is outside bounds: {median_outside_bounds:.2%}"
        )
        logging.info(
            f"Average distance between median and lower bound: {avg_distance_median_lower:.2f}"
        )
        logging.info(
            f"Average distance between median and upper bound: {avg_distance_median_upper:.2f}"
        )
        logging.info(
            f"Proportion of samples where median is closer to upper bound than lower bound: {(ratio > 0.5).mean():.2%}"
        )

    # Store main metrics in a dictionary
    main_metrics["mean_pinball_loss_lower_q"] = mean_pinball_loss_lower_perc
    main_metrics["mean_pinball_loss_upper_q"] = mean_pinball_loss_upper_perc
    main_metrics["mean_pinball_loss_median"] = mean_pinball_loss_median
    main_metrics["coverage"] = coverage
    main_metrics["interval_width"] = avg_width
    main_metrics["prop_samples_median_outside_bounds"] = median_outside_bounds
    main_metrics["prop_samples_median_closer_to_upper_bound"] = (ratio > 0.5).mean()

    return main_metrics
