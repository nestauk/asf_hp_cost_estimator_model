"""
Util functions to compute/append/log metrics such as:
- Mean Pinball Loss for lower and upper bounds
- Coverage probability
- Average, max, and min interval widths
"""

# package imports
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List
import os
from sklearn.metrics import mean_pinball_loss
import logging
import matplotlib.pyplot as plt

# local imports
from asf_hp_cost_estimator_model import PROJECT_DIR
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
)
from asf_hp_cost_estimator_model.pipeline.data_processing.process_location_data import (
    get_postcodes_data,
)
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.pipeline.data_processing.process_cpi import (
    get_df_quarterly_cpi_with_adjustment_factors,
)
from asf_hp_cost_estimator_model.getters.data_getters import get_cpi_data


def compute_metrics(
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
        plt.xlabel("Ratio of distance to lower bound / total distance")
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
            f"Average Distance between Median and Lower Bound: {avg_distance_median_lower:.2f}"
        )
        logging.info(
            f"Average Distance between Median and Upper Bound: {avg_distance_median_upper:.2f}"
        )
        logging.info(
            f"Proportion of samples where median is closer to lower bound than upper bound: {(ratio > 0.5).mean():.2%}"
        )

    # Store main metrics in a dictionary
    main_metrics["mean_pinball_loss_lower_q"] = mean_pinball_loss_lower_perc
    main_metrics["mean_pinball_loss_upper_q"] = mean_pinball_loss_upper_perc
    main_metrics["mean_pinball_loss_median"] = mean_pinball_loss_median
    main_metrics["coverage"] = coverage
    main_metrics["interval_width"] = avg_width
    main_metrics["prop_samples_median_outside_bounds"] = median_outside_bounds
    main_metrics["prop_samples_median_closer_to_lower_bound"] = (ratio > 0.5).mean()

    return main_metrics


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads and preprocesses all necessary data prior to hyperparameter tuning or cross validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Processed model data, validation set, and list of feature names.
    """

    logging.info("Loading and processing data...")
    mcs_epc_data = get_enhanced_installations_data()
    cpi_df = get_cpi_data()
    postcodes_data = get_postcodes_data()

    cpi_quarterly_df = get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=config["cpi_data"]["cpi_reference_year"],
        cpi_df=cpi_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )

    # Identifying first date of newest quarter to create a hold-out validation set from the most recent data
    valid_dates = mcs_epc_data[mcs_epc_data["commission_date"] < datetime.today()]
    max_date = valid_dates["commission_date"].max()
    first_day_of_quarter = max_date.to_period("Q").start_time

    # Creating a validation set from the newest quarter of data
    new_quarter_data = mcs_epc_data[
        mcs_epc_data["commission_date"] >= first_day_of_quarter
    ]
    validation_set = new_quarter_data.sample(
        frac=0.2, random_state=config["random_state"]
    )
    model_data = mcs_epc_data.drop(validation_set.index)

    # Processing without removing or winsorising outliers
    validation_set = process_data_before_modelling(
        mcs_epc_data=validation_set,
        postcodes_data=postcodes_data,
        cpi_quarterly_df=cpi_quarterly_df,
        min_date=config["min_date"],
        processing_validation_set=True,
    )

    # Processing the rest of the data for modelling
    model_data = process_data_before_modelling(
        mcs_epc_data=model_data,
        postcodes_data=postcodes_data,
        cpi_quarterly_df=cpi_quarterly_df,
        exclusion_criteria_dict=config["exclusion_criteria"],
        winsorise=config["winsorise_outliers"],
        min_date=config["min_date"],
    )

    logging.info(f"Model data size: {model_data.shape[0]}")
    prop_validation_set = validation_set.shape[0] / (
        model_data.shape[0] + validation_set.shape[0]
    )
    logging.info(
        f"Validation set size: {validation_set.shape[0]} installations, which is {prop_validation_set:.2%} of the total data"
    )

    features = config["numeric_features"] + config["categorical_features"]

    return model_data, validation_set, features
