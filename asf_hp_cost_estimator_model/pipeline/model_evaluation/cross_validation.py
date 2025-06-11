"""
With the the hyperparameters tuned, we do cross-validation to evaluate the model's performance
on different subsets of the data and looking at:
- Mean Pinball Loss for lower and upper bounds
- Coverage probability
- Average interval widths

Usage:
python asf_hp_cost_estimator_model/pipeline/model_evaluation/cross_validation.py --lower_quantile 0.1 --upper_quantile 0.9
"""

# package imports
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import logging

# local imports
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
from asf_hp_cost_estimator_model.utils.model_evaluation_utils import append_metrics


def argparse_setup():
    """
    Sets up the command line argument parser to allow for quantile specification
    used for estimating a cost prediction interval for an air source heat pump.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a model to estimate the cost of an air source heat pump."
    )
    parser.add_argument(
        "--lower_quantile",
        type=float,
        default=0.1,
        help="Lower quantile for cost estimation.",
    )
    parser.add_argument(
        "--upper_quantile",
        type=float,
        default=0.9,
        help="Upper quantile for cost estimation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparse_setup()
    lower_quantile = args.lower_quantile
    upper_quantile = args.upper_quantile

    # Load and process data
    mcs_epc_data = get_enhanced_installations_data()
    cpi_05_3_df = get_cpi_data()
    cpi_quarterly_df = get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=config["cpi_data"]["cpi_reference_year"],
        cpi_df=cpi_05_3_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )
    postcodes_data = get_postcodes_data()

    model_data = process_data_before_modelling(
        mcs_epc_data=mcs_epc_data,
        postcodes_data=postcodes_data,
        cpi_quarterly_df=cpi_quarterly_df,
        exclusion_criteria_dict=config["exclusion_criteria"],
        winsorise=config["winsorise_outliers"],
        min_date=config["min_date"],
    )

    # Define features and target
    numeric_features = config["numeric_features"]
    original_categorical_features = config["categorical_features_to_dummify"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    # Preparing data for modelling
    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    kf = KFold(
        n_splits=config["kfold_splits"],
        shuffle=True,
        random_state=config["random_state"],
    )

    # Lists to store metrics for each fold
    list_train_mean_minball_loss_lower_perc = []
    list_train_mean_minball_loss_upper_perc = []
    list_train_coverage = []
    list_train_avg_width = []

    list_test_mean_minball_loss_lower_perc = []
    list_test_mean_minball_loss_upper_perc = []
    list_test_coverage = []
    list_test_avg_width = []

    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_lower = GradientBoostingRegressor(
            loss="quantile",
            alpha=lower_quantile,
            n_estimators=config["n_estimators"],
            min_samples_leaf=config["min_samples_leaf"],
            min_samples_split=config["min_samples_split"],
            random_state=config["random_state"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
        )
        model_lower.fit(x_train, y_train)

        y_pred_lower_test = model_lower.predict(x_test)
        y_pred_lower_train = model_lower.predict(x_train)

        model_upper = GradientBoostingRegressor(
            loss="quantile",
            alpha=upper_quantile,
            n_estimators=config["n_estimators"],
            min_samples_leaf=config["min_samples_leaf"],
            min_samples_split=config["min_samples_split"],
            random_state=config["random_state"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
        )
        model_upper.fit(x_train, y_train)

        y_pred_upper_test = model_upper.predict(x_test)
        y_pred_upper_train = model_upper.predict(x_train)

        # Compute metrics for the current fold in the training set
        (
            list_train_mean_minball_loss_lower_perc,
            list_train_mean_minball_loss_upper_perc,
            list_train_coverage,
            list_train_avg_width,
        ) = append_metrics(
            list_mean_minball_loss_lower_perc=list_train_mean_minball_loss_lower_perc,
            list_mean_minball_loss_upper_perc=list_train_mean_minball_loss_upper_perc,
            list_coverage=list_train_coverage,
            list_avg_width=list_train_avg_width,
            y=y_train,
            y_pred_upper=y_pred_upper_train,
            y_pred_lower=y_pred_lower_train,
            alpha_lower=lower_quantile,
            alpha_upper=upper_quantile,
        )

        # Compute metrics for the current fold in the test set
        (
            list_test_mean_minball_loss_lower_perc,
            list_test_mean_minball_loss_upper_perc,
            list_test_coverage,
            list_test_avg_width,
        ) = append_metrics(
            list_mean_minball_loss_lower_perc=list_test_mean_minball_loss_lower_perc,
            list_mean_minball_loss_upper_perc=list_test_mean_minball_loss_upper_perc,
            list_coverage=list_test_coverage,
            list_avg_width=list_test_avg_width,
            y=y_test,
            y_pred_upper=y_pred_upper_test,
            y_pred_lower=y_pred_lower_test,
            alpha_lower=lower_quantile,
            alpha_upper=upper_quantile,
        )

    logging.info("----- MODEL EVALUATION RESULTS ON TRAINING SET -----")
    logging.info(
        f"Mean Pinball Loss for Lower Bound: {np.mean(list_train_mean_minball_loss_lower_perc):.2f}"
    )
    logging.info(
        f"Mean Pinball Loss for Upper Bound: {np.mean(list_train_mean_minball_loss_upper_perc):.2f}"
    )
    logging.info(f"Coverage probability: {np.mean(list_train_coverage):.2%}")
    logging.info(f"Average Interval Width: {np.mean(list_train_avg_width):.2f}")

    logging.info("\n----- MODEL EVALUATION RESULTS ON TEST SET -----")
    logging.info(
        f"Mean Pinball Loss for Lower Bound: {np.mean(list_test_mean_minball_loss_lower_perc):.2f}"
    )
    logging.info(
        f"Mean Pinball Loss for Upper Bound: {np.mean(list_test_mean_minball_loss_upper_perc):.2f}"
    )
    logging.info(f"Coverage probability: {np.mean(list_test_coverage):.2%}")
    logging.info(f"Average Interval Width: {np.mean(list_test_avg_width):.2f}")
