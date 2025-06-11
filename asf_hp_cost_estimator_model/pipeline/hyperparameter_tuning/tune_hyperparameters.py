"""
This script performs hyperparameter tuning for a Gradient Boosting Regressor model for lower and upper quantiles.

After tuning the model, it evaluates the model on both training and test sets, logging various metrics such as mean pinball loss, coverage probability, and interval widths.

Usage:
python asf_hp_cost_estimator_model/pipeline/hyperparameter_tuning/tune_hyperparameters.py --lower_quantile 0.1 --upper_quantile 0.9
"""

# package imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
import logging
import pprint

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
from asf_hp_cost_estimator_model.utils.model_evaluation_utils import compute_metrics


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

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["random_state"]
    )

    # Define the parameter grid for hyperparameter tuning
    param_grid = dict(
        learning_rate=[0.01, 0.05, 0.1, 0.2],
        max_depth=[3, 5, 10, 20],
        min_samples_leaf=[1, 5, 10, 100, 1000],
        min_samples_split=[2, 10, 50, 100, 1000],
    )

    # Set up the scoring function for lower quantile (lq) as the mean pinball loss (to minimise)
    mean_pinball_loss_lq_scorer = make_scorer(
        mean_pinball_loss,
        alpha=lower_quantile,
        greater_is_better=False,
    )

    # Search for the best hyperparameters for the lower quantile model
    # Using HalvingRandomSearchCV for efficient hyperparameter tuning
    # It does this through cross-validation (on x_train) and progressively narrowing down the search space
    gbr = GradientBoostingRegressor(
        loss="quantile", alpha=lower_quantile, random_state=config["random_state"]
    )
    search_lq = HalvingRandomSearchCV(
        gbr,
        param_grid,
        resource="n_estimators",
        max_resources=500,
        min_resources=50,
        scoring=mean_pinball_loss_lq_scorer,
        n_jobs=2,
        random_state=config["random_state"],
    ).fit(x_train, y_train)

    logging.info("Best params for lower percentile model: ")
    pprint.pprint(search_lq.best_params_)

    # Set up the scoring function for upper quantile (uq) as the mean pinball loss (to minimise)
    mean_pinball_loss_uq_scorer = make_scorer(
        mean_pinball_loss,
        alpha=upper_quantile,
        greater_is_better=False,
    )

    # The parameters to tune for the upper quantile model are the same as for the lower quantile model
    search_uq = clone(search_lq).set_params(
        estimator__alpha=upper_quantile,
        scoring=mean_pinball_loss_uq_scorer,
    )
    search_uq.fit(x_train, y_train)
    logging.info("Best params for upper percentile model: ")
    pprint.pprint(search_uq.best_params_)

    # ---- Train models fully on x_train and look at a wider range of metrics----
    model_lower = GradientBoostingRegressor(
        loss="quantile", alpha=lower_quantile, **search_lq.best_params_
    )
    model_lower.fit(x_train, y_train)

    y_pred_lower_test = model_lower.predict(x_test)
    y_pred_lower_train = model_lower.predict(x_train)

    model_upper = GradientBoostingRegressor(
        loss="quantile", alpha=upper_quantile, **search_uq.best_params_
    )
    model_upper.fit(x_train, y_train)

    y_pred_upper_test = model_upper.predict(x_test)
    y_pred_upper_train = model_upper.predict(x_train)

    logging.info("----- MODEL EVALUATION RESULTS ON TRAINING SET -----")
    compute_metrics(
        y=y_train,
        y_pred_lower=y_pred_lower_train,
        y_pred_upper=y_pred_upper_train,
        alpha_lower=lower_quantile,
        alpha_upper=upper_quantile,
    )
    logging.info("\n----- MODEL EVALUATION RESULTS ON TEST SET -----")
    compute_metrics(
        y=y_test,
        y_pred_lower=y_pred_lower_test,
        y_pred_upper=y_pred_upper_test,
        alpha_lower=lower_quantile,
        alpha_upper=upper_quantile,
    )
