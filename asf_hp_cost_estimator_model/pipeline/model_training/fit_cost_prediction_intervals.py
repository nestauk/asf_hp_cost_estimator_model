"""
Pipeline for fitting models to estimate the cost interval for air source heat pumps,
using quantile regression through Gradient Boosting Regressor with quantile loss.
It defaults to producing a 80% confidence interval by fitting models using the 10th and 90th percentiles.

This script can be run from the command line, allowing for custom quantiles to be specified:
    python asf_hp_cost_estimator_model/pipeline/model_training/fit_cost_prediction_intervals.py --lower_quantile 0.1 --upper_quantile 0.9
"""

import numpy as np
import pandas as pd
import boto3
from datetime import datetime
import pickle
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

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


def create_df_with_predictions(
    X: np.array, y: np.array, y_pred_lower: np.array, y_pred_upper: np.array
) -> pd.DataFrame:
    """
    Creates a DataFrame with predictions for the lower and upper quantiles.
    Args:
        X (np.array): Feature matrix.
        y (np.array): True values of the target variable.
        y_pred_lower (np.array): Predicted lower bounds of the intervals.
        y_pred_upper (np.array): Predicted upper bounds of the intervals.
    Returns:
        pd.DataFrame: DataFrame containing true values and predicted bounds.
    """
    predictions_df = pd.DataFrame(
        X, columns=config["numeric_features"] + config["categorical_features"]
    )
    predictions_df["y_true"] = y
    predictions_df["y_pred_lower"] = y_pred_lower
    predictions_df["y_pred_upper"] = y_pred_upper
    return predictions_df


def set_up_pipeline(quantile: float, model_bound: str) -> Pipeline:
    """
    Set up a pipeline to train a model to estimate the cost of an air source heat pump.

    Returns:
        quantile (float): The quantile to be used by GradientBoostingRegressor.
        model_bound (str): Takes "lower_bound_model" and "upper_bound_model" to specify the model for lower and upper quantiles respectively.
    """

    regressor = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=config["hyper_parameters"][model_bound]["n_estimators"],
        min_samples_leaf=config["hyper_parameters"][model_bound]["min_samples_leaf"],
        min_samples_split=config["hyper_parameters"][model_bound]["min_samples_split"],
        random_state=config["random_state"],
        learning_rate=config["hyper_parameters"][model_bound]["learning_rate"],
        max_depth=config["hyper_parameters"][model_bound]["max_depth"],
    )

    return regressor


def fit_and_save_model(lower_quantile: float = 0.1, upper_quantile: float = 0.9):
    """
    Loads data, trains model and saves model as pickle.
    """

    # Load and process data
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    cpi_05_3_df = get_cpi_data()
    cpi_quarterly_df = get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=config["cpi_data"]["cpi_reference_year"],
        cpi_df=cpi_05_3_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )

    model_data = process_data_before_modelling(
        mcs_epc_data, postcodes_data, cpi_quarterly_df
    )

    print("Model data shape:", model_data.shape)

    # Define features and target
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Train models
    regressor_lower_q = set_up_pipeline(
        quantile=lower_quantile, model_bound="lower_bound_model"
    )
    regressor_lower_q.fit(X, y)

    regressor_upper_q = set_up_pipeline(
        quantile=upper_quantile, model_bound="upper_bound_model"
    )
    regressor_upper_q.fit(X, y)

    # Printing model evaluation metrics
    y_pred_lower = regressor_lower_q.predict(X)
    y_pred_upper = regressor_upper_q.predict(X)
    compute_metrics(
        y=y,
        y_pred_lower=y_pred_lower,
        y_pred_upper=y_pred_upper,
        alpha_lower=lower_quantile,
        alpha_upper=upper_quantile,
    )

    # Save models and predictions
    today_date = datetime.today().strftime("%Y%m%d")

    predictions_df = create_df_with_predictions(
        X=X, y=y, y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )
    predictions_df.to_csv(
        f"s3://asf-hp-cost-estimator-model/outputs/model/{today_date}/predictions_{lower_quantile}_{upper_quantile}.csv",
        index=False,
    )

    s3_resource = boto3.resource("s3")
    regressor_lower_q = pickle.dumps(regressor_lower_q)
    regressor_upper_q = pickle.dumps(regressor_upper_q)

    s3_resource.Object(
        "asf-hp-cost-estimator-model",
        f"outputs/model/{today_date}/regressor_lower_q{lower_quantile}.pkl",
    ).put(Body=regressor_lower_q)
    s3_resource.Object(
        "asf-hp-cost-estimator-model",
        f"outputs/model/{today_date}/regressor_upper_q{upper_quantile}.pkl",
    ).put(Body=regressor_upper_q)

    logging.info(
        f"Models trained and saved to:\n s3://asf-hp-cost-estimator-model/outputs/model/{today_date}"
    )


if __name__ == "__main__":
    args = argparse_setup()
    lower_quantile = args.lower_quantile
    upper_quantile = args.upper_quantile

    fit_and_save_model(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
