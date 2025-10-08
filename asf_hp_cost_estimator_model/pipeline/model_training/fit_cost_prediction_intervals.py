"""
Pipeline for fitting models to estimate the cost interval for air source heat pumps,
using quantile regression through Gradient Boosting Regressor with quantile loss.
It defaults to producing a 80% confidence interval by fitting models using the 10th and 90th percentiles.
Additionally, a median model (50th percentile) is also fitted to provide a point estimate.

The pipeline includes:
- Fitting, evaluating and saving models
- Check for model drift by comparing today's model performance with the most recent model performance stored in S3.

This script can be run from the command line, allowing for custom quantiles to be specified (and whether to run in test mode):
    python asf_hp_cost_estimator_model/pipeline/model_training/fit_cost_prediction_intervals.py --lower_quantile 0.1 --upper_quantile 0.9 --test False
"""

# package imports
import numpy as np
import pandas as pd
import os
import boto3
from datetime import datetime
import pickle
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

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
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        help="Run in test mode with reduced data for quick testing.",
    )
    return parser.parse_args()


def load_and_prepare_data() -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
    """
    Loads and preprocesses all necessary data for training the models

    Returns:
        tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]: Feature matrix (X) and target vector (y) for training, and for the latest quarter validation set.
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

    # Define features and target
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Identifying first date of newest quarter to create dataset with only new quarter data
    valid_dates = mcs_epc_data[mcs_epc_data["commission_date"] < datetime.today()]
    max_date = valid_dates["commission_date"].max()
    first_day_of_quarter = max_date.to_period("Q").start_time

    # Creating a validation set from the newest quarter of data
    new_quarter_data = mcs_epc_data[
        mcs_epc_data["commission_date"] >= first_day_of_quarter
    ]
    # Processing without removing or winsorising outliers
    new_quarter_data = process_data_before_modelling(
        new_quarter_data,
        postcodes_data,
        cpi_quarterly_df,
        processing_validation_set=True,
    )
    X_latest_quarter = new_quarter_data[numeric_features + categorical_features]
    y_latest_quarter = new_quarter_data[target_feature].values.ravel()

    return X, y, X_latest_quarter, y_latest_quarter


def create_df_with_predictions(
    X: np.array,
    y: np.array,
    y_pred_lower: np.array,
    y_pred_upper: np.array,
    y_pred_median: np.array,
) -> pd.DataFrame:
    """
    Creates a DataFrame with predictions for the lower and upper quantiles.
    Args:
        X (np.array): Feature matrix.
        y (np.array): True values of the target variable.
        y_pred_lower (np.array): Predicted lower bounds of the intervals.
        y_pred_upper (np.array): Predicted upper bounds of the intervals.
        y_pred_median (np.array): Predicted median values.
    Returns:
        pd.DataFrame: DataFrame containing true values and predicted bounds.
    """
    predictions_df = pd.DataFrame(
        X, columns=config["numeric_features"] + config["categorical_features"]
    )
    predictions_df["y_true"] = y
    predictions_df["y_pred_lower"] = y_pred_lower
    predictions_df["y_pred_upper"] = y_pred_upper
    predictions_df["y_pred_median"] = y_pred_median
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


def fit_and_save_models(
    lower_quantile: float = 0.1, upper_quantile: float = 0.9, test_mode: bool = True
) -> tuple[str, dict, dict]:
    """
    Loads data, trains model and saves model as pickle.

    Args:
        lower_quantile (float, optional): Lower quantile for cost estimation. Defaults to 0.1.
        upper_quantile (float, optional): Upper quantile for cost estimation. Defaults to 0.9.
        test_mode (bool, optional): If True, runs in test mode with reduced data for quick testing. Defaults to True.
    Returns:
        tuple[str, dict]: today's date (in YYYYMMDD format, used as a folder name for saving models), dictionary of main metrics for the full dataset,
            dictionary of main metrics for the latest quarter dataset.
    """

    X, y, X_latest_quarter, y_latest_quarter = load_and_prepare_data()

    # Define quantile configurations
    quantile_configs = {
        "lower_bound": lower_quantile,
        "median": 0.5,
        "upper_bound": upper_quantile,
    }

    trained_models = {}
    predictions = {}
    predictions_latest_quarter = {}

    # Train models for each quantile
    for label, q in quantile_configs.items():
        logging.info(f"Training {label} quantile model (q={q})...")
        model = set_up_pipeline(quantile=q, model_bound=f"{label}_model")
        model.fit(X, y)
        trained_models[label] = model
        predictions[label] = model.predict(X)
        predictions_latest_quarter[label] = model.predict(X_latest_quarter)

    # Evaluate models
    logging.info("Evaluating model performance...")
    main_metrics = compute_metrics(
        y=y,
        y_pred_lower=predictions["lower_bound"],
        y_pred_upper=predictions["upper_bound"],
        y_pred_median=predictions["median"],
        alpha_lower=lower_quantile,
        alpha_upper=upper_quantile,
        log_metrics=True,
        save_histogram=True,
    )
    logging.info("Evaluating model performance on latest quarter data...")
    latest_quarter_metrics = compute_metrics(
        y=y_latest_quarter,
        y_pred_lower=predictions_latest_quarter["lower_bound"],
        y_pred_upper=predictions_latest_quarter["upper_bound"],
        y_pred_median=predictions_latest_quarter["median"],
        alpha_lower=lower_quantile,
        alpha_upper=upper_quantile,
        log_metrics=True,
        save_histogram=True,
    )

    # Prepare outputs
    today_date = datetime.today().strftime("%Y%m%d")
    s3_bucket = "asf-hp-cost-estimator-model"
    s3_prefix = f"outputs/model/{today_date}"

    # Save predictions
    if not test_mode:
        logging.info("Saving predictions to S3...")
        predictions_df = create_df_with_predictions(
            X=X,
            y=y,
            y_pred_lower=predictions["lower_bound"],
            y_pred_upper=predictions["upper_bound"],
            y_pred_median=predictions["median"],
        )

        predictions_key = os.path.join(
            s3_prefix, f"predictions_{lower_quantile}_{upper_quantile}.csv"
        )
        predictions_df.to_csv(f"s3://{s3_bucket}/{predictions_key}", index=False)

        # Save models to S3
        logging.info("Saving trained models to S3...")
        s3_resource = boto3.resource("s3")

        for label, model in trained_models.items():
            model_key = os.path.join(
                s3_prefix, f"regressor_q{quantile_configs[label]}.pkl"
            )
            try:
                s3_resource.Object(s3_bucket, str(model_key)).put(
                    Body=pickle.dumps(model)
                )
                logging.info(f"Uploaded {label} model to s3://{s3_bucket}/{model_key}")
            except Exception as e:
                logging.error(f"Failed to upload {label} model: {e}")

        logging.info(
            f"All models and predictions saved to: s3://{s3_bucket}/{s3_prefix}"
        )

    return today_date, main_metrics, latest_quarter_metrics


def run_drift_analysis(
    todays_model_performance: dict, today_date: str, test_mode: bool = True
):
    """
    Compares today's model performance with the most recent model performance stored in S3.
    Args:
        todays_model_performance (dict): Dictionary containing today's model performance metrics.
    """

    # If there is no previous model performance data, log a warning and exit
    try:
        # Get all folders in s3://asf-hp-cost-estimator-model/outputs/model/
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(
            Bucket="asf-hp-cost-estimator-model", Prefix="outputs/model/", Delimiter="/"
        )
        all_folders = [
            prefix["Prefix"].split("/")[2]
            for prefix in response.get("CommonPrefixes", [])
        ]

        # Get the most recent folder
        most_recent_folder = max(all_folders)

        # Read json file from s3://asf-hp-cost-estimator-model/outputs/ called model_performance.json
        models_performance = pd.read_json(
            f"s3://asf-hp-cost-estimator-model/outputs/model_performance.json"
        )

        # Append todays_model_performance to models_performance
        models_performance[today_date] = todays_model_performance

        # Compare results from today with the most recent folder
        logging.info(
            f"Running drift analysis comparing with folder: {most_recent_folder}"
        )

        if not test_mode:
            # Save updated models_performance to s3
            models_performance.to_json(
                f"s3://asf-hp-cost-estimator-model/outputs/model_performance.json",
                orient="columns",
            )

        previous_model_performance = models_performance[most_recent_folder]
        for df in ["full_dataset", "latest_quarter"]:
            logging.info(f"Comparing performance on {df}...")
            today_df_metrics = todays_model_performance[df]
            previous_df_metrics = previous_model_performance[df]
            for key in today_df_metrics.keys():
                # Check if todays_model_performance is more than 10% worse than previous_model_performance
                if today_df_metrics[key] > previous_df_metrics[key] * 1.1:
                    logging.warning(
                        f"Drift detected in {key}: {previous_df_metrics[key]} -> {today_df_metrics[key]}"
                    )
    except Exception as e:
        logging.error(f"No previous model performance data to compare!")


if __name__ == "__main__":
    args = argparse_setup()
    lower_quantile = args.lower_quantile
    upper_quantile = args.upper_quantile
    test = args.test

    today_date, main_metrics, latest_quarter_metrics = fit_and_save_models(
        lower_quantile=lower_quantile, upper_quantile=upper_quantile, test_mode=test
    )

    run_drift_analysis(
        todays_model_performance={
            "full_dataset": main_metrics,
            "latest_quarter": latest_quarter_metrics,
        },
        today_date=today_date,
        test_mode=test,
    )
