"""
This script performs hyperparameter tuning and cross validation for Gradient Boosting Regressor models for three different quantiles:
- lower quantile, e.g. 10th percentile
- median (50th percentile)
- upper quantile, e.g. 90th percentile

The data is split into training and two hold-out test sets. One test set is a 20% random sample of most recent quarter of data. All of the remaining data is split into training and test set.
Hyperparameter tuning is performed using Halving Random Search with cross-validation on the training set.

After tuning the models, they are evaluated on the training, test set and most recent quarter hold-out test set.
The script loggs various metrics such as mean pinball loss, coverage probability, and interval widths.

The best hyperparameters and evaluation metrics are then saved as CSV files to S3.

This script can be run from the command line, allowing for custom quantiles to be specified (and whether to run in test mode):
    python asf_hp_cost_estimator_model/pipeline/hyperparameter_tuning/tune_hyperparameters.py --lower_quantile 0.1 --upper_quantile 0.9 --test False
"""

# package imports
import yaml
import os
from typing import Union, Dict, List, Tuple
import pandas as pd
from datetime import datetime
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import logging

# local imports
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model import PROJECT_DIR
from asf_hp_cost_estimator_model.utils.model_evaluation_utils import (
    compute_metrics,
)
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


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
    param_grid: Dict[str, List[Union[int, float]]],
) -> Dict[str, Union[int, float]]:
    """
    Performs hyperparameter tuning for a Gradient Boosting Regressor for a specific quantile.
    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        quantile (float): The quantile to be predicted (e.g., 0.1 for 10th percentile).
        param_grid (Dict[str, List[Union[int, float]]]): Grid of hyperparameters to search over.

    Returns:
        Dict[str, Union[int, float]]: Best hyperparameters found during the search.
    """
    logging.info(f"--- Starting hyperparameter search for quantile: {quantile} ---")

    scorer = make_scorer(mean_pinball_loss, alpha=quantile, greater_is_better=False)

    gbr = GradientBoostingRegressor(
        loss="quantile", alpha=quantile, random_state=config["random_state"]
    )

    search = HalvingRandomSearchCV(
        gbr,
        param_grid,
        resource="n_estimators",
        max_resources=500,
        min_resources=50,
        scoring=scorer,
        n_jobs=-1,  # Use all available cores
        random_state=config["random_state"],
    ).fit(X_train, y_train)

    logging.info(search.best_params_)

    return search


def split_data(mcs_epc_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples a random 20% of the data from the newest quarter of data to create a hold-out test set.

    The remaining data (all but the hold-out set) is used for hyperparameter tuning and cross-validation.


    Args:
        mcs_epc_data (pd.DataFrame): DataFrame containing the installations data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame for CV/hyperparameter tuning and DataFrame for hold-out test set.
    """
    # Identifying first date of newest quarter to create a hold-out validation set from the most recent data
    valid_dates = mcs_epc_data[mcs_epc_data["commission_date"] < datetime.today()]
    max_date = valid_dates["commission_date"].max()
    first_day_of_quarter = max_date.to_period("Q").start_time

    # Creating a validation set from the newest quarter of data
    new_quarter_data = mcs_epc_data[
        mcs_epc_data["commission_date"] >= first_day_of_quarter
    ]
    new_quarter_hold_out_set = new_quarter_data.sample(
        frac=0.2, random_state=config["random_state"]
    )
    mcs_epc_data = mcs_epc_data.drop(new_quarter_hold_out_set.index)

    return mcs_epc_data, new_quarter_hold_out_set


def load_and_prepare_data(
    model_data: pd.DataFrame,
    test_set: pd.DataFrame,
    new_quarter_hold_out_set: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads and preprocesses all necessary data prior to hyperparameter tuning or cross validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Processed model data, validation set, and list of feature names.
    """

    logging.info("Loading and processing data...")

    cpi_df = get_cpi_data()
    postcodes_data = get_postcodes_data()

    cpi_quarterly_df = get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=config["cpi_data"]["cpi_reference_year"],
        cpi_df=cpi_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )

    # Hold-out test set from newest quarter is processed without removing or winsorising outliers
    new_quarter_hold_out_set = process_data_before_modelling(
        mcs_epc_data=new_quarter_hold_out_set,
        postcodes_data=postcodes_data,
        cpi_quarterly_df=cpi_quarterly_df,
        min_date=config["min_date"],
        remove_or_winsorise_samples=False,
    )

    # Overall test set is processed without removing or winsorising outliers
    test_set = process_data_before_modelling(
        mcs_epc_data=test_set,
        postcodes_data=postcodes_data,
        cpi_quarterly_df=cpi_quarterly_df,
        min_date=config["min_date"],
        remove_or_winsorise_samples=False,
    )

    # Processing the rest of the data for modelling
    model_data = process_data_before_modelling(
        mcs_epc_data=model_data,
        postcodes_data=postcodes_data,
        cpi_quarterly_df=cpi_quarterly_df,
        exclusion_criteria_dict=config["exclusion_criteria"],
        min_date=config["min_date"],
        remove_or_winsorise_samples=True,
        winsorise=config["winsorise_outliers"],
    )

    logging.info(f"Model data size: {model_data.shape[0]}")
    logging.info(f"Test set size: {test_set.shape[0]}")
    logging.info(f"New quarter hold-out set size: {new_quarter_hold_out_set.shape[0]}")

    features = config["numeric_features"] + config["categorical_features"]

    return model_data, test_set, new_quarter_hold_out_set, features


def run_hyperparameter_tuning(
    lower_quantile: float,
    upper_quantile: float,
    param_grid: Dict[str, List[Union[int, float]]],
    test_mode: bool = True,
):
    """
    Runs hyperparameter tuning.

    Args:
        lower_quantile (float): lower quantile for the prediction interval.
        upper_quantile (float): upper quantile for the prediction interval.
        param_grid (Dict[str, List[Union[int, float]]): grid of hyperparameters to search over.
        test_mode (bool, optional): whether to run in test mode (doesn't save data). Defaults to True.
    """
    mcs_epc_data = get_enhanced_installations_data()
    model_data, new_quarter_hold_out_set = split_data(mcs_epc_data=mcs_epc_data)

    # Further split model_data into training and test sets
    test_set = model_data.sample(frac=0.2)
    model_data = model_data.drop(test_set.index)

    # Load and prepare data for modelling
    model_data, test_set, new_quarter_hold_out_set, features = load_and_prepare_data(
        model_data, test_set, new_quarter_hold_out_set
    )

    target_feature = config["target_feature"]
    # Ensure target feature is not in features list
    if target_feature in features:
        features.remove(target_feature)

    # Separating features and target variable
    X = model_data[features]
    y = model_data[target_feature].values.ravel()

    X_test = test_set[features]
    y_test = test_set[target_feature].values.ravel()

    X_new_quarter_hold_out = new_quarter_hold_out_set[features]
    y_new_quarter_hold_out = new_quarter_hold_out_set[target_feature].values.ravel()

    # Define the models we want to tune
    models_to_tune = {
        "lower": lower_quantile,
        "median": 0.5,
        "upper": upper_quantile,
    }

    # Tune each model and store the best parameters
    best_params = {}
    cv_scores = {}
    for name, quantile in models_to_tune.items():
        # Perform hyperparameter tuning and output best model info
        tuned_model = tune_model(X, y, quantile, param_grid)

        # Best hyperparameters for the current quantile model
        best_params[name] = tuned_model.best_params_

        # Mean cross-validated score (mean pinball loss) of the best estimator
        cv_scores_name = f"mean_pinball_loss_{name}"
        cv_scores_name += "_q" if name != "median" else ""
        cv_scores[cv_scores_name] = -tuned_model.best_score_

    # Train final models on the full training set using the best parameters
    logging.info("Training final models with best hyperparameters...")
    final_models = {}
    for name, params in best_params.items():
        quantile = models_to_tune[name]
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            random_state=config["random_state"],
            **params,
        )
        model.fit(X, y)
        final_models[name] = model

    # Generate predictions for training, test, and new quarter hold-out sets
    predictions = {}
    for name, model in final_models.items():
        predictions[name] = {}
        predictions[name]["train"] = model.predict(X)
        predictions[name]["test"] = model.predict(X_test)
        predictions[name]["new_quarter_hold_out"] = model.predict(
            X_new_quarter_hold_out
        )

    # Compute and log evaluation metrics
    metrics = {}
    metrics["cross_validation_results"] = cv_scores
    logging.info(f"Cross-validated mean pinball loss scores: {cv_scores}")
    print_ = {
        "train": "Training",
        "test": "Test",
        "new_quarter_hold_out": "Test (new quarter)",
    }
    for key in print_.keys():
        metrics[key] = compute_metrics(
            dataset_name=f"{print_[key]} set",
            y=(
                y
                if key == "train"
                else (y_test if key == "test" else y_new_quarter_hold_out)
            ),
            y_pred_lower=predictions["lower"][key],
            y_pred_median=predictions["median"][key],
            y_pred_upper=predictions["upper"][key],
            alpha_lower=lower_quantile,
            alpha_upper=upper_quantile,
        )

    # Save the best hyperparameters and metrics to S3 if not in test mode
    if test_mode:
        logging.info("Saving best hyperparameters and evaluation metrics to S3...")
        today_date = datetime.today().strftime("%Y%m%d")
        best_params_df = pd.DataFrame(best_params).T
        best_params_df.to_csv(
            f"s3://asf-hp-cost-estimator-model/outputs/model/{today_date}/best_hyperparameters_{lower_quantile}_{upper_quantile}.csv",
        )
        metrics_df = pd.DataFrame(metrics).T

        metrics_df.to_csv(
            f"s3://asf-hp-cost-estimator-model/outputs/model/{today_date}/model_evaluation_metrics_{lower_quantile}_{upper_quantile}.csv",
        )

    # Update config with best hyperparameters
    # with open(os.path.join(PROJECT_DIR,"asf_hp_cost_estimator_model/config/base.yaml"), "w") as f:
    #     yaml.safe_dump(config, f)


if __name__ == "__main__":
    args = argparse_setup()
    lower_quantile = args.lower_quantile
    upper_quantile = args.upper_quantile
    test = args.test

    logging.info("Starting hyperparameter tuning process...")
    run_hyperparameter_tuning(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        param_grid=config["param_grid"],
        test_mode=test,
    )
