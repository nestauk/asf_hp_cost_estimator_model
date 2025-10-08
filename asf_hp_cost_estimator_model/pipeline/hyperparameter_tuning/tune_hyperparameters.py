"""
This script performs hyperparameter tuning for a Gradient Boosting Regressor models for three different quantiles:
- lower quantile
- median
- upper quantile

After tuning the model, it evaluates the model on training, test sets and an unseen validation set of data from.
It loggs various metrics such as mean pinball loss, coverage probability, and interval widths.

Usage:
python asf_hp_cost_estimator_model/pipeline/hyperparameter_tuning/tune_hyperparameters.py --lower_quantile 0.1 --upper_quantile 0.9
"""

# package imports
from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import logging

# local imports
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.utils.model_evaluation_utils import (
    load_and_prepare_data,
    compute_metrics,
)


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


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
    param_grid: Dict[str, Any],
    conf: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning for a Gradient Boosting Regressor for a specific quantile.

    Returns:
        Dict[str, Any]: The best parameters found by the search.
    """
    logging.info(f"--- Starting hyperparameter search for quantile: {quantile} ---")

    scorer = make_scorer(mean_pinball_loss, alpha=quantile, greater_is_better=False)

    gbr = GradientBoostingRegressor(
        loss="quantile", alpha=quantile, random_state=conf["random_state"]
    )

    search = HalvingRandomSearchCV(
        gbr,
        param_grid,
        resource="n_estimators",
        max_resources=500,
        min_resources=50,
        scoring=scorer,
        n_jobs=-1,  # Use all available cores
        random_state=conf["random_state"],
    ).fit(X_train, y_train)

    logging.info(f"Best parameters for quantile {quantile}:")
    logging.info(search.best_params_)

    return search.best_params_


def evaluate_and_log_metrics(
    dataset_name: str,
    y_true: pd.Series,
    X_data: pd.DataFrame,
    models: Dict[str, GradientBoostingRegressor],
    lower_quantile: float,
    upper_quantile: float,
):
    """Generates predictions and logs evaluation metrics for a given dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "Training set", "Test set", "Validation set").
        y_true (pd.Series): True target values.
        X_data (pd.DataFrame): Feature data.
        models (Dict[str, GradientBoostingRegressor]): Dictionary containing the trained models for lower, median, and upper quantiles.
        lower_quantile (float): The lower quantile used for the lower bound model.
        upper_quantile (float): The upper quantile used for the upper bound model.
    """

    logging.info(f"\n----- MODEL EVALUATION RESULTS ON {dataset_name.upper()} -----")

    # Generate predictions
    y_pred_lower = models["lower"].predict(X_data)
    y_pred_median = models["median"].predict(X_data)
    y_pred_upper = models["upper"].predict(X_data)

    # Compute and log metrics
    compute_metrics(
        y=y_true,
        y_pred_lower=y_pred_lower,
        y_pred_median=y_pred_median,
        y_pred_upper=y_pred_upper,
        alpha_lower=lower_quantile,
        alpha_upper=upper_quantile,
    )


def run_hyperparameter_tuning(
    lower_quantile: float, upper_quantile: float, param_grid: Dict[str, Any]
):
    """
    Runs hyperparameter tuning.

    Args:
        lower_quantile (float): lower quantile for the prediction interval.
        upper_quantile (float): upper quantile for the prediction interval.
        param_grid (Dict[str, Any]): grid of hyperparameters to search over.
    """

    # Load and prepare data for modelling
    model_data, validation_set, features = load_and_prepare_data()

    target_feature = config["target_feature"]

    X = model_data[features]
    y = model_data[target_feature].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["random_state"]
    )

    # Prepare validation set
    X_val = validation_set[features]
    y_val = validation_set[target_feature].values

    # Define the models we want to tune
    models_to_tune = {
        "lower": lower_quantile,
        "median": 0.5,
        "upper": upper_quantile,
    }

    # Tune each model and store the best parameters
    best_params = {}
    for name, quantile in models_to_tune.items():
        best_params[name] = tune_model(X_train, y_train, quantile, param_grid, config)

    # Train final models on the full training set using the best parameters
    logging.info("\nTraining final models with best hyperparameters...")
    final_models = {}
    for name, params in best_params.items():
        quantile = models_to_tune[name]
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            random_state=config["random_state"],
            **params,
        )
        model.fit(X_train, y_train)
        final_models[name] = model

    # Evaluate models on all three datasets
    evaluate_and_log_metrics(
        "Training set", y_train, X_train, final_models, lower_quantile, upper_quantile
    )
    evaluate_and_log_metrics(
        "Test set", y_test, X_test, final_models, lower_quantile, upper_quantile
    )
    evaluate_and_log_metrics(
        "Validation set", y_val, X_val, final_models, lower_quantile, upper_quantile
    )


if __name__ == "__main__":
    args = argparse_setup()
    lower_quantile = args.lower_quantile
    upper_quantile = args.upper_quantile

    # Define the parameter grid for hyperparameter tuning
    param_grid = dict(
        learning_rate=[0.01, 0.05, 0.1, 0.2],
        max_depth=[3, 5, 10, 20],
        min_samples_leaf=[1, 5, 10, 100, 1000],
        min_samples_split=[2, 10, 50, 100, 1000],
    )

    logging.info("Starting hyperparameter tuning process...")
    run_hyperparameter_tuning(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        param_grid=param_grid,
    )
