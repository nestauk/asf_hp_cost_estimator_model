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
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Any
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


def create_model(
    quantile: float, model_params: Dict[str, Any], random_state: int
) -> GradientBoostingRegressor:
    """Instantiates a GradientBoostingRegressor model for a specific quantile.

    Args:
        quantile (float): The quantile to be predicted (e.g., 0.1 for 10th percentile).
        model_params (Dict[str, Any]): Hyperparameters for the model.
        random_state (int): Random state for reproducibility.

    Returns:
        GradientBoostingRegressor: Configured model instance.
    """

    return GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        random_state=random_state,
        **model_params,
    )


def log_results(title: str, results_df: pd.DataFrame):
    """Calculates and logs the mean of the results.

    Args:
        title (str): Title for the log section.
        results_df (pd.DataFrame): DataFrame containing results from cross-validation folds.
    """
    mean_results = results_df.mean()
    logging.info(f"----- {title.upper()} -----")
    logging.info(
        f"Mean pinball loss (Lower): {mean_results['mean_pinball_loss_lower_q']:.2f}"
    )
    logging.info(
        f"Mean pinball loss (Median): {mean_results['mean_pinball_loss_median']:.2f}"
    )
    logging.info(
        f"Mean pinball loss (Upper): {mean_results['mean_pinball_loss_upper_q']:.2f}"
    )
    logging.info(f"Coverage probability: {mean_results['coverage']:.2%}")
    logging.info(f"Interval width: £{mean_results['interval_width']:,.2f}")
    logging.info(
        f"Proportion of samples where median is outside bounds: {mean_results['prop_samples_median_outside_bounds']:.2%}"
    )
    logging.info(
        f"Proportion of samples where median is closer to lower bound than upper bound: {mean_results['prop_samples_median_closer_to_lower_bound']:.2f}"
    )
    logging.info("------------")


def run_cross_validation(lower_quantile: float, upper_quantile: float):
    """Runs k-fold cross-validation to evaluate model performance.
    Args:
        lower_quantile (float): Lower quantile for prediction intervals.
        upper_quantile (float): Upper quantile for prediction intervals.
    """
    model_data, validation_set, features = load_and_prepare_data()

    X = model_data[features]
    y = model_data[config["target_feature"]].values

    # Prepare validation set features and target
    X_val = validation_set[features]
    y_val = validation_set[config["target_feature"]].values

    kf = KFold(
        n_splits=config["kfold_splits"],
        shuffle=True,
        random_state=config["random_state"],
    )

    # Define the models to train in a structured way
    models_to_train = {
        "lower": {
            "quantile": lower_quantile,
            "params": config["hyper_parameters"]["lower_bound_model"],
        },
        "median": {
            "quantile": 0.5,
            "params": config["hyper_parameters"]["median_model"],
        },
        "upper": {
            "quantile": upper_quantile,
            "params": config["hyper_parameters"]["upper_bound_model"],
        },
    }

    # Use a dictionary to store results from all folds
    results = {"train": [], "test": [], "validation": []}

    logging.info(f"Starting {config['kfold_splits']}-fold cross-validation...")
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        logging.info(f"--- Fold {i+1}/{config['kfold_splits']} ---")

        # Splitting the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Predictions for the current fold
        y_preds = {"train": {}, "test": {}, "validation": {}}

        # Train models for each quantile
        for name, model_info in models_to_train.items():
            model = create_model(
                quantile=model_info["quantile"],
                model_params=model_info["params"],
                random_state=config["random_state"],
            )
            model.fit(X_train, y_train)

            # Generate and store predictions for all data splits
            y_preds["train"][name] = model.predict(X_train)
            y_preds["test"][name] = model.predict(X_test)
            y_preds["validation"][name] = model.predict(X_val)

        # Evaluate and store metrics for each data split
        results["train"].append(
            compute_metrics(
                y=y_train,
                y_pred_lower=y_preds["train"]["lower"],
                y_pred_median=y_preds["train"]["median"],
                y_pred_upper=y_preds["train"]["upper"],
                alpha_lower=lower_quantile,
                alpha_upper=upper_quantile,
                log_metrics=False,
            )
        )
        results["test"].append(
            compute_metrics(
                y=y_test,
                y_pred_lower=y_preds["test"]["lower"],
                y_pred_median=y_preds["test"]["median"],
                y_pred_upper=y_preds["test"]["upper"],
                alpha_lower=lower_quantile,
                alpha_upper=upper_quantile,
                log_metrics=False,
            )
        )
        results["validation"].append(
            compute_metrics(
                y=y_val,
                y_pred_lower=y_preds["validation"]["lower"],
                y_pred_median=y_preds["validation"]["median"],
                y_pred_upper=y_preds["validation"]["upper"],
                alpha_lower=lower_quantile,
                alpha_upper=upper_quantile,
                log_metrics=False,
            )
        )

    logging.info("Cross-validation finished. Aggregating results...")
    results_df_train = pd.DataFrame(results["train"])
    results_df_test = pd.DataFrame(results["test"])
    results_df_validation = pd.DataFrame(results["validation"])

    # Logging the final averaged results
    log_results("Training set CV Results", results_df_train)
    log_results("Test set CV Results", results_df_test)
    log_results("Validation set CV Results", results_df_validation)


if __name__ == "__main__":
    args = argparse_setup()
    lower_quantile = args.lower_quantile
    upper_quantile = args.upper_quantile

    run_cross_validation(lower_quantile, upper_quantile)
