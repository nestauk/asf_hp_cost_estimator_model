"""
Fits a quantile regression model to the residuals of the cost model and computes prediction intervals for the cost model.
Results are visualised and metrics are calculated to evaluate the model's performance.
"""

# package imports
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging

# local imports
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model import config


def load_model():
    with open("outputs/models/staging/model.pickle", "rb") as f:
        cost_model = pickle.load(f)
    return cost_model


def load_model_data() -> pd.DataFrame:
    """
    Processes MCS-EPC data and postcode data before modelling.

    Returns:
        pd.DataFrame: model data
    """
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    model_data = process_data_before_modelling(mcs_epc_data, postcodes_data)
    return model_data


def visualise_prediction_intervals(
    y: np.array, y_pred: np.array, lower_bound: np.array, upper_bound: np.array
):
    """
    Visualise prediction intervals for the cost model.

    Args:
        y (np.array): true values
        y_pred (np.array): predicted values
        lower_bound (np.array): lower bound of prediction interval
        upper_bound (np.array): upper bound of prediction interval
    """
    plt.scatter(y, y_pred, color="blue", label="Predictions")
    plt.fill_between(
        y,
        lower_bound,
        upper_bound,
        color="lightgray",
        alpha=0.5,
        label="Quantile Range (10%-90%)",
    )
    plt.plot([min(y), max(y)], [min(y), max(y)], "r--", label="Perfect Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.show()


def visualise_histogram_of_residuals(residuals: np.array):
    """
    Visualise histogram of residuals.

    Args:
        residuals (np.array): model residuals
    """
    # Histogram of residuals
    plt.hist(residuals, bins=30, color="skyblue", edgecolor="black")
    plt.title("Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


def visualise_qq_plot(residuals: np.array):
    """
    Visualise Q-Q plot of residuals.

    Args:
        residuals (np.array): modeln residuals
    """
    # Q-Q plot for normality
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()


if __name__ == "__main__":
    logging.info("Loading data and cost model")
    # Load model data and model
    model_data = load_model_data()
    cost_model = load_model()

    # Define features and target
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    logging.info("Computing cost predictions and residuals")

    y_pred = cost_model.predict(X)
    residuals = y - y_pred

    logging.info("Fitting quantile regression to residuals (10th and 90th percentiles)")
    quantiles = [0.1, 0.5, 0.9]
    qr_models = {}

    for q in quantiles:
        qr = QuantileRegressor(quantile=q)
        qr.fit(X, residuals)
        qr_models[q] = qr

    logging.info("Making predictions for different the diffeent quantiles")

    # Predict residual quantiles
    lower_residuals = qr_models[0.1].predict(X)
    upper_residuals = qr_models[0.9].predict(X)

    # Adjust predictions based on residual quantiles
    lower_bound = y_pred - lower_residuals
    upper_bound = y_pred + upper_residuals

    logging.info("visualising prediction intervals")
    visualise_prediction_intervals(y, y_pred, lower_bound, upper_bound)

    logging.info("Checking the residuals distribution")
    # it should be a normal distribution centered around zero with constant variance
    visualise_histogram_of_residuals(residuals)

    logging.info("Visualising Q-Q plot of residuals")
    # residuals should follow a straight line
    visualise_qq_plot(residuals)

    logging.info("Computing coverage probability accuracy")
    # Check if true values fall within predicted intervals
    # For 10th to 90th percentile, about 90% of the true values should fall within the interval
    coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
    logging.info(f"Coverage Probability: {coverage:.2%}")

    logging.info("Comptuing pinball loss for quantile regression")
    # Lower pinball loss indicates better quantile regression perfromance;
    # balanced losses across quantiles suggest well calibratted intervals
    # It tells us how well our quantile regression predicts different parts of the residual distribution
    # Calculate pinball loss for each quantile
    loss_10 = mean_pinball_loss(residuals, qr_models[0.1].predict(X), alpha=0.1)
    loss_90 = mean_pinball_loss(residuals, qr_models[0.9].predict(X), alpha=0.9)

    logging.info(f"Pinball Loss (10th percentile): {loss_10:.4f}")
    logging.info(f"Pinball Loss (90th percentile): {loss_90:.4f}")
