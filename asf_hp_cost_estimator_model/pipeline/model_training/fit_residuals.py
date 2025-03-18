"""
Fits two quantile regression models to the residuals of the cost model to
compute prediction intervals for the cost model.
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
import os

# local imports
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model import config, PROJECT_DIR


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


def save_predictions_and_residuals(
    y: np.array,
    y_pred: np.array,
    lower_bound: np.array,
    upper_bound: np.array,
    lower_residuals: np.array,
    upper_residuals: np.array,
):
    """
    Save predictions and prediction intervals to a csv file.

    Args:
        y (np.array): true values
        y_pred (np.array): predicted values
        lower_bound (np.array): lower bounds of prediction interval
        upper_bound (np.array): upper bounds of prediction interval
        lower_residuals (np.array): lower bounds of residuals
        upper_residuals (np.array): upper bounds of residuals
    """
    results = pd.DataFrame(
        {
            "y": y,
            "y_pred": y_pred,
            "prediction_lower_bound": lower_bound,
            "prediction_upper_bound": upper_bound,
            "lower_residuals": lower_residuals,
            "upper_residuals": upper_residuals,
        }
    )
    results.to_csv(os.path.join(PROJECT_DIR, "outputs/data/prediction_intervals.csv"))


def visualise_prediction_against_perfect_prediction(y: np.array, y_pred: np.array):
    """
    Visualise the model's predictions against perfect predictions.

    Args:
        y (np.array): true values
        y_pred (np.array): predicted values
    """
    plt.scatter(y, y_pred, color="blue", label="Predictions")
    plt.plot(
        [y.min(), y.max()], [y.min(), y.max()], color="red", label="Perfect Predictions"
    )
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.savefig(
        os.path.join(
            PROJECT_DIR, "outputs/figures/prediction_vs_perfect_prediction.png"
        )
    )
    plt.close()


def visualise_prediction_intervals(
    y_pred: np.array, lower_bound: np.array, upper_bound: np.array
):
    """
    Visualise prediction intervals for the cost model.

    Args:
        y_pred (np.array): predicted values
        lower_bound (np.array): lower bound of prediction interval
        upper_bound (np.array): upper bound of prediction interval
    """
    plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predictions")
    plt.fill_between(
        range(len(y_pred)),
        lower_bound,
        upper_bound,
        color="lightgray",
        alpha=0.5,
        label="Quantile Range (10%-90%)",
    )
    plt.xlabel("index")
    plt.ylabel("Predicted")
    plt.legend()
    plt.savefig(os.path.join(PROJECT_DIR, "outputs/figures/prediction_intervals.png"))
    plt.close()


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
    plt.savefig(os.path.join(PROJECT_DIR, "outputs/figures/histogram_residuals.png"))
    plt.close()


def visualise_qq_plot(residuals: np.array):
    """
    Visualise Q-Q plot of residuals.

    Args:
        residuals (np.array): model residuals
    """

    sm.qqplot(residuals)  # , line="45")
    plt.title("Q-Q Plot of Residuals")
    plt.savefig(os.path.join(PROJECT_DIR, "outputs/figures/qq_plot_residuals.png"))
    plt.close()


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
    model_data["residuals"] = residuals

    logging.info("Fitting quantile regression to residuals (10th and 90th percentiles)")
    quantiles = [0.1, 0.9]
    qr_models = {}

    all_features = numeric_features + categorical_features
    model_data = model_data[all_features + ["residuals"]]
    all_features = " + ".join(all_features)

    for q in quantiles:
        print(f"quantile {q}")
        # qr = QuantileRegressor(quantile=q, solver="highs", alpha=0)
        # qr.fit(X, residuals)
        qr = smf.quantreg(f"residuals ~ {all_features}", model_data).fit(q=q)

        qr_models[q] = qr

    logging.info("Making predictions for the different quantiles")

    # Predict residual quantiles
    lower_residuals = qr_models[0.1].predict(X)
    upper_residuals = qr_models[0.9].predict(X)

    # Adjust predictions based on residual quantiles
    lower_bound = y_pred + lower_residuals
    upper_bound = y_pred + upper_residuals

    logging.info("Saving predictions and prediction intervals to a csv")
    save_predictions_and_residuals(
        y=y,
        y_pred=y_pred,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_residuals=lower_residuals,
        upper_residuals=upper_residuals,
    )

    logging.info("Visualising prediction against perfect prediction")
    visualise_prediction_against_perfect_prediction(y=y, y_pred=y_pred)

    logging.info("visualising prediction intervals")
    visualise_prediction_intervals(
        y_pred=y_pred, lower_bound=lower_bound, upper_bound=upper_bound
    )

    logging.info("Checking the residuals distribution")
    # it should be a normal distribution centered around zero with constant variance
    visualise_histogram_of_residuals(residuals=residuals)

    logging.info("Visualising Q-Q plot of residuals")
    # residuals should follow a straight line
    visualise_qq_plot(residuals=residuals)

    logging.info("Computing coverage probability accuracy")
    # Check if true values fall within predicted intervals
    # For 10th to 90th percentile, about 80% of the true values should fall within the interval
    coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
    logging.info(f"Coverage Probability: {coverage:.2%}")

    logging.info("Computing pinball loss for quantile regression")
    # It tells us how well our quantile regression predicts different parts of the residual distribution
    loss_10 = mean_pinball_loss(
        y_true=residuals, y_pred=qr_models[0.1].predict(X), alpha=0.1
    )
    loss_90 = mean_pinball_loss(
        y_true=residuals, y_pred=qr_models[0.9].predict(X), alpha=0.9
    )

    logging.info(f"Pinball Loss (10th percentile): {loss_10:.4f}")
    logging.info(f"Pinball Loss (90th percentile): {loss_90:.4f}")
