"""
Script to conduct k-fold cross-validation for the residuals model by assessing the
average coverage probability across the k-folds on the test and training data i.e.
the proportion of true values that fall within the prediction interval created for each fold.
When considering the 10th and 90th quantiles, 80% of the actual values should fall within the prediction interval.

About the residuals model (created for each fold):
- the residuals of the cost model are being modelled using a quantile regressions;
- two quantile regressions are fitted to predict the residuals of the cost model at the 10th and 90th quantiles;
- the two models are then used to predict the lower and upper bounds of the prediction interval for the cost model.
- the coverage probability of the prediction interval is calculated for the test and training data.
"""

# package imports
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import QuantileRegressor
import numpy as np
import logging
from sklearn.pipeline import Pipeline

# local imports
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.pipeline.model_training.fit_cost_model import (
    set_up_pipeline,
)


def fit_model(
    model_data: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.array,
    date_double_weights: str = config["date_double_weights"],
) -> Pipeline:
    """
    Fit the model on the training data.
    Args:
        model_data (pd.DataFrame): model data
        X_train (pd.DataFrame): training data
        y_train (np.array): training target
        date_double_weights (str, optional): date from when we double the weights. Defaults to config["date_double_weights"].
    Returns:
        Pipeline: fitted model pipeline
    """

    # To codify increased reliability in data after a certain date double the weight of the samples
    train_weights = np.where(
        model_data.loc[X_train.index]["commission_date"] >= date_double_weights,
        2,
        1,
    )

    # Set up the sklearn pipeline and fit
    model = set_up_pipeline()
    model.fit(X_train, y_train, regressor__sample_weight=train_weights)

    return model


def load_model_data() -> pd.DataFrame:
    """
    Processes MCS-EPC data and postcode data to prepare for modelling.

    Returns:
        pd.DataFrame: model data
    """
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    model_data = process_data_before_modelling(mcs_epc_data, postcodes_data)
    return model_data


# Predict residuals for train and test set
def get_prediction_intervals(
    x: np.array, y_pred: np.array, qr_models: dict
) -> tuple[np.array, np.array]:
    """
    Calculate prediction intervals for the cost model based on the
    predicted cost and the residuals predicted by the quantile regression models.

    Args:
        x (np.array): model input data
        y_pred (np.array): cost predictions
        qr_models (dict): dictionary with quantile regression models

    Returns:
        tuple[np.array, np.array]: lower and upper bounds of the prediction interval
    """
    lower_residuals = qr_models[0.1].predict(x)
    upper_residuals = qr_models[0.9].predict(x)
    lower_bound = y_pred - lower_residuals
    upper_bound = y_pred + upper_residuals
    return lower_bound, upper_bound


def calculate_coverage_probability(
    coverage_probs: list,
    y_actual: np.array,
    lower_bound: np.array,
    upper_bound: np.array,
) -> list:
    """
    Calculate the coverage probability of the prediction interval, i.e. the proportion
    of true values that fall within the prediction interval, and add to the list
    of coverage probabilities.

    Args:
        coverage_probs (list): list of coverage probabilities
        y_actual (np.array): true cost values
        lower_bound (np.array): lower bound of prediction interval
        upper_bound (np.array): upper bound of prediction interval

    Returns:
        list: updated list of coverage probabilities
    """
    prob = np.mean((y_actual >= lower_bound) & (y_actual <= upper_bound))
    coverage_probs.append(prob)
    return coverage_probs


if __name__ == "__main__":
    model_data = load_model_data()

    # Define features and target
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Store evaluation metrics for cross-validation
    coverage_probs_test = []
    coverage_probs_train = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train cost model on training data
        cost_model = fit_model()

        # Compute residuals on training data
        cost_y_train_pred = cost_model.predict(X_train)
        residuals = y_train - cost_y_train_pred

        # Fit quantile regression model on residuals
        quantiles = [0.1, 0.9]
        qr_models = {}

        for q in quantiles:
            qr = QuantileRegressor(quantile=q)
            qr.fit(X_train, residuals)
            qr_models[q] = qr

        pred_interval_lower_test, pred_interval_upper_test = get_prediction_intervals(
            X_test, cost_model.predict(X_test), qr_models
        )
        pred_interval_lower_train, pred_interval_upper_train = get_prediction_intervals(
            X_train, cost_model.predict(X_train), qr_models
        )

        coverage_probs_test = calculate_coverage_probability(
            coverage_probs=coverage_probs_test,
            y_actual=y_test,
            lower_bound=pred_interval_lower_test,
            upper_bound=pred_interval_upper_test,
        )
        coverage_probs_train = calculate_coverage_probability(
            coverage_probs=coverage_probs_train,
            y_actual=y_train,
            lower_bound=pred_interval_lower_train,
            upper_bound=pred_interval_upper_train,
        )

    # Average coverage probability across folds
    logging.info(
        f"Average Coverage Probability for test set: {np.mean(coverage_probs_test):.2%}"
    )
    logging.info(
        f"Average Coverage Probability for test set: {np.mean(coverage_probs_train):.2%}"
    )
