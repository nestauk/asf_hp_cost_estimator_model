"""
Script to analyse the model residuals.
One fold is taken from the KFold cross-validation and the residuals are plotted for the numeric and categorical variables.
"""

import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import KFold
from sklearn.inspection import (
    PartialDependenceDisplay,
)  # required to be imported before IterativeImputer
import logging
import numpy as np
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.pipeline.model_training.fit_cost_model import (
    set_up_pipeline,
)
from asf_hp_cost_estimator_model.utils.plotting_utils import (
    plot_residuals_numeric_variables,
    plot_residuals_categorical_variables,
)

min_date = config["min_date"]


def split_train_test_data(
    model_data: pd.DataFrame,
    kfold_splits: int = config["kfold_splits"],
    numeric_features: List[str] = config["numeric_features"],
    categorical_features: List[str] = config["categorical_features"],
    target_feature: str = config["target_feature"],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits the data into training and test sets for a specific fold.

    Args:
        model_data (pd.DataFrame): model ready data
        kfold_splits (int, optional): Number of splits. Defaults to config["kfold_splits"].
        numeric_features (List[str], optional): list of numeric features. Defaults to config["numeric_features"].
        categorical_features (List[str], optional): List of categorical features. Defaults to config["categorical_features"].
        target_feature (str, optional): Name of target feature. Defaults to config["target_feature"].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: tuple with X_train, X_test, y_train, y_test
    """
    # Define input and target data
    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Define a K-Fold cross-validator and take a single fold
    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)
    train_index, test_index = next(iter(kf.split(X)))

    # Split the data into train and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test


def fit_model_and_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_of_data: pd.DataFrame,
    date_double_weights: str = config["date_double_weights"],
) -> np.ndarray:
    """
    Fits the model and predicts on a new sample of data.

    Args:
        X_train (pd.DataFrame): training data
        y_train (np.ndarray): training target
        sample_of_data (pd.DataFrame): new sample of data
        date_double_weights (str, optional): date from when we double the weights. Defaults to config["date_double_weights"].

    Returns:
        np.ndarray: predictions
    """
    # set up the weights for the training set
    train_weights = np.where(
        model_data.loc[X_train.index]["commission_date"] >= date_double_weights,
        2,
        1,
    )

    # Set up the pipeline to train an air source heat pump cost model
    model = set_up_pipeline()

    # Fit the model
    model.fit(X_train, y_train, regressor__sample_weight=train_weights)

    # Predict on the sample of data
    predictions = model.predict(sample_of_data)

    return predictions


if __name__ == "__main__":
    # Load and process data
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    model_data = process_data_before_modelling(mcs_epc_data, postcodes_data)

    logging.info("Number of samples: " + str(len(model_data)))

    # Split the data into training and test set for indepedent and dependent/target variables
    X_train, X_test, y_train, y_test = split_train_test_data(model_data)

    # Set up the pipeline, fit model and predict on test set
    y_test_pred = fit_model_and_predict(X_train, y_train, X_test)

    # Plot residuals for numeric variables
    for var, label in [
        (y_test_pred, "Predicted cost (£)"),
        (X_test["TOTAL_FLOOR_AREA"], "Floor area (m^2)"),
        (X_test["NUMBER_HABITABLE_ROOMS"], "Number of habitable rooms"),
        (X_test["n_days"], f"Days since {min_date}"),
    ]:
        plot_residuals_numeric_variables(y_test, y_test_pred, var, label)

    # Plot residuals for categorical variables
    built_form_categorical_features = [
        f for f in config["categorical_features"] if "BUILT_FORM" in f
    ]
    age_band_categorical_features = [
        f for f in config["categorical_features"] if "CONSTRUCTION_AGE_BAND" in f
    ]
    property_type_categorical_features = [
        f for f in config["categorical_features"] if "PROPERTY_TYPE" in f
    ]
    region_categorical_features = [
        f for f in config["categorical_features"] if "region_name" in f
    ]
    for vars, label in [
        (
            built_form_categorical_features,
            "Built form",
        ),
        (
            age_band_categorical_features,
            "Year of construction",
        ),
        (property_type_categorical_features, "Property type"),
        (
            region_categorical_features,
            "Region",
        ),
    ]:
        plot_residuals_categorical_variables(
            X_test,
            y_test,
            y_test_pred,
            vars,
            label,
        )
