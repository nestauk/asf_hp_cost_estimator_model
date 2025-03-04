"""
Hyperparameter tuning script to find the best hyperparameters for the model.
"""

# package imports
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.inspection import (
    PartialDependenceDisplay,
)  # required to be imported before IterativeImputer
import logging
from typing import Tuple, List
import os
import datetime as dt
import numpy as np
from itertools import product
from sklearn.pipeline import Pipeline

# local imports
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.pipeline.model_evaluation.conduct_cross_validation import (
    update_error_results,
)
from asf_hp_cost_estimator_model.pipeline.model_training.fit_cost_model import (
    set_up_pipeline,
)
from asf_hp_cost_estimator_model import config, PROJECT_DIR


def get_features() -> Tuple[List[str], List[str], str]:
    """
    Loads numeric, categorical and target feature names from config file..

    Returns:
        Tuple[List[str], List[str], str]: numeric features, categorical features and target feature
    """
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]
    return numeric_features, categorical_features, target_feature


def set_data_parameters_to_tune(installations_data: pd.DataFrame) -> dict:
    """
    Set the data parameters to tune

    Args:
        installations_data (pd.DataFrame): installations data

    Returns:
        dict: dictionary of data parameters to tune
    """
    params = {
        "date_doubling_weights": [
            False,  # False means not doubling weights
            "2020-01-01",
            "2022-04-01",
        ],
        "cost_bounds": [
            (3500, 25000),  # originally set by Chris
            (
                np.nanpercentile(installations_data["cost"], 1),
                np.nanpercentile(installations_data["cost"], 99),
            ),  # 1st and 99th percentiles
        ],
        "number_rooms_bounds": [
            (2, 8),
            "categorical",  # set number of rooms as categorical
        ],
        "floor_area_bounds": [
            (20, 500),
            (
                np.nanpercentile(installations_data["TOTAL_FLOOR_AREA"], 1),
                np.nanpercentile(installations_data["TOTAL_FLOOR_AREA"], 99),
            ),  # 1st and 99th percentiles
            False,  # False means not using floor area as a feature
        ],
        "installations_start_date": [
            "2007-01-01",  # date of first installation
            "2016-01-01",  # when EPC started being made available for Scotland
        ],
    }
    return params


def fit_model(
    model_data: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.array,
    date_double_weights,
) -> Pipeline:
    """
    Fit the model on the training data.

    Args:
        model_data (Pipeline): model data
        X_train (pd.DataFrame): training data
        y_train (np.array): training target
        date_double_weights (str, optional): date from when we double the weights. Defaults to config["date_double_weights"].

    Returns:
        Pipeline: fitted model pipeline
    """

    # To codify increased reliability in data after a certain date double the weight of the samples
    if date_double_weights:
        train_weights = np.where(
            model_data.loc[X_train.index]["commission_date"] >= date_double_weights,
            2,
            1,
        )
        # Set up the sklearn pipeline and fit
        model = set_up_pipeline()
        model.fit(X_train, y_train, regressor__sample_weight=train_weights)

    # Set up the sklearn pipeline and fit
    model = set_up_pipeline()
    model.fit(X_train, y_train)

    return model


def perform_kfold_cross_validation(
    model_data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_feature: str,
    kfold_splits: int,
    date_double_weights: str,
) -> Tuple[List[dict], List[dict]]:
    """
    Performs k-fold cross-validation.

    Args:
        model_data (pd.DataFrame): _description_
        numeric_features (List[str]): _description_
        categorical_features (List[str]): _description_
        target_feature (str): _description_
        kfold_splits (int): _description_
        date_double_weights(str): _description_

    Returns:
        Tuple[List[dict], List[dict]]: _description_
    """
    # Define input and target data
    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Define a K-Fold cross-validator
    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)

    # Initialise result variables
    results_model = []
    first_fold = True

    # K-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if first_fold:
            logging.info(
                "\nSize of training dataset: "
                + str(len(y_train))
                + "\nSize of testing dataset: "
                + str(len(y_test))
                + "\n"
            )
            first_fold = False

        # Fit the model
        model = fit_model(model_data, X_train, y_train, date_double_weights)

        # Predict on the test and train sets
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Calculate the proportion of training data after a fixed date
        after_date = model_data[(model_data["commission_date"] >= "2023-01-01")]
        after_date_train = after_date[after_date.index.isin(X_train.index)]
        proportion_train_after_date = len(after_date_train) / len(X_train)

        # Calculate the proportion of testing data after a fixed date
        after_date_test = np.where(X_test.index.isin(after_date.index), True, False)
        after_date_train = np.where(X_train.index.isin(after_date.index), True, False)

        # Update the results of the model with fold specific results
        results_model_test = update_error_results(
            results_model,
            y_test,
            y_test_pred,
            proportion_train_after_date,
            after_date_test,
        )

        results_model_train = update_error_results(
            results_model,
            y_train,
            y_train_pred,
            proportion_train_after_date,
            after_date_train,
        )

    return results_model_test, results_model_train


if __name__ == "__main__":
    results_test = {}
    results_train = {}

    # Load data and set data parameters to tune
    mcs_epc_data = get_enhanced_installations_data()
    data_params = set_data_parameters_to_tune(mcs_epc_data)

    # Model hyperparameters to tune
    model_params = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.7, 0.8, 0.9],
        "min_samples_split": [2, 5, 10],
    }

    # Parameters to tune include model hyperparameters and data parameters
    all_params = data_params | model_params

    # Generating all combinations of parameters to tune through grid search
    param_combinations = list(product(*all_params.values()))

    # Grid search: Iterate through each possible combination
    for combination in param_combinations:
        param_dict = dict(zip(all_params.keys(), combination))

        # Data is required to be loaded for each combination
        mcs_epc_data = get_enhanced_installations_data()
        postcodes_data = get_postcodes_data()

        # Define exclusion criteria
        exclusion_dict = {
            "cost_lower_bound": param_dict["cost_bounds"][0],
            "cost_upper_bound": param_dict["cost_bounds"][1],
            "NUMBER_HABITABLE_ROOMS_lower_bound": param_dict["number_rooms_bounds"][0],
            "NUMBER_HABITABLE_ROOMS_upper_bound": param_dict["number_rooms_bounds"][1],
            "PROPERTY_TYPE_allowed_list": ["House", "Bungalow"],
        }

        if param_dict["floor_area_bounds"]:
            exclusion_dict["TOTAL_FLOOR_AREA_lower_bound"] = param_dict[
                "floor_area_bounds"
            ][0]
            exclusion_dict["TOTAL_FLOOR_AREA_upper_bound"] = param_dict[
                "floor_area_bounds"
            ][1]

        # Processing data before modelling
        model_data = process_data_before_modelling(
            mcs_epc_data=mcs_epc_data,
            postcodes_data=postcodes_data,
            exclusion_criteria_dict=exclusion_dict,
            min_date=param_dict["installations_start_date"],
            rooms_as_categorical=param_dict["number_rooms_bounds"],
        )

        # Defining features and target
        numeric_features, categorical_features, target_feature = get_features()

        if not param_dict["floor_area_bounds"]:  # False means not using floor area as a feature
            numeric_features = [
                feat for feat in numeric_features if feat != "TOTAL_FLOOR_AREA"
            ]

        if param_dict["number_rooms_bounds"] == "categorical":
            numeric_features = [
                feat for feat in numeric_features if feat != "NUMBER_HABITABLE_ROOMS"
            ]
            categorical_features = categorical_features + [
                "number_of_rooms_2",
                "number_of_rooms_3",
                "number_of_rooms_4",
                "number_of_rooms_5",
                "number_of_rooms_6",
                "number_of_rooms_7",
            ]

        # Performing k-fold cross-validation
        results_model_test, results_model_train = perform_kfold_cross_validation(
            model_data=model_data,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_feature=target_feature,
            kfold_splits=5,
            date_double_weights=param_dict["date_doubling_weights"],
        )

        # Averaging the results for the specific combination across the different folds
        results_model_test = pd.DataFrame(results_model_test).mean(axis=0).round(2)
        results_test[combination] = results_model_test
        results_train[combination] = results_model_train

    # Saving the results
    pd.DataFrame(results_test).to_csv(
        os.path.join(
            PROJECT_DIR, "outputs/model_evaluation/hyperparameter_tuning_test.csv"
        )
    )
    pd.DataFrame(results_train).to_csv(
        os.path.join(
            PROJECT_DIR, "outputs/model_evaluation/hyperparameter_tuning_train.csv"
        )
    )
