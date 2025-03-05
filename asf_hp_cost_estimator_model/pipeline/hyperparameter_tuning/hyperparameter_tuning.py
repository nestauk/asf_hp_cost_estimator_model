"""
Hyperparameter tuning script to find the best hyperparameters for the model.
"""

# package imports
import numpy as np
import pandas as pd
import json
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
from asf_hp_cost_estimator_model.pipeline.hyperparameter_tuning.params_to_tune import (
    model_params,
    data_params,
)
from asf_hp_cost_estimator_model import config, PROJECT_DIR


def fit_model(
    model_data: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.array,
    date_double_weights: str,
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
    # Set up the sklearn pipeline and fit
    model = set_up_pipeline()

    # To codify increased reliability in data after a certain date double the weight of the samples
    if date_double_weights:
        train_weights = np.where(
            model_data.loc[X_train.index]["commission_date"] >= date_double_weights,
            2,
            1,
        )
        model.fit(X_train, y_train, regressor__sample_weight=train_weights)
    else:
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
        model_data (pd.DataFrame): model data
        numeric_features (List[str]): list of numeric features
        categorical_features (List[str]): list of categorical features
        target_feature (str): target feature
        kfold_splits (int): number of folds
        date_double_weights(str): date from when we start doubling the weights for installations

    Returns:
        Tuple[List[dict], List[dict]]: results on the test and train sets
    """
    # Define input and target data
    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Define a K-Fold cross-validator
    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)

    # Initialise result variables
    results_model_train = []
    results_model_test = []
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

        # Calculate the proportion of training data after a fixed date (for evaluating performance on latest prediction)
        after_date = model_data[
            (model_data["commission_date"] >= config["date_for_latest_predictions"])
        ]
        after_date_train = after_date[after_date.index.isin(X_train.index)]
        proportion_train_after_date = len(after_date_train) / len(X_train)

        # Calculate the proportion of testing data after a fixed date
        after_date_test = np.where(X_test.index.isin(after_date.index), True, False)
        after_date_train = np.where(X_train.index.isin(after_date.index), True, False)

        # Update the results of the model with fold specific results
        results_model_test = update_error_results(
            results_model_test,
            y_test,
            y_test_pred,
            proportion_train_after_date,
            after_date_test,
        )

        results_model_train = update_error_results(
            results_model_train,
            y_train,
            y_train_pred,
            proportion_train_after_date,
            after_date_train,
        )

    return results_model_test, results_model_train


if __name__ == "__main__":
    results_test = {}
    results_train = {}

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

        # Getting the pre-defined features and target
        numeric_features = config["numeric_features"]
        categorical_features = config["categorical_features"]

        # Define exclusion criteria
        exclusion_dict = {
            "cost_lower_bound": param_dict["cost_bounds"][0],
            "cost_upper_bound": param_dict["cost_bounds"][1],
            "PROPERTY_TYPE_allowed_list": ["House", "Bungalow"],
        }

        # We exclude the floor area feature if floor_area_bounds is set to False
        if not param_dict[
            "floor_area_bounds"
        ]:  # False means not using floor area as a feature
            numeric_features = [
                feat for feat in numeric_features if feat != "TOTAL_FLOOR_AREA"
            ]
        else:  # otherwise we update the exclusion_dict with floor area bounds provided
            exclusion_dict["TOTAL_FLOOR_AREA_lower_bound"] = param_dict[
                "floor_area_bounds"
            ][0]
            exclusion_dict["TOTAL_FLOOR_AREA_upper_bound"] = param_dict[
                "floor_area_bounds"
            ][1]

        # Defining rooms_as_categorical variable according to number_rooms_bounds param
        if param_dict["number_rooms_bounds"] == "categorical":
            rooms_as_categorical = True

            # We create dummies from NUMBER_HABITABLE_ROOMS feature if number_rooms_bounds is set to categorical
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
        else:  # otherwise we update the exclusion_dict
            rooms_as_categorical = False
            exclusion_dict["NUMBER_HABITABLE_ROOMS_lower_bound"] = param_dict[
                "number_rooms_bounds"
            ][0]
            exclusion_dict["NUMBER_HABITABLE_ROOMS_upper_bound"] = param_dict[
                "number_rooms_bounds"
            ][1]

        # Processing data before modelling
        model_data = process_data_before_modelling(
            mcs_epc_data=mcs_epc_data,
            postcodes_data=postcodes_data,
            exclusion_criteria_dict=exclusion_dict,
            min_date=param_dict["installations_start_date"],
            rooms_as_categorical=rooms_as_categorical,
        )

        # Performing k-fold cross-validation
        results_model_test, results_model_train = perform_kfold_cross_validation(
            model_data=model_data,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_feature=config["target_feature"],
            kfold_splits=5,
            date_double_weights=param_dict["date_doubling_weights"],
        )

        # Averaging the results for the specific combination across the different folds
        results_test[", ".join(map(str, combination))] = (
            pd.DataFrame(results_model_test).mean().round(2).to_dict()
        )
        results_train[", ".join(map(str, combination))] = (
            pd.DataFrame(results_model_test).mean().round(2).to_dict()
        )

    # Saving the results
    with open(
        os.path.join(
            PROJECT_DIR, "outputs/model_evaluation/hyperparameter_tuning_test.json"
        ),
        "w",
    ) as f:
        json.dump(results_test, f)
    with open(
        os.path.join(
            PROJECT_DIR, "outputs/model_evaluation/hyperparameter_tuning_train.json"
        ),
        "w",
    ) as f:
        json.dump(results_train, f)
