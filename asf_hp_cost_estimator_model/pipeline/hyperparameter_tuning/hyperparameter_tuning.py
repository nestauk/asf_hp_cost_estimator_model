"""
Script to analyse the residuals of the model.
It plots the residuals for the numeric variables and the categorical variables of one of the folds.
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
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
)
import datetime as dt
from sklearn.pipeline import Pipeline
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
import numpy as np

numeric_features = config["numeric_features"]
categorical_features = config["categorical_features"]
target_feature = config["target_feature"]
mcs_epc_data = get_enhanced_installations_data()
postcodes_data = get_postcodes_data()

date_doubling_weights = [
    False,  # False means not doubling weights
    "2020-01-01",
    "2022-04-01",
]
cost_bounds = [
    (3500, 25000),  # originally set by Chris
    (
        np.percentile(mcs_epc_data["cost"], 1),
        np.percentile(mcs_epc_data["cost"], 99),
    ),  # 1st and 93*49th percentiles
]
number_rooms_bounds = [
    (2, 8),
    (
        np.percentile(mcs_epc_data["NUMBER_HABITABLE_ROOMS"], 1),
        np.percentile(mcs_epc_data["NUMBER_HABITABLE_ROOMS"], 99),
    ),  # 1st and 99th percentiles
]
floor_area_bounds = [
    (20, 500),
    (
        np.percentile(mcs_epc_data["TOTAL_FLOOR_AREA"], 1),
        np.percentile(mcs_epc_data["TOTAL_FLOOR_AREA"], 99),
    ),  # 1st and 99th percentiles
    False,  # False means not using floor area as a feature
]
number_rooms_type = ["numeric", "categorical"]
installations_start_date = [
    "2007-01-01",  # date of first installation
    "2016-01-01",  # when EPC started being made available for Scotland
]

from asf_hp_cost_estimator_model.pipeline.model_evaluation.conduct_cross_validation import (
    update_error_results,
    compare_average_to_model,
)
from asf_hp_cost_estimator_model.pipeline.model_training.fit_cost_model import (
    set_up_pipeline,
)


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
    if date_double_weights != False:
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
) -> Tuple[List, List, dict]:
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
        Tuple[List, List, dict]: _description_
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

        # Predict on the test set
        y_test_pred = model.predict(X_test)

        # Calculate the proportion of training data after a fixed date
        after_date = model_data[(model_data["commission_date"] >= "2023-01-01")]
        after_date_train = after_date[after_date.index.isin(X_train.index)]
        proportion_train_after_date = len(after_date_train) / len(X_train)

        # Calculate the proportion of testing data after a fixed date
        after_date_test = np.where(X_test.index.isin(after_date.index), True, False)

        # Update the results of the model with fold specific results
        results_model = update_error_results(
            results_model,
            y_test,
            y_test_pred,
            proportion_train_after_date,
            after_date_test,
        )

    return results_model


results = {}
for d in date_doubling_weights:
    for c in cost_bounds:
        for r in number_rooms_bounds:
            for f in floor_area_bounds:
                for inst_d in installations_start_date:
                    exclusion_dict = {
                        "cost_lower_bound": c[0],
                        "cost_upper_bound": c[1],
                        "NUMBER_HABITABLE_ROOMS_lower_bound": r[0],
                        "NUMBER_HABITABLE_ROOMS_upper_bound": r[1],
                        "TOTAL_FLOOR_AREA_lower_bound": f[0],
                        "TOTAL_FLOOR_AREA_upper_bound": f[1],
                        "PROPERTY_TYPE_allowed_list": ["House", "Bungalow"],
                    }
                    mcs_epc_data = get_enhanced_installations_data()
                    postcodes_data = get_postcodes_data()

                    model_data = process_data_before_modelling(
                        mcs_epc_data=mcs_epc_data,
                        postcodes_data=postcodes_data,
                        exclusion_criteria_dict=exclusion_dict,
                        min_date=inst_d,
                    )
                    if f == False:
                        numeric_features = numeric_features.remove("TOTAL_FLOOR_AREA")

                    results_model = perform_kfold_cross_validation(
                        model_data=model_data,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        target_feature=target_feature,
                        kfold_splits=5,
                        date_double_weights=d,
                    )
                    results_model = pd.DataFrame(results_model).mean(axis=0).round(2)
                    print(results_model)
                    results[(d, c, r, f, inst_d)] = results_model
