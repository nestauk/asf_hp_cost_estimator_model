"""
Script to conduct k-fold cross-validation.
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

# local imports
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.pipeline.model_training.fit_cost_model import (
    set_up_pipeline,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)

numeric_features = config["numeric_features"]
categorical_features = config["categorical_features"]
target_feature = config["target_feature"]
model_pickle_local_path = config["model_pickle_local_path"]
model_evaluation_folder = config["model_evaluation_folder"]
kfold_splits = config["kfold_splits"]
min_date = config["min_date"]


if not os.path.isdir(model_evaluation_folder):
    os.makedirs(model_evaluation_folder)


def update_error_results(
    results: List[dict],
    actual: np.array,
    predicted: np.array,
    proportion_train_after_date: float,
    after_date_test: np.array,
) -> List[dict]:
    """
    Update error results after testing the model trained on a new fold.

    Args:
        results (List[dict]): List of dictionaries containing the results of the model evaluation
        actual (np.array): true values of the target variable
        predicted (np.array): predicted values of the target variable
        proportion_train_after_date (float): proportion of training data after a fixed date
        after_date_test (np.array): array of booleans indicating which samples of the testing data is after a fixed date

    Returns:
        List[dict]: updated error results
    """

    under_predictions = np.where(predicted < actual, True, False)
    over_predictions = np.where(predicted > actual, True, False)

    results.append(
        {
            "Proportion of train set after date: ": proportion_train_after_date,
            "Mean absolute error": round(mean_absolute_error(actual, predicted), 2),
            "Median absolute error": round(
                median_absolute_error(actual, predicted),
                2,
            ),
            "Percentage of over-predictions": np.mean(predicted > actual) * 100,
            "Mean absolute error for over-prediction": round(
                mean_absolute_error(
                    actual[over_predictions], predicted[over_predictions]
                ),
                2,
            ),
            "Median absolute error for over-prediction": round(
                median_absolute_error(
                    actual[over_predictions], predicted[over_predictions]
                ),
                2,
            ),
            "Percentage of under-predictions": np.mean(predicted < actual) * 100,
            "Mean absolute error for under-prediction": round(
                mean_absolute_error(
                    actual[under_predictions], predicted[under_predictions]
                ),
                2,
            ),
            "Median absolute error for under-prediction": round(
                median_absolute_error(
                    actual[under_predictions], predicted[under_predictions]
                ),
                2,
            ),
            "Proportion of test set after date: ": after_date_test.sum() / len(actual),
            "Mean absolute error after date": round(
                mean_absolute_error(
                    actual[after_date_test], predicted[after_date_test]
                ),
                2,
            ),
            "Median absolute error after date": round(
                median_absolute_error(
                    actual[after_date_test], predicted[after_date_test]
                ),
                2,
            ),
        }
    )
    return results


def update_error_results_by_dummy_group(
    model_data: pd.DataFrame,
    test_index: np.array,
    y_test: np.array,
    y_test_pred: np.array,
    list_features: List[str],
    results: dict,
):
    """
    Update error results after a new testing the model trained on a new fold.

    Args:
        model_data (pd.DataFrame): model data
        test_index (np.array): indices of the test data
        y_test (np.array): test set true values of the target variable
        y_test_pred (np.array): predicted values of the target variable
        list_features (List[str]): _description_
        results (dict): dictionary containing the results of the model evaluation

    Returns:
        dict: updated error results
    """

    for feature in list_features:
        group_data = model_data.iloc[test_index]
        group_data["actual"] = y_test
        group_data["predicted"] = y_test_pred
        group_data = group_data[group_data[feature]]

        results[feature].append(
            {
                "Mean absolute error": round(
                    mean_absolute_error(group_data["actual"], group_data["predicted"]),
                    2,
                ),
                "Median absolute error": round(
                    median_absolute_error(
                        group_data["actual"], group_data["predicted"]
                    ),
                    2,
                ),
                "Percentage of over-predictions": np.mean(
                    group_data["predicted"] > group_data["actual"]
                )
                * 100,
                "Percentage of under-predictions": np.mean(
                    group_data["predicted"] < group_data["actual"]
                )
                * 100,
            }
        )

    return results


def fit_model(
    model_data: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.array,
    date_double_weights: str = config["date_double_weights"],
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
    train_weights = np.where(
        model_data.loc[X_train.index]["commission_date"] >= date_double_weights,
        2,
        1,
    )

    # Set up the sklearn pipeline and fit
    model = set_up_pipeline()
    model.fit(X_train, y_train, regressor__sample_weight=train_weights)

    return model


def perform_kfold_cross_validation(
    model_data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_feature: str,
    kfold_splits: int,
) -> Tuple[List, List, dict]:
    """
    Performs k-fold cross-validation.

    Args:
        model_data (pd.DataFrame): _description_
        numeric_features (List[str]): _description_
        categorical_features (List[str]): _description_
        target_feature (str): _description_
        kfold_splits (int): _description_

    Returns:
        Tuple[List, List, dict]: _description_
    """
    # Define input and target data
    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Define a K-Fold cross-validator
    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)

    # Initialise result variables
    results_constant = []
    results_model = []
    results_per_categorical_feature = {f: [] for f in categorical_features}

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
        model = fit_model(model_data, X_train, y_train)

        # Predict on the test set
        y_test_pred = model.predict(X_test)

        # Predict with a constant model (based on the mean of the training set)
        # this is used to assess how good the model is compared to a simple model
        y_test_pred_constant = np.full_like(y_test, np.mean(y_train))

        # Calculate the proportion of training data after a fixed date
        after_date = model_data[
            (model_data["commission_date"] >= config["date_double_weights"])
        ]
        after_date_train = after_date[after_date.index.isin(X_train.index)]
        proportion_train_after_date = len(after_date_train) / len(X_train)

        # Calculate the proportion of testing data after a fixed date
        after_date_test = np.where(X_test.index.isin(after_date.index), True, False)

        # Update the results of the constant model with fold specific results
        results_constant = update_error_results(
            results_constant,
            y_test,
            y_test_pred_constant,
            proportion_train_after_date,
            after_date_test,
        )

        # Update the results of the model with fold specific results
        results_model = update_error_results(
            results_model,
            y_test,
            y_test_pred,
            proportion_train_after_date,
            after_date_test,
        )

        # Update the results of the model with fold specific results for each dummy group
        results_per_categorical_feature = update_error_results_by_dummy_group(
            model_data,
            test_index,
            y_test,
            y_test_pred,
            categorical_features,
            results_per_categorical_feature,
        )

    return results_constant, results_model, results_per_categorical_feature


def compare_average_to_model(
    results_constant: List[dict], results_model: List[dict]
) -> pd.DataFrame:
    """
    Compares the results of the model to the constant model.

    Args:
        results_constant (List[dict]): List of dictionaries containing the results of the constant model evaluation
        results_model (List[dict]): List of dictionaries containing the results of the model evaluation

    Returns:
        pd.DataFrame: Dataframe with the average results of the model and the constant model
    """
    results_model = pd.DataFrame(results_model).mean(axis=0).round(2)

    results_constant = pd.DataFrame(results_constant).mean(axis=0).round(2)

    results = pd.DataFrame(
        {"constant_model": results_constant, "modelled": results_model}
    )
    return results


def summarise_results_categories(results_per_categorical_feature: dict) -> pd.DataFrame:
    """
    Summarises the results of the model evaluation for ach dummy feature.

    Args:
        results_per_categorical_feature (dict): dictionary containing the results of the model evaluation for each dummy feature

    Returns:
        pd.DataFrame: Dataframe with the average results of the model for each dummy feature
    """
    results_categories = pd.DataFrame()
    for feature in categorical_features:
        aux = (
            pd.DataFrame(results_per_categorical_feature[feature]).mean(axis=0).round(2)
        )
        results_categories[feature] = aux
    return results_categories


if __name__ == "__main__":
    # Load and process data
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    model_data = process_data_before_modelling(mcs_epc_data, postcodes_data)

    logging.info("Number of samples: " + str(len(model_data)))

    # Perform k-fold cross-validation
    results_constant, results_model, results_per_categorical_feature = (
        perform_kfold_cross_validation(model_data, categorical_features, kfold_splits)
    )

    # Summarise results from k-fold cross-validation
    overall_results = compare_average_to_model(results_constant, results_model)
    results_per_category = summarise_results_categories(results_per_categorical_feature)

    # Save results
    overall_results.to_csv(
        os.path.join(model_evaluation_folder, "cross_validation_results.csv")
    )
    results_per_category.to_csv(
        os.path.join(
            model_evaluation_folder, "cross_validation_categorical_feature_results.csv"
        )
    )
