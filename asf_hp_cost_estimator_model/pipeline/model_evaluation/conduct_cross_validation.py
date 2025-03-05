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
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
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
    actual_train: np.array,
    predicted_train: np.array,
    train_dates: np.array,
    actual_test: np.array,
    predicted_test: np.array,
    test_dates: np.array,
    after_date: str,
) -> List[dict]:
    """
    Update error results after testing the model trained on a new fold for both train and test sets.

    Args:
        results (List[dict]): List of dictionaries containing the results of the model evaluation
        actual_train (np.array): True values of the target variable for training set
        predicted_train (np.array): Predicted values of the target variable for training set
        train_dates (np.array): Dates corresponding to the training set samples
        actual_test (np.array): True values of the target variable for testing set
        predicted_test (np.array): Predicted values of the target variable for testing set
        test_dates (np.array): Dates corresponding to the testing set samples
        after_date (str): Specifies a date to split the evaluation. The model's performance will be
        evaluated separately on instances that occur after this date.

    Returns:
        List[dict]: Updated error results
    """

    # Compute 'after_date' boolean arrays
    after_date_train = train_dates >= np.datetime64(after_date)
    after_date_test = test_dates >= np.datetime64(after_date)

    # Compute proportion of training/test data after the date
    proportion_train_after_date = after_date_train.sum() / len(train_dates)
    proportion_test_after_date = after_date_test.sum() / len(test_dates)

    # Compute errors for training set
    under_train = predicted_train < actual_train
    over_train = predicted_train > actual_train

    # Compute errors for testing set
    under_test = predicted_test < actual_test
    over_test = predicted_test > actual_test

    # Append results with consistent naming
    results.append(
        {
            # Training set results
            "train_r2_score": r2_score(actual_train, predicted_train),
            "train_mae": mean_absolute_error(actual_train, predicted_train),
            "train_mdae": median_absolute_error(actual_train, predicted_train),
            "train_percent_over_predictions": np.mean(over_train) * 100,
            "train_mae_over_predictions": mean_absolute_error(
                actual_train[over_train], predicted_train[over_train]
            ),
            "train_mdae_over_predictions": median_absolute_error(
                actual_train[over_train], predicted_train[over_train]
            ),
            "train_percent_under_predictions": np.mean(under_train) * 100,
            "train_mae_under_predictions": mean_absolute_error(
                actual_train[under_train], predicted_train[under_train]
            ),
            "train_mdae_under_predictions": median_absolute_error(
                actual_train[under_train], predicted_train[under_train]
            ),
            f"prop_train_set_after_{after_date}": proportion_train_after_date,
            f"train_mae_after_{after_date}": mean_absolute_error(
                actual_train[after_date_train], predicted_train[after_date_train]
            ),
            f"train_mdae_after_{after_date}": median_absolute_error(
                actual_train[after_date_train], predicted_train[after_date_train]
            ),
            # Test set results (updated to match train)
            "test_r2_score": r2_score(actual_test, predicted_test),
            "test_mae": mean_absolute_error(actual_test, predicted_test),
            "test_mdae": median_absolute_error(actual_test, predicted_test),
            "test_percent_over_predictions": np.mean(over_test) * 100,
            "test_mae_over_predictions": mean_absolute_error(
                actual_test[over_test], predicted_test[over_test]
            ),
            "test_mdae_over_predictions": median_absolute_error(
                actual_test[over_test], predicted_test[over_test]
            ),
            "test_percent_under_predictions": np.mean(under_test) * 100,
            "test_mae_under_predictions": mean_absolute_error(
                actual_test[under_test], predicted_test[under_test]
            ),
            "test_mdae_under_predictions": median_absolute_error(
                actual_test[under_test], predicted_test[under_test]
            ),
            f"prop_test_set_after_{after_date}": proportion_test_after_date,
            f"test_mae_after_{after_date}": mean_absolute_error(
                actual_test[after_date_test], predicted_test[after_date_test]
            ),
            f"test_mdae_after_{after_date}": median_absolute_error(
                actual_test[after_date_test], predicted_test[after_date_test]
            ),
        }
    )
    return results


def create_group_data(
    data: pd.DataFrame,
    index: np.array,
    feature: str,
    actual: np.array,
    predicted: np.array,
) -> pd.DataFrame:
    """
    Create a slice of data given a specfic set of indices and for a specific feature
    with information about actual and predicted values.
    This is used to evaluate the model performance on a specific group of data.

    Args:
        data (pd.DataFrame): model data
        index (np.array): indices of the data
        feature (str): feature to separate the data
        actual (np.array): true values of the target variable
        predicted (np.array): predicted values of the target variable
    Returns:
        pd.DataFrame: group data
    """

    group_data = data.iloc[index].copy()
    group_data["actual"] = actual
    group_data["predicted"] = predicted
    group_data["over_prediction"] = group_data["predicted"] > group_data["actual"]
    group_data["under_prediction"] = group_data["predicted"] < group_data["actual"]
    group_data = group_data[group_data[feature]]
    return group_data


def update_error_results_for_each_feature(
    model_data: pd.DataFrame,
    test_index: np.array,
    train_index: np.array,
    actual_test: np.array,
    predicted_test: np.array,
    actual_train: np.array,
    predicted_train: np.array,
    list_features: List[str],
    results: dict,
) -> dict:
    """
    Update error results after testing the model trained on a new fold
    for each feature in the list of features.

    Args:
        model_data (pd.DataFrame): model data
        test_index (np.array): indices of the test data
        train_index (np.array): indices of the train data
        actual_test (np.array): set of true values of the target variable on the test set
        predicted_test (np.array): predicted values of the target variable on the test set
        actual_train (np.array): set of true values of the target variable on the train set
        predicted_train (np.array): predicted values of the target variable on the train set
        list_features (List[str]): list of categorical features where to separatelly evaluate the model
        results (dict): dictionary containing the results of the model evaluation

    Returns:
        dict: updated error results
    """

    for feature in list_features:
        group_data_test = create_group_data(
            data=model_data,
            index=test_index,
            feature=feature,
            actual=actual_test,
            predicted=predicted_test,
        )
        group_data_train = create_group_data(
            data=model_data,
            index=train_index,
            feature=feature,
            actual=actual_train,
            predicted=predicted_train,
        )

        results[feature].append(
            {
                # Training set results
                "train_r2_score": r2_score(
                    group_data_train["actual"], group_data_train["predicted"]
                ),
                "train_mae": mean_absolute_error(
                    group_data_train["actual"], group_data_train["predicted"]
                ),
                "train_mdae": median_absolute_error(
                    group_data_train["actual"], group_data_train["predicted"]
                ),
                "train_percent_over_predictions": np.mean(
                    group_data_train["over_prediction"]
                )
                * 100,
                "train_mae_over_predictions": mean_absolute_error(
                    group_data_train[group_data_train["over_prediction"]]["actual"],
                    group_data_train[group_data_train["over_prediction"]]["predicted"],
                ),
                "train_mdae_over_predictions": median_absolute_error(
                    group_data_train[group_data_train["over_prediction"]]["actual"],
                    group_data_train[group_data_train["over_prediction"]]["predicted"],
                ),
                "train_percent_under_predictions": np.mean(
                    group_data_train["under_prediction"]
                )
                * 100,
                "train_mae_under_predictions": mean_absolute_error(
                    group_data_train[group_data_train["under_prediction"]]["actual"],
                    group_data_train[group_data_train["under_prediction"]]["predicted"],
                ),
                "train_mdae_under_predictions": median_absolute_error(
                    group_data_train[group_data_train["under_prediction"]]["actual"],
                    group_data_train[group_data_train["under_prediction"]]["predicted"],
                ),
                # Test set results (updated to match train format)
                "test_r2_score": r2_score(
                    group_data_test["actual"], group_data_test["predicted"]
                ),
                "test_mae": mean_absolute_error(
                    group_data_test["actual"], group_data_test["predicted"]
                ),
                "test_mdae": median_absolute_error(
                    group_data_test["actual"], group_data_test["predicted"]
                ),
                "test_percent_over_predictions": np.mean(
                    group_data_test["over_prediction"]
                )
                * 100,
                "test_mae_over_predictions": mean_absolute_error(
                    group_data_test[group_data_test["over_prediction"]]["actual"],
                    group_data_test[group_data_test["over_prediction"]]["predicted"],
                ),
                "test_mdae_over_predictions": median_absolute_error(
                    group_data_test[group_data_test["over_prediction"]]["actual"],
                    group_data_test[group_data_test["over_prediction"]]["predicted"],
                ),
                "test_percent_under_predictions": np.mean(
                    group_data_test["under_prediction"]
                )
                * 100,
                "test_mae_under_predictions": mean_absolute_error(
                    group_data_test[group_data_test["under_prediction"]]["actual"],
                    group_data_test[group_data_test["under_prediction"]]["predicted"],
                ),
                "test_mdae_under_predictions": median_absolute_error(
                    group_data_test[group_data_test["under_prediction"]]["actual"],
                    group_data_test[group_data_test["under_prediction"]]["predicted"],
                ),
            }
        )

    return results


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


def perform_kfold_cross_validation(
    model_data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_feature: str,
    kfold_splits: int,
    date_double_weights: str = config["date_double_weights"],
) -> Tuple[List, List, dict]:
    """
    Performs k-fold cross-validation across kfold splits by:
        - computing a series of metrics for the model on training and testing sets
        - computing a series of metrics for a constant model on training and testing sets
        - computing a series of metrics for the model on training and testing sets for each dummy feature
        to understand how the model performs on different groups of data.

    Metrics include: r2_score, mean_absolute_error, median_absolute_error,
    percentage of over and under predictions, mean absolute error for over and under predictions.

    Args:
        model_data (pd.DataFrame): model data
        numeric_features (List[str]): list of numeric features
        categorical_features (List[str]): list of categorical features
        target_feature (str): target feature
        kfold_splits (int): number of folds
        date_double_weights(str): date from when we start doubling the weights for instances
    Returns:
        Tuple[List[dict], List[dict]]: results on the test and train sets
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
        model = fit_model(model_data, X_train, y_train, date_double_weights)

        # Model predictions
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Constant model predictions (based on the mean of the training set)
        # this is used to assess how good the model is compared to a simple model
        y_test_pred_constant = np.full_like(y_test, np.mean(y_train), dtype=np.double)
        y_train_pred_constant = np.full_like(y_train, np.mean(y_train), dtype=np.double)

        # Update the results of the model with fold specific results
        results_model = update_error_results(
            results=results_model,
            actual_train=y_train,
            predicted_train=y_train_pred,
            train_dates=model_data.iloc[X_train.index]["commission_date"].values,
            actual_test=y_test,
            predicted_test=y_test_pred,
            test_dates=model_data.iloc[X_test.index]["commission_date"].values,
            after_date=config["date_for_latest_predictions"],
        )

        # Update the results of the constant model with fold specific results
        results_constant = update_error_results(
            results=results_constant,
            actual_train=y_train,
            predicted_train=y_train_pred_constant,
            train_dates=model_data.iloc[X_train.index]["commission_date"].values,
            actual_test=y_test,
            predicted_test=y_test_pred_constant,
            test_dates=model_data.iloc[X_test.index]["commission_date"].values,
            after_date=config["date_for_latest_predictions"],
        )

        # Update the results of the model with fold specific results for each dummy group
        results_per_categorical_feature = update_error_results_for_each_feature(
            model_data=model_data,
            test_index=test_index,
            train_index=train_index,
            actual_test=y_test,
            predicted_test=y_test_pred,
            actual_train=y_train,
            predicted_train=y_train_pred,
            list_features=categorical_features,
            results=results_per_categorical_feature,
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
    Summarises the results of the model evaluation for each dummy feature.

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
        perform_kfold_cross_validation(
            model_data=model_data,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_feature=target_feature,
            kfold_splits=kfold_splits,
        )
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
