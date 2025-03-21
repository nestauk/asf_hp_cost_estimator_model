"""
Hyperparameter tuning script to find the best hyperparameters for the model.

To run the script:
python asf_hp_cost_estimator_model/pipeline/hyperparameter_tuning/hyperparameter_tuning.py --datastore=s3 --package-suffixes=.txt run --max-num-splits 2000 --max-workers 100
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
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/ht_flow_requirements.txt 1> /dev/null"
)

from metaflow import FlowSpec, step, S3, batch, Parameter

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
) -> List[dict]:
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
        List[dict]: results on the test and train sets
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
    return results_model


class CostModelHyperparameterTuning(FlowSpec):
    # Setting all parameters to tune as a parameter for ease of use
    all_params = Parameter(
        name="all_param",
        help="Parameters to tune include model hyperparameters and data parameters",
        default=data_params | model_params,
    )

    @step
    def start(self):
        """
        Starts the flow.
        """
        # Generating all combinations of parameters to tune through grid search
        self.param_combinations = list(product(*self.all_params.values()))

        self.next(self.create_chunks_of_params)

    @step
    def create_chunks_of_params(self):
        """
        Creates chunks of 25 parameter combinations to be processed in parallel.
        """
        self.chunks_of_param_combinations = [
            self.param_combinations[i : i + 25]
            for i in range(0, len(self.param_combinations), 25)
        ]
        self.next(
            self.perform_hyperparameter_tuning, foreach="chunks_of_param_combinations"
        )

    @batch
    @step
    def perform_hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning through grid search.
        """
        self.chunk_results = {}
        # Grid search: Iterate through each possible combination
        for combination in self.input:
            param_dict = dict(zip(self.all_params.keys(), combination))

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
                    feat
                    for feat in numeric_features
                    if feat != "NUMBER_HABITABLE_ROOMS"
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
            results_model = perform_kfold_cross_validation(
                model_data=model_data,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_feature=config["target_feature"],
                kfold_splits=5,
                date_double_weights=param_dict["date_doubling_weights"],
            )

            # Averaging the results for the specific combination across the different folds
            self.chunk_results[", ".join(map(str, combination))] = (
                pd.DataFrame(results_model).mean().round(2).to_dict()
            )

        self.next(self.join_results)

    @step
    def join_results(self, inputs):
        """
        Joins results from different chunks (parallel steps).
        """
        self.results = {}
        for input in inputs:
            self.results.update(input)
        self.next(self.save_results)

    @step
    def save_results(self):
        """
        Save the results to a json file.
        """
        with open(
            os.path.join(
                PROJECT_DIR,
                "outputs/model_evaluation/hyperparameter_tuning_results.json",
            ),
            "w",
        ) as f:
            json.dump(self.results, f)

        self.next(self.end)

    @step
    def end(self):
        """
        Ends the flow.
        """
        pass


if __name__ == "__main__":
    CostModelHyperparameterTuning()
