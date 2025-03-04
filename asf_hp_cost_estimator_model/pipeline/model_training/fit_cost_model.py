"""
Pipeline for fitting a model to estimate the cost of an air source heat pump.
"""

import numpy as np
import pickle
import os
import logging
from sklearn.experimental import (
    enable_iterative_imputer,
)  # required to be imported before IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model import config, PROJECT_DIR


if not os.path.isdir(config["staging_folder"]):
    os.makedirs(config["staging_folder"])


def set_up_pipeline() -> Pipeline:
    """
    Set up a pipeline to training a model to estimate the cost of an air source heat pump.

    Returns:
        Pipeline: pipeline consisting of an imputation step and a regression model step.
    """

    # This preprocessor will enable filling in missing values
    preprocessor = IterativeImputer(random_state=0)

    # The regression model trained to predict the cost of an air source heat pump
    regressor = GradientBoostingRegressor(
        loss="absolute_error",
        n_estimators=500,
        min_samples_leaf=64,
        random_state=0,
        max_features="sqrt",
        verbose=0,
    )

    # The pipeline consists of our imputation and regression model steps
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return pipeline


def fit_and_save_model():
    """
    Loads data, trains model and saves model as pickle.
    """

    # Load and process data
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    model_data = process_data_before_modelling(mcs_epc_data, postcodes_data)

    logging.info("Number of samples used to train the model: " + str(len(model_data)))

    # Define features and target
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # To codify increased reliability in data after a certain date we double their weightd
    train_weights = np.where(
        model_data.loc[X.index]["commission_date"] >= config["date_double_weights"],
        2,
        1,
    )

    # Train model
    gb_forest = set_up_pipeline()
    gb_forest.fit(X, y, regressor__sample_weight=train_weights)

    # Save model
    model_pickle_local_path = config["model_pickle_local_path"]
    with open(model_pickle_local_path, "wb") as f:
        pickle.dump(gb_forest, f)

    logging.info(
        "Model trained and saved to: "
        + os.path.join(PROJECT_DIR, model_pickle_local_path)
    )


if __name__ == "__main__":
    fit_and_save_model()
