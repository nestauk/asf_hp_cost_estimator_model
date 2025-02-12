"""
Assessing feature importance.
"""

# package imports
import pandas as pd
import pickle
import logging
from typing import List
from sklearn.base import BaseEstimator

# local imports
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
    get_postcodes_data,
)
from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model import config


def assess_feature_importance(
    features: List[str], model: BaseEstimator
) -> pd.DataFrame:
    """
    Assess feature importance of a model.

    Args:
        features (List[str]): model features
        model (BaseEstimator): an sklearn regression model

    Returns:
        pd.DataFrame: a dataframw with feature importance values for each feature
    """
    feat_imp = pd.DataFrame(index=features)
    feat_imp["feature_importance"] = model.feature_importances_
    feat_imp.sort_values("feature_importance", ascending=False, inplace=True)
    return feat_imp


if __name__ == "__main__":
    # Loading and processing data
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    model_data = process_data_before_modelling(mcs_epc_data, postcodes_data)

    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]

    # Loading model
    model_pickle_local_path = config["model_pickle_local_path"]
    with open(model_pickle_local_path, "rb") as f:
        pipeline = pickle.load(f)

    # We're actually loading a pipeline, and model is the second step of the pipeline
    # So we access pipeline[1] to get the model itself
    model = pipeline[1]

    feat_imp = assess_feature_importance(X.columns, model)

    logging.info("----FEATURE IMPORTANCE----\n")
    logging.info(feat_imp)
