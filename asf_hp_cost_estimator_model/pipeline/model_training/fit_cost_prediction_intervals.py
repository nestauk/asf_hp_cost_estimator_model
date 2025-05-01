"""
Pipeline for fitting a model to estimate the cost of an air source heat pump.
"""

import boto3
from datetime import datetime
import pickle
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from asf_hp_cost_estimator_model.pipeline.data_processing.process_installations_data import (
    process_data_before_modelling,
)
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
)
from asf_hp_cost_estimator_model.pipeline.data_processing.process_location_data import (
    get_postcodes_data,
)
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.pipeline.data_processing.process_cpi import (
    get_df_quarterly_cpi_with_adjustment_factors,
)
from asf_hp_cost_estimator_model.getters.data_getters import get_cpi_data


def set_up_pipeline(quantile: int) -> Pipeline:
    """
    Set up a pipeline to training a model to estimate the cost of an air source heat pump.

    Returns:
        Pipeline: pipeline consisting of an imputation step and a regression model step.
    """

    regressor = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        min_samples_split=config["min_samples_split"],
        random_state=config["random_state"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
    )

    return regressor


def fit_and_save_model():
    """
    Loads data, trains model and saves model as pickle.
    """

    # Load and process data
    mcs_epc_data = get_enhanced_installations_data()
    postcodes_data = get_postcodes_data()
    cpi_05_3_df = get_cpi_data()
    cpi_quarterly_df = get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=config["cpi_data"]["cpi_reference_year"],
        cpi_df=cpi_05_3_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )

    model_data = process_data_before_modelling(
        mcs_epc_data, postcodes_data, cpi_quarterly_df
    )

    print("Model data shape:", model_data.shape)

    # Define features and target
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_feature = config["target_feature"]

    X = model_data[numeric_features + categorical_features]
    y = model_data[target_feature].values.ravel()

    # Train models
    regressor_10p = set_up_pipeline(0.1)
    regressor_10p.fit(X, y)

    regressor_90p = set_up_pipeline(0.9)
    regressor_90p.fit(X, y)

    # Save models
    today_date = datetime.today().strftime("%Y%m%d")

    s3_resource = boto3.resource("s3")
    regressor_10p = pickle.dumps(regressor_10p)
    regressor_90p = pickle.dumps(regressor_90p)

    s3_resource.Object(
        "asf-hp-cost-estimator-model", f"outputs/model/{today_date}/regressor10p.pkl"
    ).put(Body=regressor_10p)
    s3_resource.Object(
        "asf-hp-cost-estimator-model", f"outputs/model/{today_date}/regressor90p.pkl"
    ).put(Body=regressor_90p)

    logging.info(
        f"Model trained and saved to:\n s3://asf-hp-cost-estimator-model/outputs/model/{today_date}"
    )


if __name__ == "__main__":
    fit_and_save_model()
