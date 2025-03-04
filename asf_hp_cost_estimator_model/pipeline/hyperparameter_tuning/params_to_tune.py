"""
This module contains the hyperparameters to tune for the GradientBoostingRegressor model
and some data-specific parameters to use for the tuning.
"""

# package imports
import numpy as np

# local imports
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_enhanced_installations_data,
)

# Model hyperparameters to tune
model_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.8, 0.9, 1],
    "min_samples_split": [2, 5, 10],
}

# Data params
installations_data = get_enhanced_installations_data()
data_params = {
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
