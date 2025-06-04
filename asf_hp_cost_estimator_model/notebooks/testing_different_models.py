# %% [markdown]
# # Investigating if there's a strong enough signal in the data
#
# If cost is noisy or there's not enough signal, we might not be able to get a good estimate of the cost.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import uniform, randint

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

# %% [markdown]
# ### Data imports

# %%
mcs_epc_data = get_enhanced_installations_data()

cpi_05_3_df = get_cpi_data()
cpi_quarterly_df = get_df_quarterly_cpi_with_adjustment_factors(
    ref_year=2024,
    cpi_df=cpi_05_3_df,
    cpi_col_header=config["cpi_data"]["cpi_column_header"],
)

postcodes_data = get_postcodes_data()

# %% [markdown]
# ### Processing data

# %%
target_feature = "cost"
# target_feature = "adjusted_cost"

# %%
if target_feature == "cost":
    exclusion_criteria_dict = {
        # just setting a negative lower bound to not exclude any data
        "cost_lower_bound": -100,  # 3500,
        # "cost_upper_bound": 25000,
        "NUMBER_HABITABLE_ROOMS_lower_bound": 2,
        # "NUMBER_HABITABLE_ROOMS_upper_bound": 8,
        "TOTAL_FLOOR_AREA_lower_bound": 20,  # 20
        "TOTAL_FLOOR_AREA_upper_bound": 500,
        "PROPERTY_TYPE_allowed_list": ["House", "Bungalow"],
    }

else:  # adjusted_cost
    exclusion_criteria_dict = {
        # just setting a negative lower bound to not exclude any data
        "cost_lower_bound": -100,  # 3500,
        # "cost_upper_bound": 25000,
        "NUMBER_HABITABLE_ROOMS_lower_bound": 2,
        # "NUMBER_HABITABLE_ROOMS_upper_bound": 8,
        "TOTAL_FLOOR_AREA_lower_bound": 20,  # 20
        "TOTAL_FLOOR_AREA_upper_bound": 500,
        "PROPERTY_TYPE_allowed_list": ["House", "Bungalow"],
    }

# %%


# %%
model_data = process_data_before_modelling(
    mcs_epc_data,
    postcodes_data,
    cpi_quarterly_df,
    exclusion_criteria_dict=exclusion_criteria_dict,
    min_date=config["min_date"],
)

# Define features and target
numeric_features = config["numeric_features"]
original_categorical_features = config["categorical_features_to_dummify"]
categorical_features = config["categorical_features"]

# %%
len(model_data)

# %%
len(model_data[(model_data["cost"] >= 3500) & (model_data["cost"] <= 25000)])

# %%
model_data["NUMBER_HABITABLE_ROOMS"].value_counts()

# %%
len(model_data[model_data["NUMBER_HABITABLE_ROOMS"] >= 8]), len(
    model_data[model_data["NUMBER_HABITABLE_ROOMS"] >= 8]
) / len(model_data) * 100

# %%
model_data["NUMBER_HABITABLE_ROOMS"] = model_data["NUMBER_HABITABLE_ROOMS"].apply(
    lambda x: 8 if x > 8 else x
)

# %%
archetypes = (
    model_data.groupby(
        ["PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND", "NUMBER_HABITABLE_ROOMS"]
    )[["original_mcs_index"]]
    .nunique()
    .sort_values("original_mcs_index")
    .reset_index()
    .rename(columns={"original_mcs_index": "number_of_installations"})
)

archetypes["archetype"] = ["archetype_" + str(i) for i in range(1, len(archetypes) + 1)]

model_data = model_data.merge(
    archetypes[
        [
            "PROPERTY_TYPE",
            "CONSTRUCTION_AGE_BAND",
            "NUMBER_HABITABLE_ROOMS",
            "archetype",
        ]
    ],
    on=["PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND", "NUMBER_HABITABLE_ROOMS"],
    how="left",
)

# %%
from sklearn.ensemble import IsolationForest


def remove_outliers_within_archetypes(
    df, archetype_col, feature_cols, winsorise, contamination=0.1
):
    cleaned_df = pd.DataFrame()

    for archetype in df[archetype_col].unique():
        subset = df[df[archetype_col] == archetype]

        # ---- Uncomment one of the approaches below
        # APROACH 1: Apply Isolation Forest
        # iso = IsolationForest(contamination=contamination, random_state=42)
        # preds = iso.fit_predict(subset[feature_cols])

        # # Keep only inliers (prediction == 1)
        # subset_cleaned = subset[preds == 1]

        # APROACH 2: Use IQR to filter out outliers
        Q1 = subset[feature_cols].quantile(0.25)
        Q3 = subset[feature_cols].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # APPROACH 3: use quantiles directly if you prefer
        # lower_bound = subset[target_feature].quantile(0.05)
        # upper_bound = subset[target_feature].quantile(0.90)

        # print(lower_bound, upper_bound)

        # --------
        # Filter out outliers
        print("Processing archetype:", archetype)
        print("Lower bound:", lower_bound)
        print("Upper bound:", upper_bound)

        subset_clean = subset.copy()
        if winsorise == "none":
            # Filter out outliers on both ends
            subset_clean = subset_clean[
                (subset_clean[target_feature] >= lower_bound)
                & (subset_clean[target_feature] <= upper_bound)
            ]
        elif winsorise == "both":
            # Winsorise all outliers
            subset_clean[target_feature] = subset_clean[target_feature].apply(
                lambda x: upper_bound if (x > upper_bound) else x
            )
            subset_clean[target_feature] = subset_clean[target_feature].apply(
                lambda x: lower_bound if (x < lower_bound) else x
            )
        elif winsorise == "upper":
            # Filter out outliers on the lower end
            subset_clean = subset_clean[(subset_clean[target_feature] >= lower_bound)]
            # Winsorise outliers on the upper end
            subset_clean[target_feature] = subset_clean[target_feature].apply(
                lambda x: upper_bound if (x > upper_bound) else x
            )
        elif winsorise == "lower":
            # Winsorise outliers on the lower end
            subset_clean[target_feature] = subset_clean[target_feature].apply(
                lambda x: lower_bound if (x < lower_bound) else x
            )
            # Filter out outliers on upper end
            subset_clean = subset_clean[(subset_clean[target_feature] <= upper_bound)]

        cleaned_df = pd.concat([cleaned_df, subset_clean], ignore_index=True)

    return cleaned_df


# Example usage:
model_data = remove_outliers_within_archetypes(
    model_data, archetype_col="archetype", feature_cols="cost"
)

# %%
len(model_data)

# %% [markdown]
# ### Functions


# %%
def print_metrics(y_train, y_train_pred, y_test, y_test_pred):
    print(f"R^2 on training data: {r2_score(y_train, y_train_pred)}")
    print(f"R^2 on test data: {r2_score(y_test, y_test_pred)}")
    print(f"MAE on training data: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"MAE on test data: {mean_absolute_error(y_test, y_test_pred):.4f}")


# %%
def plot_true_vs_predicted_against_perfect_prediction(true_values, predicted_values):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.plot(
        [true_values.min(), true_values.max()],
        [true_values.min(), true_values.max()],
        "r--",
        lw=2,
        color="red",
        label="Perfect Predictions",
    )
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()


# %%
def run_print_hyp_param_tune_results(
    estimator, param_grid, n_comb, x_train, y_train, x_test, y_test
):
    grid_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_comb,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(x_train, y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print("Best MAE: ", -grid_search.best_score_)

    final_model = grid_search.best_estimator_
    y_train_pred = final_model.predict(x_train)
    y_test_pred = final_model.predict(x_test)
    print("MAE on train data: ", mean_absolute_error(y_train, y_train_pred))
    print("MAE on test data: ", mean_absolute_error(y_test, y_test_pred))
    print("r2 on train data: ", r2_score(y_train, y_train_pred))
    print("r2 on test data: ", r2_score(y_test, y_test_pred))


# %% [markdown]
# ## 2. Testing different models to predict adjusted cost

# %% [markdown]
# ### 2.0. BASELINES (for comparison)

# %% [markdown]
# ### 2.0.1. Median adjusted cost per archetype (reference year 20223)
#
#
# Using the archetypes available and respective costs available here: https://www.getaheatpump.org.uk/heat-pump-costs
#
# ![image.png](attachment:image.png)
#

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
def get_archetype_median_cost(is_bungalow, number_of_rooms, is_detached):
    if is_bungalow:
        if number_of_rooms == 2:
            return 9900
        elif number_of_rooms == 3:
            return 10650
        elif number_of_rooms == 4:
            return 11950
        elif number_of_rooms == 5:
            return 12780
        elif number_of_rooms == 6:
            return 13710
        elif number_of_rooms == 7:
            return 14480
        else:
            return np.nan
    elif not is_bungalow and is_detached:
        if number_of_rooms == 4:
            return 12640
        elif number_of_rooms == 5:
            return 13210
        elif number_of_rooms == 6:
            return 13800
        elif number_of_rooms == 7:
            return 14370
        elif number_of_rooms == 8:
            return 15150
        else:
            return np.nan
    elif not is_bungalow and not is_detached:
        if number_of_rooms == 3:
            return 10850
        elif number_of_rooms == 5:
            return 11380
        elif number_of_rooms == 5:
            return 11600
        elif number_of_rooms == 6:
            return 12850
        elif number_of_rooms == 7:
            return 14070
        else:
            return np.nan
    else:
        return np.nan


# %%
x_test["predicted_cost_archetypes"] = x_test.apply(
    lambda v: get_archetype_median_cost(
        v["PROPERTY_TYPE_bungalow"],
        v["NUMBER_HABITABLE_ROOMS"],
        v["BUILT_FORM_detached"],
    ),
    axis=1,
)

# %%
x_train["predicted_cost_archetypes"] = x_train.apply(
    lambda v: get_archetype_median_cost(
        v["PROPERTY_TYPE_bungalow"],
        v["NUMBER_HABITABLE_ROOMS"],
        v["BUILT_FORM_detached"],
    ),
    axis=1,
)

# %%
x_test = (
    pd.concat([x_test.reset_index(), pd.DataFrame(y_test)], axis=1)
    .rename(columns={0: target_feature})
    .drop(columns=["index"])
)

# %%
x_train = (
    pd.concat([x_train.reset_index(), pd.DataFrame(y_train)], axis=1)
    .rename(columns={0: target_feature})
    .drop(columns=["index"])
)

# %%
x_train = x_train[~pd.isnull(x_train["predicted_cost_archetypes"])]
x_test = x_test[~pd.isnull(x_test["predicted_cost_archetypes"])]

# %%
r2_score(x_train[target_feature], x_train["predicted_cost_archetypes"]), r2_score(
    x_test[target_feature], x_test["predicted_cost_archetypes"]
)

# %%
mean_absolute_error(
    x_train[target_feature], x_train["predicted_cost_archetypes"]
), mean_absolute_error(x_test[target_feature], x_test["predicted_cost_archetypes"])

# %%
plot_true_vs_predicted_against_perfect_prediction(
    x_train[target_feature], x_train["predicted_cost_archetypes"]
)

# %% [markdown]
# ### 2.0.2. cost through moving average (average of past 10 installations in similar homes)

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
def get_average_cost(bedrooms, floor_area, data):
    similar_installations = data[
        (data["NUMBER_HABITABLE_ROOMS"] >= bedrooms - 1)
        & (data["NUMBER_HABITABLE_ROOMS"] <= bedrooms + 1)
        & (data["TOTAL_FLOOR_AREA"] <= floor_area + 50)
        & (data["TOTAL_FLOOR_AREA"] >= floor_area - 50)
    ]

    if len(similar_installations) > 10:

        avg = (
            similar_installations.sort_values("n_days", ascending=False)
            .head(10)[target_feature]
            .mean()
        )
        return avg
    else:
        return np.nan


# %%
data = (
    pd.concat([x_train.reset_index(), pd.DataFrame(y_train)], axis=1)
    .rename(columns={0: target_feature})
    .drop(columns=["index"])
)

x_test["predicted_cost_moving_average"] = x_test.apply(
    lambda v: get_average_cost(
        v["NUMBER_HABITABLE_ROOMS"], v["TOTAL_FLOOR_AREA"], data
    ),
    axis=1,
)

x_train["predicted_cost_moving_average"] = x_train.apply(
    lambda v: get_average_cost(
        v["NUMBER_HABITABLE_ROOMS"], v["TOTAL_FLOOR_AREA"], data
    ),
    axis=1,
)

# %%
x_train = (
    pd.concat([x_train.reset_index(), pd.DataFrame(y_train)], axis=1)
    .rename(columns={0: target_feature})
    .drop(columns=["index"])
)
x_test = (
    pd.concat([x_test.reset_index(), pd.DataFrame(y_test)], axis=1)
    .rename(columns={0: target_feature})
    .drop(columns=["index"])
)

# %%
x_train = x_train[~pd.isnull(x_train["predicted_cost_moving_average"])]
x_test = x_test[~pd.isnull(x_test["predicted_cost_moving_average"])]

# %%
r2_score(x_train[target_feature], x_train["predicted_cost_moving_average"]), r2_score(
    x_test[target_feature], x_test["predicted_cost_moving_average"]
)

# %%
mean_absolute_error(
    x_train[target_feature], x_train["predicted_cost_moving_average"]
), mean_absolute_error(
    x_train[target_feature], x_train["predicted_cost_moving_average"]
)

# %%
plot_true_vs_predicted_against_perfect_prediction(
    x_train[target_feature], x_train["predicted_cost_moving_average"]
)

# %%


# %%


# %% [markdown]
# ### 2.1. Tree-based models:

# %% [markdown]
# #### Gradient Boosting Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
from sklearn.ensemble import GradientBoostingRegressor

rf = GradientBoostingRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)
plot_true_vs_predicted_against_perfect_prediction(y_train, y_pred)

# %%
param_grid = {
    "n_estimators": [100, 200, 500],  # default is 100
    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # default is 0.1
    "max_depth": [3, 5, 10, 20],  # default is 3
    "min_samples_leaf": [1, 5, 10, 100, 1000],  # default is 1
    "min_samples_split": [2, 10, 50, 100, 1000],  # default is 2
}

estimator = GradientBoostingRegressor()


run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %%
from sklearn.ensemble import GradientBoostingRegressor

rf = GradientBoostingRegressor(
    n_estimators=200,
    min_samples_split=100,
    min_samples_leaf=1000,
    max_depth=10,
    learning_rate=0.2,
)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)
plot_true_vs_predicted_against_perfect_prediction(y_train, y_pred)

# %% [markdown]
# #### HistGradientBoosting Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.ensemble import HistGradientBoostingRegressor


rf = HistGradientBoostingRegressor(random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)
plot_true_vs_predicted_against_perfect_prediction(y_train, y_pred)

# %%
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],  # default is 0.1
    "max_depth": [None, 3, 5, 10],  # default is None
    "min_samples_leaf": [20, 100, 1000],  # default is 20
    "early_stopping": [True],  # default is False
}

estimator = HistGradientBoostingRegressor(random_state=42)


run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %% [markdown]
# #### Decision Tree Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.tree import DecisionTreeRegressor

rf = DecisionTreeRegressor(random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)
plot_true_vs_predicted_against_perfect_prediction(y_train, y_pred)

# %%
param_grid = {
    "max_depth": [3, 5, 10, None],  # default is None
    "min_samples_leaf": [1, 5, 10, 100, 1000],  # default is 1
    "min_samples_split": [2, 100, 1000],  # default is 2
    "max_features": [None, "sqrt", "log2"],
    "ccp_alpha": [
        0.0,
        0.001,
        0.01,
        0.1,
    ],  # defaults to 0 - Cost complexity pruning parameter
}

estimator = DecisionTreeRegressor(random_state=42)


run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %% [markdown]
# #### Random Forest Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %%
param_grid = {
    "n_estimators": [100, 200, 500],  # default is 100
    "max_depth": [3, 5, 10, None],  # default is None
    "min_samples_leaf": [1, 5, 10, 100, 1000],  # default is 1
    "min_samples_split": [2, 100, 1000],  # default is 2
    "max_features": [None, "sqrt", "log2"],  # default is None or 1
    "ccp_alpha": [
        0.0,
        0.001,
        0.01,
        0.1,
    ],  # defaults to 0 - Cost complexity pruning parameter
}

estimator = RandomForestRegressor(random_state=42)

run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %% [markdown]
# #### XBoost Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from xgboost import XGBRegressor

rf = XGBRegressor(random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %%
param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 10],
}

estimator = XGBRegressor(random_state=42)

run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %% [markdown]
# #### LGBM Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from lightgbm import LGBMRegressor

rf = LGBMRegressor(random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %%
param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [-1, 3, 5, 10],
}

estimator = LGBMRegressor(random_state=42)

run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %% [markdown]
# #### CatBoost Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from catboost import CatBoostRegressor

rf = CatBoostRegressor(random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %%
param_grid = {
    "iterations": [100, 200, 500, 1000],  # same as n_estimators in other models
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [3, 5, 10],
}


estimator = CatBoostRegressor(random_state=42)

run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train, y_train, x_test, y_test
)

# %% [markdown]
# ### 2.2. K neighbors

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rf = KNeighborsRegressor()
rf.fit(x_train_scaled, y_train)


y_pred = rf.predict(x_train_scaled)
y_pred_test = rf.predict(x_test_scaled)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %%
param_grid = {
    "n_neighbors": [5, 10, 50, 100, 1000],
    "weights": ["uniform", "distance"],
    "leaf_size": [10, 30, 50, 100],
}

estimator = KNeighborsRegressor()

run_print_hyp_param_tune_results(
    estimator, param_grid, 15, x_train_scaled, y_train, x_test_scaled, y_test
)

# %% [markdown]
# ### 2.3. SVRs

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.svm import SVR

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize and fit the SVR model
rf = SVR()

rf.fit(x_train_scaled, y_train)

y_pred = rf.predict(x_train_scaled)
y_pred_test = rf.predict(x_test_scaled)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %%
# param_grid = {
#     'C': [0.1, 1, 10, 100],              # Regularization parameter
#     'epsilon': [0.01, 0.1, 0.2, 0.3],   # Epsilon in the epsilon-SVR model
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
# }

# estimator = SVR()

# run_print_hyp_param_tune_results(estimator, param_grid, 15, x_train_scaled, y_train, x_test_scaled, y_test)

# %% [markdown]
# ### 2.4. Linear models

# %% [markdown]
# #### 2.4.1. Linear regrression

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.linear_model import LinearRegression

rf = LinearRegression()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %% [markdown]
# ### 2.4.2. Regularised regression models

# %% [markdown]
# Lasso

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.linear_model import Lasso

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rf = Lasso()
rf.fit(x_train_scaled, y_train)

y_pred = rf.predict(x_train_scaled)
y_pred_test = rf.predict(x_test_scaled)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %% [markdown]
# Ridge

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.linear_model import Ridge

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rf = Ridge()
rf.fit(x_train_scaled, y_train)

y_pred = rf.predict(x_train_scaled)
y_pred_test = rf.predict(x_test_scaled)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %% [markdown]
# ElasticNet

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.linear_model import ElasticNet

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rf = ElasticNet()
rf.fit(x_train_scaled, y_train)

y_pred = rf.predict(x_train_scaled)
y_pred_test = rf.predict(x_test_scaled)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %% [markdown]
# ### 2.5. Multi layer Perceptron - NN

# %%
from sklearn.neural_network import MLPRegressor

rf = MLPRegressor(random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print_metrics(y_train, y_pred, y_test, y_pred_test)

# %% [markdown]
# ### 2.6. Creating intervals with quantile regression

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint
from sklearn.metrics import mean_pinball_loss, mean_squared_error

# %% [markdown]
# #### 2.6.1. Quantile regressor:

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
import statsmodels.formula.api as smf

quantiles = [0.1, 0.9]
qr_models = {}

all_features = numeric_features + categorical_features
x_train = (
    pd.concat([x_train.reset_index(), pd.DataFrame(y_train)], axis=1)
    .rename(columns={0: target_feature})
    .drop(columns=["index"])
)
all_features = " + ".join(all_features)

for q in quantiles:
    print(f"quantile {q}")
    qr = smf.quantreg(f"{target_feature} ~ {all_features}", x_train).fit(q=q)
    qr_models[q] = qr

y_pred_upper = qr_models[0.9].predict(x_test)
y_pred_lower = qr_models[0.1].predict(x_test)


# Evaluate using MAE (for both upper and lower bounds)
mae_upper = mean_absolute_error(y_test, y_pred_upper)
mae_lower = mean_absolute_error(y_test, y_pred_lower)

print(f"MAE for Upper Bound (90th Percentile): {mae_upper:.4f}")
print(f"MAE for Lower Bound (10th Percentile): {mae_lower:.4f}")

coverage = ((y_test >= y_pred_lower) & (y_test <= y_pred_upper)).mean()
print(f"Coverage: {coverage:.4f}")

avg_width = np.mean(y_pred_upper - y_pred_lower)
print(f"Average Interval Width: {avg_width:.2f}")


mean_pinball_loss_01p = mean_pinball_loss(y_test, y_pred_lower, alpha=0.1)
print(
    f"Mean Pinball Loss for Lower Bound (10th Percentile): {mean_pinball_loss_01p:.4f}"
)
mean_pinball_loss_90p = mean_pinball_loss(y_test, y_pred_upper, alpha=0.9)
print(
    f"Mean Pinball Loss for Upper Bound (90th Percentile): {mean_pinball_loss_90p:.4f}"
)

# %%


# %% [markdown]
# #### 2.6.2. LGBMRegressor quantile regression:

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
param_grid = {
    # 'n_estimators': [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [-1, 3, 5, 10],
}
alpha = 0.1
mean_pinball_loss_01p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,
)
gbr = LGBMRegressor(loss="quantile", alpha=alpha, random_state=42)
search_01p = HalvingRandomSearchCV(
    gbr,
    param_grid,
    resource="n_estimators",
    max_resources=500,
    min_resources=50,
    scoring=mean_pinball_loss_01p_scorer,
    n_jobs=2,
    random_state=0,
).fit(x_train, y_train)
pprint(search_01p.best_params_)

# %%
from sklearn.base import clone

alpha = 0.9
mean_pinball_loss_90p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,  # maximize the negative loss
)
search_90p = clone(search_01p).set_params(
    estimator__alpha=alpha,
    scoring=mean_pinball_loss_90p_scorer,
)
search_90p.fit(x_train, y_train)
pprint(search_90p.best_params_)

# %%
from lightgbm import LGBMRegressor

# Lower bound model (e.g., 10th percentile)
model_lower = LGBMRegressor(
    objective="quantile", alpha=0.1, random_state=42, **search_01p.best_params_
)
model_lower.fit(x_train, y_train)
y_pred_lower = model_lower.predict(x_test)

# Upper bound model (e.g., 90th percentile)
model_upper = LGBMRegressor(
    objective="quantile", alpha=0.9, random_state=42, **search_90p.best_params_
)
model_upper.fit(x_train, y_train)
y_pred_upper = model_upper.predict(x_test)

# Evaluate using MAE (for both upper and lower bounds)
mae_upper = mean_absolute_error(y_test, y_pred_upper)
mae_lower = mean_absolute_error(y_test, y_pred_lower)

print(f"MAE for Upper Bound (90th Percentile): {mae_upper:.4f}")
print(f"MAE for Lower Bound (10th Percentile): {mae_lower:.4f}")

coverage = ((y_test >= y_pred_lower) & (y_test <= y_pred_upper)).mean()
print(f"Coverage: {coverage:.2%}")

avg_width = np.mean(y_pred_upper - y_pred_lower)
print(f"Average Interval Width: {avg_width:.2f}")

mean_pinball_loss_01p = mean_pinball_loss(y_test, y_pred_lower, alpha=0.1)
print(
    f"Mean Pinball Loss for Lower Bound (10th Percentile): {mean_pinball_loss_01p:.4f}"
)
mean_pinball_loss_90p = mean_pinball_loss(y_test, y_pred_upper, alpha=0.9)
print(
    f"Mean Pinball Loss for Upper Bound (90th Percentile): {mean_pinball_loss_90p:.4f}"
)

# %% [markdown]
# #### 2.6.3. Predicting intervals with Gradient Boosting Regressor

# %%
# Preparing data for modelling

X = model_data[numeric_features + categorical_features]
y = model_data[target_feature].values.ravel()


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
from sklearn.ensemble import GradientBoostingRegressor

# %%
param_grid = dict(
    learning_rate=[0.01, 0.05, 0.1, 0.2],
    max_depth=[3, 5, 10, 20],
    min_samples_leaf=[1, 5, 10, 100, 10000],
    min_samples_split=[2, 10, 50, 100, 1000],
)
alpha = 0.1
mean_pinball_loss_01p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,
)
gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=0)
search_01p = HalvingRandomSearchCV(
    gbr,
    param_grid,
    resource="n_estimators",
    max_resources=500,
    min_resources=50,
    scoring=mean_pinball_loss_01p_scorer,
    n_jobs=2,
    random_state=0,
).fit(x_train, y_train)
pprint(search_01p.best_params_)

# %%
from sklearn.base import clone

alpha = 0.9
mean_pinball_loss_90p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,
)
search_90p = clone(search_01p).set_params(
    estimator__alpha=alpha,
    scoring=mean_pinball_loss_90p_scorer,
)
search_90p.fit(x_train, y_train)
pprint(search_90p.best_params_)

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

model_lower = GradientBoostingRegressor(
    loss="quantile", alpha=0.1, **search_01p.best_params_
)
model_lower.fit(x_train, y_train)

y_pred_lower = model_lower.predict(x_test)

mae_upper = GradientBoostingRegressor(
    loss="quantile", alpha=0.9, **search_90p.best_params_
)
mae_upper.fit(x_train, y_train)

y_pred_upper = mae_upper.predict(x_test)


# Evaluate using MAE (for both upper and lower bounds)
mae_upper = mean_absolute_error(y_test, y_pred_upper)
mae_lower = mean_absolute_error(y_test, y_pred_lower)

print(f"MAE for Upper Bound (90th Percentile): {mae_upper:.4f}")
print(f"MAE for Lower Bound (10th Percentile): {mae_lower:.4f}")

coverage_test = ((y_test >= y_pred_lower) & (y_test <= y_pred_upper)).mean()
print(f"Coverage test: {coverage_test:.2%}")

avg_width = np.mean(y_pred_upper - y_pred_lower)
print(f"Average Interval Width: {avg_width:.2f}")

mean_pinball_loss_01p = mean_pinball_loss(y_test, y_pred_lower, alpha=0.1)
print(
    f"Mean Pinball Loss for Lower Bound (10th Percentile): {mean_pinball_loss_01p:.4f}"
)
mean_pinball_loss_90p = mean_pinball_loss(y_test, y_pred_upper, alpha=0.9)
print(
    f"Mean Pinball Loss for Upper Bound (90th Percentile): {mean_pinball_loss_90p:.4f}"
)

# %%
print(
    f"Mean Pinball Loss for Lower Bound (10th Percentile): {mean_pinball_loss(y_test, y_pred_lower, alpha=0.5)}"
)
print(
    f"Mean Pinball Loss for Upper Bound (90th Percentile): {mean_pinball_loss(y_test, y_pred_upper, alpha=0.5)}"
)

# %%


# %%
