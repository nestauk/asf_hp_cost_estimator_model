# %% [markdown]
# # Investigating if there's a strong enough signal in the data
#
# If cost is noisy or there's not enough signal, we might not be able to get a good estimate of the cost.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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
model_data = process_data_before_modelling(
    mcs_epc_data, postcodes_data, cpi_quarterly_df
)

# Define features and target
numeric_features = config["numeric_features"]
categorical_features = config["categorical_features"]
target_feature = config["target_feature"]

# %% [markdown]
# ## 1. Investigating signal in data

# %%
model_data.plot("cost", "adjusted_cost", kind="scatter", alpha=0.5)

# %%
model_data[categorical_features].sum().sort_values(ascending=False)

# %% [markdown]
# ### Looking at the difference between commision date and inspection date

# %%
(model_data["commission_date"] - model_data["INSPECTION_DATE"]).median()

# %%
(model_data["commission_date"] - model_data["INSPECTION_DATE"]).mean()

# %%
(model_data["commission_date"] - model_data["INSPECTION_DATE"]).max()

# %%
np.corrcoef(
    (model_data["commission_date"] - model_data["INSPECTION_DATE"]).dt.days,
    model_data["cost"],
)

# %%
plt.plot(
    (model_data["commission_date"] - model_data["INSPECTION_DATE"]).dt.days,
    model_data["adjusted_cost"],
    "o",
    alpha=0.1,
)

# %%
model_data["days_betweeen_comission_inspection"] = (
    model_data["commission_date"] - model_data["INSPECTION_DATE"]
).dt.days

# %%


# %% [markdown]
# ### On average cost increases with time, which is expected
#
#

# %%
model_data["year"] = model_data["commission_date"].dt.year
model_data.groupby("year")["cost"].mean().plot()
plt.title("Mean cost per year")

# %%
model_data["year"] = model_data["commission_date"].dt.year
model_data.groupby("year")["adjusted_cost"].mean().plot()
plt.title("Mean ADJUSTED cost per year")

# %%
model_data.groupby("year")["original_mcs_index"].count().plot()
plt.title("Number of installations per year")

# %% [markdown]
# ### Looking at the cost distribution and at the cost distribution for specific categories in the data (built forms, property types, construction age bands, etc)

# %%
plt.figure(figsize=(10, 5))
plt.hist(model_data["cost"], bins=range(3500, 25500, 500), density=True)
plt.title("Cost Distribution")
plt.xlabel("Cost")

median_cost = model_data["cost"].median()

categorical_features_full = categorical_features + [
    "BUILT_FORM_end_terrace",
    "PROPERTY_TYPE_house",
    "region_name_yorkshire_and_the_humber",
    "CONSTRUCTION_AGE_BAND_2007_onwards",
]
for var in categorical_features_full:

    aux = model_data.loc[model_data[var]]
    median = aux["cost"].median().round()
    plt.figure(figsize=(10, 6))
    plt.hist(
        model_data["cost"],
        bins=range(3500, 25500, 500),
        label="all installations: " + str(median_cost),
        density=True,
    )
    plt.hist(
        aux["cost"],
        bins=range(3500, 25500, 500),
        label=var + ": " + str(median),
        density=True,
        alpha=0.5,
    )
    plt.legend()
    plt.title(var)
    plt.xlabel("Cost")

# %%
plt.figure(figsize=(10, 5))
plt.hist(model_data["adjusted_cost"], bins=range(3500, 25500, 500), density=True)
plt.title("Adjusted Cost Distribution")
plt.xlabel("Cost")

median_cost = model_data["adjusted_cost"].median()

categorical_features_full = categorical_features + [
    "BUILT_FORM_end_terrace",
    "PROPERTY_TYPE_house",
    "region_name_yorkshire_and_the_humber",
    "CONSTRUCTION_AGE_BAND_2007_onwards",
]
for var in categorical_features_full:

    aux = model_data.loc[model_data[var]]
    median = aux["adjusted_cost"].median().round()
    plt.figure(figsize=(10, 6))
    plt.hist(
        model_data["adjusted_cost"],
        bins=range(3500, 25500, 500),
        label="all installations: " + str(median_cost),
        density=True,
    )
    plt.hist(
        aux["adjusted_cost"],
        bins=range(3500, 25500, 500),
        label=var + ": " + str(median),
        density=True,
        alpha=0.5,
    )
    plt.legend()
    plt.title(var)
    plt.xlabel("Adjusted cost")

# %% [markdown]
# ### Looking at the distribution of years/ floor area/ number of rooms/ for all installations vs. installations with cost above £15K

# %%
plt.hist(
    model_data["commission_date"], bins=50, label="All installations", density=True
)

plt.hist(
    model_data[model_data["cost"] > 15000]["commission_date"],
    bins=50,
    label="Costs above £15k",
    density=True,
    alpha=0.5,
)
plt.legend()
plt.title("Distribution of commision date")

# %%
plt.figure(figsize=(10, 5))
plt.hist(model_data["TOTAL_FLOOR_AREA"], bins=50, label="All costs", density=True)
plt.hist(
    model_data[model_data["cost"] > 15000]["TOTAL_FLOOR_AREA"],
    bins=50,
    label="Costs above £15k",
    density=True,
    alpha=0.5,
)
plt.legend()
plt.title("Distribution of floor area")

# %%
len(model_data[model_data["cost"] > 15000]) / len(model_data) * 100

# %%
plt.figure(figsize=(10, 5))
plt.hist(model_data["n_days"], bins=50, label="All costs", density=True)
plt.hist(
    model_data[model_data["cost"] > 15000]["n_days"],
    bins=50,
    label="Costs above £15k",
    density=True,
    alpha=0.5,
)
plt.legend()
plt.title("Distribution of number of days")

# %%
plt.figure(figsize=(10, 5))
plt.hist(model_data["NUMBER_HABITABLE_ROOMS"], bins=50, label="All costs", density=True)
plt.hist(
    model_data[model_data["cost"] > 15000]["NUMBER_HABITABLE_ROOMS"],
    bins=50,
    label="Costs above £15k",
    density=True,
    alpha=0.5,
)
plt.legend()
plt.title("Distribution of number of rooms")

# %%


# %% [markdown]
# ### Correlations between numeric features and target

# %%
correlations = (
    model_data[numeric_features + [target_feature]]
    .corr()[target_feature]
    .sort_values(ascending=False)
)
print(correlations)

# %%
correlations = (
    model_data[numeric_features + ["adjusted_cost"]]
    .corr()["adjusted_cost"]
    .sort_values(ascending=False)
)
print(correlations)

# %% [markdown]
# ### Relationship between numeric features and cost


# %%
def aggregate_days(n_days):

    for i in range(500, 7000, 500):
        if n_days <= i:
            return i


model_data["n_days_grouped"] = model_data["n_days"].apply(aggregate_days)


def aggregate_floor_area(floor_area):

    for i in range(10, 600, 10):
        if floor_area <= i:
            return i


model_data["TOTAL_FLOOR_AREA_grouped"] = model_data["TOTAL_FLOOR_AREA"].apply(
    aggregate_floor_area
)


# %%
for feature in ["n_days_grouped", "TOTAL_FLOOR_AREA_grouped", "NUMBER_HABITABLE_ROOMS"]:
    data = model_data.groupby(feature)["cost"].mean().reset_index()
    plt.scatter(data[feature], data["cost"], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("cost")
    plt.title(f"{feature} vs cost")
    plt.show()

# %% [markdown]
# ### Relationship between numeric features and adjusted cost

# %%
for feature in ["n_days_grouped", "TOTAL_FLOOR_AREA_grouped", "NUMBER_HABITABLE_ROOMS"]:
    data = model_data.groupby(feature)["adjusted_cost"].mean().reset_index()
    plt.scatter(data[feature], data["adjusted_cost"], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("adjusted cost")
    plt.title(f"{feature} vs adjusted cost")
    plt.show()

# %%
