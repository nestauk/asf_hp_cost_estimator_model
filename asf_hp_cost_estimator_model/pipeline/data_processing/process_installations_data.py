"""
Functions to process data required to train a model to estimate the cost of an air source heat pump,
including cleaning and filtering the joined MCS-EPC.
"""

import numpy as np
import pandas as pd

from asf_hp_cost_estimator_model import config


def identify_new_dwellings(mcs_epc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies new dwellings in MCS-EPC data.

    Args:
        mcs_epc_data (pd.DataFrame): MCS HP records joined to EPC. Assumed to
        contain INSPECTION_DATE, TRANSACTION_TYPE, TENURE, commission_date and
        original_mcs_index columns.

    Returns:
        pd.DataFrame: MCS-EPC data corresponding to new dwellings
    """

    # Identifying first EPC records for each dwelling
    first_records = (
        mcs_epc_data.sort_values("INSPECTION_DATE")
        .groupby("original_mcs_index")
        .head(1)
    )

    # Identifying new dwellings
    new_dwellings = first_records.loc[
        (first_records["TRANSACTION_TYPE"] == "new dwelling")
    ]

    return new_dwellings


def updates_construction_age_band(mcs_epc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Update CONSTRUCTION_AGE_BAND for new dwellings when missing.

    Args:
        mcs_epc_data (pd.DataFrame): MCS HP records joined to EPC. Assumed to
        contain INSPECTION_DATE, TRANSACTION_TYPE, TENURE, commission_date and
        original_mcs_index columns.

    Returns:
        pd.DataFrame: updated records.
    """

    new_dwellings = identify_new_dwellings(mcs_epc_data=mcs_epc_data)

    # Filling in CONSTRUCTION_AGE_BAND for new dwellings
    mcs_epc_data.loc[
        mcs_epc_data["original_mcs_index"].isin(new_dwellings["original_mcs_index"]),
        "CONSTRUCTION_AGE_BAND",
    ] = "2007 onwards"

    # Identify construction age band for each dwelling so that if it is missing in some records
    # it can be filled in with the most common value found
    most_common_construction_age_band = (
        mcs_epc_data[
            ~pd.isnull(mcs_epc_data["CONSTRUCTION_AGE_BAND"])
        ]  # remove records with missing age band
        .groupby(
            ["original_mcs_index", "CONSTRUCTION_AGE_BAND"]
        )  # group by installation index and age band
        .size()  # count number of records for each age band
        .sort_values(ascending=False)  # sort by count
        .reset_index()
        .groupby("original_mcs_index")  # group by installation index
        .head(1)  # get most common age band
        .drop(columns=0)  # drop count column
    )

    mcs_epc_data = mcs_epc_data.join(
        most_common_construction_age_band,
        how="left",
        on="original_mcs_index",
        rsuffix="_2",
    )

    # Fill in missing CONSTRUCTION_AGE_BAND with most common value
    mcs_epc_data["CONSTRUCTION_AGE_BAND"] = mcs_epc_data[
        "CONSTRUCTION_AGE_BAND"
    ].fillna(mcs_epc_data["CONSTRUCTION_AGE_BAND_2"])

    return mcs_epc_data


def remove_properties_with_hp_when_built(
    mcs_epc_data: pd.DataFrame, hp_when_built_threshold: int
) -> pd.DataFrame:
    """
    Removes records assumed to relate to dwellings that were built
    with a HP already installed from a dataframe of "fully joined"
    MCS-EPC data.

    For new dwellings, the number of days between the
    installation date and the date of the first EPC certificate is
    calculated. If this number is below a certain threshold, the
    dwelling is assumed to have been built with a HP and the
    associated records are removed.

    Args:
        mcs_epc_data (pd.DataFrame): MCS HP records joined to EPC. Assumed to
        contain INSPECTION_DATE, TRANSACTION_TYPE, TENURE, commission_date and
        original_mcs_index columns.
        hp_when_built_threshold (int): threshold for assuming a dwelling was built with a HP.

    Returns:
        pd.DataFrame: Joined records filtered to dwellings assumed not
        to have been built with a HP.
    """

    new_dwellings = identify_new_dwellings(mcs_epc_data=mcs_epc_data)

    new_dwellings["days_between_inspection_and_hp_comission"] = abs(
        new_dwellings["commission_date"] - new_dwellings["INSPECTION_DATE"]
    )

    # Assume dwelling was built with HP if time difference between EPC inspection
    # when dwelling was built and HP installation is less than threshold
    new_dwellings["assumed_hp_when_built"] = (
        new_dwellings["days_between_inspection_and_hp_comission"].dt.days
        < hp_when_built_threshold
    )

    # Get installation indices of dwellings assumed to have a HP when built
    invalid_indices = new_dwellings.loc[
        new_dwellings["assumed_hp_when_built"], "original_mcs_index"
    ]

    # Filter data to remove records associated with dwellings assumed to have a HP when built
    filtered_data = mcs_epc_data.loc[
        ~(mcs_epc_data["original_mcs_index"].isin(invalid_indices))
    ].reset_index(drop=True)

    return filtered_data


def filter_to_relevant_samples(mcs_epc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter HP installation data to samples that are useful for modelling.
    Samples useful for modelling are:
    - ASHP installations
    - Cost not missing
    - original_epc_index is not missing (installation is linked to an EPC)
    - INSPECTION_DATE is not missing
    - remove non-domestic installations
    - Not part of a "cluster" of installations within the same postcode and time interval

    Args:
        mcs_epc_data (pd.DataFrame): MCS-EPC records.

    Returns:
        pd.DataFrame: Records relevant for modelling.
    """

    filtered_data = mcs_epc_data.copy()
    # Filter to ASHP installations

    key_variables = ["cost", "original_epc_index", "INSPECTION_DATE"]
    filtered_data = filtered_data.dropna(subset=key_variables, how="any")

    filtered_data = filtered_data.loc[
        (filtered_data["tech_type"] == "Air Source Heat Pump")
    ]

    # Filter to samples that are not non-domestic installations
    filtered_data = filtered_data.loc[
        ~(
            filtered_data["installation_type"]
            .str.lower()
            .isin(["commercial", "non-domestic"])
        )
    ]

    # Filter out samples that are part of a "cluster" of installations
    filtered_data = filtered_data.loc[~filtered_data["cluster"]]

    filtered_data = filtered_data.reset_index(drop=True)
    return filtered_data


def remove_samples_exclusion_criteria(
    mcs_epc_data: pd.DataFrame, exclusion_criteria_dict: dict, winsorise: str = "upper"
) -> pd.DataFrame:
    """
    - Cost in a sensible range for an ASHP installation
    - Number of habitable rooms and total floor area within a certain range
    - Only selected property types

    Args:
        mcs_epc_data (pd.DataFrame): MCS-EPC records.
        exclusion_criteria_dict (dict): Dictionary of exclusion criteria.
        winsorise (str, optional): whether to winsorise outliers. Defaults to "upper".Takes "upper", "lower", "both" and "none" as values.
        "upper": upper outliers are replaced with the upper bound and lower outliers are removed
        "lower": lower outliers are replaced with the lower bound and upper outliers are removed
        "both": upper outliers are replaced with the upper bound and lower outliers are replaced with the lower bound
        "none": both lower and upper outliers are removed



    Returns:
        pd.DataFrame: Records relevant for modelling.
    """

    filtered_data = mcs_epc_data.copy()

    if "PROPERTY_TYPE_allowed_list" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            filtered_data["PROPERTY_TYPE"].isin(
                exclusion_criteria_dict["PROPERTY_TYPE_allowed_list"]
            )
        ]

    if "TOTAL_FLOOR_AREA_lower_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                filtered_data["TOTAL_FLOOR_AREA"]
                >= exclusion_criteria_dict["TOTAL_FLOOR_AREA_lower_bound"]
            )
        ]

    if "TOTAL_FLOOR_AREA_upper_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                filtered_data["TOTAL_FLOOR_AREA"]
                <= exclusion_criteria_dict["TOTAL_FLOOR_AREA_upper_bound"]
            )
        ]

    if "adjusted_cost_lower_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                filtered_data["adjusted_cost"]
                >= exclusion_criteria_dict["adjusted_cost_lower_bound"]
            )
        ]

    if "adjusted_cost_upper_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                filtered_data["adjusted_cost"]
                <= exclusion_criteria_dict["adjusted_cost_lower_bound"]
            )
        ]

    if "cost_lower_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (filtered_data["cost"] >= exclusion_criteria_dict["cost_lower_bound"])
        ]

    if "cost_upper_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (filtered_data["cost"] <= exclusion_criteria_dict["cost_upper_bound"])
        ]

    if (
        ("ajusted_cost_upper_bound" not in exclusion_criteria_dict)
        and ("cost_upper_bound" not in exclusion_criteria_dict)
        and ("adjusted_cost_lower_bound" not in exclusion_criteria_dict)
        and ("cost_lower_bound" not in exclusion_criteria_dict)
    ):
        filtered_data = remove_or_winsorise_cost_outliers_per_group(
            data=filtered_data,
            target_feature=config["target_feature"],
            winsorise=winsorise,
        )

    if "NUMBER_HABITABLE_ROOMS_lower_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                filtered_data["NUMBER_HABITABLE_ROOMS"]
                >= exclusion_criteria_dict["NUMBER_HABITABLE_ROOMS_lower_bound"]
            )
        ]

    if "NUMBER_HABITABLE_ROOMS_upper_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                filtered_data["NUMBER_HABITABLE_ROOMS"]
                <= exclusion_criteria_dict["NUMBER_HABITABLE_ROOMS_upper_bound"]
            )
        ]

    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def choose_epc_records(mcs_epc_data: pd.DataFrame) -> pd.DataFrame:
    """
    From a fully joined MCS-EPC dataset (i.e. each MCS record joined to all possible
    EPC records), chooses the "most relevant" EPC for each HP installation record
    (i.e. the one that is assumed to best reflect the status of the property at the
    time of HP installation). The EPC chosen is the latest one before the installation
    if it exists; otherwise it is the first one after the installation.

    Args:
        mcs_epc_data (pd.DataFrame): Joined MCS-EPC data. Assumed to
        contain INSPECTION_DATE, commission_date and original_mcs_index columns.

    Returns:
        pd.DataFrame: HP installation records with the "most relevant" EPC.
    """

    # Sort data by INSPECTION_DATE
    mcs_epc_data = mcs_epc_data.sort_values("INSPECTION_DATE").reset_index(drop=True)

    # Identify rows where the EPC inspection is before the MCS installation
    mcs_epc_data["epc_before_mcs"] = (
        mcs_epc_data["INSPECTION_DATE"] <= mcs_epc_data["commission_date"]
    )

    # Identify indices of last EPC record before MCS
    last_epc_before_mcs_indices = (
        mcs_epc_data.reset_index()
        .loc[mcs_epc_data["epc_before_mcs"]]
        .groupby("original_mcs_index")
        .tail(1)["index"]
        .values
    )

    # Flag last EPC before MCS
    mcs_epc_data["last_epc_before_mcs"] = False
    mcs_epc_data["last_epc_before_mcs"].iloc[last_epc_before_mcs_indices] = True

    # Filter to either "last EPC before MCS" or "EPC after MCS",
    # then group by installation and take first record -
    # this will be the last EPC before MCS
    # if it exists, otherwise the first EPC after MCS
    filtered_data = (
        mcs_epc_data.loc[
            mcs_epc_data["last_epc_before_mcs"] | ~mcs_epc_data["epc_before_mcs"]
        ]
        .groupby("original_mcs_index")
        .head(1)
        .reset_index(drop=True)
        .drop(columns=["epc_before_mcs", "last_epc_before_mcs"])
    )

    return filtered_data


def join_postcode_data(
    mcs_epc_data: pd.DataFrame, postcodes_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge easting/northing and LA code data to data featuring a postcode column.

    Args:
        mcs_epc_data (pd.DataFrame): joined MCS-EPC data with "postcode" column.
        postcodes_data (pd.DataFrame): Dataframe with easting/northing and LA codes corresponding to postcodes.

    Returns:
        pd.DataFrame: MCS-EPC data enhanced with location data
    """
    mcs_epc_data = mcs_epc_data.merge(postcodes_data, on="postcode", how="left")

    return mcs_epc_data


def add_n_days_col(mcs_epc_data: pd.DataFrame, min_date: str) -> pd.DataFrame:
    """
    Adds column to the MCS-EPC installations data corresponding to:
    number of days elapsed between a set minimum date and
    the date when the HP was commissioned.

    Args:
        mcs_epc_data (pd.DataFrame): Dataframe with "commission_date" column.
        min_date (str): Minimum date to calculate days elapsed from.

    Returns:
        pd.DataFrame: Dataframe with added "n_days" column.
    """

    mcs_epc_data["n_days"] = (
        mcs_epc_data["commission_date"] - pd.to_datetime(min_date)
    ).dt.days

    return mcs_epc_data


# Need dummy variables to flag when a variable is in one of several categories
# (e.g. construction age bands such as "pre-1929") so easier to dummify variables
# manually rather than include this step as part of the modelling pipeline
def dummify_variables(
    mcs_epc_data: pd.DataFrame, rooms_as_categorical: bool = False
) -> pd.DataFrame:
    """
    Transform columns in data by dummifying according to dictionary.

    Args:
        mcs_epc_data (pd.Dataframe): Dataframe containing keys of var_dict as columns.
        rooms_as_categorical (bool): Whether to treat number of rooms as a categorical
        variable which is then dummified (True) or to treat it as a continuous variable (False).

    Returns:
        pd.Dataframe: Dataframe with columns dummified.
    """

    # Remove records with missing values in key features before creating dummies
    key_variables = (
        config["categorical_features_to_dummify"] + config["numeric_features"]
    )
    mcs_epc_data = mcs_epc_data.dropna(subset=key_variables, how="any")

    # renaming and aggregating age bands
    age_bands_mapping = {
        "England and Wales: before 1900": "pre_1929",
        "Scotland: before 1919": "pre_1929",
        "1900-1929": "pre_1929",
        "1930-1949": "between_1930_1966",
        "1950-1966": "between_1930_1966",
        "1965-1975": "between_1965_1983",
        "1976-1983": "between_1965_1983",
        "1983-1991": "between_1983_2007",
        "1996-2002": "between_1983_2007",
        "1991-1998": "between_1983_2007",
        "2003-2007": "between_1983_2007",
        "2007 onwards": "2007_onwards",
    }
    mcs_epc_data["CONSTRUCTION_AGE_BAND"] = mcs_epc_data["CONSTRUCTION_AGE_BAND"].map(
        age_bands_mapping
    )

    # aggregating BUILT forms ("End-Terrace" with "Enclosed End-Terrace" and "Mid-Terrace" with "Enclosed Mid-Terrace")
    mcs_epc_data["BUILT_FORM"] = mcs_epc_data["BUILT_FORM"].apply(
        lambda x: x.replace(
            x,
            (
                "End-Terrace"
                if "End-Terrace" in x
                else x.replace(x, "Mid-Terrace" if "Mid-Terrace" in x else x)
            ),
        )
    )

    # replacing "-" and " " with "_" in feature values
    for col in config["categorical_features_to_dummify"]:
        mcs_epc_data[col] = mcs_epc_data[col].str.lower()
        mcs_epc_data[col] = mcs_epc_data[col].str.replace(" ", "_")
        mcs_epc_data[col] = mcs_epc_data[col].str.replace("-", "_")

    original_feature_data = mcs_epc_data[config["categorical_features_to_dummify"]]

    if rooms_as_categorical:
        rooms_mapping = {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8+",
        }

        mcs_epc_data["number_of_rooms"] = mcs_epc_data["NUMBER_HABITABLE_ROOMS"].apply(
            lambda x: "8+" if x > 8 else rooms_mapping.get(x, np.nan)
        )

        mcs_epc_data = pd.get_dummies(
            mcs_epc_data,
            columns=config["categorical_features_to_dummify"]
            + [
                "number_of_rooms",
            ],
            dtype=int,
        )
    else:
        mcs_epc_data = pd.get_dummies(
            mcs_epc_data,
            columns=config["categorical_features_to_dummify"],
            dtype=int,
        )

    mcs_epc_data = pd.concat([original_feature_data, mcs_epc_data], axis=1)

    return mcs_epc_data


def _generate_series_year_quarters(commission_date_series: pd.Series) -> pd.Series:
    """
    Generate a series of years and quarters from a series of dates.

    Args
        commission_date_series (pd.Series): commission dates with year, month, and day

    Returns
        pd.Series: series of year and quarter values in the form `YYYY QN`
    """
    return (
        commission_date_series.pipe(pd.to_datetime).dt.year.astype(str)
        + " Q"
        + commission_date_series.pipe(pd.to_datetime).dt.quarter.astype(str)
    )


def generate_df_adjusted_costs(
    mcs_epc_df: pd.DataFrame, cpi_quarters_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join CPI (consumer price index) dataframe containing quarterly adjustment factors to MCS-EPC dataframe and
    calculate adjusted installation costs for each row.

    Args
        mcs_epc_df (pd.DataFrame): joined MCS-EPC dataframe
        cpi_quarters_df (pd.DataFrame): quarterly CPI data with adjustment factors for each quarter

    Returns
        pd.DataFrame: MCS-EPC dataframe with CPI values, adjustment factors, and adjusted costs
    """
    mcs_epc_df["year_quarter"] = _generate_series_year_quarters(
        commission_date_series=mcs_epc_df["commission_date"]
    )

    mcs_epc_inf = mcs_epc_df.merge(
        cpi_quarters_df, how="left", left_on="year_quarter", right_on="Title"
    )

    mcs_epc_inf["adjusted_cost"] = (
        mcs_epc_inf["cost"] * mcs_epc_inf["adjustment_factor"]
    )

    return mcs_epc_inf


def remove_or_winsorise_cost_outliers_per_group(
    data: pd.DataFrame,
    target_feature: str = config["target_feature"],
    winsorise: str = "upper",
):
    """
    Remove or winsorise outliers from cost data for each group, where a group is a combination of
    PROPERTY_TYPE, CONSTRUCTION_AGE_BAND, and NUMBER_HABITABLE_ROOMS.

    Args:
        data (pd.DataFrame): installations data
        target_feature (str, optional): cost feature to use as target. Defaults to config["target_feature"].
        winsorise (str, optional): whether to winsorise outliers. Defaults to "upper". Takes "upper", "lower",
        "both" and "none" as values.
            "upper": upper outliers are replaced with the upper bound and lower outliers are removed
            "lower": lower outliers are replaced with the lower bound and upper outliers are removed
            "both": upper outliers are replaced with the upper bound and lower outliers are replaced with the lower bound
            "none": both lower and upper outliers are removed
    """
    # Creating a "8+" category, while keeping number of habitable rooms as a numeric variable
    data["NUMBER_HABITABLE_ROOMS"] = data["NUMBER_HABITABLE_ROOMS"].apply(
        lambda x: 8 if x > 8 else x
    )
    combinations = (
        data.groupby(
            ["PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND", "NUMBER_HABITABLE_ROOMS"]
        )[["original_mcs_index"]]
        .nunique()
        .sort_values("original_mcs_index")
        .reset_index()
        .rename(columns={"original_mcs_index": "number_of_installations"})
    )

    combinations["combination"] = [
        "combination_" + str(i) for i in range(1, len(combinations) + 1)
    ]

    data = data.merge(
        combinations[
            [
                "PROPERTY_TYPE",
                "CONSTRUCTION_AGE_BAND",
                "NUMBER_HABITABLE_ROOMS",
                "combination",
            ]
        ],
        on=["PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND", "NUMBER_HABITABLE_ROOMS"],
        how="left",
    )

    cleaned_df = pd.DataFrame()

    for comb in data["combination"].unique():
        subset = data[data["combination"] == comb]

        Q1 = subset[target_feature].quantile(0.25)
        Q3 = subset[target_feature].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

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
            # Filter out outliers on both ends
            subset_clean = subset_clean[(subset_clean[target_feature] <= upper_bound)]

        cleaned_df = pd.concat([cleaned_df, subset_clean], ignore_index=True)

    return cleaned_df


def process_data_before_modelling(
    mcs_epc_data: pd.DataFrame,
    postcodes_data: pd.DataFrame,
    cpi_quarterly_df: pd.DataFrame,
    hp_when_built_threshold: int = config["hp_when_built_threshold"],
    exclusion_criteria_dict: dict = config["exclusion_criteria"],
    winsorise: str = config["winsorise_outliers"],
    min_date: dict = config["min_date"],
    rooms_as_categorical: bool = False,
) -> pd.DataFrame:
    """
    Get clean MCS-EPC data in suitable format for modelling.

    Args:
        mcs_epc_data (pd.DataFrame): MCS-EPC data.
        postcodes_data (pd.DataFrame): Postcode data.
        hp_when_built_threshold (int, optional): Threshold for assuming a dwelling was built with a HP.
        exclusion_criteria_dict (dict, optional): Dictionary of exclusion criteria.
        min_date (str, optional): Minimum date to calculate days elapsed from.
        rooms_as_categorical (bool): Whether to treat number of rooms as a categorical
        variable which is then dummified (True) or to treat it as a continuous variable (False).

    Returns:
        pd.Dataframe: Suitable MCS-EPC data.
    """
    enhanced_installations_data = mcs_epc_data.copy()

    enhanced_installations_data = enhanced_installations_data.replace(
        "(?i)unknown", np.nan, regex=True
    )

    enhanced_installations_data = updates_construction_age_band(
        enhanced_installations_data
    )

    enhanced_installations_data = remove_properties_with_hp_when_built(
        enhanced_installations_data, hp_when_built_threshold
    )

    enhanced_installations_data = filter_to_relevant_samples(
        enhanced_installations_data
    )

    enhanced_installations_data = remove_samples_exclusion_criteria(
        mcs_epc_data=enhanced_installations_data,
        exclusion_criteria_dict=exclusion_criteria_dict,
        winsorise=winsorise,
    )

    enhanced_installations_data = generate_df_adjusted_costs(
        mcs_epc_df=enhanced_installations_data, cpi_quarters_df=cpi_quarterly_df
    )

    enhanced_installations_data = choose_epc_records(enhanced_installations_data)

    enhanced_installations_data = join_postcode_data(
        enhanced_installations_data, postcodes_data
    )

    enhanced_installations_data = add_n_days_col(enhanced_installations_data, min_date)

    enhanced_installations_data = dummify_variables(
        enhanced_installations_data, rooms_as_categorical
    )

    enhanced_installations_data.reset_index(drop=True, inplace=True)

    return enhanced_installations_data
