"""
Functions to process data required to train a model to estimate the cost of an air source heat pump,
including cleaning and filtering the joined MCS-EPC.
"""

import numpy as np
import pandas as pd

from asf_hp_cost_estimator_model import config


def updates_construction_age_band(mcs_epc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Update CONSTRUCTION_AGE_BAND for new dwellings when missing.

    Args:
        mcs_epc_data (pd.DataFrame):  MCS HP records joined to EPC. Assumed to
        contain INSPECTION_DATE, TRANSACTION_TYPE, TENURE, commission_date and

    Returns:
        pd.DataFrame: updated records.
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
        | (
            first_records["TENURE"]
            == "Not defined - use in the case of a new dwelling for which the intended tenure in not known. It is no"  # note: typos intentional!
        )
        | (first_records["TENURE"] == "unknown")
        | pd.isnull(first_records["TENURE"])
    ]

    # Filling in CONSTRUCTION_AGE_BAND for new dwellings
    print("Before updating CONSTRUCTION_AGE_BAND:")
    print(mcs_epc_data["CONSTRUCTION_AGE_BAND"].value_counts(dropna=False))
    mcs_epc_data.loc[
        mcs_epc_data["original_mcs_index"].isin(new_dwellings["original_mcs_index"]),
        "CONSTRUCTION_AGE_BAND",
    ] = "2007 onwards"
    print("After updating CONSTRUCTION_AGE_BAND:")
    print(mcs_epc_data["CONSTRUCTION_AGE_BAND"].value_counts(dropna=False))

    print("Values of tenure when CONSTRUCTION_AGE_BAND is missing:")
    print(
        mcs_epc_data[mcs_epc_data["CONSTRUCTION_AGE_BAND"] == "unknown"][
            "TENURE"
        ].value_counts(dropna=False)
    )
    print("Values of TRANSACTION_TYPE when CONSTRUCTION_AGE_BAND is missing:")
    print(
        mcs_epc_data[mcs_epc_data["CONSTRUCTION_AGE_BAND"] == "unknown"][
            "TRANSACTION_TYPE"
        ].value_counts(dropna=False)
    )

    return mcs_epc_data


def remove_properties_with_hp_when_built(
    mcs_epc_data: pd.DataFrame, hp_when_built_threshold: int
) -> pd.DataFrame:
    """
    Removes records assumed to relate to dwellings that were built
    with a HP already installed from a dataframe of "fully joined"
    MCS-EPC data.

    New dwellings are identified by considering the TRANSACTION_TYPE
    and TENURE fields of their first EPC certificate.
    For these dwellings, the number of days between the
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

    # Identifying first EPC records for each dwelling
    first_records = (
        mcs_epc_data.sort_values("INSPECTION_DATE")
        .groupby("original_mcs_index")
        .head(1)
    )

    # Idenifying new dwellings
    new_dwellings = first_records.loc[
        (first_records["TRANSACTION_TYPE"] == "new dwelling")
        | (
            first_records["TENURE"]
            == "Not defined - use in the case of a new dwelling for which the intended tenure in not known. It is no"  # note: typos intentional!
        )
        | (first_records["TENURE"] == "unknown")
        | (first_records["TENURE"] == np.nan)
    ]

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
    - Cost not NA
    - INSPECTION_DATE is not null (installation is linked to an EPC)

    Args:
        mcs_epc_data (pd.DataFrame): MCS-EPC records.

    Returns:
        pd.DataFrame: Records relevant for modelling.
    """
    filtered_data = mcs_epc_data.loc[
        (mcs_epc_data["tech_type"] == "Air Source Heat Pump")
        & ~mcs_epc_data["INSPECTION_DATE"].isnull()  # has an EPC certificate
        & ~mcs_epc_data["cost"].isna()
    ].reset_index(drop=True)

    return filtered_data


def remove_samples_exclusion_criteria(
    mcs_epc_data: pd.DataFrame, exclusion_criteria_dict: dict
) -> pd.DataFrame:
    """
    - Cost in a sensible range for an ASHP installation
    - Number of habitable rooms within a certain range
    - Only selected property types
    - Not part of a "cluster" of installations within the same postcode and time interval

    Args:
        mcs_epc_data (pd.DataFrame): MCS-EPC records.
        exclusion_criteria_dict (dict): Dictionary of exclusion criteria.

    Returns:
        pd.DataFrame: Records relevant for modelling.
    """

    filtered_data = mcs_epc_data.copy()

    if "TOTAL_FLOOR_AREA_lower_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (
                mcs_epc_data["TOTAL_FLOOR_AREA"]
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

    if "cost_lower_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (filtered_data["cost"] >= exclusion_criteria_dict["cost_lower_bound"])
        ]

    if "cost_upper_bound" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            (filtered_data["cost"] <= exclusion_criteria_dict["cost_upper_bound"])
        ]

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

    if "PROPERTY_TYPE_allowed_list" in exclusion_criteria_dict:
        filtered_data = filtered_data.loc[
            filtered_data["PROPERTY_TYPE"].isin(
                exclusion_criteria_dict["PROPERTY_TYPE_allowed_list"]
            )
        ]

    filtered_data = filtered_data.loc[~filtered_data["cluster"]]

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
        np.nan: "unknown",
    }

    mcs_epc_data["CONSTRUCTION_AGE_BAND"] = mcs_epc_data["CONSTRUCTION_AGE_BAND"].map(
        age_bands_mapping
    )

    print(
        "CONSTRUCTION_AGE_BAND values after mapping:",
        mcs_epc_data["CONSTRUCTION_AGE_BAND"].value_counts(dropna=False),
    )

    for col in ["BUILT_FORM", "PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND", "region_name"]:
        mcs_epc_data[col] = mcs_epc_data[col].replace(np.nan, "unknown")

    mcs_epc_data["region_name"] = (
        mcs_epc_data["region_name"].str.lower().str.replace(" ", "_")
    )
    mcs_epc_data["BUILT_FORM"] = (
        mcs_epc_data["BUILT_FORM"].str.lower().str.replace(" ", "_")
    )
    mcs_epc_data["BUILT_FORM"] = mcs_epc_data["BUILT_FORM"].apply(
        lambda x: x.replace(
            x,
            (
                "end-terrace"
                if "end-terrace" in x
                else x.replace(x, "mid-terrace" if "mid-terrace" in x else x)
            ),
        )
    )

    for col in ["BUILT_FORM", "PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND", "region_name"]:
        mcs_epc_data[col] = mcs_epc_data[col].str.replace("-", "_")

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
            columns=[
                "BUILT_FORM",
                "PROPERTY_TYPE",
                "CONSTRUCTION_AGE_BAND",
                "region_name",
                "number_of_rooms",
            ],
        )
    else:
        mcs_epc_data = pd.get_dummies(
            mcs_epc_data,
            columns=[
                "BUILT_FORM",
                "PROPERTY_TYPE",
                "CONSTRUCTION_AGE_BAND",
                "region_name",
            ],
        )

    return mcs_epc_data


def process_data_before_modelling(
    mcs_epc_data: pd.DataFrame,
    postcodes_data: pd.DataFrame,
    hp_when_built_threshold: int = config["hp_when_built_threshold"],
    exclusion_criteria_dict: dict = config["exclusion_criteria"],
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
        enhanced_installations_data, exclusion_criteria_dict
    )

    enhanced_installations_data = choose_epc_records(enhanced_installations_data)

    enhanced_installations_data = join_postcode_data(
        enhanced_installations_data, postcodes_data
    )

    enhanced_installations_data = add_n_days_col(enhanced_installations_data, min_date)

    enhanced_installations_data = dummify_variables(
        enhanced_installations_data, rooms_as_categorical
    )

    return enhanced_installations_data
