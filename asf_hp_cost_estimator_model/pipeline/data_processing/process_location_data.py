"""
Functions for processing location data.
"""

import pandas as pd
import numpy as np
from asf_hp_cost_estimator_model import config
from asf_hp_cost_estimator_model.getters.data_getters import (
    get_postcode_to_lad_data,
    get_lad_to_region_data,
)


def join_location_dfs_on_lad(
    postcode_to_lad_data_filename: str, lad_to_region_filename: str
) -> pd.DataFrame:
    """
    Joins postcode to LAD data with LAD to region data:
        - gets postcode to LAD data
        - gets LAD to region data
        - merges the two datasets on LAD code
        - keeps only relevant columns: postcode, LAD code, LAD name, region code, and region name

    Returns:
        pd.DataFrame: location data with postcode, LAD, and region information.
    """
    # UK postcode to LAD data
    postcode_to_lad_data = get_postcode_to_lad_data(postcode_to_lad_data_filename)

    # LAD to region data - England only
    lad_to_region_data = get_lad_to_region_data(lad_to_region_filename)

    # Removing ladnm because it exists in both datasets
    lad_to_region_data.drop(columns=["ladnm"], inplace=True)

    postcode_to_region = postcode_to_lad_data.merge(
        lad_to_region_data, on="ladcd", how="left"
    )

    postcode_to_region = postcode_to_region[
        ["postcode", "ladcd", "ladnm", "rgncd", "rgnnm"]
    ]

    return postcode_to_region


def get_location_data() -> pd.DataFrame:
    """
    Processes location data to match postcode to region name:
        - first uses census 2021 data to create postcode to region mapping
        - for postcodes with missing region, uses census 2011 data to fill in the gaps
        - fills missing region with country name (for Scotland and Wales)

    Returns
        pd.DataFrame: location data with postcode, LAD, and region information.
    """
    # 2021 Census data location data
    postcode_to_region = join_location_dfs_on_lad(
        config["postcode_to_lad_census_2021_filename"],
        config["lad_to_region_census_2021_filename"],
    )

    # postcodes with missing region (it might be because they are in Scotland or Wales or because they are no
    # longer postcodes in use, so we look at 2011 location data from census)
    postcodes_with_missing_region = postcode_to_region[
        pd.isnull(postcode_to_region["rgncd"])
    ]["postcode"].unique()

    # 2011 Census data location data
    postcode_to_region_2011 = join_location_dfs_on_lad(
        config["postcode_to_lad_census_2011_filename"],
        config["lad_to_region_census_2011_filename"],
    )

    postcode_to_region_2011 = postcode_to_region_2011[
        postcode_to_region_2011["postcode"].isin(postcodes_with_missing_region)
    ]
    postcode_to_region = postcode_to_region[
        ~postcode_to_region["postcode"].isin(postcodes_with_missing_region)
    ]

    # Final location information
    postcode_to_region = pd.concat(
        [postcode_to_region, postcode_to_region_2011], ignore_index=True
    )

    # Get country names from LAD codes - country can be inferred from first character of LAD code
    country_dict = {
        "E": "England",
        "W": "Wales",
        "S": "Scotland",
        "N": "Northern Ireland",
        "L": "Channel Islands",
        "M": "Isle of Man",
        " ": np.nan,
    }
    postcode_to_region["country"] = (
        postcode_to_region["ladcd"]
        .fillna(" ")
        .apply(lambda code: country_dict[code[0]])
    )

    # LAD to region mapping only available for England, so for Scotland and Wales we use region as the country
    postcode_to_region["region_name"] = postcode_to_region["rgnnm"].fillna(
        postcode_to_region["country"]
    )

    return postcode_to_region
