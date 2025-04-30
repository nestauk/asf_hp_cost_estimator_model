"""Data getters for MCS-EPC data and postcode/location data."""

import pandas as pd
import numpy as np
import os
from asf_hp_cost_estimator_model import config, PROJECT_DIR
from asf_hp_cost_estimator_model.getters.getter_utils import get_df_from_csv_url

postcode_path = config["postcode_path"]  # TODO: DELETE AFTER CODE REVIEW
regions_path = config["regions_path"]  # TODO: DELETE AFTER CODE REVIEW


def get_enhanced_installations_data():
    mcs_enhanced_with_epc = pd.read_csv(
        "s3://asf-core-data/outputs/MCS/mcs_installations_epc_full_250310.csv",
        usecols=config["relevant_mcs_epc_fields"],
        parse_dates=["INSPECTION_DATE", "commission_date"],
    )
    return mcs_enhanced_with_epc


def get_postcodes_data():  # TODO: DELETE AFTER CODE REVIEW
    """Get dataset of all UK postcodes with easting/northing coordinates,
    top-level region, and country columns. Save postcode and region columns
    as csv to enable lookup.

    Returns:
        Dataframe: Postcode geographic data.
    """
    # Read postcode data
    postcode_folder = PROJECT_DIR / postcode_path
    files = os.listdir(postcode_folder)
    postcode_df = pd.concat(
        # Only need postcode, coordinates and LA code cols
        (
            pd.read_csv(postcode_folder / file, header=None)[[0, 2, 3, 8]]
            for file in files
        ),
        ignore_index=True,
    )
    postcode_df.columns = ["postcode", "easting", "northing", "la_code"]

    postcode_df["postcode"] = postcode_df["postcode"].str.replace(" ", "")

    # Read English regions data so that LA codes can be associated with region names
    regions = pd.read_csv(PROJECT_DIR / regions_path)
    regions = regions[["LAD21CD", "RGN21NM"]]
    regions.columns = ["la_code", "region_name"]

    pc_regions = postcode_df.merge(regions, on="la_code", how="left")

    # Get country names from LA codes - country can be inferred from
    # first character of LA code
    country_dict = {
        "E": "England",
        "W": "Wales",
        "S": "Scotland",
        "N": "Northern Ireland",
        " ": np.nan,
    }
    pc_regions["country"] = (
        pc_regions["la_code"].fillna(" ").apply(lambda code: country_dict[code[0]])
    )
    # Outside England, region = country
    pc_regions["region_name"] = pc_regions["region_name"].fillna(pc_regions["country"])

    return pc_regions


def get_cpi_data() -> pd.DataFrame:
    """
    Get CPI data.

    Returns:
        pd.DataFrame: CPI data
    """
    cpi_05_3_df = get_df_from_csv_url(config["cpi_data"]["cpi_source_url"])

    return cpi_05_3_df


def get_postcode_to_lad_data(
    postcode_to_lad_data_file_name: str, s3_path: str = config["location_data_s3_path"]
) -> pd.DataFrame:
    """
    Gets location data with postcode matched to LAD and processes the postcode column.

    Args:
        s3_path (str, optional): Path to the location data. Defaults to config["location_data_s3_path"].
        postcode_to_lad_data_file_name (str, optional): File name of postcode to LAD data. Defaults to config["postcode_to_lad_data_file_name"].

    Returns:
        pd.DataFrame: postcode to LAD data
    """
    postcode_to_lad_data = pd.read_csv(
        os.path.join(s3_path, postcode_to_lad_data_file_name), encoding="latin-1"
    )
    postcode_to_lad_data["postcode"] = postcode_to_lad_data["pcds"].str.replace(" ", "")

    return postcode_to_lad_data


def get_lad_to_region_data(
    lad_to_region_file_name: str, s3_path: str = config["location_data_s3_path"]
) -> pd.DataFrame:
    """
    Gets location data with LAD matched to region and renames the columns.

    Args:
        s3_path (str, optional): Path to the location data. Defaults to config["location_data_s3_path"].
        lad_to_region_file_name (str, optional): File name of LAD to region data. Defaults to config["lad_to_region_file_name"].

    Returns:
        pd.DataFrame: LAD to region data
    """
    if lad_to_region_file_name.endswith(".csv"):
        lad_to_region_data = pd.read_csv(os.path.join(s3_path, lad_to_region_file_name))
    else:
        lad_to_region_data = pd.read_excel(
            os.path.join(s3_path, lad_to_region_file_name)
        )

    lad_to_region_data.columns = ["ladcd", "ladnm", "rgncd", "rgnnm", "FID"]

    return lad_to_region_data
