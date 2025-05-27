"""Data getters for MCS-EPC data, postcode/location data and CPI."""

import pandas as pd
import numpy as np
import os
from asf_hp_cost_estimator_model import config, PROJECT_DIR
from asf_hp_cost_estimator_model.getters.getter_utils import get_df_from_csv_url

postcode_path = config["postcode_path"]  # TODO: DELETE AFTER CODE REVIEW
regions_path = config["regions_path"]  # TODO: DELETE AFTER CODE REVIEW


def get_enhanced_installations_data(
    date: str = config["mcs_epc_filename_date"],
    usecols: list = config["relevant_mcs_epc_fields"],
) -> pd.DataFrame:
    """
    Get MCS-EPC data.

    Args:
        date (str, optional): processing date reference, YYMMDD. Defaults to config["mcs_epc_filename_date"].
        usecols (list, optional): columns to import. Defaults to config["relevant_mcs_epc_fields"].

    Returns:
        pd.DataFrame: MCS-EPC data
    """
    mcs_enhanced_with_epc = pd.read_csv(
        f"s3://asf-core-data/outputs/MCS/mcs_installations_epc_full_{date}.csv",
        usecols=usecols,
        parse_dates=["INSPECTION_DATE", "commission_date"],
    )
    return mcs_enhanced_with_epc


def get_cpi_data() -> pd.DataFrame:
    """
    Get the Consumer Price Index (CPI) data from the source URL.

    Returns:
        pd.DataFrame: CPI data
    """
    cpi_05_3_df = get_df_from_csv_url(config["cpi_data"]["cpi_source_url"])

    return cpi_05_3_df


def get_postcode_to_lad_data(
    postcode_to_lad_data_file_name: str, s3_dir: str = config["location_data_s3_dir"]
) -> pd.DataFrame:
    """
    Gets location data with postcode matched to LAD and processes the postcode column.

    Args:
        postcode_to_lad_data_file_name (str, optional): File name of postcode to LAD data.
        s3_dir (str, optional): Path to the location data directory. Defaults to config["location_data_s3_dir"].
    Returns:
        pd.DataFrame: postcode to LAD data
    """
    postcode_to_lad_data = pd.read_csv(
        os.path.join(s3_dir, postcode_to_lad_data_file_name), encoding="latin-1"
    )
    postcode_to_lad_data["postcode"] = postcode_to_lad_data["pcds"].str.replace(" ", "")

    return postcode_to_lad_data


def get_lad_to_region_data(
    lad_to_region_file_name: str, s3_dir: str = config["location_data_s3_dir"]
) -> pd.DataFrame:
    """
    Gets location data with LAD matched to region and renames the columns.

    Args:
        lad_to_region_file_name (str, optional): File name of LAD to region data.
        s3_dir (str, optional): Path to the location data directory. Defaults to config["location_data_s3_dir"].
    Returns:
        pd.DataFrame: LAD to region data
    """
    if lad_to_region_file_name.endswith(".csv"):
        lad_to_region_data = pd.read_csv(os.path.join(s3_dir, lad_to_region_file_name))
    else:
        lad_to_region_data = pd.read_excel(
            os.path.join(s3_dir, lad_to_region_file_name)
        )

    lad_to_region_data.columns = ["ladcd", "ladnm", "rgncd", "rgnnm", "FID"]

    return lad_to_region_data
