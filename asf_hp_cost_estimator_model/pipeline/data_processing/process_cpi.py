import pandas as pd
import logging

from asf_hp_cost_estimator_model import config


def get_df_quarterly_cpi_with_adjustment_factors(
    ref_year: int, cpi_df: pd.DataFrame, cpi_col_header: str
) -> pd.DataFrame:
    """
    Get dataframe of CPI (consumer price index) data with adjustment factors calculated for a given reference year.

    Args
        ref_year (int): reference year to take whole year CPI value from
        cpi_df (pd.DataFrame): CPI time series data
        cpi_col_header (str): name of column containing CPI values in cpi_df

    Returns
        pd.DataFrame: quarterly CPI values with adjustment factors for a given reference year
    """
    cpi_ref_value = _get_int_ref_cpi_value(ref_year, cpi_df)
    logging.info(f"CPI reference value for year {ref_year}: {cpi_ref_value}")
    cpi_quarterly_df = _get_df_quarterly_cpi_data(cpi_df)
    cpi_quarterly_df["adjustment_factor"] = _compute_series_cpi_adjustment_factors(
        ref_cpi=cpi_ref_value, cpi_series=cpi_quarterly_df[cpi_col_header]
    )

    return cpi_quarterly_df


def _get_int_ref_cpi_value(ref_year: int, cpi_df: pd.DataFrame) -> int:
    """
    Get whole year CPI (consumer price index) value for a given reference year.

    Args
        ref_year (int): reference year to take whole year CPI value from
        cpi_df (pd.DataFrame): CPI time series data

    Returns
        int: whole year CPI value
    """
    ref_cpi = cpi_df.loc[
        cpi_df["Title"] == str(ref_year),
        config["cpi_data"]["cpi_column_header"],
    ].values.astype(float)[0]

    return ref_cpi


def _get_df_quarterly_cpi_data(cpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get dataframe of quarterly CPI (consumer price index) values.

    Args
        cpi_df (pd.DataFrame): CPI time series data

    Returns
        pd.DataFrame: quarterly CPI data
    """
    return cpi_df.loc[lambda df: df["Title"].str.contains("Q")].reset_index(drop=True)


def _compute_series_cpi_adjustment_factors(
    ref_cpi: int, cpi_series: pd.Series
) -> pd.Series:
    """
    Compute series of adjustment factors for a series of CPI (consumer price index) values against a CPI reference value.

    Args
        ref_cpi (int): CPI reference value
        cpi_series (pd.Series): CPI values to generate adjustment factors for

    Returns
        pd.Series: adjustment factors
    """
    return 1 + ((ref_cpi - cpi_series.astype(float)) / cpi_series.astype(float))
