import requests
import pandas as pd
import numpy as np
from io import BytesIO


def _get_content_from_url(url: str) -> BytesIO:
    """
    Get BytesIO stream from URL.
    Args
        url (str): URL
    Returns
        io.BytesIO: content of URL as BytesIO stream
    """
    with requests.Session() as session:
        res = session.get(url)
    content = BytesIO(res.content)

    return content


def get_df_from_csv_url(url: str, **kwargs) -> pd.DataFrame:
    """
    Get dataframe from CSV file stored at URL.

    Args
        url (str): URL location of CSV file download
        **kwargs for pandas.read_csv()

    Returns
        pd.DataFrame: dataframe from CSV file
    """
    content = _get_content_from_url(url)
    df = pd.read_csv(content, **kwargs)

    return df
