"""
Plotting utils.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
from typing import List
import logging

# Fonts and colours
FONT_NAME = "Averta-Regular"
FONT = "Averta"
TITLE_FONT = "Averta"
FONTSIZE_TITLE = 16
FONTSIZE_NORMAL = 14


def set_spines():
    """
    Function to add or remove spines from plots.
    """
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = True


def set_plotting_styles():
    """
    Function that sets plotting styles.
    """

    sns.set_context("talk")

    set_spines()

    all_font_files = font_manager.findSystemFonts()

    try:
        mpl.rcParams["font.family"] = "sans-serif"
        font_files = [f for f in all_font_files if "Averta-Regular" in f]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.sans-serif"] = "Averta"
    except:
        logging.info(
            "Averta" + " font could not be located. Using 'DejaVu Sans' instead"
        )
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f][0]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = "DejaVu Sans"

    mpl.rcParams["xtick.labelsize"] = FONTSIZE_NORMAL
    mpl.rcParams["ytick.labelsize"] = FONTSIZE_NORMAL
    mpl.rcParams["axes.titlesize"] = FONTSIZE_TITLE
    mpl.rcParams["axes.labelsize"] = FONTSIZE_NORMAL
    mpl.rcParams["legend.fontsize"] = FONTSIZE_NORMAL
    mpl.rcParams["figure.titlesize"] = FONTSIZE_TITLE


def plot_residuals_numeric_variables(
    actual: np.array, predicted: np.array, x: np.array, x_label: str
):
    """
    Plot model residuals against the predicted values.
    Residual = actual - predicted

    Args:
        actual (np.array): Actual costs
        predicted (np.array): Predicted costs
        x (np.array): numeric variable values
        x_label (str): numeric variable name
    """
    set_plotting_styles()
    plt.figure(figsize=(10, 5))

    # Add horizontal line at 0
    plt.axhline(0, color="blue", linestyle="--", linewidth=1)

    # Plot residuals and regression trend line
    sns.regplot(
        x=x,
        y=actual - predicted,
        scatter_kws={"s": 2, "alpha": 0.1},
    )

    # Add title and labels
    plt.title("Residuals for " + x_label)
    plt.xlabel(x_label)
    plt.ylabel("Actual - Predicted (£)")

    plt.tight_layout()
    plt.show()


def plot_residuals_categorical_variables(
    data: pd.DataFrame,
    actual: np.array,
    predicted: np.array,
    features: List[str],
    x_label: str,
):
    """
    Plot boxplot of model residuals by dummified categorical feature.
    Residual = actual - predicted

    Args:
        data (pd.DataFrame): Data used for splitting by category. Must contain all columns in 'features'
            and column values must be 0/1.
        actual (np.array): Actual costs.
        predicted (np.array): Predicted costs.
        features (List[str]): List of features to consider.
        x_label (str): Text to display on x axis of plot.
        file_path (str): Location where file should be saved.
    """
    set_plotting_styles()
    plt.figure(figsize=(10, 5))

    # Add vertical line at 0
    plt.axvline(0, color="blue", linestyle="--", linewidth=1)

    # Get indices of rows where feature is True
    feature_indices = {f: np.where(data[f])[0] for f in features}

    # Calculate residuals for each feature
    boxplot_dict = {
        f: actual[feature_indices[f]] - predicted[feature_indices[f]] for f in features
    }

    # Plotting residuals for each feature
    plt.boxplot(boxplot_dict.values(), vert=False)

    # Add title and labels
    plt.title("Residuals for " + x_label)
    plt.yticks(
        range(1, len(features) + 1), features
    )  # ensure feature names are displayed in the correct place
    plt.xlabel("Actual - Predicted (£)")

    plt.tight_layout()
    plt.show()
