# Create a heatmap visualization of the well plate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import numpy as np


def create_well_plate_heatmap(
    results: Dict[str, float],
    num_rows: int = 8,
    num_cols: int = 12,
    title: str = "Well Plate Heatmap - Mean Intensity",
    cmap: str = "viridis",
    figsize: tuple = (12, 6),
    annot: bool = True,
    fmt: str = ".1f",
) -> plt.Figure:
    """
    Create a heatmap visualization of well plate data.

    Parameters
    ----------
    results : Dict[str, float]
        Dictionary with well positions as keys (format: "row/col", e.g., "B/4")
        and intensity values as values
    num_rows : int, optional
        Number of rows in the well plate (default: 8 for 96-well plate)
    num_cols : int, optional
        Number of columns in the well plate (default: 12 for 96-well plate)
    title : str, optional
        Title for the heatmap (default: "Well Plate Heatmap - Mean Intensity")
    cmap : str, optional
        Matplotlib/seaborn colormap name (default: "viridis")
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 6))
    annot : bool, optional
        Whether to show values in cells (default: True)
    fmt : str, optional
        String formatting for annotations (default: ".1f")

    Returns
    -------
    plt.Figure
        Matplotlib figure object containing the heatmap

    Examples
    --------
    >>> results = {"A/1": 100.5, "A/2": 120.3, "B/1": 95.7}
    >>> fig = create_well_plate_heatmap(results, num_rows=8, num_cols=12)
    >>> plt.show()
    """
    # Generate row labels (A, B, C, ...)
    rows_labels = [chr(65 + i) for i in range(num_rows)]  # 65 is ASCII for 'A'

    # Generate column labels (1, 2, 3, ...)
    cols_labels = list(range(1, num_cols + 1))

    # Create a matrix for the heatmap (filled with NaN for empty wells)
    heatmap_data = np.full((num_rows, num_cols), np.nan)

    # Fill the matrix with intensity values from results
    for well_key, intensity in results.items():
        row_name, col_name = well_key.split("/")
        row_idx = rows_labels.index(row_name)
        col_idx = cols_labels.index(int(col_name))
        heatmap_data[row_idx, col_idx] = intensity

    # Create DataFrame for better labeling
    df_heatmap = pd.DataFrame(heatmap_data, index=rows_labels, columns=cols_labels)

    # Create the heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        df_heatmap,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": "Mean Intensity"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Column", fontsize=12)
    plt.ylabel("Row", fontsize=12)
    plt.tight_layout()

    return fig
