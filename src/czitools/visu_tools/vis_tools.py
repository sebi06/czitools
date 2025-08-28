# -*- coding: utf-8 -*-

#################################################################
# File        : vis_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple
from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def scatterplot_mpl(
    planetable: pd.DataFrame,
    s: int = 0,
    t: int = 0,
    z: int = 0,
    c: int = 0,
    msz2d: int = 35,
    normz: bool = True,
    fig1savename: str = "zsurface2d.png",
    fig2savename: str = "zsurface3d.png",
    msz3d: int = 20,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates 2D and 3D scatter plots of XYZ positions from a given table and saves them as PNG files.
    Parameters:
    -----------
    planetable : pandas.DataFrame
        DataFrame containing the XYZ positions with columns 'X[micron]', 'Y[micron]', and 'Z[micron]'.
    s : int, optional
        Unused parameter, default is 0.
    t : int, optional
        Time point, default is 0.
    z : int, optional
        Z position, default is 0.
    c : int, optional
        Channel, default is 0.
    msz2d : int, optional
        Marker size for the 2D scatter plot, default is 35.
    normz : bool, optional
        If True, normalize Z data by subtracting the minimum value, default is True.
    fig1savename : str, optional
        Filename for saving the 2D scatter plot, default is "zsurface2d.png".
    fig2savename : str, optional
        Filename for saving the 3D scatter plot, default is "zsurface3d.png".
    msz3d : int, optional
        Marker size for the 3D scatter plot, default is 20.
    Returns:
    --------
    fig1 : matplotlib.figure.Figure
        The 2D scatter plot figure.
    fig2 : matplotlib.figure.Figure
        The 3D scatter plot figure.
    Raises:
    -------
    KeyError
        If the required columns 'X[micron]', 'Y[micron]', or 'Z[micron]' are not found in the DataFrame.
    Notes:
    ------
    - The function attempts to find a "good" aspect ratio for the figures based on the data.
    - The Y-axis is inverted to have the origin (0,0) at the top-left.
    - The color of the scatter points represents the Z positions.
    - The function logs the filenames of the saved figures.
    """

    # extract XYZ positions
    try:
        xpos = planetable["X[micron]"]
        ypos = planetable["Y[micron]"]
        zpos = planetable["Z[micron]"]
    except KeyError:
        xpos = planetable["X [micron]"]
        ypos = planetable["Y [micron]"]
        zpos = planetable["Z [micron]"]

    # normalize z-data by substracting the minimum value
    if normz:
        zpos = zpos - zpos.min()

    # create a name for the figure
    figtitle = "XYZ-Positions: T=" + str(t) + " Z=" + str(z) + " CH=" + str(c)

    # try to find a "good" aspect ratio for the figures
    dx = xpos.max() - xpos.min()
    dy = ypos.max() - ypos.min()
    fsy = 8
    fsx = int(np.ceil(fsy * dx / dy))

    # create figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(fsx + 1, fsy))

    # invert the Y-axis --> O,O = Top-Left
    ax1.invert_yaxis()

    # configure the axis
    ax1.set_title(figtitle)
    ax1.set_xlabel("Stage X-Axis [micron]", fontsize=12, fontweight="normal")
    ax1.set_ylabel("Stage Y-Axis [micron]", fontsize=12, fontweight="normal")
    ax1.grid(True)
    ax1.set_aspect("equal", "box")

    # plot data and label the colorbar
    sc1 = ax1.scatter(
        xpos,
        ypos,
        marker="s",
        c=zpos,
        s=msz2d,
        facecolor=cm.coolwarm,
        edgecolor="black",
    )

    # add the colorbar on the right-hand side
    cb1 = plt.colorbar(sc1, fraction=0.046, shrink=0.8, pad=0.04)

    # add a label
    if normz:
        cb1.set_label("Z-Offset [micron]", labelpad=20, fontsize=12, fontweight="normal")
    if not normz:
        cb1.set_label("Z-Position [micron]", labelpad=20, fontsize=12, fontweight="normal")

    # save figure as PNG
    fig1.savefig(fig1savename, dpi=100)
    logger.info(f"Saved: {fig1savename}")

    # 3D plot of surface
    fig2 = plt.figure(figsize=(fsx + 1, fsy))
    ax2 = fig2.add_subplot(111, projection="3d")

    # invert the Y-axis --> O,O = Top-Left
    ax2.invert_yaxis()

    # define the labels
    ax2.set_xlabel("Stage X-Axis [micron]", fontsize=12, fontweight="normal")
    ax2.set_ylabel("Stage Y-Axis [micron]", fontsize=12, fontweight="normal")
    ax2.set_title(figtitle)

    # plot data and label the colorbar
    sc2 = ax2.scatter(
        xpos,
        ypos,
        zpos,
        marker=".",
        s=msz3d,
        c=zpos,
        facecolor=cm.coolwarm,
        depthshade=False,
    )

    # add colorbar to the 3d plot
    cb2 = plt.colorbar(sc2, shrink=0.8)
    # add a label
    if normz:
        cb2.set_label("Z-Offset [micron]", labelpad=20, fontsize=12, fontweight="normal")
    if not normz:
        cb2.set_label("Z-Position [micron]", labelpad=20, fontsize=12, fontweight="normal")

    # save figure as PNG
    fig2.savefig(fig2savename, dpi=100)
    logger.info(f"Saved: {fig2savename}")

    return fig1, fig2


def scatterplot_plotly(
    planetable: pd.DataFrame,
    s: int = 0,
    t: int = 0,
    z: int = 0,
    c: int = 0,
    msz2d: int = 35,
    normz: bool = True,
    fig1savename: str = "zsurface2d.html",
    fig2savename: str = "zsurface3d.html",
    msz3d: int = 20,
) -> Tuple[go.Figure, go.Figure]:
    """
    Generates 2D and 3D scatter plots using Plotly and saves them as HTML files.
    Parameters:
    -----------
    planetable : pandas.DataFrame
        DataFrame containing the XYZ positions and other relevant data.
    s : int, optional
        Placeholder parameter (default is 0).
    t : int, optional
        Placeholder parameter (default is 0).
    z : int, optional
        Placeholder parameter (default is 0).
    c : int, optional
        Placeholder parameter (default is 0).
    msz2d : int, optional
        Marker size for the 2D scatter plot (default is 35).
    normz : bool, optional
        If True, normalize the Z data by subtracting the minimum value (default is True).
    fig1savename : str, optional
        Filename for saving the 2D scatter plot HTML file (default is "zsurface2d.html").
    fig2savename : str, optional
        Filename for saving the 3D scatter plot HTML file (default is "zsurface3d.html").
    msz3d : int, optional
        Marker size for the 3D scatter plot (default is 20).
    Returns:
    --------
    fig1 : plotly.graph_objs._figure.Figure
        The generated 2D scatter plot figure.
    fig2 : plotly.graph_objs._figure.Figure
        The generated 3D scatter plot figure.
    """

    # extract XYZ position for the selected channel
    try:
        xpos = planetable["X[micron]"]
        ypos = planetable["Y[micron]"]
        zpos = planetable["Z[micron]"]
    except KeyError:
        xpos = planetable["X [micron]"]
        ypos = planetable["Y [micron]"]
        zpos = planetable["Z [micron]"]

    # normalize z-data by substracting the minimum value
    if normz:
        zpos = zpos - zpos.min()
        scalebar_title = "Z-Offset [micron]"
    if not normz:
        scalebar_title = "Z-Position [micron]"

    # create a name for the figure
    figtitle = "XYZ-Positions: T=" + str(t) + " Z=" + str(z) + " CH=" + str(c)

    fig1 = go.Figure(
        data=go.Scatter(
            x=xpos,
            y=ypos,
            mode="markers",
            text=np.round(zpos, 1),
            marker_symbol="square",
            marker_size=msz2d,
            marker=dict(
                color=zpos,
                colorscale="Viridis",
                line_width=2,
                showscale=True,
                colorbar=dict(thickness=10, title=dict(text=scalebar_title, side="right")),
            ),
        )
    )

    fig1.update_xaxes(showgrid=True, zeroline=True, automargin=True)
    fig1.update_yaxes(showgrid=True, zeroline=True, automargin=True)
    fig1["layout"]["yaxis"]["autorange"] = "reversed"
    fig1.update_layout(
        title=figtitle,
        xaxis_title="StageX Position [micron]",
        yaxis_title="StageY Position [micron]",
        font=dict(size=16, color="Black"),
    )

    # save the figure
    fig1.write_html(fig1savename)
    logger.info(f"Saved: {fig1savename}")

    fig2 = go.Figure(
        data=[
            go.Scatter3d(
                x=xpos,
                y=ypos,
                z=zpos,
                mode="markers",
                marker=dict(
                    size=msz3d,
                    color=zpos,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(thickness=10, title=dict(text=scalebar_title, side="right")),
                ),
            )
        ]
    )

    fig2.update_xaxes(showgrid=True, zeroline=True, automargin=True)
    fig2.update_yaxes(showgrid=True, zeroline=True, automargin=True)
    fig2["layout"]["yaxis"]["autorange"] = "reversed"
    fig2.update_layout(
        title=figtitle,
        xaxis_title="StageX Position [micron]",
        yaxis_title="StageY Position [micron]",
        font=dict(size=16, color="Black"),
    )

    # save the figure
    fig2.write_html(fig2savename)
    logger.info(f"Saved: {fig2savename}")

    return fig1, fig2
