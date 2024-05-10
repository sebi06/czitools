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
from czitools.tools import logger

logger = logger.get_logger()


def scatterplot_mpl(
    planetable,
    s=0,
    t=0,
    z=0,
    c=0,
    msz2d=35,
    normz=True,
    fig1savename="zsurface2d.png",
    fig2savename="zsurface3d.png",
    msz3d=20,
):

    # extract XYZ positions
    try:
        xpos = planetable["X[micron]"]
        ypos = planetable["Y[micron]"]
        zpos = planetable["Z[micron]"]
    except KeyError as e:
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
        cb1.set_label(
            "Z-Offset [micron]", labelpad=20, fontsize=12, fontweight="normal"
        )
    if not normz:
        cb1.set_label(
            "Z-Position [micron]", labelpad=20, fontsize=12, fontweight="normal"
        )

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
        cb2.set_label(
            "Z-Offset [micron]", labelpad=20, fontsize=12, fontweight="normal"
        )
    if not normz:
        cb2.set_label(
            "Z-Position [micron]", labelpad=20, fontsize=12, fontweight="normal"
        )

    # save figure as PNG
    fig2.savefig(fig2savename, dpi=100)
    logger.info(f"Saved: {fig2savename}")

    return fig1, fig2


def scatterplot_plotly(
    planetable,
    s=0,
    t=0,
    z=0,
    c=0,
    msz2d=35,
    normz=True,
    fig1savename="zsurface2d.html",
    fig2savename="zsurface3d.html",
    msz3d=20,
):

    # extract XYZ position for the selected channel
    try:
        xpos = planetable["X[micron]"]
        ypos = planetable["Y[micron]"]
        zpos = planetable["Z[micron]"]
    except KeyError as e:
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
                colorbar=dict(
                    thickness=10, title=dict(text=scalebar_title, side="right")
                ),
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
                    colorbar=dict(
                        thickness=10, title=dict(text=scalebar_title, side="right")
                    ),
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
