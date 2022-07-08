# -*- coding: utf-8 -*-

#################################################################
# File        : visutools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm, use
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from skimage.color import label2rgb
import plotly.graph_objects as go
#import plotly.express as px
#from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Type, Any, Union
import re


def create_heatmap(platetype: int) -> np.ndarray:
    """Create empty heatmap array based on the platetype

    :param platetype: [description], defaults to 96
    :type platetype: int, optional
    :return: heatmap_array - empty array with the shape of the wellplate
    :rtype: NumPy.Array
    """

    # create heatmap based on the platetype
    nr, nc = getrowandcolumn(platetype=platetype)
    heatmap_array = np.full([nr, nc], np.nan)

    return heatmap_array


def extract_wellstats(results, well: str = "A1",
                      wellstr: str = "WellID",
                      wellcolstr: str = "WellColumnID",
                      wellrowstr: str = "WellRowID") -> Tuple[pd.DataFrame, int, int]:

    # extract all entries for specific well
    well_results = results.loc[results[wellstr] == well]

    # get the descriptive statistics for specific well
    stats = well_results.describe(include="all")

    # get the column an row indices for specific well
    col = int(stats[wellcolstr]["mean"])
    row = int(stats[wellrowstr]["mean"])

    return (stats, row, col)


def showheatmap(heatmap, parameter2display,
                fontsize_title=12,
                fontsize_label=10,
                colormap='Blues',
                linecolor='black',
                linewidth=1.0,
                save=False,
                # savename='Heatmap.png',
                robust=True,
                filename='Test.czi',
                dpi=100,
                show=False):
    """Plot a heatmap for a wellplate from a dataframe.

    :param heatmap: Pandas DataFrame with the heatmap data for a wellplate
    :type heatmap: Pandas.dataFrame
    :param parameter2display: Measurement parameter to be displays as heatmap
    :type parameter2display: str
    :param fontsize_title: font size of title, defaults to 12
    :type fontsize_title: int, optional
    :param fontsize_label: font size of labels, defaults to 10
    :type fontsize_label: int, optional
    :param colormap: Specifies which colormap to use for the heatmap, defaults to 'Blues'
    :type colormap: str, optional
    :param linecolor: Specifies the color of the line between the wells, defaults to 'black'
    :type linecolor: str, optional
    :param linewidth: Specifies the line width between the wells, defaults to 1.0
    :type linewidth: float, optional
    :param save: Option to save the heapmap as PNG image, defaults to False
    :type save: bool, optional
    :param robust: If True and vmin or vmax are absent, the colormap range is
    computed with robust quantiles instead of the extreme values., defaults to True
    :type robust: bool, optional
    :param filename: filename of the original image - will be used to derive
    the filename for the PNG image to be saved, defaults to 'Test.czi'
    :type filename: str, optional
    :param dpi: dpi, defaults to 100
    :type dpi: int, optional
    :param show: show the plot, defaults to False
    :type show: bool, optional
    :return: savename - filename of the saved plot
    :rtype: str
    """

    # create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # create the heatmap
    ax = sns.heatmap(heatmap,
                     ax=ax,
                     cmap=colormap,
                     linecolor=linecolor,
                     linewidths=linewidth,
                     square=True,
                     robust=robust,
                     annot=False,
                     cbar_kws={"shrink": 0.68})

    # customize the plot to your needs
    ax.set_title(parameter2display,
                 fontsize=fontsize_title,
                 fontweight='normal')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label)

    # modify the labels of the colorbar
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=fontsize_label)

    if save:
        savename = filename[:-4] + '_HM_' + parameter2display + '.png'
        fig.savefig(savename,
                    dpi=dpi,
                    orientation='portrait',
                    transparent=False)
        print('Heatmap image saved as: ', savename)
    else:
        savename = None

    if show:

        plt.show()

    return savename


def getrowandcolumn(platetype=96):
    """Returns the number of rows and columns for a given wellplate type

    :param platetype: number total wells of plate (6, 24, 96, 384 or 1536), defaults to 96
    :type platetype: int, optional
    :return: nr - number of rows of wellplate
    :rtype: int
    :return: nc - number of columns for wellplate
    :rtype: int
    """

    platetype = int(platetype)

    if platetype == 6:
        nr = 2
        nc = 3
    elif platetype == 24:
        nr = 4
        nc = 6
    elif platetype == 96:
        nr = 8
        nc = 12
    elif platetype == 384:
        nr = 16
        nc = 24
    elif platetype == 1536:
        nr = 32
        nc = 48

    return nr, nc


def convert_array_to_heatmap(hmarray, nr, nc):
    """Get the labels for a well plate and create a data frame
    from the numpy array.

    :param hmarray: The numpy array containing the actual heatmap
    :type hmarray: NumPy.Array
    :param nr: number of rows for the well plate
    :type nr: int
    :param nc: number of columns for the well plate
    :type nc: int
    :return: A Pandas dataframe with the respective row and columns labels
    :rtype: Pandas.DataFrame
    """

    lx, ly = extract_labels(nr, nc)
    heatmap_dataframe = pd.DataFrame(hmarray, index=ly, columns=lx)

    return heatmap_dataframe


def extract_labels(nr, nc):
    """[summary]

    :param nr: number of rows of the wellplate, e.g. 8 (A-H) for a 96 wellplate
    :type nr: [type]
    :param nc: number of columns of the wellplate, e.g. 12 (1-12) for a 96 wellplate
    :type nc: [type]
    :return: lx - list containing the actual row IDs for the selected wellplate
    :rtype: list
    :return: ly - list containing the actual column IDs for the selected wellplate
    :rtype: list
    """

    # labeling schemes
    labelX = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
              '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
              '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
              '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', ]

    labelY = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF']

    lx = labelX[0:nc]
    ly = labelY[0:nr]

    return lx, ly


def plot_segresults(image, mask, props, add_bbox=True):
    """Display the results of the segmentation

    :param image: 2d image with the original data
    :type image: NumPy.Array
    :param mask: binary mask from the segmentation
    :type mask: NumPy.Array
    :param props: region props with scikit-image
    :type props: list (of region properties)
    :param add_bbox: show the bounding box on the original image, defaults to True
    :type add_bbox: bool, optional
    """

    # create overlay image
    image_label_overlay = label2rgb(mask, image=image, bg_label=0)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(image,
                 cmap=plt.cm.gray,
                 interpolation='nearest',
                 clim=[image.min(), image.max() * 0.5])

    ax[1].imshow(image_label_overlay,
                 clim=[image.min(), image.max() * 0.5])

    ax[0].set_title('Original', fontsize=12)
    ax[1].set_title('Masks', fontsize=12)

    if add_bbox:
        ax[0] = add_boundingbox(props, ax[0])

    plt.show()

    return ax


def add_boundingbox(props, ax2plot):
    """Add bounding boxes for objects to the current axes

    :param props: list of measured regions
    :type props: list
    :param ax2plot: matplot axis where the bounding boxes should be added
    :type ax2plot: axes
    :return: ax2plot - the axes including the bounding boxes
    :rtype: axes
    """

    for index, row in props.iterrows():

        minr = row['bbox-0']
        minc = row['bbox-1']
        maxr = row['bbox-2']
        maxc = row['bbox-3']
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False,
                                  edgecolor='red',
                                  linewidth=1)
        ax2plot.add_patch(rect)

    return ax2plot


def scatterplot_mpl(planetable,
                    s=0, t=0, z=0, c=0,
                    msz2d=35,
                    normz=True,
                    fig1savename='zsurface2d.png',
                    fig2savename='zsurface3d.png',
                    show3d=False,
                    msz3d=20):

    # extract XYZ positions
    try:
        xpos = planetable['X[micron]']
        ypos = planetable['Y[micron]']
        zpos = planetable['Z[micron]']
    except KeyError as e:
        xpos = planetable['X [micron]']
        ypos = planetable['Y [micron]']
        zpos = planetable['Z [micron]']

    # normalize z-data by substracting the minimum value
    if normz:
        zpos = zpos - zpos.min()

    # create a name for the figure
    figtitle = 'XYZ-Positions:  S=' + str(s) + ' T=' + str(t) + ' Z=' + str(z) + ' CH=' + str(c)

    # try to find a "good" aspect ration for the figures
    dx = xpos.max() - xpos.min()
    dy = ypos.max() - ypos.min()
    fsy = 8
    fsx = np.ceil(fsy * dx / dy).astype(np.int)

    # create figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(fsx + 1, fsy))

    # invert the Y-axis --> O,O = Top-Left
    ax1.invert_yaxis()

    # configure the axis
    ax1.set_title(figtitle)
    ax1.set_xlabel('Stage X-Axis [micron]', fontsize=12, fontweight='normal')
    ax1.set_ylabel('Stage Y-Axis [micron]', fontsize=12, fontweight='normal')
    ax1.grid(True)
    ax1.set_aspect('equal', 'box')

    # plot data and label the colorbar
    sc1 = ax1.scatter(xpos, ypos,
                      marker='s',
                      c=zpos,
                      s=msz2d,
                      facecolor=cm.coolwarm,
                      edgecolor='black')

    # add the colorbar on the right-hand side
    cb1 = plt.colorbar(sc1,
                       fraction=0.046,
                       shrink=0.8,
                       pad=0.04)

    # add a label
    if normz:
        cb1.set_label('Z-Offset [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')
    if not normz:
        cb1.set_label('Z-Position [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')

    # save figure as PNG
    fig1.savefig(fig1savename, dpi=100)
    print('Saved: ', fig1savename)

    # 3D plot of surface
    fig2 = plt.figure(figsize=(fsx + 1, fsy))
    ax2 = fig2.add_subplot(111, projection='3d')

    # invert the Y-axis --> O,O = Top-Left
    ax2.invert_yaxis()

    # define the labels
    ax2.set_xlabel('Stage X-Axis [micron]',
                   fontsize=12,
                   fontweight='normal')
    ax2.set_ylabel('Stage Y-Axis [micron]',
                   fontsize=12,
                   fontweight='normal')
    ax2.set_title(figtitle)

    # plot data and label the colorbar
    sc2 = ax2.scatter(xpos, ypos, zpos,
                      marker='.',
                      s=msz3d,
                      c=zpos,
                      facecolor=cm.coolwarm,
                      depthshade=False)

    # add colorbar to the 3d plot
    cb2 = plt.colorbar(sc2, shrink=0.8)
    # add a label
    if normz:
        cb2.set_label('Z-Offset [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')
    if not normz:
        cb2.set_label('Z-Position [micron]',
                      labelpad=20,
                      fontsize=12,
                      fontweight='normal')

    # save figure as PNG
    fig2.savefig(fig2savename, dpi=100)
    print('Saved: ', fig2savename)

    return fig1, fig2


def scatterplot_plotly(planetable,
                       s=0, t=0, z=0, c=0,
                       msz2d=35,
                       normz=True,
                       fig1savename='zsurface2d.html',
                       fig2savename='zsurface3d.html',
                       msz3d=20):

    # extract XYZ position for the selected channel
    try:
        xpos = planetable['X[micron]']
        ypos = planetable['Y[micron]']
        zpos = planetable['Z[micron]']
    except KeyError as e:
        xpos = planetable['X [micron]']
        ypos = planetable['Y [micron]']
        zpos = planetable['Z [micron]']

    # normalize z-data by substracting the minimum value
    if normz:
        zpos = zpos - zpos.min()
        scalebar_title = 'Z-Offset [micron]'
    if not normz:
        scalebar_title = 'Z-Position [micron]'

    # create a name for the figure
    figtitle = 'XYZ-Positions:  S=' + str(s) + ' T=' + str(t) + ' Z=' + str(z) + ' CH=' + str(c)

    fig1 = go.Figure(
        data=go.Scatter(
            x=xpos,
            y=ypos,
            mode='markers',
            text=np.round(zpos, 1),
            marker_symbol='square',
            marker_size=msz2d,
            marker=dict(
                color=zpos,
                colorscale='Viridis',
                line_width=2,
                showscale=True,
                colorbar=dict(thickness=10,
                              title=dict(
                                  text=scalebar_title,
                                  side='right'))
            )
        )
    )

    fig1.update_xaxes(showgrid=True, zeroline=True, automargin=True)
    fig1.update_yaxes(showgrid=True, zeroline=True, automargin=True)
    fig1['layout']['yaxis']['autorange'] = "reversed"
    fig1.update_layout(title=figtitle,
                       xaxis_title="StageX Position [micron]",
                       yaxis_title="StageY Position [micron]",
                       font=dict(size=16,
                                 color='Black')
                       )

    # save the figure
    fig1.write_html(fig1savename)
    print('Saved: ', fig1savename)

    fig2 = go.Figure(data=[go.Scatter3d(
        x=xpos,
        y=ypos,
        z=zpos,
        mode='markers',
        marker=dict(
            size=msz3d,
            color=zpos,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(thickness=10,
                          title=dict(
                              text=scalebar_title,
                              side='right')
                          )
        )
    )])

    fig2.update_xaxes(showgrid=True, zeroline=True, automargin=True)
    fig2.update_yaxes(showgrid=True, zeroline=True, automargin=True)
    fig2['layout']['yaxis']['autorange'] = "reversed"
    fig2.update_layout(title=figtitle,
                       xaxis_title="StageX Position [micron]",
                       yaxis_title="StageY Position [micron]",
                       font=dict(size=16,
                                 color='Black')
                       )

    # save the figure
    fig2.write_html(fig2savename)
    print('Saved: ', fig2savename)

    return fig1, fig2


def rename_columns(df: pd.DataFrame,
                   paramlist: List,
                   verbose: bool = False) -> pd.DataFrame:

    for i in range(0, len(paramlist)):
        # rename the columns with measured parameters and correct types
        if verbose:
            print('Renamed : ', df.columns[i], ' to ', paramlist[i])
        try:
            df.rename(columns={df.columns[i]: paramlist[i]}, inplace=True)
        except:
            print('Column not find inside table for renaming. Doing nothing.')

    return df


def clean_zenia_headernames(df: pd.DataFrame,
                            match: str = "::") -> pd.DataFrame:

    # get headers and make names readable
    new_headers = []
    for h in list(df.columns.values):
        try:
            new_headers.append(h[0:h.find(match)])
        except Exception as e:
            new_headers.append(h)

    return new_headers


def get_wellID(wellid: str) -> Tuple[int, int]:

    # get the strings before the digits - A12: A
    rowid = re.split("\d+", wellid)[0]

    # get the columin index - A12: 12
    colindex = int(re.search("\d+", wellid).group()) - 1

    rowids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF']

    # translate the "letter" into an index
    rowindex = rowids.index(rowid)

    return rowindex, colindex


def check_separator(csvfile):

    reader = pd.read_csv(csvfile, sep=None, engine='python', iterator=True)
    sep = reader._engine.data.dialect.delimiter
    reader.close()

    return sep


def determine_plotgrid(num_parameter, columns=2):

    if np.mod(num_parameter, columns) == 0:
        plotrows = np.int(num_parameter / columns)
        empty = False
    if np.mod(num_parameter, columns) == 1:
        plotrows = np.int(num_parameter / columns) + 1
        empty = True

    plotgrid = [plotrows, columns]

    return plotgrid, empty


def showheatmap_all(heatmap_dict, subplots,
                    fontsize_title=16,
                    fontsize_label=12,
                    colormap='Blues',
                    linecolor='black',
                    linewidth=1.0,
                    save=False,
                    robust=True,
                    filename='Test.czi',
                    dpi=100,
                    deletelast=False):

    # create figure with subplots
    fig, axn = plt.subplots(subplots[0], subplots[1], figsize=(12, 10))
    # counter for the graphs to plot while iterating over all measured params
    plotid = -1

    # cycle heatmaps heatmaps
    for key in heatmap_dict.keys():  # python 3

        plotid = plotid + 1

        # get the desired heatmap from the dictionary containing all heatmaps
        heatmap_test = heatmap_dict[key]
        # create the actual heatmap
        ax = sns.heatmap(heatmap_test, ax=axn.flat[plotid],
                         cmap=colormap,
                         linecolor=linecolor,
                         linewidths=linewidth,
                         square=True,
                         robust=robust,
                         annot=False,
                         cbar_kws={"shrink": 1.0})

        # customize the plot to your needs
        ax.set_title(key, fontsize=fontsize_title)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_label)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_label)

    # delete last subplot for an uneven number of parameters
    if deletelast:
        axn[subplots[0] - 1, subplots[1] - 1].remove()

    # modify the layout so that titles do not overlap
    plt.tight_layout()

    if save:
        savename = filename[:-4] + '_HM_all.png'
        fig.savefig(savename, dpi=dpi, orientation='portrait', transparent=False, frameon=False)
        print('Heatmap image saved as: ', savename)
    else:
        savename = False

    return savename


def df_to_plotly(df: pd.DataFrame) -> Dict:
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


def create_heatmap_data(xyz: Dict,
                        unit: str = "[]",
                        colorscale: str = "Viridis",
                        showscale: bool = True) -> go.Heatmap:

    data = go.Heatmap(xyz,
                      xgap=3,
                      ygap=3,
                      colorbar=dict(title=dict(text=unit,
                                               side="right",
                                               font=dict(size=10,
                                                         color='Black')
                                               )
                                    ),
                      colorscale=colorscale,
                      showscale=showscale,
                      # hovertemplate = "Value:%{z:.2f}<extra></extra>"
                      hovertemplate="%{z:.2f}<extra></extra>",
                      )

    return data


def create_heatmap_plotly(heatmap_dataframe: pd.DataFrame,
                          parameter: str,
                          unit: str,
                          xaxis_template: Dict,
                          yaxis_template: Dict,
                          showscale: bool = True,
                          colorscale: str = "Viridis",
                          ) -> go.Figure:

    # create xyz dictionary required by plotly to create the heatmap
    xyz = df_to_plotly(heatmap_dataframe)
    data = create_heatmap_data(xyz,
                               colorscale=colorscale,
                               showscale=showscale,
                               unit=unit)
    fig = go.Figure(data)

    # use this template if nothing else was provided
    if xaxis_template == {}:
        xaxis_template = dict(constrain="domain",
                              side="top",
                              # autorange=False,
                              showgrid=False,
                              zeroline=False,
                              showticklabels=True,
                              # scaleanchor="x",
                              scaleratio=1,
                              )

    if yaxis_template == {}:
        yaxis_template = dict(constrain="domain",
                              side="left",
                              autorange="reversed",
                              showgrid=False,
                              zeroline=False,
                              showticklabels=True,
                              scaleanchor="x",
                              scaleratio=1,
                              )

    fig.update_layout(xaxis=xaxis_template,
                      yaxis=yaxis_template,
                      # showlegend=True,
                      autosize=True,
                      title=dict(text=parameter + " [" + unit + "]",
                                 x=0.5,
                                 font=dict(size=12,
                                           color='Black')
                                 )
                      )

    return fig
