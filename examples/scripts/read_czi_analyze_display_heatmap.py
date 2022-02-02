# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_analyze_display_heatmap.py
# Version     : 0.3
# Author      : sebi06
# Date        : 15.12.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################


# required imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from czitools import segmentation_tools as sgt
from czitools import visutools as vt
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
from tqdm.contrib.itertools import product
from IPython.display import display, HTML
from MightyMosaic import MightyMosaic
from aicspylibczi import CziFile
from czitools import czi_metadata as czimd_aics
from czitools import pylibczirw_metadata as czimd
from czitools import czi_read as czird
from pylibCZIrw import czi as pyczi
from czitools import misc, napari_tools
from IPython.display import display, HTML
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Type, Any, Union


# specify the filename of the CZI file
filename = os.path.abspath("../../testdata/WP96_4Pos_B4-10_DAPI.czi")
print(filename)

# define platetype and get number of rows and columns
platetype = 96
nr, nc = vt.getrowandcolumn(platetype=platetype)
print(nr, nc)

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# get the CZI metadata dictionary directly from filename
mdict = czimd.create_mdict_complete(filename, sort=False)

# convert metadata dictionary to a pandas dataframe
mdframe = misc.md2dataframe(mdict)

# and display it nicely as a HTML inside the jupyter notebook
display(HTML(mdframe.to_html()))


### Define General Options ###

# define channel and size filter for objects
chindex = 0  # channel containing the objects, e.g. the nuclei in DAPI channel
minsize = 100  # minimum object size [pixel]
maxsize = 500  # maximum object size [pixel]

# optional dipslay of "some" results - empty list = no display
show_image = [0]

# toggle additional printed output
verbose = False

# set number of Scenes for testing
#SizeS = 1


# Define Segmentation Method - Uncomment as needed

######### Scikit-Image #############

# Use classical processing function from scikit-image
use_method = 'scikit'
# Define threshold parameters
#filtermethod = 'median'
filtermethod = None
filtersize = 3
threshold = 'triangle'

# use watershed for splitting after segmentation- 'ws' or 'ws_adv'
use_ws = True
#ws_method = 'ws_adv'
ws_method = 'ws'
filtersize_ws = 3
min_distance = 5
radius_dilation = 1

# define columns names for dataframe for the measure objects
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)

# set image counter to zero and create empty dataframe
image_counter = 0
results = pd.DataFrame()

# check if dimensions are None (because the do not exist for that image)
sizeC = misc.check_dimsize(mdata.dims.SizeC, set2value=1)
sizeZ = misc.check_dimsize(mdata.dims.SizeZ, set2value=1)
sizeT = misc.check_dimsize(mdata.dims.SizeT, set2value=1)
sizeS = misc.check_dimsize(mdata.dims.SizeS, set2value=1)

# define measure region properties
to_measure = ('label',
              'area',
              'centroid',
              'max_intensity',
              'mean_intensity',
              'min_intensity',
              'bbox'
              )

units = ["micron**2", "pixel", "pixel", "cts", "counts", "cts",]

# open the original CZI document to read 2D image planes
with pyczi.open_czi(filename) as czidoc_r:

    # read array for the scene
    for s, t, z, c in product(range(sizeS),
                              range(sizeT),
                              range(sizeZ),
                              range(sizeC)):

        # get the current plane indices and store them
        values = {'S': s, 'T': t, 'Z': z, 'C': chindex, 'Number': 0}
        #print('Analyzing S-T-Z-C: ', s, t, z, chindex)

        # read 2D plane in case there are (no) scenes
        if mdata.dims.SizeS is None:
            image2d = czidoc_r.read(plane={'T': t, 'Z': z, 'C': c})
        else:
            image2d = czidoc_r.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)
        
        # make sure to remove the A dim
        image2d = image2d[..., 0]

        # use scikit-image tools to segment the nuclei
        if use_method == 'scikit':
            mask = sgt.segment_threshold(image2d,
                                         filtermethod=filtermethod,
                                         filtersize=filtersize,
                                         threshold=threshold,
                                         split_ws=use_ws,
                                         min_distance=min_distance,
                                         ws_method=ws_method,
                                         radius=radius_dilation)

        # clear the border by removing "touching" objects
        mask = segmentation.clear_border(mask)

        # measure the specified parameters store in dataframe
        props = pd.DataFrame(
            measure.regionprops_table(
                mask,
                intensity_image=image2d,
                properties=to_measure
            )
        ).set_index('label')

        # filter objects by size
        props = props[(props['area'] >= minsize) & (props['area'] <= maxsize)]

        # add well information for CZI metadata
        props['WellID'] = mdict['WellArrayNames'][s]
        props['WellColumnID'] = mdict['WellColumnID'][s]
        props['WellRowID'] = mdict['WellRowID'][s]

        # add plane indices
        props['S'] = s
        props['T'] = t
        props['Z'] = z
        props['C'] = chindex

        # count the number of objects
        values['Number'] = props.shape[0]

        if verbose:
            print('Well:', props['WellID'].iloc[0], 'Objects: ', values['Number'])

        # update dataframe containing the number of objects
        objects = objects.append(pd.DataFrame(values, index=[0]), ignore_index=True)
        results = results.append(props, ignore_index=True)

        image_counter += 1
        # optional display of results
        if image_counter - 1 in show_image:
            print('Well:', props['WellID'].iloc[0],
                  'Index S-T-Z-C:', s, t, z, chindex,
                  'Objects:', values['Number'])

            ax = vt.plot_segresults(image2d, mask, props, add_bbox=True)

# reorder dataframe with single objects
new_order = list(results.columns[-7:]) + list(results.columns[:-7])
results = results.reindex(columns=new_order)

# Filter and Inspect Data

# get a specific scene from a well and a specific scene
well2show = results[(results['WellID'] == 'B4') & (results['S'] == 0)]

# create heatmap array with NaNs
heatmap_numobj = vt.create_heatmap(platetype=platetype)
heatmap_param = vt.create_heatmap(platetype=platetype)

# get the updated headers, units, number of parameters etc.
headers = list(results.columns.values)
params = headers[7:13]
numparams = len(params)
print(params)

# create heatmap dictionary with empty array based on the plate type
heatmap_dict = {}
for p in range(0, numparams):
    heatmap_dict[params[p]] = np.full([nr, nc], np.nan)


# loop over all measured parameters
for p in params:
    for well in mdict['WellCounter']:

        stats, row, col = vt.extract_wellstats(results,
                                               well=well,
                                               wellstr="WellID",
                                               wellcolstr="WellColumnID",
                                               wellrowstr="WellRowID" )

        # add value for specifics params to heatmap - e.g. "area"
        heatmap_param[row - 1, col - 1] = stats[p]['mean']

        # add value for number of objects to heatmap_numobj - e.g. "count"
        heatmap_numobj[row - 1, col - 1] = stats["WellID"]["count"]

    # store heatmap in dict for all the parameter heatmaps
    heatmap_dict[p] = vt.convert_array_to_heatmap(heatmap_param, nr, nc)

# store heatmap with object number inside a separate dict
heatmap_obj = vt.convert_array_to_heatmap(heatmap_numobj, nr, nc)


# show the heatmap for a single parameter
savename_single = vt.showheatmap(heatmap_obj, "Objects - Number",
                                 fontsize_title=16,
                                 fontsize_label=14,
                                 colormap="cividis_r",
                                 linecolor='black',
                                 linewidth=3.0,
                                 save=True,
                                 filename=filename,
                                 dpi=100,
                                 show=True)

# show the heatmap for a single parameter
savename_single = vt.showheatmap(heatmap_dict["area"], "Objects - Mean Area",
                                 fontsize_title=16,
                                 fontsize_label=14,
                                 colormap='cividis_r',
                                 linecolor='black',
                                 linewidth=3.0,
                                 save=True,
                                 filename=filename,
                                 dpi=100,
                                 show=True)


# create the figure of a single heatmap plot using plotly
xaxis_template = dict(constrain="domain",
                      side="top",
                      #autorange=False,
                      showgrid=False,
                      zeroline=False,
                      showticklabels=True,
                      #scaleanchor="x",
                      scaleratio=1,
                      )

yaxis_template = dict(constrain="domain",
                      side="left",
                      autorange="reversed",
                      showgrid=False,
                      zeroline=False,
                      showticklabels=True,
                      scaleanchor="x",
                      scaleratio=1,
                      )

fig = vt.create_heatmap_plotly(heatmap_obj,
                               "Objects - Number",
                               "[]",
                               xaxis_template=xaxis_template,
                               yaxis_template=yaxis_template,
                               showscale=True,
                               colorscale="Viridis")

# display the figure
fig.show()
# save the figure
fig.write_html("test.html")


####################################################################

# determine the require plot grid
plotgrid, deletelast = vt.determine_plotgrid(numparams, columns=2)

savename = vt.showheatmap_all(heatmap_dict, plotgrid,
                              fontsize_title=14,
                              fontsize_label=12,
                              colormap='Blues',
                              linecolor='black',
                              linewidth=1.0,
                              save=False,
                              robust=True,
                              deletelast=deletelast
                              )

plt.show()


# create titles for subplots
subtitles = []
for st, un in zip(params, units):
    subtitles.append(st + " [" + un + "]")


# subplots with plotly
fig2 = make_subplots(rows=plotgrid[0],
                     cols=plotgrid[1],
                     shared_yaxes=False,
                     shared_xaxes=False,
                     subplot_titles=subtitles,
                     horizontal_spacing=0.01,
                     vertical_spacing=0.1,
                     )
# create zero-based counter for subplots
plotid = -1

# cycle heatmaps heatmaps
for r, c in product(range(plotgrid[0]), range(plotgrid[1])):
    plotid = plotid + 1

    # check if is a parameter, e.g. a 2x3 grid but only 5 parameters
    if plotid < numparams:

        # get the desired heatmap from the dictionary containing all heatmaps
        heatmap_test = heatmap_dict[params[plotid]]

        # create dictionary with XYZ data for the heatmap
        xyz = vt.df_to_plotly(heatmap_test)

        # create the data for the individual heatmaps
        data = vt.create_heatmap_data(xyz,
                                      colorscale="Viridis",
                                      showscale=False,
                                      unit=units[plotid])

        # add the individual subplots
        fig2.add_trace(data, row=r+1, col=c+1)


# style x and y axis for all subplots by iterating over them
for i in range(len(params)):

    fig2.layout[f'yaxis{i + 1}'].update(dict(showgrid=False,
                                             #side="top",
                                             constrain="domain",
                                             autorange='reversed',
                                             scaleanchor="x",
                                             scaleratio=1
                                             )
                                        )

    fig2.layout[f'xaxis{i + 1}'].update(dict(showgrid=False,
                                             #side="bottom",
                                             constrain="domain",
                                             #autorange='reversed',
                                             scaleanchor="x",
                                             scaleratio=1
                                             )
                                        )

# display the figure
fig2.show()

# save the figure
fig2.write_html("test2.html")


