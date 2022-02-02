# -*- coding: utf-8 -*-

#################################################################
# File        : test_zenheatmap.py
# Version     : 0.1
# Author      : sebi06
# Date        : 04.12.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import pandas as pd
from czitools import visutools as vt
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from itertools import product

filename = r"../../experimental/Full Plate 20x_heatmap.csv"
#filename = "fixed endpoint 3C 25 384well_Entire Carrier.csv"

# get the separator and use it to read the CSV data table
separator = vt.check_separator(filename)
print('Separator used:', separator)

# read the CSV table containing all the single object data for
df = pd.read_csv(filename, sep=separator)

# get the size of the wellplate
platetype = df.shape[0] - 1
print("Type of WellPlate: ", platetype)

num_rows, num_cols = vt.getrowandcolumn(platetype=platetype)
print("Number Rows - Cols:", num_rows, num_cols)

# correct decimal separator of the CSV table
df = df.apply(lambda x: x.astype(str).str.replace(',', '.'))

# replace nan inside units with empty string
df = df.applymap(lambda x: np.nan if x == " " else x)

# get headers and make names readable
new_headers = vt.clean_zenia_headernames(df, match="::")

# rename columns with "cleaned" header names
df = vt.rename_columns(df, new_headers, verbose=False)

# check for the existence of well category
if "ImageSceneCategoryName" in new_headers:
    category_exist = True
else:
    category_exist = False
    df.insert(1, "ImageSceneCategoryName", [" "] * (platetype + 1))

# get the updated headers, units, number of parameters etc.
headers = list(df.columns.values)

# get a list of the used "units" for the measured parameters
units = list(df.loc[0, :])[2:]

# replace empty string with NaN
units = list(map(lambda x: "" if x == "nan" else x, units))


numparams = len(headers) - 2
params = headers[2:]

# use meaningful categories for wells
for w in range(platetype):
    ch = df.iloc[w + 1, 1]
    if ch == " ":
        df.iloc[w + 1, 1] = None
    if pd.isnull(ch):
        df.iloc[w + 1, 1] = "default"

# get well categories
well_categories = {}
for w in range(platetype):
    wellid = df.iloc[w+1, 0]
    category = df.iloc[w+1, 1]
    well_categories[wellid] = df.iloc[w+1, 1]
    
# create heatmap dictionary with empty array based on the platetype
heatmap_dict = {}
for p in range(0, numparams):
    heatmap_dict[params[p]] = np.full([num_rows, num_cols], np.nan)

# loop over all measured parameters
for p in params:
    for i in range(platetype):

        # read values
        well = df.iloc[i+1, 0]

        # get row and col index from well string
        rowindex, colindex = vt.get_wellID(well)

        # read the actual value measure for a specific well
        value = df[p][i+1]

        # store the value inside the correct heatmap
        heatmap_dict[p][rowindex, colindex] = value
    
    # convert numpy array into pd.dataframe with labels
    heatmap_dict[p] = vt.convert_array_to_heatmap(heatmap_dict[p],
                                                  num_rows,
                                                  num_cols)


p2d = 1

# created a single heatmap using matplotlib
savename = vt.showheatmap(heatmap_dict[params[p2d]], params[p2d],
                          fontsize_title=14,
                          fontsize_label=12,
                          colormap='Blues',
                          linecolor='black',
                          linewidth=1.0,
                          save=True,
                          robust=True
                          )

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

fig = vt.create_heatmap_plotly(heatmap_dict[params[p2d]],
                               params[p2d],
                               units[p2d],
                               xaxis_template=xaxis_template,
                               yaxis_template=yaxis_template,
                               showscale=True,
                               colorscale="Viridis")

# display the figure
fig.show()

# save the figure
fig.write_html("test.html")

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

