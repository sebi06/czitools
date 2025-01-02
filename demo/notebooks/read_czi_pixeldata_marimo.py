import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from czitools.metadata_tools import czi_metadata as czimd
    from czitools.utils import misc
    from czitools.read_tools import read_tools as czird
    import dask.array as da
    from pathlib import Path
    import os
    import glob
    import stackview
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    return (
        Path,
        cm,
        czimd,
        czird,
        da,
        glob,
        misc,
        mo,
        os,
        pd,
        plt,
        stackview,
    )


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(multiple=False)

    # Display the file browser
    mo.vstack([file_browser])
    return (file_browser,)


@app.cell
def _(file_browser):
    filepath = str(file_browser.path(0))
    print(f"Filepath: {filepath}")
    return (filepath,)


@app.cell
def _(czird, da, filepath):
    # return array with dimension order STCZYX(A)
    array6d, mdata = czird.read_6darray(filepath, use_dask=False, chunk_zyx=True)

    # show dask array structure
    if isinstance(array6d, da.Array):
        print(array6d)
    else:
        print("Shape:", array6d.shape, "dtype:", array6d.dtype)

    # Get array dimensions
    dims = array6d.shape[:-2]
    dims_names = ["S", "T", "C", "Z"]
    return array6d, dims, dims_names, mdata


@app.cell
def _(dims, mo):
    scene = mo.ui.slider(start=0, 
            stop=dims[0] - 1, 
            step=1, 
            label=f"scene [0 - {dims[0]-1}]",
            show_value=True
        )

    time = mo.ui.slider(start=0, 
            stop=dims[1] - 1, 
            step=1,  
            label=f"time [0 - {dims[1]-1}]",
            show_value=True
        )

    channel = mo.ui.slider(start=0, 
            stop=dims[2] - 1, 
            step=1, 
            label=f"channel [0 - {dims[2]-1}]",
            show_value=True
        )

    zplane = mo.ui.slider(start=0, 
            stop=dims[3] - 1, 
            step=1, 
            label=f"zplane [0 - {dims[3]-1}]",
            show_value=True
        )

    mo.vstack([scene, time, channel, zplane])
    return channel, scene, time, zplane


@app.cell
def _(channel, scene, show_2dplane, time, zplane):
    show_2dplane(scene.value, time.value, channel.value, zplane.value)
    return


@app.cell
def _(array6d, cm, plt):
    def show_2dplane(s, t, c, z):
        plt.figure(figsize=(8, 8))
        plt.imshow(array6d[s, t, c, z, ...], cmap=cm.gray)
        plt.tight_layout()
        return plt.gca()
    return (show_2dplane,)


if __name__ == "__main__":
    app.run()
