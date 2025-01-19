import marimo

__generated_with = "0.10.14"
app = marimo.App(auto_download=["ipynb"])


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
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap
    return (
        LinearSegmentedColormap,
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
def _(LinearSegmentedColormap, czird, da, filepath):
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """
        Convert a hexadecimal color string to an RGB tuple.
        Args:
            hex_color (str): A string representing a color in hexadecimal format (e.g., "#RRGGBB").
        Returns:
            tuple[int, int, int]: A tuple containing the RGB values as floats in the range [0, 1].
        """

        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

        return rgb

    # get 6d array with dimension order STCZYX(A)
    array6d, mdata = czird.read_6darray(filepath, use_dask=False, chunk_zyx=True)

    # show dask array structure
    if isinstance(array6d, da.Array):
        print(array6d)
    else:
        print("Shape:", array6d.shape, "dtype:", array6d.dtype)

    # get array dimensions
    dims = array6d.shape[:-2]
    dims_names = ["S", "T", "C", "Z"]

    cmaps = []

    for ch in range(mdata.image.SizeC):
        chname = mdata.channelinfo.names[ch]
        rgb = hex_to_rgb(mdata.channelinfo.colors[ch][3:])
        cmaps.append(LinearSegmentedColormap.from_list(chname, [(0, 0, 0), rgb]))
    return (
        array6d,
        ch,
        chname,
        cmaps,
        dims,
        dims_names,
        hex_to_rgb,
        mdata,
        rgb,
    )


@app.cell
def _(czimd, mdata, misc, mo):
    # get the CZI metadata dictionary directly from filename
    mdict = czimd.create_md_dict_red(mdata, sort=False, remove_none=True)

    # convert metadata dictionary to a pandas dataframe
    mdframe = misc.md2dataframe(mdict)

    mo.vstack([mo.ui.table(mdframe)])
    return mdframe, mdict


@app.cell
def _(dims, mo):
    scene = mo.ui.slider(
        start=0,
        stop=dims[0] - 1,
        step=1,
        label=f"scene [0 - {dims[0]-1}]",
        show_value=True,
    )

    time = mo.ui.slider(
        start=0,
        stop=dims[1] - 1,
        step=1,
        label=f"time [0 - {dims[1]-1}]",
        show_value=True,
    )

    channel = mo.ui.slider(
        start=0,
        stop=dims[2] - 1,
        step=1,
        label=f"channel [0 - {dims[2]-1}]",
        show_value=True,
    )

    zplane = mo.ui.slider(
        start=0,
        stop=dims[3] - 1,
        step=1,
        label=f"zplane [0 - {dims[3]-1}]",
        show_value=True,
    )

    mo.vstack([scene, time, channel, zplane])
    return channel, scene, time, zplane


@app.cell
def _(channel, scene, show_2dplane, time, zplane):
    show_2dplane(scene.value, time.value, channel.value, zplane.value)
    return


@app.cell
def _(array6d, cmaps, plt):
    def show_2dplane(s, t, c, z):
        plt.figure(figsize=(8, 8))
        plt.imshow(array6d[s, t, c, z, ...], cmap=cmaps[c], vmin=None, vmax=None)
        plt.tight_layout()
        return plt.gca()
    return (show_2dplane,)


if __name__ == "__main__":
    app.run()
