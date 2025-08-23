import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from czitools.metadata_tools import czi_metadata as czimd
    from czitools.utils import misc

    return czimd, misc, mo


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
def _(czimd, filepath):
    # get only specific metadata
    czi_dimensions = czimd.CziDimensions(filepath)
    print(f"SizeS: {czi_dimensions.SizeS}")
    print(f"SizeT: {czi_dimensions.SizeT}")
    print(f"SizeZ: {czi_dimensions.SizeZ}")
    print(f"SizeC: {czi_dimensions.SizeC}")
    print(f"SizeY: {czi_dimensions.SizeY}")
    print(f"SizeX: {czi_dimensions.SizeX}")
    return


@app.cell
def _(czimd, filepath, misc, mo):
    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filepath)

    # get the CZI metadata dictionary directly from filename
    mdict = czimd.create_md_dict_red(mdata, sort=False, remove_none=True)

    # convert metadata dictionary to a pandas dataframe
    mdframe = misc.md2dataframe(mdict)

    mo.vstack([mo.ui.table(mdframe)])
    return


@app.cell
def _(filepath, misc, mo):
    # get the planetable for the CZI file
    pt, savepath = misc.get_planetable(filepath, norm_time=True, save_table=True, time=0, channel=0, zplane=0)

    mo.vstack([mo.ui.table(pt)])
    return


if __name__ == "__main__":
    app.run()
