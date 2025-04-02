from czitools.read_tools import read_tools
import napari
import xarray as xr
from napari.utils.colormaps import Colormap

# filepath = r"/datadisk1/testpictures/Testdata_Zeiss/CZI_Testfiles/WP96_2Pos_B2+B4_S=2_T=2_Z=4_C=3_X=512_Y=256.czi"
filepath = r"/datadisk1/testpictures/Testdata_Zeiss/CZI_Testfiles/CellDivision_T=10_Z=15_CH=2_DCV_small.czi"

# return an array with dimension order STCZYX(A)
array6d, mdata = read_tools.read_6darray(filepath, use_dask=True)

use_channel_axis = False
if array6d is None:
    print("Empty array6d. Nothing to display in Napari")
    exit()

# Define the dimension names and coordinates
dims = ("S", "T", "C", "Z", "Y", "X")
coords = {
    "S": range(array6d.shape[0]),
    "T": range(array6d.shape[1]),
    "C": range(array6d.shape[2]),
    "Z": range(array6d.shape[3]),
    "Y": range(array6d.shape[4]),
    "X": range(array6d.shape[5]),
}

# Create the xarray.DataArray
data_array = xr.DataArray(array6d, dims=dims, coords=coords)

# Set attributes for the DataArray
data_array.attrs = {
    "description": "6D image data from CZI file",
    "source": filepath,
    "axes": "STCZYX",
    "metadata": mdata,  # Include metadata if it's a dictionary or serializable
}

print(data_array.shape)
print(data_array.dims)

# show in napari
viewer = napari.Viewer()

if not use_channel_axis:

    for ch in range(0, data_array.sizes["C"]):
        print(f"Channel: {ch}")
        sub_array = data_array.sel(C=ch)

        # get the scaling factors for that channel
        scalefactors = [1.0] * len(sub_array.shape)
        scalefactors[sub_array.get_axis_num("Z")] = mdata.scale.ratio["zx_sf"] * 1.00001

        # get colors and channel name
        chname = mdata.channelinfo.names[ch]
        # inside the CZI metadata_tools colors are defined as ARGB hexstring
        rgb = "#" + mdata.channelinfo.colors[ch][3:]
        ncmap = Colormap(["#000000", rgb], name="cm_" + chname)

        # add the channel to the viewer
        viewer.add_image(
            sub_array,
            name=chname,
            colormap=ncmap,
            blending="additive",
            scale=scalefactors,
            gamma=0.85,
        )
        viewer.dims.axis_labels = sub_array.dims


if use_channel_axis:

    scalefactors = [1.0] * len(data_array.shape)
    scalefactors[data_array.get_axis_num("Z")] = mdata.scale.ratio["zx_sf"] * 1.00001
    scalefactors.pop(data_array.get_axis_num("C"))

    # get colors and channel name
    ncmaps = []

    for ch in range(0, data_array.sizes["C"]):
        # inside the CZI metadata_tools colors are defined as ARGB hexstring
        rgb = "#" + mdata.channelinfo.colors[ch][3:]
        ncmaps.append(
            Colormap(["#000000", rgb], name="cm_" + mdata.channelinfo.names[ch])
        )

    # Show the raw (not resampled) model data
    viewer.add_image(
        data_array,
        channel_axis=data_array.get_axis_num("C"),
        name=mdata.channelinfo.names,
        colormap=ncmaps,
        blending="additive",
        scale=scalefactors,
        gamma=0.85,
    )

    # Set the axis labels for the viewer
    dims = list(data_array.dims)
    dims.pop(data_array.get_axis_num("C"))

    viewer.dims.axis_labels = dims

# show the napari viewer
napari.run()
