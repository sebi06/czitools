from czitools.read_tools import read_tools
from pathlib import Path
from ome_zarr_utils import write_omezarr_ngff


# Point to the main data folder (two directories up from demo/omezarr_testing)
filepath1: str = str(Path(__file__).parent.parent.parent / "data" / "CellDivision_T10_Z15_CH2_DCV_small.czi")
filepath2: str = str(Path(__file__).parent.parent.parent / "data" / "WP96_4Pos_B4-10_DAPI.czi")
filepaths = [Path(filepath1), Path(filepath2)]

show_napari = False  # Whether to display the result in napari viewer

for filepath in filepaths:

    try:
        # return a 6D array with dimension order STCZYX(A)
        array, mdata = read_tools.read_6darray(filepath, use_xarray=True)
        # use 5D subset for NGFF
        array = array[0, ...]
        print(f"Array Type: {type(array)}, Shape: {array.shape}, Dtype: {array.dtype}")

        zarr_path = Path(str(filepath)[:-4] + "_ngff.ome.zarr")

        ngff_image = write_omezarr_ngff(array, zarr_path, mdata, scale_factors=[2, 4], overwrite=True)
        print(f"NGFF Image: {ngff_image}")
        print(f"Written OME-ZARR using ngff-zarr: {zarr_path}")

        if show_napari:
            import napari

            viewer = napari.Viewer()
            viewer.open(zarr_path, plugin="napari-ome-zarr")
            napari.run()

    except KeyError as e:
        print(f"Could not convert: {filepath} KeyError: {e}")

print("Done")
