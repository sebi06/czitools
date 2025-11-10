from ome_zarr_utils import convert_czi2hcs_omezarr, convert_czi2hcs_ngff, omezarr_package
import ngff_zarr as nz
from pathlib import Path
from czitools.read_tools import read_tools
from typing import Optional
from ome_zarr_utils import write_omezarr, write_omezarr_ngff


# Main execution
if __name__ == "__main__":

    # Configuration parameters
    show_napari = False  # Whether to display the result in napari viewer
    write_hcs = True
    ome_package = omezarr_package.OME_ZARR  # Choose between OME_ZARR and NGFF_ZARR
    scene_id: int = 0  # Scene ID will be only used for non HCS format

    # Point to the main data folder (two directories up from demo/omezarr_testing)
    # filepath: str = str(Path(__file__).parent.parent.parent / "data" / "WP96_4Pos_B4-10_DAPI.czi")
    filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi"

    if write_hcs:

        if ome_package == omezarr_package.OME_ZARR:
            # Convert CZI file to HCS-ZARR format and get the output path
            zarr_output_path = convert_czi2hcs_omezarr(filepath, overwrite=True)

        if ome_package == omezarr_package.NGFF_ZARR:
            # Convert CZI file to HCS-ZARR format and get the output path
            zarr_output_path = convert_czi2hcs_ngff(filepath, overwrite=True)

        print(f"Converted to OME-ZARR HCS format at: {zarr_output_path}")

        # Load the ZARR file as a plate object with metadata validation
        # This ensures the HCS (High Content Screening) metadata follows the specification
        print("Validating created HCS-ZARR file against schema...")
        plate = nz.from_hcs_zarr(zarr_output_path, validate=True)
        print("Validation successful.")

    if not write_hcs:

        # Read the CZI file and return a 6D array with dimension order STCZYX(A)
        array, mdata = read_tools.read_6darray(filepath, use_xarray=True)
        array = array[scene_id, ...]
        print(f"Array Type: {type(array)}, Shape: {array.shape}, Dtype: {array.dtype}")

        if ome_package == omezarr_package.OME_ZARR:

            zarr_path: Path = Path(str(filepath)[:-4] + ".ome.zarr")

            # Write OME-ZARR using utility function
            result_zarr_path: Optional[str] = write_omezarr(
                array, zarr_path=str(zarr_path), metadata=mdata, overwrite=True
            )
            print(f"Written OME-ZARR using ome-zarr-py: {result_zarr_path}")

        if omezarr_package == omezarr_package.NGFF_ZARR:

            zarr_path = Path(str(filepath)[:-4] + "_ngff.ome.zarr")

            # Write OME-ZARR using utility function
            ngff_image = write_omezarr_ngff(array, zarr_path, mdata, scale_factors=[2, 4], overwrite=True)
            print(f"Written OME-ZARR using ngff-zarr: {zarr_path}")

    # Optional: Visualize the plate data using napari
    if show_napari:
        import napari

        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
