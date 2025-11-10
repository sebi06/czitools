from ome_zarr_utils import convert_czi2hcs_omezarr, convert_czi2hcs_ngff, omezarr_package
import ngff_zarr as nz
from pathlib import Path


# Main execution
if __name__ == "__main__":

    # Configuration parameters
    show_napari = True  # Whether to display the result in napari viewer
    ome_package = omezarr_package.OME_ZARR  # Choose between OME_ZARR and NGFF_ZARR

    # Point to the main data folder (two directories up from demo/omezarr_testing)
    filepath: str = str(Path(__file__).parent.parent.parent / "data" / "WP96_4Pos_B4-10_DAPI.czi")
    filepath = r"F:\Testdata_Zeiss\CD7\10X_HeLa_Kyto_Obl\HeLa_Kyoto_Obl_B2+B3_5xwell_T=31.czi"

    if ome_package == omezarr_package.OME_ZARR:
        # Convert CZI file to HCS-ZARR format and get the output path
        zarr_output_path = convert_czi2hcs_omezarr(filepath, overwrite=True)

    if ome_package == omezarr_package.NGFF_ZARR:
        # Convert CZI file to HCS-ZARR format and get the output path
        zarr_output_path = convert_czi2hcs_ngff(filepath, overwrite=True)

    # Load the ZARR file as a plate object with metadata validation
    # This ensures the HCS (High Content Screening) metadata follows the specification
    print("Validating created HCS-ZARR file against schema...")
    plate = nz.from_hcs_zarr(zarr_output_path, validate=True)
    print("Validation successful.")

    # Optional: Visualize the plate data using napari
    if show_napari:
        import napari

        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
