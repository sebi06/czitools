from ome_zarr_utils import convert_czi_to_hcs_zarr
from plotting_utils import create_well_plate_heatmap
import ngff_zarr as nz
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from processing_tools import ArrayProcessor


# Main execution
if __name__ == "__main__":

    # Configuration parameters
    show_napari = False  # Whether to display the result in napari viewer

    # Point to the main data folder (two directories up from demo/omezarr_testing)
    filepath: str = str(Path(__file__).parent.parent.parent / "data" / "WP96_4Pos_B4-10_DAPI.czi")

    # Convert CZI file to HCS-ZARR format and get the output path
    zarr_output_path = convert_czi_to_hcs_zarr(filepath, overwrite=True)

    # Load the ZARR file as a plate object with metadata validation
    # This ensures the HCS (High Content Screening) metadata follows the specification
    plate = nz.from_hcs_zarr(zarr_output_path, validate=True)

    # Dictionary to store results: keys are well positions (e.g., "B/4"), values are mean intensities
    results_obj = {}
    results_mean = {}
    measure_properties = ("label", "area", "centroid", "bbox")

    # Iterate through all wells in the plate
    for well_meta in plate.metadata.wells:

        # Get row (e.g., "B") and column (e.g., "4") names from the well path
        # This is more robust than using indices, as the path is always correct
        row, col = well_meta.path.split("/")
        print(f"Processing well: {well_meta.path} (Row: {row}, Column: {col})")

        # Get the well object for the current row/column position
        well = plate.get_well(row, col)

        if well:
            # Store intensities for all fields (positions) within the well
            field_intensities = []
            field_num_objects = []

            # Process each field (microscope position) in the current well
            for field_idx in range(len(well.images)):

                image = well.get_image(field_idx)

                if image:

                    # Load the image data into memory (compute() for dask arrays)
                    data = image.images[0].data.compute()
                    print(
                        f"Processing well: {well_meta.path} - Field {field_idx} data shape: {data.shape}, dtype: {data.dtype}"
                    )

                    # count objects
                    ap = ArrayProcessor(np.squeeze(data))  # 2D data as input
                    pro2d = ap.apply_otsu_threshold()
                    ap = ArrayProcessor(pro2d)
                    pro2d, num_objects, props = ap.label_objects(
                        min_size=100,
                        label_rgb=False,
                        orig_image=None,
                        bg_label=0,
                        measure_params=True,
                        measure_properties=measure_properties,
                    )

                    # store number of objects for this field
                    field_num_objects.append(int(num_objects))

                    # Calculate mean intensity for this field
                    mean_intensity = np.mean(data)
                    field_intensities.append(mean_intensity)

            # Store the average intensity across all fields for this well
            results_mean[f"{row}/{col}"] = np.mean(field_intensities)

            # Store the total number of objects across all fields for this well
            results_obj[f"{row}/{col}"] = np.sum(field_num_objects)

    # Report the number of wells processed
    print(f"Total Size of results: {len(results_mean)}")
    print(f"Results: {results_mean}")
    print(f"Results: {results_obj}")

    # Create and display heatmap visualization using the dedicated function
    fig = create_well_plate_heatmap(
        results=results_obj,
        num_rows=8,  # Standard 96-well plate
        num_cols=12,  # Standard 96-well plate
        title="96-Well Plate Heatmap - Mean Intensity per Well",
        cmap="viridis",
        figsize=(12, 6),
    )
    plt.show()

    # Optional: Visualize the plate data using napari
    if show_napari:
        import napari

        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
