from ome_zarr_utils import convert_czi_to_hcsplate
from plotting_utils import create_well_plate_heatmap
import ngff_zarr as nz
from czitools.read_tools import read_tools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from processing_tools import ArrayProcessor


# Main execution
if __name__ == "__main__":

    overwrite = True
    validate = True
    show_napari = False

    # # Point to the main data folder (two directories up from demo/omezarr_testing)
    # czi_filepath: str = str(Path(__file__).parent.parent.parent / "data" / "WP96_4Pos_B4-10_DAPI.czi")

    # # Read CZI file
    # array6d, mdata = read_tools.read_6darray(czi_filepath, use_xarray=True)
    # print(f"Array Type: {type(array6d)}, Shape: {array6d.shape}, Dtype: {array6d.dtype}")

    # zarr_output_path = convert_czi_to_hcsplate(czi_filepath, plate_name="Automated Plate", overwrite=overwrite)

    zarr_output_path = r"F:\Github\czitools\data\WP96_4Pos_B4-10_DAPI_ngff_plate.ome.zarr"

    if validate:
        print("Validating created HCS-ZARR file against schema...")
        hcs_plate = nz.from_hcs_zarr(zarr_output_path, validate=True)
        print("Validation successful.")

    # run some processing
    results_obj = {}
    results_mean = {}
    measure_properties = ("label", "area", "centroid", "bbox")

    # Debug: Print plate metadata information
    print(f"Number of wells in metadata: {len(hcs_plate.metadata.wells)}")
    print(f"Wells in metadata: {[w.path for w in hcs_plate.metadata.wells]}")

    # Iterate through all wells that actually have data
    # Use the well path directly since it's always correct (e.g., "B/4")
    for well_meta in hcs_plate.metadata.wells:

        # Extract row and column from the path (format: "B/4")
        row, col = well_meta.path.split("/")
        print(f"\nProcessing well: {well_meta.path} (Row: {row}, Column: {col})")

        # Get the well object for the current row/column position
        well = hcs_plate.get_well(row, col)

        # Only process if the well exists and has data
        if not well:
            print(f"  Warning: Well {well_meta.path} not found in plate, skipping")
            continue

        if not well.images or len(well.images) == 0:
            print(f"  Warning: Well {well_meta.path} has no images, skipping")
            continue

        # Store intensities for all fields (positions) within the well
        field_intensities = []
        field_num_objects = []

        print(f"  Found {len(well.images)} field(s) in well {well_meta.path}")

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
