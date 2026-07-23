from czitools.analysis_tools.hcs_analysis import process_hcs_omezarr
from czitools.analysis_tools.plotting import create_well_plate_heatmap
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Main execution
if __name__ == "__main__":

    # adapt the path to your needs
    hcs_omezarr_path = r"F:\Testdata_Zeiss\OME_ZARR_Testfiles\testwell96-bin2_HCSplate_zarr3.ome.zarr"

    # Index of the channel to analyze
    channel2analyze = 0

    # define measurement properties to extract
    measure_properties = ("label", "area", "centroid", "bbox")

    results_obj = process_hcs_omezarr(
        hcs_omezarr_path=hcs_omezarr_path, channel2analyze=channel2analyze, measure_properties=measure_properties
    )

    # Create and display heatmap visualization using the dedicated function
    fig = create_well_plate_heatmap(
        results=results_obj,
        num_rows=8,  # Standard 96-well plate
        num_cols=12,  # Standard 96-well plate
        title="96-Well Plate Heatmap",
        parameter="Objects",
        cmap="viridis",
        figsize=(12, 6),
        fmt=".0f",
    )
    plt.show()
