import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import logging
    from plotting_utils import create_well_plate_heatmap
    import ngff_zarr as nz
    import numpy as np
    import matplotlib.pyplot as plt
    from processing_tools import ArrayProcessor
    import pandas as pd
    from typing import Dict
    try:
        import altair as alt
    except ImportError as e:
        print(f"Altair package not found: {e}")

    return (
        ArrayProcessor,
        Dict,
        alt,
        create_well_plate_heatmap,
        logging,
        mo,
        np,
        nz,
        pd,
    )


@app.cell
def _(logging):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return (logger,)


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(multiple=False, restrict_navigation=False, selection_mode="directory", initial_path=None)

    # Display the file browser
    mo.vstack([file_browser])
    return (file_browser,)


@app.cell
def _(file_browser):
    hcs_omezarr_path = str(file_browser.path(0))
    print(f"OME-ZARR Path: {hcs_omezarr_path}")
    return (hcs_omezarr_path,)


@app.cell
def _(ArrayProcessor, hcs_omezarr_path, logger, np, nz):
    channel2analyze = 0  # Index of the channel to analyze

    try:
        logger.info("Validating created HCS-ZARR file against schema...")
        hcs_plate = nz.from_hcs_zarr(hcs_omezarr_path, validate=True)
        logger.info("Validation successful.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise e

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
                logger.info(
                    f"Processing well: {well_meta.path} - Field {field_idx} data shape: {data.shape}, dtype: {data.dtype}"
                )

                # count objects
                ap = ArrayProcessor(np.squeeze(data[:, channel2analyze, ...]))  # 2D data as input
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
    logger.info(f"Total Size of results: {len(results_mean)}")
    return (results_obj,)


@app.cell
def _(Dict, alt, pd):
    def create_well_plate_altair_heatmap(
        results: Dict[str, float],
        num_rows: int = 8,
        num_cols: int = 12,
        title: str = "Well Plate Heatmap",
        parameter: str = "Objects",
        cmap_range: str = "plasma",
        annot: bool = True,
    ) -> alt.Chart:
        """
        Create an Altair heatmap visualization of well plate data with Marimo reactivity.
        (Fixes: Annotations use conditional color for maximum visibility.)
        """
    
        # 1. Prepare Data into "Long" Format
        data_list = []
        for well_key, value in results.items():
            row_name, col_name = well_key.split("/")
            data_list.append({
                'Row': row_name,
                'Column': int(col_name),
                parameter: value
            })
        
        df = pd.DataFrame(data_list)
        df['Column'] = df['Column'].astype(str)
    
        # Define row and column order
        rows_labels = [chr(65 + i) for i in range(num_rows)]
        cols_labels = [str(i) for i in range(1, num_cols + 1)]

        # Calculate median for dynamic text color contrast
        median_value = df[parameter].median()

        # 2. Define the Marimo Selection Parameter (CRITICAL FOR REACTIVITY)
        marimo_selection = alt.selection_point(
            name="marimo_selection", 
            fields=['Row', 'Column'], 
            empty=True
        )

        # 3. Create the Base Chart
        base_chart = alt.Chart(df, title=title).encode(
            x=alt.X('Column:O', sort=cols_labels, axis=alt.Axis(title="Column")),
            y=alt.Y('Row:O', sort=rows_labels, axis=alt.Axis(title="Row")),
            color=alt.Color(
                f'{parameter}:Q',
                title=parameter,
                scale=alt.Scale(range=cmap_range)
            ),
            tooltip=['Row', 'Column', alt.Tooltip(f'{parameter}:Q', title=parameter, format='.2f')]
        ).add_params(
            marimo_selection
        )

        # 4. Add the Rectangle Layer (The Cells)
        rect_layer = base_chart.mark_rect().encode(
            # Visual highlighting logic
            stroke=alt.condition(
                marimo_selection,  # If selected...
                alt.value('red'),  # Use red stroke
                alt.value('gray')   # Otherwise, use gray stroke
            ),
            strokeWidth=alt.condition(
                marimo_selection,  # If selected...
                alt.value(2),      # Use thicker border
                alt.value(0.5)     # Otherwise, use thin border
            ),
            # Dim unselected cells slightly
            opacity=alt.condition(marimo_selection, alt.value(1.0), alt.value(0.8))
        )
    
        # 5. Add Annotation Layer (if requested)
        if annot:
            text_layer = base_chart.mark_text(
                baseline='middle',
                fontSize=10,
            ).encode(
                text=alt.Text(f'{parameter}:Q', format='.0f'),
                # NEW FIX: Conditional coloring for text contrast
                color=alt.condition(
                    alt.datum[parameter] > median_value,
                    alt.value('black'),  # For bright cells (high values), use black text
                    alt.value('white')   # For dark cells (low values), use white text
                )
            )
        
            final_chart = rect_layer + text_layer
        else:
            final_chart = rect_layer

        # Final styling and configuration
        return final_chart.configure_view(
            stroke=None 
        ).configure_title(
            fontSize=14,
            anchor='start'
        )
    return (create_well_plate_altair_heatmap,)


@app.cell
def _(create_well_plate_altair_heatmap, mo, results_obj):
    chart = create_well_plate_altair_heatmap(
        results=results_obj,
        parameter="Object Count",
        title="96-Well Plate Analysis",
        cmap_range="heatmap"
    )

    # 2. Reactive UI Definition (From original Cell 1)
    well_selector = mo.ui.altair_chart(
        chart,
    )

    # We define well_selector as the last expression to ensure its variable 
    # is exported for other cells to use.
    well_selector
    return (well_selector,)


@app.cell
def _(mo, well_selector):
    # 3. Value Consumption (From original Cell 2)
    # Marimo will execute this block every time 'well_selector' changes.
    # Note: well_selector is defined in the previous cell (Marimo Cell 1).
    selected_well = well_selector.value

    # 4. Conditional Output Logic
    output_md = mo.md("""
        ### 🧪 Well Plate Detail Panel
        Click on any well in the heatmap above to see its details.
    """)

    if not selected_well.empty:
        # Extract the data from the selected row (always index 0 for "point" selection)
        well_name = f"{selected_well['Row'].iloc[0]}/{selected_well['Column'].iloc[0]}"
        # Ensure 'Object Count' is accessed correctly
        value = selected_well.iloc[0]["Object Count"]
    
        output_md = mo.md(f"""
            ### 🧪 Selected Well: **{well_name}**
            The **Object Count** for this well is: **{value:.0f}**
        """)

    # 5. Final Output (Display both the chart and detail panel side-by-side)
    # mo.hstack REQUIRES a single list of elements. We use mo.vstack to group 
    # the title and content for each column before stacking them horizontally.
    mo.hstack([
        # LEFT SIDE: Heatmap container (Title + Chart UI)
        mo.vstack([
            mo.md("## Well Plate Heatmap"),
            well_selector
        ]),
        # RIGHT SIDE: Details Panel container (Title + Detail Output)
        mo.vstack([
            mo.md("## Selected Well Details"),
            output_md
        ])
    ])
    return


@app.cell
def _(create_well_plate_heatmap, results_obj):
    # Create heatmap visualization
    fig = create_well_plate_heatmap(
        results=results_obj,
        num_rows=8,
        num_cols=12,
        title="96-Well Plate Heatmap",
        parameter="Objects",
        cmap="viridis",
        figsize=(12, 6),
        fmt=".0f",
    )

    # This will display the plot in the cell's output
    fig
    return


if __name__ == "__main__":
    app.run()
