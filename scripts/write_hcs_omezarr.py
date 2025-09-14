from hcs_zarr_utils import convert_czi_to_hcs_zarr
import ngff_zarr as nz
import numpy as np

# Main execution
if __name__ == "__main__":

    # Configuration parameters
    show_napari = False  # Whether to display the result in napari viewer

    # filepath to original CZI file
    filepath = r"data/WP96_4Pos_B4-10_DAPI.czi"
    # filepath = r"/home/sebi06/github/pymicrotools/data/WP96_4Pos_B4-10_DAPI.czi"

    # Convert CZI file to HCS-ZARR format and get the output path
    zarr_output_path = convert_czi_to_hcs_zarr(filepath, overwrite=True)

    # Load the ZARR file as a plate object with metadata validation
    # This ensures the HCS (High Content Screening) metadata follows the specification
    plate = nz.from_hcs_zarr(zarr_output_path, validate=True)

    # Dictionary to store results: keys are well positions (e.g., "B/4"), values are mean intensities
    results = {}

    # Iterate through all wells in the plate
    for well_meta in plate.metadata.wells:

        # Get row (e.g., "B") and column (e.g., "4") names for the current well
        row = plate.metadata.rows[well_meta.rowIndex].name
        col = plate.metadata.columns[well_meta.columnIndex].name

        # Get the well object for the current row/column position
        well = plate.get_well(row, col)
        if well:
            # Store intensities for all fields (positions) within the well
            field_intensities = []

            # Process each field (microscope position) in the current well
            for field_idx in range(len(well.images)):
                image = well.get_image(field_idx)
                if image:
                    # Load the image data into memory (compute() for dask arrays)
                    data = image.images[0].data.compute()
                    print(
                        f"Processing well: {well_meta.path} - Field {field_idx} data shape: {data.shape}, dtype: {data.dtype}"
                    )

                    # Calculate mean intensity for this field
                    mean_intensity = np.mean(data)
                    field_intensities.append(mean_intensity)

            # Store the average intensity across all fields for this well
            results[f"{row}/{col}"] = np.mean(field_intensities)

    # Report the number of wells processed
    print(f"Total Size of results: {len(results)}")

    # Optional: Visualize the plate data using napari
    if show_napari:
        import napari

        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
