from hcs_zarr_utils import convert_czi_to_hcs_zarr
import ngff_zarr as nz
import numpy as np

# Main execution
if __name__ == "__main__":

    # Configuration parameters
    show_napari = False  # Whether to display the result in napari viewer
    filepath = r"data/WP96_4Pos_B4-10_DAPI.czi"
    zarr_output_path = convert_czi_to_hcs_zarr(filepath, overwrite=True)

    # Validate HCS metadata during loading
    plate = nz.from_hcs_zarr(zarr_output_path, validate=True)

    results = {}

    for well_meta in plate.metadata.wells:
        print(f"Processing well: {well_meta.path}")
        row = plate.metadata.rows[well_meta.rowIndex].name
        col = plate.metadata.columns[well_meta.columnIndex].name

        well = plate.get_well(row, col)
        if well:
            # Analyze all fields in the well
            field_intensities = []
            for field_idx in range(len(well.images)):
                image = well.get_image(field_idx)
                if image:
                    data = image.images[0].data.compute()
                    print(f"Field {field_idx} data shape: {data.shape}, dtype: {data.dtype}")
                    mean_intensity = np.mean(data)
                    field_intensities.append(mean_intensity)

            results[f"{row}/{col}"] = np.mean(field_intensities)

    print(f"Total Size of results: {len(results)}")

    # Optional: Display the result in napari viewer
    if show_napari:

        import napari

        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
