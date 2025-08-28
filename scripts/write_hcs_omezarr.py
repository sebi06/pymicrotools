import napari
from hcs_zarr_utils import convert_czi_to_hcs_zarr


# Main execution
if __name__ == "__main__":

    # Configuration parameters
    show_napari = False  # Whether to display the result in napari viewer
    filepath = r"data/WP96_4Pos_B4-10_DAPI.czi"
    zarr_output_path = convert_czi_to_hcs_zarr(filepath, overwrite=True)

    # Optional: Display the result in napari viewer
    if show_napari:
        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
