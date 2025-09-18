from ome_zarr_utils import convert_czi_to_hcsplate
import ngff_zarr as nz
from czitools.read_tools import read_tools


# Main execution
if __name__ == "__main__":

    overwrite = True
    validate = True
    show_napari = False
    czi_filepath = r"data/WP96_4Pos_B4-10_DAPI.czi"

    # zarr_output_path = convert_czi_to_hcsplate(czi_filepath, plate_name="Automated Plate", overwrite=overwrite)

    # Read CZI file
    array6d, mdata = read_tools.read_6darray(czi_filepath, use_xarray=True)
    print(f"Array Type: {type(array6d)}, Shape: {array6d.shape}, Dtype: {array6d.dtype}")

    zarr_output_path = convert_czi_to_hcsplate(czi_filepath, plate_name="Automated Plate", overwrite=overwrite)

    if validate:
        print("Validating created HCS-ZARR file against schema...")
        hcs_plate = nz.from_hcs_zarr(zarr_output_path, validate=True)
        print("Validation successful.")

        # Optional: Visualize the plate data using napari
    if show_napari:
        import napari

        viewer = napari.Viewer()
        viewer.open(zarr_output_path, plugin="napari-ome-zarr")
        napari.run()
