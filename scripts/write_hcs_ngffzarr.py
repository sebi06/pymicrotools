from hcs_zarr_utils import extract_well_coordinates
from pathlib import Path
import ngff_zarr as nz
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
import shutil
from czitools.read_tools import read_tools
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell

# Main execution
if __name__ == "__main__":

    overwrite = True
    validate = True
    show_napari = False

    # Configuration parameters
    show_napari = False  # Whether to display the result in napari viewer
    czi_filepath = r"data/WP96_4Pos_B4-10_DAPI.czi"

    # Read CZI file
    array6d, mdata = read_tools.read_6darray(czi_filepath, use_xarray=False)
    print(f"Array Type: {type(array6d)}, Shape: {array6d.shape}, Dtype: {array6d.dtype}")

    # Define output path
    zarr_output_path = Path(czi_filepath[:-4] + "_ngff_plate.zarr")

    # Handle existing files
    if zarr_output_path.exists():
        if overwrite:
            shutil.rmtree(zarr_output_path)
        else:
            print(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            exit()

    # Extract plate layout
    row_names, col_names, well_paths = extract_well_coordinates(mdata.sample.well_counter)
    field_paths = [str(i) for i in range(mdata.sample.well_counter[mdata.sample.well_array_names[0]])]

    # Create plate layout
    columns = [
        PlateColumn(name="4"),
        PlateColumn(name="5"),
        PlateColumn(name="6"),
        PlateColumn(name="7"),
        PlateColumn(name="8"),
        PlateColumn(name="9"),
        PlateColumn(name="10"),
    ]
    rows = [PlateRow(name="B")]
    wells = [
        PlateWell(path="B/4", rowIndex=1, columnIndex=3),
        PlateWell(path="B/5", rowIndex=1, columnIndex=4),
        PlateWell(path="B/6", rowIndex=1, columnIndex=5),
        PlateWell(path="B/7", rowIndex=1, columnIndex=6),
        PlateWell(path="B/8", rowIndex=1, columnIndex=7),
        PlateWell(path="B/9", rowIndex=1, columnIndex=8),
        PlateWell(path="B/10", rowIndex=1, columnIndex=9),
    ]

    plate_96 = Plate(columns=columns, rows=rows, wells=wells, name="Example Plate96", field_count=4)

    # Create the HCS plate structure
    hcs_plate = HCSPlate(store=zarr_output_path, plate_metadata=plate_96)
    to_hcs_zarr(hcs_plate, zarr_output_path)

    for well in wells:
        print(f"Processing well: {well.path}")
        row_name, col_name = well.path.split("/")
        current_well_id = well.path.replace("/", "")
        print(f"Current WellID: {current_well_id} Row: {row_name}, Column: {col_name}")
        for fi, field in enumerate(field_paths):
            current_scene_index = mdata.sample.well_scene_indices[current_well_id][fi]

            # create current field image
            current_field_image = nz.NgffImage(
                data=array6d[current_scene_index, ...],
                dims=["t", "c", "z", "y", "x"],
                scale={"y": mdata.scale.Y, "x": mdata.scale.X, "z": mdata.scale.Z},
                translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
                name=mdata.filename,
            )

            # create multi-scaled, chunked data structure from the image
            multiscales = nz.to_multiscales(current_field_image, [2, 4], method=nz.Methods.DASK_IMAGE_GAUSSIAN)

            # write to wells
            nz.write_hcs_well_image(
                store=zarr_output_path,
                multiscales=multiscales,
                plate_metadata=plate_96,
                row_name=row_name,
                column_name=col_name,
                field_index=fi,  # First field of view
                acquisition_id=0,
            )

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
