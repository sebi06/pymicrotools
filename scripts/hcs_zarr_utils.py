import zarr
from pathlib import Path
from czitools.read_tools import read_tools
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
import shutil


def convert_czi_to_hcs_zarr(czi_filepath: str, overwrite: bool = True) -> str:
    """Convert CZI file to OME-ZARR HCS (High Content Screening) format.

    This function converts a CZI (Carl Zeiss Image) file containing plate data into
    the OME-ZARR HCS format. It handles multi-well plates with multiple fields per well.

    Args:
        czi_filepath: Path to the input CZI file
        overwrite: If True, removes existing zarr files at the output path.
                  If False, skips conversion if output exists.

    Returns:
        str: Path to the output ZARR file (.ngff_plate.zarr)

    Note:
        The output format follows the OME-NGFF specification for HCS data,
        organizing the data in a plate/row/column/field hierarchy.
    """
    # Define output path
    zarr_output_path = Path(czi_filepath[:-4] + "_ngff_plate.zarr")

    # Handle existing files
    if zarr_output_path.exists():
        if overwrite:
            shutil.rmtree(zarr_output_path)
        else:
            print(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            return str(zarr_output_path)

    # Read CZI file
    array6d, mdata = read_tools.read_6darray(czi_filepath, use_dask=False)

    # Extract plate layout
    row_names, col_names, well_paths = extract_well_coordinates(mdata.sample.well_counter)
    field_paths = [str(i) for i in range(mdata.sample.well_counter[mdata.sample.well_array_names[0]])]

    # Initialize zarr storage and write plate metadata
    store = parse_url(zarr_output_path, mode="w").store
    root = zarr.group(store=store)
    write_plate_metadata(root, row_names, col_names, well_paths)

    # Process wells
    for wp in well_paths:
        row, col = wp.split("/")
        well_group = root.require_group(row).require_group(col)
        write_well_metadata(well_group, field_paths)

        current_well_id = wp.replace("/", "")
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            current_scene_index = mdata.sample.well_scene_indices[current_well_id][fi]

            write_image(
                image=array6d[current_scene_index, ...],
                group=image_group,
                axes=array6d.axes[1:].lower(),
                storage_options=dict(chunks=(1, 1, 1, array6d.Y.size, array6d.X.size)),
            )
    return str(zarr_output_path)


def extract_well_coordinates(
    well_counter: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Extract unique row and column names from a well counter dictionary.

    This function parses well positions (e.g., 'B4', 'B5') to extract unique row letters
    and column numbers, and generates corresponding well paths.

    Args:
        well_counter (dict): Dictionary with well positions as keys (e.g., {'B4': 4, 'B5': 4})

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing:
            - row_names: Sorted list of unique row letters
            - col_names: Sorted list of unique column numbers
            - well_paths: List of well paths in format "row/column"
    """
    # Initialize empty sets for rows and columns
    rows = set()
    cols = set()

    # Iterate through well names
    for well in well_counter.keys():
        # Extract row (letters) and column (numbers)
        row = "".join(filter(str.isalpha, well))
        col = "".join(filter(str.isdigit, well))

        rows.add(row)
        cols.add(col)

    # Convert to sorted lists
    row_names = sorted(list(rows))
    col_names = sorted(list(cols))

    # Generate well_paths from the extracted coordinates
    well_paths = [f"{row}/{col}" for row in row_names for col in col_names]

    return row_names, col_names, well_paths
