import numpy as np
import zarr
import napari
from czitools.read_tools import read_tools
from czitools.metadata_tools import czi_metadata as czimd
from czitools.metadata_tools.czi_metadata import CziSampleInfo
from czitools.metadata_tools.sample import get_scenes_for_well
from ome_zarr.format import FormatV04
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata


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


def get_scenes_for_well(sample: CziSampleInfo, well_id: str) -> list[int]:
    """
    Returns a list of scene indices for a given well ID.

    Args:
        sample: The CziSampleInfo object containing well information.
        well_id: The ID of the well.

    Returns:
        list[int]: List of scene indices corresponding to the given well ID.
    """

    if sample.multipos_per_well:
        scene_indices = [
            i for i, x in enumerate(sample.well_array_names) if x == well_id
        ]
    else:
        scene_indices = [
            i for i, x in enumerate(sample.well_position_names) if x == well_id
        ]

    return scene_indices


show_napari = False
filepath = r"data/WP96_4Pos_B4-10_DAPI.czi"

# read comnplete CZI file as 6D array (STCZYX)
array6d, mdata = read_tools.read_6darray(filepath, use_dask=False)


zarr_output_path = filepath[:-4] + "ngff_plate.zarr"

# row_names = ["A", "B"]
# col_names = ["1", "2", "3"]
# well_paths = ["A/2", "B/3"]
# field_paths = ["0", "1", "2"]

row_names, col_names, well_paths = extract_well_coordinates(mdata.sample.well_counter)
field_paths = [
    str(i) for i in range(mdata.sample.well_counter[mdata.sample.well_array_names[0]])
]


# generate data
mean_val = 10
num_wells = len(well_paths)
num_fields = len(field_paths)
size_xy = 128
size_z = 20
rng = np.random.default_rng(0)
data = rng.poisson(
    mean_val, size=(num_wells, num_fields, size_z, size_xy, size_xy)
).astype(np.uint8)

# write the plate of images and corresponding metadata
# Use fmt=FormatV04() in parse_url() to write v0.4 format (zarr v2)
store = parse_url(zarr_output_path, mode="w").store
root = zarr.group(store=store)
write_plate_metadata(root, row_names, col_names, well_paths)
for wi, wp in enumerate(well_paths):
    row, col = wp.split("/")
    row_group = root.require_group(row)
    well_group = row_group.require_group(col)
    write_well_metadata(well_group, field_paths)
    for fi, field in enumerate(field_paths):
        image_group = well_group.require_group(str(field))
        write_image(
            image=data[wi, fi],
            group=image_group,
            axes="zyx",
            storage_options=dict(chunks=(1, size_xy, size_xy)),
        )

if show_napari:

    viewer = napari.Viewer()
    viewer.open(zarr_output_path, plugin="napari-ome-zarr")

    napari.run()
