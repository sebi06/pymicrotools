import zarr
from pathlib import Path
from czitools.read_tools import read_tools
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
import shutil
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
from dataclasses import dataclass
from typing import Dict
from enum import Enum


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

@dataclass
class PlateConfiguration:
    """Configuration for standard microplate formats"""

    rows: int
    columns: int
    name: str

    @property
    def total_wells(self) -> int:
        return self.rows * self.columns

    @property
    def row_labels(self) -> list:
        """Generate row labels (A, B, C, ...)"""
        return [chr(ord("A") + i) for i in range(self.rows)]

    @property
    def column_labels(self) -> list:
        """Generate column labels (1, 2, 3, ...)"""
        return [str(i) for i in range(1, self.columns + 1)]


class PlateType(Enum):
    """Standard microplate formats with their configurations"""

    PLATE_6 = PlateConfiguration(2, 3, "6-Well Plate")
    PLATE_24 = PlateConfiguration(4, 6, "24-Well Plate")
    PLATE_48 = PlateConfiguration(6, 8, "48-Well Plate")
    PLATE_96 = PlateConfiguration(8, 12, "96-Well Plate")
    PLATE_384 = PlateConfiguration(16, 24, "384-Well Plate")
    PLATE_1536 = PlateConfiguration(32, 48, "1536-Well Plate")


# Dictionary for easy lookup by well count
PLATE_FORMATS: Dict[int, PlateConfiguration] = {
    6: PlateType.PLATE_6.value,
    24: PlateType.PLATE_24.value,
    48: PlateType.PLATE_48.value,
    96: PlateType.PLATE_96.value,
    384: PlateType.PLATE_384.value,
    1536: PlateType.PLATE_1536.value,
}


def define_plate(plate_type: PlateType, field_count: int = 1) -> Plate:
    """
    Create a plate metadata object for any standard plate format

    Args:
        plate_type: PlateType enum value specifying the plate format
        field_count: Number of fields per well (default: 1)

    Returns:
        Plate metadata object
    """
    config = plate_type.value

    # Create columns and rows based on configuration
    columns = [PlateColumn(name=label) for label in config.column_labels]
    rows = [PlateRow(name=label) for label in config.row_labels]

    # Generate all wells
    wells = [
        PlateWell(path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, col in enumerate(columns)
    ]

    # Create plate metadata
    plate_metadata = Plate(name=config.name, columns=columns, rows=rows, wells=wells, field_count=field_count)

    return plate_metadata


def define_plate_by_well_count(well_count: int, field_count: int = 1) -> Plate:
    """
    Create a plate by specifying the number of wells

    Args:
        well_count: Number of wells (6, 24, 48, 96, 384, or 1536)
        field_count: Number of fields per well (default: 1)

    Returns:
        Plate metadata object

    Raises:
        ValueError: If well_count is not a standard format
    """
    if well_count not in PLATE_FORMATS:
        available = list(PLATE_FORMATS.keys())
        raise ValueError(f"Unsupported well count: {well_count}. Available formats: {available}")

    config = PLATE_FORMATS[well_count]

    # Create columns and rows based on configuration
    columns = [PlateColumn(name=label) for label in config.column_labels]
    rows = [PlateRow(name=label) for label in config.row_labels]

    # Generate all wells
    wells = [
        PlateWell(path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, col in enumerate(columns)
    ]

    # Create plate metadata
    plate_metadata = Plate(name=config.name, columns=columns, rows=rows, wells=wells, field_count=field_count)

    return plate_metadata
