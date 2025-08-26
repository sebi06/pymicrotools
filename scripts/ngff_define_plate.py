from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
from dataclasses import dataclass
from typing import Dict
from enum import Enum


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


# Example usage:
if __name__ == "__main__":

    # Method 1: Using PlateType enum (default field_count=1)
    plate_96 = define_plate(PlateType.PLATE_96)
    print(f"96-well plate: {plate_96.name}, Wells: {len(plate_96.wells)}, Fields: {plate_96.field_count}")

    # Method 2: Using well count with custom field count
    plate_384 = define_plate_by_well_count(384, field_count=4)
    print(f"384-well plate: {plate_384.name}, Wells: {len(plate_384.wells)}, Fields: {plate_384.field_count}")

    # Method 3: Using PlateType enum with custom field count
    plate_24 = define_plate(PlateType.PLATE_24, field_count=2)
    print(f"24-well plate: {plate_24.name}, Wells: {len(plate_24.wells)}, Fields: {plate_24.field_count}")

    # Show all available formats
    print("\nAvailable plate formats:")
    for plate_type in PlateType:
        config = plate_type.value
        print(f"  {config.name}: {config.rows}x{config.columns} = {config.total_wells} wells")

    # Example: Create all plate types with varied field counts
    print("\nCreating all plate types with varied field counts:")
    field_counts = {6: 1, 24: 2, 48: 1, 96: 4, 384: 9, 1536: 16}  # Example field counts
    for well_count in PLATE_FORMATS.keys():
        field_count = field_counts[well_count]
        plate = define_plate_by_well_count(well_count, field_count=field_count)
        print(f"  {plate.name}: {len(plate.wells)} wells, {plate.field_count} field(s)")
