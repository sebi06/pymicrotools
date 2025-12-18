from ome_zarr_utils import PlateConfiguration, PlateType, PLATE_FORMATS

plate = PlateConfiguration(8, 12, "well96")
print(f"Plate Name: {plate.name}")
print(f"Rows: {plate.rows}, Columns: {plate.columns}, Total Wells: {plate.total_wells}")
print(f"Row Labels: {plate.row_labels}")
print(f"Column Labels: {plate.column_labels}")
