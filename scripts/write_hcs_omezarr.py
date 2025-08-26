import numpy as np
import zarr
import napari

from ome_zarr.format import FormatV04
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

path = "test_ngff_plate.zarr"
row_names = ["A", "B"]
col_names = ["1", "2", "3"]
well_paths = ["A/2", "B/3"]
field_paths = ["0", "1", "2"]

# generate data
mean_val = 10
num_wells = len(well_paths)
num_fields = len(field_paths)
size_xy = 128
size_z = 20
rng = np.random.default_rng(0)
data = rng.poisson(mean_val, size=(num_wells, num_fields, size_z, size_xy, size_xy)).astype(np.uint8)

# write the plate of images and corresponding metadata
# Use fmt=FormatV04() in parse_url() to write v0.4 format (zarr v2)
store = parse_url(path, mode="w").store
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
            image=data[wi, fi], group=image_group, axes="zyx", storage_options=dict(chunks=(1, size_xy, size_xy))
        )

viewer = napari.Viewer()
viewer.open(path, plugin="napari-ome-zarr")

napari.run()
