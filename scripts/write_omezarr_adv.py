from czitools.read_tools import read_tools
from czitools.metadata_tools.czi_metadata import CziMetadata
from pathlib import Path
from typing import Union, Optional
import zarr
import dask.array as da
import ome_zarr.writer
import ome_zarr.format
from ome_zarr.io import parse_url
import shutil
import numpy as np
import xarray as xr


def get_display(metadata: CziMetadata, channel_index):

    # try to read the display settings embedded in the CZI
    try:
        lower = np.round(
            metadata.channelinfo.clims[channel_index][0] * metadata.maxvalue_list[channel_index],
            0,
        )
        higher = np.round(
            metadata.channelinfo.clims[channel_index][1] * metadata.maxvalue_list[channel_index],
            0,
        )

        maxvalue = metadata.maxvalue_list[channel_index]

    except IndexError:
        print("Calculation from display setting from CZI failed. Use 0-Max instead.")
        lower = 0
        higher = metadata.maxvalue[channel_index]
        maxvalue = higher

    return lower, higher, maxvalue


def write_omezarr(
    array5d: Union[np.ndarray, xr.DataArray, da.Array],
    zarr_path: Union[str, Path],
    metadata: CziMetadata,
    overwrite: bool = False,
) -> Optional[str]:
    """
    Write a 5D array to OME-ZARR format.

    This function writes a multi-dimensional array (typically from microscopy data)
    to the OME-ZARR format, which is a cloud-optimized format for storing and
    accessing large microscopy datasets.

    Args:
        array5d: Input array with up to 5 dimensions. Can be a numpy array or
                xarray DataArray or dask Array. Expected dimension order is typically TCZYX
                (Time, Channel, Z, Y, X) or similar.
        zarr_path: Path where the OME-ZARR file should be written. Can be a
                  string or Path object.
        metadata: Metadata object containing information about the image.
        overwrite: If True, remove existing file at zarr_path before writing.
                  If False and file exists, return None without writing.
                  Default is False.

    Returns:
        str: Path to the written OME-ZARR file if successful, None if failed.

    Raises:
        None: Function handles errors gracefully and returns None on failure.

    Examples:
        >>> import numpy as np
        >>> data = np.random.rand(10, 2, 5, 512, 512)  # TCZYX
        >>> result = write_omezarr(data, "output.ome.zarr", madata, overwrite=True)
        >>> print(f"Written to: {result}")

    Notes:
        - The function uses chunking strategy (1, 1, 1, Y, X) which keeps
          individual Z-slices as chunks for efficient access.
        - Requires the array to have an 'axes' attribute (typical for xarray)
          or the function will use default axes handling.
        - Uses the current NGFF (Next Generation File Format) specification.
    """

    # check number of dimension of input array
    if len(array5d.shape) > 5:
        print("Input array as more than 5 dimensions.")
        return None

    # check if zarr_path already exits
    if Path(zarr_path).exists() and overwrite:
        shutil.rmtree(zarr_path, ignore_errors=False, onerror=None)
    elif Path(zarr_path).exists() and not overwrite:
        print(f"File already exists at {zarr_path}. Set overwrite=True to remove.")
        return None

    # show currently used version of NGFF specification
    ngff_version = ome_zarr.format.CurrentFormat().version
    print(f"Using ngff format version: {ngff_version}")

    # write the image data
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store, overwrite=overwrite)

    # write the OME-ZARR file
    ome_zarr.writer.write_image(
        image=array5d,
        group=root,
        axes=array5d.axes[1:].lower(),
        storage_options=dict(chunks=(1, 1, 1, array5d.Y.size, array5d.X.size)),
    )

    channels_list = []

    for ch_index in range(mdata.image.SizeC):

        rgb = mdata.channelinfo.colors[ch_index][3:]
        chname = mdata.channelinfo.names[ch_index]

        lower, higher, maxvalue = get_display(metadata, ch_index)

        channels_list.append(
            {
                "color": rgb,
                "label": chname,
                "active": True,
                "window": {"min": lower, "start": lower, "end": higher, "max": maxvalue},
            }
        )
    ome_zarr.writer.add_metadata(root, {"omero": {"name": metadata.filename, "channels": channels_list}})

    return zarr_path


# ----------------------------------------------------------------

# Configuration parameters
filepath: str = r"data/CellDivision_T10_Z15_CH2_DCV_small.czi"
# filepath: str = r"data/WP96_4Pos_B4-10_DAPI.czi"
show_napari: bool = True  # Whether to display the result in napari viewer
scene_id: int = 0

# Read the CZI file and return a 6D array with dimension order STCZYX(A)
array, mdata = read_tools.read_6darray(filepath, use_xarray=True)
array = array[scene_id, ...]
zarr_path: Path = Path(str(filepath)[:-4] + ".ome.zarr")

print(f"Array Type: {type(array)}, Shape: {array.shape}, Dtype: {array.dtype}")

# Write OME-ZARR using utility function
result_zarr_path: Optional[str] = write_omezarr(array, zarr_path=str(zarr_path), metadata=mdata, overwrite=True)
print(f"Written OME-ZARR using ome-zarr-py: {result_zarr_path}")

# Optional: Visualize the plate data using napari
if show_napari:
    import napari

    viewer: napari.Viewer = napari.Viewer()
    viewer.open(result_zarr_path, plugin="napari-ome-zarr")
    napari.run()
