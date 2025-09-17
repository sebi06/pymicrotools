# -*- coding: utf-8 -*-

#################################################################
# File        : write_omezarr_ngff.py
# Author      : sebi06
#
# Requires: ome-zarr, ngff-zarr
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.read_tools import read_tools
from czitools.metadata_tools import czi_metadata as czimd
import ngff_zarr as nz
from pathlib import Path
import dask.array as da
import zarr
import ome_zarr.writer
import ome_zarr.format
from ome_zarr.io import parse_url
from typing import Union
import shutil
import numpy as np
from czitools.utils import logging_tools
from czitools.metadata_tools.czi_metadata import CziMetadata

logger = logging_tools.set_logging()


def write_omezarr(
    array5d: Union[np.ndarray, da.Array],
    zarr_path: str,
    metadata: CziMetadata,
    overwrite: bool = False,
) -> str:
    """
     Writes a 5D array to an OME-ZARR file.
    Parameters:
    -----------
    array5d : Union[np.ndarray, da.Array]
        The 5D array to be written. The dimensions should not exceed 5.
    zarr_path : str
        The path where the OME-ZARR file will be saved.
    metadata : CziMetadata
        Metadata object containing information about the image.
    overwrite : bool, optional
        If True, the existing file at zarr_path will be overwritten. Default is False.
    Returns:
    --------
    str
        The path to the written OME-ZARR folder if successful, otherwise None.
    Notes:
    ------
    - The function ensures the axes are in lowercase and removes any invalid dimensions.
    - If the zarr_path already exists and overwrite is True, the existing directory will be removed.
    - The function logs the NGFF format version being used.
    - The function writes the image data to the specified zarr_path.
    - If the writing process is successful, the function returns the zarr_path; otherwise, it returns None.
    """

    # check number of dimension of input array
    if len(array5d.shape) > 5:
        logger.warning("Input array as more than 5 dimensions.")
        return None

    # check if zarr_path already exits
    if Path(zarr_path).exists() and overwrite:
        shutil.rmtree(zarr_path, ignore_errors=False, onerror=None)
    elif Path(zarr_path).exists() and not overwrite:
        logger.warning(
            f"File already exists at {zarr_path}. Set overwrite=True to remove."
        )
        return None

    # show currently used version of NGFF specification
    ngff_version = ome_zarr.format.CurrentFormat().version
    logger.info(f"Using ngff format version: {ngff_version}")

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

        channels_list.append(
            {
                "color": rgb,
                "label": chname,
                "active": True,
            }
        )
    ome_zarr.writer.add_metadata(
        root, {"omero": {"name": metadata.filename, "channels": channels_list}}
    )

    logger.info(f"Finished writing OME-ZARR to: {zarr_path}")

    return zarr_path


# ----------------------------------------------------------------

# open s simple dialog to select a CZI file
filepath = r"data/CellDivision_T10_Z15_CH2_DCV_small.czi"

show_napari = True  # Whether to display the result in napari viewer

# get the metadata_tools at once as one big class
mdata = czimd.CziMetadata(filepath)
print("Number of Scenes: ", mdata.image.SizeS)
scene_id = 0

# return a 6D array with dimension order STCZYX(A)
array, mdata = read_tools.read_6darray(filepath, use_dask=True)
array = array[scene_id, ...]


# # Approach 1: Use ome-zarr-py to write OME-ZARR
# zarr_path1 = Path(str(filepath)[:-4] + "_1.ome.zarr")

# # write OME-ZARR using utility function
# zarr_path1 = write_omezarr(array, zarr_path=str(zarr_path1), metadata=mdata, overwrite=True)
# print(f"Written OME-ZARR using ome-zarr.py: {zarr_path1}")


# Approach 2: Use ngff-zarr to create NGFF structure and write using ome-zarr-py
zarr_path2 = Path(str(filepath)[:-4] + "_2.ome.zarr")

# create NGFF image from the array
image = nz.to_ngff_image(
    array.data,
    dims=["t", "c", "z", "y", "x"],
    scale={"y": mdata.scale.Y, "x": mdata.scale.X, "z": mdata.scale.Z},
    name=mdata.filename,
)

# create multi-scaled, chunked data structure from the image
multiscales = nz.to_multiscales(image, [2, 4], method=nz.Methods.DASK_IMAGE_GAUSSIAN)

# write using ngff-zarr
nz.to_ngff_zarr(zarr_path2, multiscales)
print(f"NGFF Image: {image}")
print(f"Written OME-ZARR using ngff-zarr: {zarr_path2}")

# Optional: Visualize the plate data using napari
if show_napari:
    import napari

    viewer = napari.Viewer()
    viewer.open(zarr_path2, plugin="napari-ome-zarr")
    napari.run()
