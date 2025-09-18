from czitools.read_tools import read_tools
import ngff_zarr as nz
from pathlib import Path
import ome_zarr.format
import shutil
from czitools.metadata_tools.czi_metadata import CziMetadata
import napari


def write_omezarr_ngff(array5d, zarr_output_path: str, metadata: CziMetadata, overwrite: bool = False):

    # check if zarr_path already exits
    if Path(zarr_output_path).exists() and overwrite:
        shutil.rmtree(zarr_output_path, ignore_errors=False, onerror=None)
    elif Path(zarr_output_path).exists() and not overwrite:
        print(f"File already exists at {zarr_output_path}. Set overwrite=True to remove.")
        return None

    # show currently used version of NGFF specification
    ngff_version = ome_zarr.format.CurrentFormat().version
    print(f"Using ngff format version: {ngff_version}")

    # create NGFF image from the array
    image = nz.to_ngff_image(
        array5d,
        dims=["t", "c", "z", "y", "x"],
        scale={"y": metadata.scale.Y, "x": metadata.scale.X, "z": metadata.scale.Z},
        name=metadata.filename[:-4] + ".ome.zarr",
    )

    # create multi-scaled, chunked data structure from the image
    multiscales = nz.to_multiscales(image, scale_factors=128, method=nz.Methods.DASK_IMAGE_GAUSSIAN)

    # write using ngff-zarr
    nz.to_ngff_zarr(zarr_output_path, multiscales)

    return image


# ----------------------------------------------------------------

# open s simple dialog to select a CZI file
filepaths = [r"data/CellDivision_T10_Z15_CH2_DCV_small.czi", r"data/WP96_4Pos_B4-10_DAPI.czi"]
show_napari = False  # Whether to display the result in napari viewer

for filepath in filepaths:

    try:
        # return a 6D array with dimension order STCZYX(A)
        array, mdata = read_tools.read_6darray(filepath, use_xarray=True)
        # use 5D subset for NGFF
        array = array[0, ...]
        print(f"Array Type: {type(array)}, Shape: {array.shape}, Dtype: {array.dtype}")

        zarr_path = Path(str(filepath)[:-4] + "_ngff.ome.zarr")

        ngff_image = write_omezarr_ngff(array, zarr_path, mdata, overwrite=True)
        print(f"NGFF Image: {ngff_image}")
        print(f"Written OME-ZARR using ngff-zarr: {zarr_path}")

        if show_napari:

            viewer = napari.Viewer()
            viewer.open(zarr_path, plugin="napari-ome-zarr")
            napari.run()

    except KeyError as e:
        print(f"Could not convert: {filepath} KeyError: {e}")

print("Done")
