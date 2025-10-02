import dask.array as da
from skimage.measure import label
import numpy as np
import itertools
import inspect
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, NamedTuple, Union, Tuple, Callable
from misc import measure_execution_time, measure_memory_usage
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


# def process_nd(func):

#     def wrapper(md_array: np.ndarray, *args, **kwargs):

#         # print("Before function call")
#         arg_names = inspect.getfullargspec(func).args
#         print(f"Arguments: {arg_names}")

#         shape_woXY = md_array.shape[:-2]

#         # processed_md = da.zeros_like(md_array, chunks=md_array.shape)
#         processed_md = np.zeros_like(md_array)

#         # create the "values" each for-loop iterates over
#         loopover = [range(s) for s in shape_woXY]
#         prod = itertools.product(*loopover)

#         # loop over all dimensions
#         for idx in prod:

#             # create list of slice objects based on the shape
#             sl = len(shape_woXY) * [np.s_[0:1]]

#             # insert the correct index into the respective slice objects for all dimensions
#             for nd in range(len(shape_woXY)):
#                 sl[nd] = idx[nd]

#             # extract the 2D image from the n-dims stack using the list of slice objects
#             array2d = np.squeeze(md_array[tuple(sl)])

#             # process the whole 2d image - make sure to use the correct **kwargs

#             # insert new 2D after tile-wise processing into nd array
#             processed_md[tuple(sl)] = func(array2d, *args, **kwargs)

#         return processed_md

#     return wrapper


def process_nd(func):
    """Processes an n-dimensional NumPy array by applying the given function to each 2D slice.

    Args:
        func: The function to apply to each 2D slice.

    Returns:
        A function that takes an n-dimensional NumPy array and returns the processed array.
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Applies the function to each 2D slice of the n-dimensional array.

        Args:
            array: The n-dimensional NumPy array to process.
            *args: Additional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The processed n-dimensional NumPy array.
        """

        # Get the shape without the last two dimensions
        shape_woXY = array.shape[:-2]

        # Initialize output array
        processed_array = np.zeros_like(array)

        # Create iterators for all dimensions except the last two
        loopover = [range(s) for s in shape_woXY]
        prod = itertools.product(*loopover)

        # Process each 2D slice
        for idx in prod:
            # Create slice objects for all dimensions except last two
            sl = list(idx) + [slice(None), slice(None)]

            # Extract and process the 2D slice
            array2d = array[tuple(sl)]

            processed_array[tuple(sl)] = func(array2d, *args, **kwargs)

        return processed_array

    return wrapper


def process_nd_dask(func, chunks="auto", scheduler="threads"):
    """
    Dask-based implementation using map_blocks - BEST for very large arrays.

    WHEN TO USE:
    - Arrays larger than available RAM
    - Want lazy evaluation and computation graphs
    - Need to scale across multiple machines (with distributed scheduler)
    - Working with arrays that have thousands of 2D slices

    ADVANTAGES:
    - Memory efficient (processes chunks at a time)
    - Lazy evaluation (no computation until .compute())
    - Can handle arrays larger than RAM
    - Built-in parallelization and optimization
    - Works with distributed computing

    Args:
        func: Function to apply to each 2D slice
        chunks: Chunking strategy ('auto', tuple, or dict)
        scheduler: 'threads', 'processes', or 'synchronous'

    Returns:
        Decorator function
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:

        # Convert to dask array if not already
        if not isinstance(array, da.Array):
            dask_array = da.from_array(array, chunks=chunks)
        else:
            dask_array = array

        def apply_func_to_chunk(chunk, *args, **kwargs):
            """
            Apply function to a dask chunk.

            The key insight: each chunk might contain multiple 2D slices,
            so we need to process each 2D slice within the chunk.
            """
            # Get shape without last two dimensions
            shape_woXY = chunk.shape[:-2]

            # If this chunk is already a 2D slice, apply function directly
            if len(shape_woXY) == 0:
                return func(chunk, *args, **kwargs)

            # Otherwise, process each 2D slice in the chunk
            processed_chunk = np.zeros_like(chunk)
            loopover = [range(s) for s in shape_woXY]
            prod = itertools.product(*loopover)

            for idx in prod:
                sl = list(idx) + [slice(None), slice(None)]
                array2d = chunk[tuple(sl)]
                processed_chunk[tuple(sl)] = func(array2d, *args, **kwargs)

            return processed_chunk

        # Apply function to each chunk using map_blocks
        result = da.map_blocks(
            apply_func_to_chunk,
            dask_array,
            *args,
            dtype=dask_array.dtype,
            chunks=dask_array.chunks,  # Preserve chunking structure
            **kwargs,
        )

        # Compute result with specified scheduler
        with da.config.set(scheduler=scheduler):
            return result.compute()

    return wrapper


def process_nd_dask_optimized(func, chunks_param="auto", scheduler="threads", slice_based_chunking=True):
    """
    Optimized dask implementation that chunks along slice dimensions.

    RECOMMENDED for very large arrays with many slices.

    This version is smarter about chunking - it tries to put complete 2D slices
    in each chunk, which reduces the overhead of processing partial slices.

    Args:
        func: Function to apply to each 2D slice
        chunks_param: Base chunking strategy
        scheduler: 'threads', 'processes', or 'synchronous'
        slice_based_chunking: If True, optimize chunks for 2D slice processing
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:

        # Optimize chunking for slice-based processing
        if slice_based_chunking and chunks_param == "auto":
            # Create chunks that align with 2D slices
            # Keep the last two dimensions intact, chunk the others
            shape = array.shape
            optimal_chunks = list(shape)

            # For the first n-2 dimensions, use smaller chunks
            for i in range(len(shape) - 2):
                optimal_chunks[i] = min(optimal_chunks[i], 4)  # Max 4 slices per chunk

            chunks = tuple(optimal_chunks)
        else:
            chunks = chunks_param

        # Convert to dask array
        if not isinstance(array, da.Array):
            dask_array = da.from_array(array, chunks=chunks)
        else:
            dask_array = array

        print(f"Dask array chunks: {dask_array.chunks}")
        print(f"Number of chunks: {dask_array.npartitions}")

        def apply_func_to_chunk(chunk, *args, **kwargs):
            """Optimized chunk processing."""
            shape_woXY = chunk.shape[:-2]

            if len(shape_woXY) == 0:
                # Single 2D slice
                return func(chunk, *args, **kwargs)

            # Multiple 2D slices in chunk
            processed_chunk = np.zeros_like(chunk)
            loopover = [range(s) for s in shape_woXY]

            # Process each 2D slice
            for idx in itertools.product(*loopover):
                sl = list(idx) + [slice(None), slice(None)]
                array2d = chunk[tuple(sl)]
                processed_chunk[tuple(sl)] = func(array2d, *args, **kwargs)

            return processed_chunk

        # Use map_blocks with optimized settings
        result = da.map_blocks(
            apply_func_to_chunk,
            dask_array,
            *args,
            dtype=dask_array.dtype,
            chunks=dask_array.chunks,
            drop_axis=None,  # Keep all dimensions
            new_axis=None,  # Don't add dimensions
            **kwargs,
        )

        # Compute with progress bar and specified scheduler
        with da.config.set(scheduler=scheduler):
            return result.compute()

    return wrapper


def process_nd_parallel(func, max_workers=None):
    """
    Parallelized version of process_nd using ThreadPoolExecutor.

    WHEN TO USE:
    - Large arrays with many 2D slices (>50 slices)
    - CPU-intensive operations (like Gaussian filtering)
    - When you have multiple CPU cores available

    Args:
        func: Function to apply to each 2D slice
        max_workers: Number of parallel workers (None for auto)

    Returns:
        Decorator function
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        processed_array = np.zeros_like(array)

        # Get all indices to process
        loopover = [range(s) for s in shape_woXY]
        indices = list(itertools.product(*loopover))

        # For small numbers of slices, use serial processing
        if len(indices) < 8:
            for idx in indices:
                sl = list(idx) + [slice(None), slice(None)]
                array2d = array[tuple(sl)]
                processed_array[tuple(sl)] = func(array2d, *args, **kwargs)
            return processed_array

        def process_single_slice(idx):
            """Process a single 2D slice."""
            sl = list(idx) + [slice(None), slice(None)]
            array2d = array[tuple(sl)]
            return idx, func(array2d, *args, **kwargs)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_slice, idx): idx for idx in indices}

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                sl = list(idx) + [slice(None), slice(None)]
                processed_array[tuple(sl)] = result

        return processed_array

    return wrapper


def process_nd_auto(func, parallel_threshold=8, max_workers=None):
    """
    Automatic selection between serial and parallel processing.

    RECOMMENDED: Use this for production code.

    Args:
        func: Function to apply to each 2D slice
        parallel_threshold: Minimum number of slices to use parallel processing
        max_workers: Number of parallel workers (None for auto)
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        num_slices = np.prod(shape_woXY)

        if num_slices >= parallel_threshold:
            # Use parallel processing for many slices
            return process_nd_parallel(func, max_workers)(array, *args, **kwargs)
        else:
            # Use serial processing for few slices
            return process_nd(func)(array, *args, **kwargs)

    return wrapper


# @measure_memory_usage
@measure_execution_time
@process_nd
def apply_filter(array: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a 2D Gaussian filter to a 2D NumPy array.

    Args:
      array: The 2D NumPy array to filter.
      sigma: The standard deviation of the Gaussian kernel.

    Returns:
      The filtered 2D NumPy array.
    """

    return gaussian_filter(array, sigma=sigma)
    # return array**2


# @measure_memory_usage
@measure_execution_time
@process_nd_auto
def apply_gauss(array: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a 2D Gaussian filter to a 2D NumPy array.

    Args:
      array: The 2D NumPy array to filter.
      sigma: The standard deviation of the Gaussian kernel.

    Returns:
      The filtered 2D NumPy array.
    """

    return gaussian_filter(array, sigma=sigma)
    # return array**2


# Example usage:
array = np.random.randint(0, high=100, size=(1, 2, 20, 2000, 1500), dtype=int)

# Apply function to array
filtered_array1 = apply_filter(array, sigma=2)
filtered_array2 = apply_gauss(array, sigma=2)

print(f"Equal: {np.array_equal(filtered_array1, filtered_array2)}")
