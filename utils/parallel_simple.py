import numpy as np
import itertools
from scipy.ndimage import gaussian_filter
import time
from joblib import Parallel, delayed


def process_nd_serial(func):
    """Original serial implementation."""

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        processed_array = np.zeros_like(array)

        loopover = [range(s) for s in shape_woXY]
        prod = itertools.product(*loopover)

        for idx in prod:
            sl = list(idx) + [slice(None), slice(None)]
            array2d = array[tuple(sl)]
            processed_array[tuple(sl)] = func(array2d, *args, **kwargs)

        return processed_array

    return wrapper


def process_nd_parallel(func, n_jobs=-1):
    """
    RECOMMENDED: Parallelized version using joblib.

    Pros:
    - Easy to use and reliable
    - Good performance for CPU-bound tasks
    - Handles shared memory efficiently
    - Works well with numpy arrays

    Args:
        func: Function to apply to each 2D slice
        n_jobs: Number of parallel jobs (-1 uses all cores)
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        processed_array = np.zeros_like(array)

        # Get all indices to process
        loopover = [range(s) for s in shape_woXY]
        indices = list(itertools.product(*loopover))

        def process_single_slice(idx):
            """Process a single 2D slice."""
            sl = list(idx) + [slice(None), slice(None)]
            array2d = array[tuple(sl)]
            return idx, func(array2d, *args, **kwargs)

        # Process all slices in parallel
        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_single_slice)(idx) for idx in indices)

        # Assign results back to output array
        for idx, result in results:
            sl = list(idx) + [slice(None), slice(None)]
            processed_array[tuple(sl)] = result

        return processed_array

    return wrapper


# Example usage and comparison
def demonstrate_parallelization():
    """Demonstrate the performance difference between serial and parallel processing."""

    # Create test functions
    @process_nd_serial
    def apply_filter_serial(array: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter(array, sigma=sigma)

    @process_nd_parallel
    def apply_filter_parallel(array: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter(array, sigma=sigma)

    # Create test data
    print("Creating test array...")
    array = np.random.randint(0, 100, size=(2, 3, 4, 200, 150)).astype(float)
    print(f"Array shape: {array.shape}")
    print(f"Number of 2D slices: {np.prod(array.shape[:-2])}")
    print(f"Each slice size: {array.shape[-2]} x {array.shape[-1]}")

    sigma = 2.0

    # Test serial version
    print("\n--- Serial Processing ---")
    start_time = time.time()
    result_serial = apply_filter_serial(array, sigma=sigma)
    serial_time = time.time() - start_time
    print(f"Serial time: {serial_time:.2f} seconds")

    # Test parallel version
    print("\n--- Parallel Processing ---")
    start_time = time.time()
    result_parallel = apply_filter_parallel(array, sigma=sigma)
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.2f} seconds")

    # Verify results are identical
    arrays_equal = np.allclose(result_serial, result_parallel, rtol=1e-10)
    print(f"\nResults identical: {arrays_equal}")

    # Calculate speedup
    if parallel_time > 0:
        speedup = serial_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")

        if speedup > 1:
            print("✅ Parallel processing is faster!")
        else:
            print("⚠️  Serial processing is faster (overhead dominates)")

    return result_parallel


if __name__ == "__main__":
    result = demonstrate_parallelization()

    print("\n" + "=" * 50)
    print("BEST PRACTICES FOR PARALLELIZATION:")
    print("=" * 50)
    print("1. Use process_nd_parallel for CPU-bound operations")
    print("2. Parallel processing helps most when:")
    print("   - You have many 2D slices to process")
    print("   - Each slice operation is computationally expensive")
    print("   - Your machine has multiple CPU cores")
    print("3. For very large arrays, consider chunking or dask")
    print("4. Always verify results are identical!")
