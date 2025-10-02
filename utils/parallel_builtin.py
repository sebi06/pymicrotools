import numpy as np
import itertools
from scipy.ndimage import gaussian_filter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


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


def process_nd_parallel(func, max_workers=None):
    """
    RECOMMENDED: Parallelized version using ThreadPoolExecutor.

    This is the best approach for parallelizing process_nd because:
    - Uses only standard library (no extra dependencies)
    - ThreadPoolExecutor works well with numpy/scipy (releases GIL)
    - Good performance for CPU-bound tasks like image filtering
    - Easy to control number of workers

    Args:
        func: Function to apply to each 2D slice
        max_workers: Number of threads (None uses default based on CPU count)
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


def process_nd_chunked(func, chunk_size_param=None, max_workers=None):
    """
    Advanced: Process in chunks for very large arrays.

    This approach is useful when you have thousands of 2D slices
    and want to balance memory usage with parallelization.
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        processed_array = np.zeros_like(array)

        # Get all indices
        loopover = [range(s) for s in shape_woXY]
        indices = list(itertools.product(*loopover))

        # Determine chunk size
        if chunk_size_param is None:
            num_workers = max_workers or min(32, cpu_count() * 2)
            chunk_size = max(1, len(indices) // num_workers)
        else:
            chunk_size = chunk_size_param

        # Split indices into chunks
        chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

        def process_chunk(chunk_indices):
            """Process a chunk of indices."""
            chunk_results = []
            for idx in chunk_indices:
                sl = list(idx) + [slice(None), slice(None)]
                array2d = array[tuple(sl)]
                result = func(array2d, *args, **kwargs)
                chunk_results.append((idx, result))
            return chunk_results

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                for idx, result in chunk_results:
                    sl = list(idx) + [slice(None), slice(None)]
                    processed_array[tuple(sl)] = result

        return processed_array

    return wrapper


def demonstrate_parallelization():
    """Demonstrate different parallelization approaches."""

    # Create test functions
    @process_nd_serial
    def apply_filter_serial(array: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter(array, sigma=sigma)

    @process_nd_parallel
    def apply_filter_parallel(array: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter(array, sigma=sigma)

    @process_nd_chunked
    def apply_filter_chunked(array: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter(array, sigma=sigma)

    # Create test data
    print("Setting up test...")
    print(f"Available CPU cores: {cpu_count()}")

    # Use smaller array for demo
    array = np.random.randint(0, 100, size=(2, 3, 4, 100, 80)).astype(float)
    print(f"Array shape: {array.shape}")
    print(f"Number of 2D slices: {np.prod(array.shape[:-2])}")
    print(f"Each slice size: {array.shape[-2]} x {array.shape[-1]}")

    sigma = 2.0

    implementations = [
        ("Serial", apply_filter_serial),
        ("Parallel (ThreadPool)", apply_filter_parallel),
        ("Chunked Parallel", apply_filter_chunked),
    ]

    results = {}

    for name, func in implementations:
        print(f"\n--- {name} ---")
        start_time = time.time()

        try:
            result = func(array, sigma=sigma)
            elapsed = time.time() - start_time

            results[name] = {"result": result, "time": elapsed, "success": True}

            print(f"✅ Time: {elapsed:.3f} seconds")

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results[name] = {"success": False}

    # Verify all results are identical
    print("\n--- Verification ---")
    successful = [name for name, data in results.items() if data.get("success")]

    if len(successful) > 1:
        reference = results[successful[0]]["result"]
        for name in successful[1:]:
            result = results[name]["result"]
            if np.allclose(reference, result, rtol=1e-10):
                print(f"✅ {name} matches reference")
            else:
                print(f"❌ {name} differs from reference")

    # Performance comparison
    print("\n--- Performance Summary ---")
    times = {name: data["time"] for name, data in results.items() if data.get("success")}

    if times:
        fastest_time = min(times.values())
        fastest_name = min(times.items(), key=lambda x: x[1])[0]

        print(f"Fastest: {fastest_name} ({fastest_time:.3f}s)")

        for name, time_taken in sorted(times.items(), key=lambda x: x[1]):
            speedup = fastest_time / time_taken if time_taken > 0 else 1.0
            print(f"{name:20}: {time_taken:.3f}s (speedup: {speedup:.2f}x)")


# Updated version of your process_nd with optional parallelization
def process_nd_optimized(func, parallel=True, max_workers=None):
    """
    Drop-in replacement for your process_nd with optional parallelization.

    Args:
        func: Function to apply to each 2D slice
        parallel: Whether to use parallel processing
        max_workers: Number of parallel workers (None for auto)

    Usage:
        @process_nd_optimized  # Uses parallel processing
        def my_filter(array, sigma):
            return gaussian_filter(array, sigma=sigma)

        @process_nd_optimized(parallel=False)  # Forces serial
        def my_filter_serial(array, sigma):
            return gaussian_filter(array, sigma=sigma)
    """
    if parallel:
        return process_nd_parallel(func, max_workers=max_workers)
    else:
        return process_nd_serial(func)


if __name__ == "__main__":
    demonstrate_parallelization()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR YOUR USE CASE:")
    print("=" * 60)
    print("1. 🎯 BEST: Replace your process_nd with process_nd_optimized")
    print("2. 🚀 For Gaussian filtering: Use process_nd_parallel")
    print("3. 💾 For huge arrays: Use process_nd_chunked")
    print("4. 🔧 Always test both serial and parallel on your actual data")
    print("\nExample integration:")
    print("@process_nd_optimized(parallel=True, max_workers=4)")
    print("def apply_filter(array, sigma):")
    print("    return gaussian_filter(array, sigma=sigma)")
