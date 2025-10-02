"""
SUMMARY: Best Parallelization Strategies for process_nd

Based on the analysis, here are the recommended approaches for parallelizing
your process_nd function when working with large multi-dimensional arrays:
"""

import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# RECOMMENDED APPROACH: process_nd_auto
# =============================================================================


def process_nd_auto(func, parallel_threshold=8, max_workers=None):
    """
    🎯 BEST CHOICE: Automatically chooses serial vs parallel processing.

    This decorator intelligently decides whether to use parallel processing
    based on the number of 2D slices in your array.

    Args:
        func: Function to apply to each 2D slice
        parallel_threshold: Minimum slices needed for parallel processing (default: 8)
        max_workers: Number of parallel workers (None = auto-detect)

    Usage:
        @process_nd_auto
        def apply_gaussian_filter(array, sigma):
            return gaussian_filter(array, sigma=sigma)
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        num_slices = np.prod(shape_woXY)

        if num_slices >= parallel_threshold:
            return process_nd_parallel(func, max_workers)(array, *args, **kwargs)
        else:
            return process_nd_serial(func)(array, *args, **kwargs)

    return wrapper


def process_nd_serial(func):
    """Your original implementation - fast for small arrays."""

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
    """Parallel version using ThreadPoolExecutor."""

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape_woXY = array.shape[:-2]
        processed_array = np.zeros_like(array)

        loopover = [range(s) for s in shape_woXY]
        indices = list(itertools.product(*loopover))

        def process_single_slice(idx):
            sl = list(idx) + [slice(None), slice(None)]
            array2d = array[tuple(sl)]
            return idx, func(array2d, *args, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_slice, idx): idx for idx in indices}

            for future in as_completed(future_to_idx):
                idx, result = future.result()
                sl = list(idx) + [slice(None), slice(None)]
                processed_array[tuple(sl)] = result

        return processed_array

    return wrapper


# =============================================================================
# PERFORMANCE GUIDELINES
# =============================================================================

"""
🚀 WHEN PARALLEL PROCESSING HELPS MOST:

1. Large number of 2D slices (>20-50 slices)
2. Each slice operation is computationally expensive
3. Array shapes like (10, 20, 30, 2000, 1500) - many slices, large slices
4. Operations like Gaussian filtering, morphological operations, etc.

⚠️  WHEN TO STICK WITH SERIAL:

1. Small arrays with few slices (<8 slices)
2. Very fast operations (simple arithmetic)
3. When overhead dominates processing time
4. Memory-constrained environments

📊 PERFORMANCE CHARACTERISTICS:

Array Shape               | Slices | Recommended Approach
(1, 2, 3, 100, 80)       |   6    | Serial (overhead too high)
(5, 10, 20, 500, 400)    |  1000  | Parallel (good speedup)
(1, 1, 1, 2000, 1500)    |   1    | Serial (only one slice)
(2, 20, 15, 1000, 800)   |  600   | Parallel (excellent speedup)

🎯 INTEGRATION EXAMPLE:

# Replace your current decorator:
@process_nd  # OLD
def apply_filter(array, sigma):
    return gaussian_filter(array, sigma=sigma)

# With the auto version:
@process_nd_auto  # NEW
def apply_filter(array, sigma):
    return gaussian_filter(array, sigma=sigma)

# Or force parallel processing:
@process_nd_parallel(max_workers=4)
def apply_filter_parallel(array, sigma):
    return gaussian_filter(array, sigma=sigma)

💡 ADVANCED TIPS:

1. Use max_workers=cpu_count() for CPU-bound tasks
2. Use max_workers=cpu_count()*2 for I/O-bound tasks
3. Monitor memory usage - parallel processing uses more RAM
4. Always verify results match between serial and parallel versions
5. Benchmark with your actual data sizes and operations

🔧 DEBUGGING:

To check if parallelization is working:
- Add print statements showing which approach is chosen
- Time both serial and parallel versions
- Monitor CPU usage during processing
- Check that all CPU cores are being utilized
"""

if __name__ == "__main__":
    print(__doc__)
    print("📚 Parallelization guide for process_nd completed!")
    print("💡 Key takeaway: Use @process_nd_auto as a drop-in replacement")
    print("🚀 It will automatically choose the best approach for your data!")
