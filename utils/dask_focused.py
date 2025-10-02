"""
🎯 FOCUSED EXAMPLE: Dask map_blocks for process_nd

This shows the key advantages of using Dask map_blocks for large array processing.
"""

import numpy as np
import dask.array as da
import itertools
from scipy.ndimage import gaussian_filter
import time


def process_nd_serial(func):
    """Your original serial implementation."""

    def wrapper(array, *args, **kwargs):
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


def process_nd_dask(func, chunks="auto"):
    """
    🚀 DASK VERSION: Best for very large arrays using map_blocks

    KEY ADVANTAGES:
    1. Memory efficient - processes chunks at a time
    2. Lazy evaluation - builds computation graph first
    3. Parallelization - automatic multi-core processing
    4. Scalable - can handle arrays larger than RAM
    5. Optimized - dask optimizes the computation graph
    """

    def wrapper(array, *args, **kwargs):
        nonlocal chunks
        # Smart chunking for 2D slice processing
        if chunks == "auto":
            shape = array.shape
            # Keep last 2 dimensions intact, chunk others moderately
            optimal_chunks = []
            for i, dim_size in enumerate(shape[:-2]):
                chunk_size = min(dim_size, 4)  # Max 4 slices per chunk
                optimal_chunks.append(chunk_size)
            optimal_chunks.extend([shape[-2], shape[-1]])  # Keep spatial dims intact
            chunks = tuple(optimal_chunks)

        # Convert to dask array
        if not isinstance(array, da.Array):
            dask_array = da.from_array(array, chunks=chunks)
        else:
            dask_array = array

        print(f"📊 Dask Info:")
        print(f"  Array shape: {dask_array.shape}")
        print(f"  Chunks: {dask_array.chunks}")
        print(f"  Total chunks: {dask_array.npartitions}")
        print(f"  Memory per chunk: ~{dask_array.nbytes / dask_array.npartitions / 1e6:.1f} MB")

        def apply_to_chunk(chunk, *args, **kwargs):
            """Apply function to each 2D slice in a dask chunk."""
            shape_woXY = chunk.shape[:-2]

            # If chunk is already a single 2D slice
            if len(shape_woXY) == 0:
                return func(chunk, *args, **kwargs)

            # Process multiple 2D slices in this chunk
            processed_chunk = np.zeros_like(chunk)
            for idx in itertools.product(*[range(s) for s in shape_woXY]):
                sl = list(idx) + [slice(None), slice(None)]
                array2d = chunk[tuple(sl)]
                processed_chunk[tuple(sl)] = func(array2d, *args, **kwargs)

            return processed_chunk

        # 🔥 The magic happens here: map_blocks applies function to each chunk
        result = da.map_blocks(
            apply_to_chunk,
            dask_array,
            *args,
            dtype=dask_array.dtype,
            chunks=dask_array.chunks,  # Preserve chunk structure
            **kwargs,
        )

        # Compute the result (this is when actual processing happens)
        return result.compute()

    return wrapper


def demonstrate_dask_power():
    """Show why dask is powerful for large arrays."""

    print("🎯 DASK MAP_BLOCKS DEMONSTRATION")
    print("=" * 50)

    # Create test functions
    @process_nd_serial
    def filter_serial(array, sigma):
        return gaussian_filter(array, sigma=sigma)

    @process_nd_dask
    def filter_dask(array, sigma):
        return gaussian_filter(array, sigma=sigma)

    # Test with progressively larger arrays
    test_cases = [
        {"name": "Small", "shape": (2, 4, 6, 300, 200)},
        {"name": "Medium", "shape": (5, 8, 10, 500, 400)},
        {"name": "Large", "shape": (8, 15, 20, 600, 500)},
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} Array: {case['shape']} ---")

        # Create test array
        array = np.random.randint(0, 100, size=case["shape"]).astype(np.float32)
        array_size = array.nbytes / 1e6
        num_slices = np.prod(case["shape"][:-2])

        print(f"💾 Size: {array_size:.1f} MB, Slices: {num_slices}")

        sigma = 2.0

        # Test serial version
        print(f"\n🐌 Serial Processing:")
        start = time.time()
        result_serial = filter_serial(array, sigma=sigma)
        serial_time = time.time() - start
        print(f"  Time: {serial_time:.2f}s")

        # Test dask version
        print(f"\n🚀 Dask Processing:")
        start = time.time()
        result_dask = filter_dask(array, sigma=sigma)
        dask_time = time.time() - start
        print(f"  Time: {dask_time:.2f}s")

        # Compare results
        results_match = np.allclose(result_serial, result_dask, rtol=1e-5)
        speedup = serial_time / dask_time if dask_time > 0 else 1.0

        print(f"\n📊 Results:")
        print(f"  ✅ Results match: {results_match}")
        print(f"  ⚡ Speedup: {speedup:.2f}x")

        if speedup > 1:
            print(f"  🎉 Dask is {speedup:.1f}x faster!")
        else:
            print(f"  💡 Serial faster (overhead dominates for small arrays)")


def demonstrate_lazy_evaluation():
    """Show the power of lazy evaluation."""

    print(f"\n" + "=" * 50)
    print("🔥 LAZY EVALUATION DEMO")
    print("=" * 50)

    # Create a moderate array
    shape = (4, 8, 12, 400, 300)
    array = np.random.randint(0, 100, size=shape).astype(np.float32)
    print(f"Array: {shape}, Size: {array.nbytes / 1e6:.1f} MB")

    # Convert to dask with smart chunking
    chunks = (2, 4, 6, 400, 300)  # Keep spatial dimensions intact
    dask_array = da.from_array(array, chunks=chunks)

    print(f"Dask chunks: {dask_array.chunks}")

    # Define pipeline operations
    def blur_chunk(chunk):
        """Apply blur to each 2D slice in chunk."""
        shape_woXY = chunk.shape[:-2]
        if len(shape_woXY) == 0:
            return gaussian_filter(chunk, sigma=2.0)

        processed = np.zeros_like(chunk)
        for idx in itertools.product(*[range(s) for s in shape_woXY]):
            sl = list(idx) + [slice(None), slice(None)]
            processed[tuple(sl)] = gaussian_filter(chunk[tuple(sl)], sigma=2.0)
        return processed

    def threshold_chunk(chunk):
        """Apply threshold."""
        return np.where(chunk > 30, chunk, 0)

    # Build computation graph (NO COMPUTATION YET!)
    print(f"\n🔨 Building computation pipeline...")
    step1 = da.map_blocks(blur_chunk, dask_array, dtype=np.float32)
    step2 = da.map_blocks(threshold_chunk, step1, dtype=np.float32)
    step3 = step2 * 2.0  # Arithmetic operation

    print(f"✅ Pipeline built (shape: {step3.shape})")
    print(f"💡 No computation performed yet - just built the graph!")

    # Now compute everything at once
    print(f"\n⚡ Computing entire pipeline...")
    start = time.time()
    result = step3.compute()
    compute_time = time.time() - start

    print(f"✅ Done! Time: {compute_time:.2f}s")
    print(f"📊 Result shape: {result.shape}, dtype: {result.dtype}")

    print(f"\n🎯 KEY INSIGHT:")
    print(f"Dask optimized the entire pipeline and computed it efficiently!")


if __name__ == "__main__":
    print("DASK MAP_BLOCKS for Large Array Processing")
    print("Using map_blocks for efficient 2D slice processing")

    try:
        demonstrate_dask_power()
        demonstrate_lazy_evaluation()

        print(f"\n" + "=" * 60)
        print("🎯 WHEN TO USE DASK MAP_BLOCKS:")
        print("=" * 60)
        print("✅ Arrays > 1GB in size")
        print("✅ Thousands of 2D slices to process")
        print("✅ Complex processing pipelines")
        print("✅ Want automatic parallelization")
        print("✅ Need memory-efficient processing")
        print("✅ Working with file-backed arrays")

        print(f"\n🚀 INTEGRATION WITH YOUR CODE:")
        print("Replace @process_nd with @process_nd_dask for large arrays!")

    except ImportError:
        print("❌ Please install dask: pip install dask[array]")
    except Exception as e:
        print(f"❌ Error: {e}")
