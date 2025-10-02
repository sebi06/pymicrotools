"""
Comprehensive Dask Example for process_nd with map_blocks

This demonstrates how to use Dask for processing very large multi-dimensional arrays
that might not fit in memory, using the map_blocks function.
"""

import numpy as np
import dask.array as da
import itertools
from scipy.ndimage import gaussian_filter
import time
import tempfile


def process_nd_dask_simple(func, chunks="auto"):
    """Simple dask implementation using map_blocks."""

    def wrapper(array, *args, **kwargs):
        # Convert to dask array
        if not isinstance(array, da.Array):
            dask_array = da.from_array(array, chunks=chunks)
        else:
            dask_array = array

        def apply_to_chunk(chunk, *args, **kwargs):
            """Apply function to each 2D slice in the chunk."""
            shape_woXY = chunk.shape[:-2]

            if len(shape_woXY) == 0:
                # Single 2D slice - apply directly
                return func(chunk, *args, **kwargs)

            # Multiple slices - process each one
            processed_chunk = np.zeros_like(chunk)
            for idx in itertools.product(*[range(s) for s in shape_woXY]):
                sl = list(idx) + [slice(None), slice(None)]
                processed_chunk[tuple(sl)] = func(chunk[tuple(sl)], *args, **kwargs)

            return processed_chunk

        # Use map_blocks to apply function
        result = da.map_blocks(apply_to_chunk, dask_array, *args, dtype=dask_array.dtype, **kwargs)

        return result.compute()

    return wrapper


def process_nd_dask_optimized(func, chunks_param="auto", persist_input=False):
    """
    Optimized dask implementation with smart chunking.

    Args:
        func: Function to apply
        chunks_param: Chunking strategy
        persist_input: Whether to persist input in memory (good for multiple operations)
    """

    def wrapper(array, *args, **kwargs):
        # Smart chunking: keep 2D slices intact, chunk other dimensions
        if chunks_param == "auto":
            shape = array.shape
            # Chunk size strategy: aim for ~100MB chunks
            target_chunk_size = 100 * 1024 * 1024  # 100MB
            pixel_size = np.dtype(array.dtype).itemsize

            # Calculate optimal chunk size for non-spatial dimensions
            spatial_size = shape[-2] * shape[-1] * pixel_size
            max_slices_per_chunk = max(1, target_chunk_size // spatial_size)

            optimal_chunks = []
            for i, dim_size in enumerate(shape[:-2]):
                chunk_size = min(dim_size, max_slices_per_chunk)
                optimal_chunks.append(chunk_size)

            # Keep spatial dimensions intact
            optimal_chunks.extend([shape[-2], shape[-1]])
            chunks = tuple(optimal_chunks)
        else:
            chunks = chunks_param

        # Convert to dask array
        if not isinstance(array, da.Array):
            dask_array = da.from_array(array, chunks=chunks)
        else:
            dask_array = array

        # Optionally persist input for multiple operations
        if persist_input:
            dask_array = dask_array.persist()

        print(f"Dask array shape: {dask_array.shape}")
        print(f"Chunk structure: {dask_array.chunks}")
        print(f"Number of chunks: {dask_array.npartitions}")
        print(f"Estimated memory per chunk: {dask_array.nbytes / dask_array.npartitions / 1e6:.1f} MB")

        def process_chunk(chunk, *args, **kwargs):
            """Process all 2D slices in a chunk."""
            shape_woXY = chunk.shape[:-2]

            if len(shape_woXY) == 0:
                return func(chunk, *args, **kwargs)

            processed = np.zeros_like(chunk)
            for idx in itertools.product(*[range(s) for s in shape_woXY]):
                sl = list(idx) + [slice(None), slice(None)]
                processed[tuple(sl)] = func(chunk[tuple(sl)], *args, **kwargs)

            return processed

        # Apply function using map_blocks
        result = da.map_blocks(
            process_chunk, dask_array, *args, dtype=dask_array.dtype, chunks=dask_array.chunks, **kwargs
        )

        return result

    return wrapper


def demonstrate_dask_approaches():
    """Demonstrate different dask approaches with various array sizes."""

    print("🚀 Dask map_blocks Demonstration for process_nd")
    print("=" * 60)

    # Create test functions
    @process_nd_dask_simple
    def apply_filter_simple(array, sigma):
        return gaussian_filter(array, sigma=sigma)

    @process_nd_dask_optimized
    def apply_filter_optimized(array, sigma):
        return gaussian_filter(array, sigma=sigma)

    # Test different array sizes
    test_cases = [
        {
            "name": "Small Array (in-memory)",
            "shape": (2, 3, 4, 200, 150),
            "description": "Small enough to fit in memory easily",
        },
        {
            "name": "Medium Array",
            "shape": (5, 8, 10, 500, 400),
            "description": "Moderate size, good for testing chunking",
        },
        {
            "name": "Large Array (chunked)",
            "shape": (10, 20, 30, 800, 600),
            "description": "Large array that benefits from chunking",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Shape: {test_case['shape']}")
        print(f"Description: {test_case['description']}")

        # Create test array
        array = np.random.randint(0, 100, size=test_case["shape"]).astype(np.float32)
        array_size_mb = array.nbytes / 1e6
        num_slices = np.prod(test_case["shape"][:-2])

        print(f"Array size: {array_size_mb:.1f} MB")
        print(f"Number of 2D slices: {num_slices}")

        sigma = 2.0

        # Test simple dask approach
        print("\n  Simple Dask:")
        start_time = time.time()
        result_simple = apply_filter_simple(array, sigma=sigma)
        simple_time = time.time() - start_time
        print(f"  ✅ Time: {simple_time:.2f}s")

        # Test optimized dask approach
        print("\n  Optimized Dask:")
        start_time = time.time()
        result_optimized = apply_filter_optimized(array, sigma=sigma)
        optimized_time = time.time() - start_time
        print(f"  ✅ Time: {optimized_time:.2f}s")

        # Verify results match
        arrays_match = np.allclose(
            result_simple, result_optimized.compute() if hasattr(result_optimized, "compute") else result_optimized
        )
        print(f"  Results match: {arrays_match}")


def demonstrate_lazy_evaluation():
    """Demonstrate the power of lazy evaluation with dask."""

    print("\n" + "=" * 60)
    print("🔥 Lazy Evaluation Demo - Multiple Operations")
    print("=" * 60)

    # Create a moderately large array
    shape = (5, 10, 15, 400, 300)
    array = np.random.randint(0, 100, size=shape).astype(np.float32)
    print(f"Array shape: {shape}")
    print(f"Array size: {array.nbytes / 1e6:.1f} MB")

    # Convert to dask array with smart chunking
    chunks = (2, 4, 5, 400, 300)  # Keep spatial dims intact, chunk others
    dask_array = da.from_array(array, chunks=chunks)

    print(f"Dask chunks: {dask_array.chunks}")
    print(f"Number of chunks: {dask_array.npartitions}")

    # Define processing pipeline using map_blocks
    def gaussian_blur(chunk, sigma=2.0):
        """Apply Gaussian blur to each 2D slice in chunk."""
        shape_woXY = chunk.shape[:-2]
        if len(shape_woXY) == 0:
            return gaussian_filter(chunk, sigma=sigma)

        processed = np.zeros_like(chunk)
        for idx in itertools.product(*[range(s) for s in shape_woXY]):
            sl = list(idx) + [slice(None), slice(None)]
            processed[tuple(sl)] = gaussian_filter(chunk[tuple(sl)], sigma=sigma)
        return processed

    def threshold_filter(chunk, threshold=50):
        """Apply threshold to each chunk."""
        return np.where(chunk > threshold, chunk, 0)

    # Build computation graph (no computation yet!)
    print("\n🔨 Building computation graph...")

    step1 = da.map_blocks(gaussian_blur, dask_array, dtype=np.float32)
    step2 = da.map_blocks(threshold_filter, step1, threshold=30, dtype=np.float32)
    step3 = step2 * 1.5  # Simple arithmetic operation

    print("✅ Graph built (no computation performed yet)")
    print(f"Final result shape: {step3.shape}")

    # Now compute the entire pipeline
    print("\n⚡ Computing entire pipeline...")
    start_time = time.time()
    final_result = step3.compute()
    compute_time = time.time() - start_time

    print(f"✅ Pipeline completed in {compute_time:.2f}s")
    print(f"Result shape: {final_result.shape}")
    print(f"Result dtype: {final_result.dtype}")


def demonstrate_out_of_core_processing():
    """Demonstrate processing arrays larger than memory using file-backed arrays."""

    print("\n" + "=" * 60)
    print("💾 Out-of-Core Processing Demo")
    print("=" * 60)

    # Create a large array and save to disk
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save a moderately large array
        shape = (8, 12, 20, 600, 400)
        print(f"Creating array with shape: {shape}")

        # Create in chunks to avoid memory issues
        chunks = (2, 4, 5, 600, 400)
        dask_array = da.random.randint(0, 100, size=shape, chunks=chunks, dtype=np.uint8)

        print(f"Array size: {dask_array.nbytes / 1e6:.1f} MB")
        print(f"Chunk structure: {dask_array.chunks}")

        # Save to file
        print("💾 Saving to disk...")
        da.to_npy_stack(temp_dir / "array_stack", dask_array)

        # Read back as memory-mapped array
        print("📖 Reading as memory-mapped array...")
        mmap_array = da.from_npy_stack(temp_dir / "array_stack")

        print(f"Memory-mapped array loaded: {mmap_array.shape}")

        # Process using map_blocks (processes chunks from disk)
        def simple_filter(chunk):
            """Simple processing function."""
            return gaussian_filter(chunk.astype(np.float32), sigma=1.0)

        print("⚡ Processing with map_blocks...")
        start_time = time.time()

        result = da.map_blocks(simple_filter, mmap_array, dtype=np.float32, chunks=mmap_array.chunks)

        # Compute only a small portion to demonstrate
        sample_result = result[0, 0, 0, :100, :100].compute()
        process_time = time.time() - start_time

        print(f"✅ Sample processing completed in {process_time:.2f}s")
        print(f"Sample result shape: {sample_result.shape}")
        print("🎯 Full array could be processed chunk-by-chunk without loading into memory!")


if __name__ == "__main__":
    print("Dask map_blocks Examples for Large Array Processing")
    print("=" * 60)

    try:
        demonstrate_dask_approaches()
        demonstrate_lazy_evaluation()
        demonstrate_out_of_core_processing()

        print("\n" + "=" * 60)
        print("🎯 KEY TAKEAWAYS:")
        print("=" * 60)
        print("1. 🚀 Dask map_blocks is perfect for very large arrays")
        print("2. 💾 Can process arrays larger than available RAM")
        print("3. ⚡ Lazy evaluation allows building complex pipelines")
        print("4. 🔧 Smart chunking keeps 2D slices intact for efficiency")
        print("5. 📊 Excellent for distributed computing and scaling")
        print("\n🎯 WHEN TO USE DASK:")
        print("- Arrays > 1GB in size")
        print("- Need to process thousands of 2D slices")
        print("- Want to build processing pipelines")
        print("- Working with file-backed arrays")
        print("- Need distributed computing capabilities")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Make sure you have dask installed: pip install dask[array]")
