import torch
import time
import numpy as np
import math

# Adjust path to import from reconlib if not installed
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.operators import NUFFTOperator

def setup_nufft_op(dim, device='cpu', image_size=128, k_points_factor=2):
    # Helper to set up a NUFFTOperator for benchmarking.
    if dim == 2:
        image_shape = (image_size, image_size)
        # For 2D, k_points_factor determines total points relative to N^2
        num_k_points = image_size * image_size * k_points_factor 
    elif dim == 3:
        # For 3D, use a smaller base for image_size to keep it manageable
        # e.g., if image_size=64, true shape is (32, 64, 64)
        actual_z_size = image_size // 2 if image_size > 16 else image_size # Ensure Z is not too large
        image_shape = (actual_z_size, image_size, image_size)
        # For 3D, k_points_factor determines total points relative to N_z*N_y*N_x
        num_k_points = actual_z_size * image_size * image_size * k_points_factor // 2 # Adjusted for 3D
    else:
        raise ValueError("Dimension must be 2 or 3")

    # NUFFT implementation expects k-trajectory to be in [-0.5, 0.5] range
    k_traj = torch.rand(num_k_points, dim, dtype=torch.float32, device=device) - 0.5

    # Standard MIRT-like parameters
    oversamp_factor = tuple([2.0] * dim)
    kb_J = tuple([4] * dim) # Kernel width
    kb_alpha = tuple([2.34 * J for J in kb_J]) # Shape parameter
    Ld = tuple([2**8] * dim) # Smaller table oversampling for benchmarks for speed
    # Kd (oversampled grid dimensions) will be calculated by NUFFT operator if None, or can be explicit:
    Kd = tuple(int(N * os) for N, os in zip(image_shape, oversamp_factor))
    
    nufft_op = NUFFTOperator(
        k_trajectory=k_traj,
        image_shape=image_shape,
        oversamp_factor=oversamp_factor,
        kb_J=kb_J,
        kb_alpha=kb_alpha,
        Ld=Ld,
        Kd=Kd,
        device=device,
        # nufft_type_3d is only relevant for dim=3, NUFFTOperator handles default
        nufft_type_3d='table' if dim == 3 else None 
    )
    return nufft_op, image_shape # Return image_shape for clarity in print

def benchmark_op(nufft_op, op_image_shape, dim_for_print, num_repeats=10):
    # Benchmarks forward and adjoint operations.
    device = nufft_op.device
    
    img_data = torch.randn(op_image_shape, dtype=torch.complex64, device=device)
    ksp_data_shape = (nufft_op.k_trajectory.shape[0],)
    ksp_data = torch.randn(ksp_data_shape, dtype=torch.complex64, device=device)

    # Warm-up runs
    _ = nufft_op.op(img_data)
    _ = nufft_op.op_adj(ksp_data)
    if device.type == 'cuda': # Check device type
        torch.cuda.synchronize()

    # Forward op benchmark
    start_time = time.time()
    for _ in range(num_repeats):
        _ = nufft_op.op(img_data)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    avg_forward_time = (end_time - start_time) / num_repeats
    print(f"  Forward Op ({dim_for_print}D, shape={op_image_shape}): {avg_forward_time*1000:.2f} ms / op")

    # Adjoint op benchmark
    start_time = time.time()
    for _ in range(num_repeats):
        _ = nufft_op.op_adj(ksp_data)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    avg_adjoint_time = (end_time - start_time) / num_repeats
    print(f"  Adjoint Op ({dim_for_print}D, shape={op_image_shape}): {avg_adjoint_time*1000:.2f} ms / op")
    
    return avg_forward_time, avg_adjoint_time

if __name__ == '__main__':
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running NUFFT benchmarks on device: {test_device}")
    num_repeats_benchmark = 10

    # 2D Benchmarks
    print("\n--- Benchmarking 2D NUFFTOperator (Table based) ---")
    nufft_op_2d_bench, shape_2d = setup_nufft_op(dim=2, device=test_device, image_size=128)
    benchmark_op(nufft_op_2d_bench, shape_2d, "2D", num_repeats=num_repeats_benchmark)

    nufft_op_2d_bench_large, shape_2d_large = setup_nufft_op(dim=2, device=test_device, image_size=256)
    benchmark_op(nufft_op_2d_bench_large, shape_2d_large, "2D", num_repeats=num_repeats_benchmark)

    # 3D Benchmarks (Table based)
    print("\n--- Benchmarking 3D NUFFTOperator (Table based) ---")
    # Smaller size for 3D table due to potential slowness of Python loops
    nufft_op_3d_table_bench, shape_3d_table = setup_nufft_op(dim=3, device=test_device, image_size=32) 
    benchmark_op(nufft_op_3d_table_bench, shape_3d_table, "3D Table", num_repeats=num_repeats_benchmark)
    
    nufft_op_3d_table_bench_larger, shape_3d_table_larger = setup_nufft_op(dim=3, device=test_device, image_size=64) # Slightly larger 3D
    benchmark_op(nufft_op_3d_table_bench_larger, shape_3d_table_larger, "3D Table", num_repeats=max(1, num_repeats_benchmark // 2)) # Fewer repeats if larger


    # 3D Benchmarks (Direct NDFT) - Will be slower, use very small size
    print("\n--- Benchmarking 3D NUFFTOperator (Direct NDFT) ---")
    image_size_direct_3d = 16 
    num_k_points_direct_3d = image_size_direct_3d**3 // 4 
    
    k_traj_direct_3d = torch.rand(num_k_points_direct_3d, 3, dtype=torch.float32, device=test_device) - 0.5
    image_shape_direct_3d = (image_size_direct_3d, image_size_direct_3d, image_size_direct_3d)
    
    oversamp_factor_direct = (1.0, 1.0, 1.0) 
    kb_J_direct = (2, 2, 2) # Minimal kernel, less relevant for direct but NUFFTOperator expects it
    kb_alpha_direct = tuple(2.34 * J for J in kb_J_direct)
    Ld_direct = (2**6, 2**6, 2**6) # Not used by direct, but expected by NUFFTOperator
    Kd_direct = image_shape_direct_3d

    nufft_op_3d_direct_bench = NUFFTOperator(
        k_trajectory=k_traj_direct_3d,
        image_shape=image_shape_direct_3d,
        oversamp_factor=oversamp_factor_direct,
        kb_J=kb_J_direct,
        kb_alpha=kb_alpha_direct,
        Ld=Ld_direct,
        Kd=Kd_direct, 
        device=test_device,
        nufft_type_3d='direct' 
    )
    benchmark_op(nufft_op_3d_direct_bench, image_shape_direct_3d, "3D Direct", num_repeats=3)

    print("\nAll benchmarks completed.")
