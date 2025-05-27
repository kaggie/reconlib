import torch
import numpy as np # For np.pi
from reconlib.nufft import NUFFT2D, NUFFT3D # Ensure this import path is correct

def simulate_b0_corrupted_data(
    phantom: torch.Tensor,
    k_trajectory: torch.Tensor,
    b0_map_hz: torch.Tensor,
    time_per_kspace_point: torch.Tensor,
    oversamp_factor: tuple[float,...],
    kb_J: tuple[int,...],
    kb_alpha: tuple[float,...],
    # Ld can be calculated internally if not provided or based on image_shape & oversamp_factor
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Simulates k-space data acquisition including B0 inhomogeneity effects
    by applying phase errors at each k-space point's acquisition time.
    This is computationally intensive as it notionally performs one NUFFT per k-point.
    """
    dev = torch.device(device)
    phantom = phantom.to(dev)
    k_trajectory = k_trajectory.to(dev)
    b0_map_hz = b0_map_hz.to(dev)
    time_per_kspace_point = time_per_kspace_point.to(dev)

    if not phantom.is_complex():
        phantom = phantom.to(torch.complex64)
    if b0_map_hz.shape != phantom.shape:
        raise ValueError("B0 map shape must match phantom shape.")
    if time_per_kspace_point.shape[0] != k_trajectory.shape[0]:
        raise ValueError("time_per_kspace_point length must match k_trajectory length.")

    num_k_points = k_trajectory.shape[0]
    output_kspace = torch.zeros(num_k_points, dtype=torch.complex64, device=dev)
    
    b0_map_rad_s = 2 * np.pi * b0_map_hz

    dimensionality = phantom.ndim
    nufft_class = None
    if dimensionality == 2:
        nufft_class = NUFFT2D
    elif dimensionality == 3:
        nufft_class = NUFFT3D
    else:
        raise ValueError(f"Unsupported phantom dimensionality: {dimensionality}")

    # Calculate Ld based on image_shape and oversamp_factor for NUFFT internal use
    current_Ld = tuple(int(ims * ovs) for ims, ovs in zip(phantom.shape, oversamp_factor))

    print_interval = num_k_points // 10 if num_k_points > 10 else 1
    if num_k_points == 0: # Handle case with no k-space points
        print("Simulation of B0-corrupted k-space data complete (0 points).")
        return output_kspace
        
    for i in range(num_k_points):
        t_i = time_per_kspace_point[i]
        k_i_trajectory_point = k_trajectory[i:i+1] # NUFFT expects (num_points_in_batch, ndims)

        phase_error = torch.exp(-1j * b0_map_rad_s * t_i) # b0_map_rad_s is (image_shape)
        phantom_at_time_t_i = phantom * phase_error     # Element-wise multiplication
        
        # Instantiate NUFFT for a single k-space point.
        # This is inefficient but simulates the continuous time evolution accurately.
        nufft_for_point_i = nufft_class(
            k_trajectory=k_i_trajectory_point,
            image_shape=phantom.shape,
            oversamp_factor=oversamp_factor,
            kb_J=kb_J,
            kb_alpha=kb_alpha,
            Ld=current_Ld, 
            device=device
        )
        # The forward method of NUFFT2D/3D should return a tensor of k-space values.
        # Since k_i_trajectory_point has only one point, the output should be a single k-space value.
        kspace_val_tensor = nufft_for_point_i.forward(phantom_at_time_t_i)
        output_kspace[i] = kspace_val_tensor[0] # Assuming forward returns (1,) tensor
        
        if (i + 1) % print_interval == 0:
             print(f"Simulating B0-corrupted k-space point {i+1}/{num_k_points}")
    
    print(f"Simulation of B0-corrupted k-space data complete.")
    return output_kspace
