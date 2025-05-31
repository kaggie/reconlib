import torch
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any

# Try-except for NUFFTOperator for dynamic data generation
try:
    from reconlib.operators import NUFFTOperator
except ImportError:
    print("Warning: reconlib.operators.NUFFTOperator not found. Using a placeholder for toy_datasets.")
    class NUFFTOperator: # Placeholder
        def __init__(self, k_trajectory: torch.Tensor, image_shape: Tuple[int, ...], device: str = 'cpu', **kwargs: Any):
            self.k_trajectory = k_trajectory
            self.image_shape = image_shape
            self.device = torch.device(device)
            print(f"MockNUFFTOperator (toy_datasets) initialized for image shape {image_shape} on device {self.device}.")
            # Store kwargs if needed, e.g. oversamp_factor, kb_J, kb_alpha, Ld
            self.kwargs = kwargs

        def op(self, x: torch.Tensor) -> torch.Tensor: # Expects (H, W) or other image_shape
            if x.shape != self.image_shape:
                raise ValueError(f"MockNUFFTOperator.op input shape mismatch. Expected {self.image_shape}, got {x.shape}")
            num_k_points = self.k_trajectory.shape[0]
            # Simulate some k-space data, make it complex
            output_kspace = torch.randn(num_k_points, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(x))
            return output_kspace
        
        # op_adj might not be needed for these stubs if only forward is used for k-space generation
        def op_adj(self, y: torch.Tensor) -> torch.Tensor:
            if y.ndim != 1 or y.shape[0] != self.k_trajectory.shape[0]:
                 raise ValueError(f"MockNUFFTOperator.op_adj input shape mismatch. Expected ({self.k_trajectory.shape[0]},), got {y.shape}")
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(y))


def generate_dynamic_phantom_data(
    image_size: Tuple[int, int] = (64, 64), 
    num_frames: int = 10, 
    num_coils: int = 4, 
    device: torch.device = torch.device('cpu'), 
    k_trajectory: Optional[torch.Tensor] = None, 
    nufft_op_class: Optional[Callable[..., NUFFTOperator]] = None, 
    nufft_op_params: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a simple dynamic phantom, corresponding multi-coil k-space data,
    and dummy sensitivity maps.

    Args:
        image_size: Spatial dimensions of the phantom, e.g., (H, W).
        num_frames: Number of temporal frames.
        num_coils: Number of coils.
        device: PyTorch device.
        k_trajectory: Optional k-space trajectory for NUFFT. Shape (num_kpoints, num_dims).
                      If None, Cartesian FFT is used.
        nufft_op_class: The class for the NUFFT operator (e.g., reconlib.operators.NUFFTOperator).
                        Required if k_trajectory is provided.
        nufft_op_params: Dictionary of parameters for instantiating nufft_op_class.
                         Should include image_shape, device, and k_trajectory if not passed directly.

    Returns:
        A tuple containing:
            - dynamic_phantom_4d: (num_frames, H, W), complex64
            - dynamic_kspace_multi_coil_4d: (num_frames, num_coils, k_space_samples_dim), complex64
            - sensitivity_maps_3d: (num_coils, H, W), complex64
    """
    H, W = image_size
    dynamic_phantom_4d = torch.zeros((num_frames,) + image_size, dtype=torch.complex64, device=device)

    # Create a simple base phantom (e.g., a disk)
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_x, center_y = W // 2, H // 2
    
    for t in range(num_frames):
        current_frame_phantom = torch.zeros(image_size, dtype=torch.complex64, device=device)
        # Modify the base phantom: e.g., change radius of disk
        radius = min(H, W) // 4 + (t * 2) # Radius changes over time
        mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 < radius**2
        current_frame_phantom[mask] = 1.0 + 0.1 * t # Intensity changes over time
        dynamic_phantom_4d[t] = current_frame_phantom

    # Generate dummy sensitivity maps
    sensitivity_maps_3d = torch.rand((num_coils,) + image_size, dtype=torch.complex64, device=device) + 1e-3 # Add small epsilon
    # Normalize maps (sum-of-squares = 1 along coil dimension)
    sos_maps = torch.sqrt(torch.sum(torch.abs(sensitivity_maps_3d)**2, dim=0, keepdim=True))
    sensitivity_maps_3d = sensitivity_maps_3d / (sos_maps + 1e-8) # Avoid division by zero

    k_space_samples_dim: int
    if k_trajectory is not None and nufft_op_class is not None:
        if nufft_op_params is None:
            nufft_op_params = {}
        # Ensure required params are present for NUFFTOperator
        nufft_op_params.update({'k_trajectory': k_trajectory, 'image_shape': image_size, 'device': str(device)})
        # Instantiate once if params don't change per coil/frame (typical)
        try:
            nufft_op = nufft_op_class(**nufft_op_params)
            k_space_samples_dim = nufft_op.k_trajectory.shape[0]
        except Exception as e:
            print(f"Error instantiating NUFFT operator {nufft_op_class} with params {nufft_op_params}: {e}")
            print("Falling back to Cartesian FFT for dynamic data generation.")
            k_trajectory = None # Force Cartesian fallback
    
    if k_trajectory is None or nufft_op_class is None: # Fallback or default to Cartesian
        k_space_samples_dim = image_size[0] * image_size[1]

    dynamic_kspace_multi_coil_4d = torch.zeros((num_frames, num_coils, k_space_samples_dim), dtype=torch.complex64, device=device)

    for t in range(num_frames):
        current_phantom_frame = dynamic_phantom_4d[t]
        coil_images = sensitivity_maps_3d * current_phantom_frame.unsqueeze(0) # (num_coils, H, W)

        if k_trajectory is not None and nufft_op_class is not None:
            # Re-use or re-init nufft_op; for this stub, assume re-init or it's stateless for op
            # This part assumes nufft_op.op works on single coil images (H,W)
            nufft_op = nufft_op_class(**nufft_op_params) # Re-instantiate for safety if params could change (they don't here)
            for c in range(num_coils):
                dynamic_kspace_multi_coil_4d[t, c] = nufft_op.op(coil_images[c])
        else: # Simple Cartesian FFT
            kspace_coils = torch.fft.fftshift(torch.fft.fft2(coil_images, norm='ortho'), dims=(-2, -1))
            dynamic_kspace_multi_coil_4d[t] = kspace_coils.reshape(num_coils, -1)
            
    return dynamic_phantom_4d, dynamic_kspace_multi_coil_4d, sensitivity_maps_3d


def generate_nlinv_data_stubs(
    image_size: Tuple[int, int] = (64, 64), 
    acs_region_shape: Tuple[int, int] = (16, 16), 
    num_coils: int = 8, 
    device: torch.device = torch.device('cpu'), 
    undersampling_factor: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates stub data for NLINV-like scenarios.

    Args:
        image_size: Spatial dimensions (H, W).
        acs_region_shape: Shape of the ACS region (ACS_H, ACS_W).
        num_coils: Number of coils.
        device: PyTorch device.
        undersampling_factor: Factor for undersampling the outer k-space region.
                              E.g., 2.0 means sample 1/2 of the outer points.

    Returns:
        A tuple containing:
            - phantom: (H, W), complex64
            - sensitivity_maps: (num_coils, H, W), complex64
            - full_acs_kspace: (num_coils, ACS_H, ACS_W), complex64
            - undersampled_kspace_coils: (num_coils, H, W), complex64 (k-space domain)
            - undersampling_mask: (H, W), bool (True where k-space is sampled)
    """
    H, W = image_size
    ACS_H, ACS_W = acs_region_shape

    # Generate a simple phantom
    phantom = torch.zeros(image_size, dtype=torch.complex64, device=device)
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_x, center_y = W // 2, H // 2
    radius = min(H,W) // 3
    phantom[(x_coords - center_x)**2 + (y_coords - center_y)**2 < radius**2] = 1.0
    phantom[(x_coords - center_x + radius//2)**2 + (y_coords - center_y)**2 < (radius//2)**2] = 0.5


    # Generate dummy sensitivity maps
    sensitivity_maps = torch.rand((num_coils,) + image_size, dtype=torch.complex64, device=device) + 1e-3
    sos_maps = torch.sqrt(torch.sum(torch.abs(sensitivity_maps)**2, dim=0, keepdim=True))
    sensitivity_maps = sensitivity_maps / (sos_maps + 1e-8)

    # Simulate full k-space (Cartesian FFT)
    coil_images = sensitivity_maps * phantom.unsqueeze(0)
    full_kspace_coils = torch.fft.fftshift(torch.fft.fft2(coil_images, norm='ortho'), dims=(-2, -1))

    # Extract full_acs_kspace
    center_y_start = (H - ACS_H) // 2
    center_y_end = center_y_start + ACS_H
    center_x_start = (W - ACS_W) // 2
    center_x_end = center_x_start + ACS_W
    
    full_acs_kspace = full_kspace_coils[:, center_y_start:center_y_end, center_x_start:center_x_end].clone()

    # Create undersampled_outer_kspace_mask
    mask = torch.zeros_like(full_kspace_coils[0], dtype=torch.bool, device=device) # Mask for one coil
    
    # ACS region is fully sampled
    mask[center_y_start:center_y_end, center_x_start:center_x_end] = True
    
    # Undersample outer region
    outer_region_mask = torch.ones_like(mask, dtype=torch.bool)
    outer_region_mask[center_y_start:center_y_end, center_x_start:center_x_end] = False
    
    num_outer_points = outer_region_mask.sum().item()
    num_samples_outer = int(num_outer_points / undersampling_factor)
    
    outer_indices = torch.where(outer_region_mask.flatten())[0]
    permuted_outer_indices = outer_indices[torch.randperm(len(outer_indices), device=device)]
    selected_outer_indices = permuted_outer_indices[:num_samples_outer]
    
    mask.flatten()[selected_outer_indices] = True
    
    undersampled_kspace_coils = full_kspace_coils * mask.unsqueeze(0) # Apply mask to all coils

    return phantom, sensitivity_maps, full_acs_kspace, undersampled_kspace_coils, mask


if __name__ == '__main__':
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Testing Toy Dataset Generation on {test_device} ---")

    # Test generate_dynamic_phantom_data (Cartesian FFT path)
    print("\nTesting generate_dynamic_phantom_data (Cartesian FFT)...")
    dyn_phantom, dyn_kspace, dyn_smaps = generate_dynamic_phantom_data(
        image_size=(32, 32), num_frames=3, num_coils=2, device=test_device
    )
    print(f"  Dynamic Phantom shape: {dyn_phantom.shape}, dtype: {dyn_phantom.dtype}")
    print(f"  Dynamic K-space shape: {dyn_kspace.shape}, dtype: {dyn_kspace.dtype}")
    print(f"  Dynamic SMaps shape: {dyn_smaps.shape}, dtype: {dyn_smaps.dtype}")
    assert dyn_phantom.shape == (3, 32, 32)
    assert dyn_kspace.shape == (3, 2, 32*32)
    assert dyn_smaps.shape == (2, 32, 32)
    assert dyn_phantom.device == test_device
    assert dyn_kspace.device == test_device
    assert dyn_smaps.device == test_device

    # Test generate_dynamic_phantom_data (NUFFT path with Mock)
    print("\nTesting generate_dynamic_phantom_data (NUFFT with Mock)...")
    num_kpoints_nufft = 128
    mock_traj = torch.rand(num_kpoints_nufft, 2, device=test_device) # (K, ndims)
    mock_nufft_params = {'oversamp_factor': (1.5,1.5)} # Dummy params for mock
    
    dyn_phantom_nufft, dyn_kspace_nufft, dyn_smaps_nufft = generate_dynamic_phantom_data(
        image_size=(30, 30), num_frames=2, num_coils=1, device=test_device,
        k_trajectory=mock_traj,
        nufft_op_class=NUFFTOperator, # Using the placeholder/mock defined in this file if actual not found
        nufft_op_params=mock_nufft_params
    )
    print(f"  Dynamic Phantom (NUFFT) shape: {dyn_phantom_nufft.shape}, dtype: {dyn_phantom_nufft.dtype}")
    print(f"  Dynamic K-space (NUFFT) shape: {dyn_kspace_nufft.shape}, dtype: {dyn_kspace_nufft.dtype}")
    print(f"  Dynamic SMaps (NUFFT) shape: {dyn_smaps_nufft.shape}, dtype: {dyn_smaps_nufft.dtype}")
    assert dyn_phantom_nufft.shape == (2, 30, 30)
    assert dyn_kspace_nufft.shape == (2, 1, num_kpoints_nufft)
    assert dyn_smaps_nufft.shape == (1, 30, 30)
    assert torch.sum(torch.abs(dyn_kspace_nufft)) > 0 # Check if k-space is not all zero

    # Test generate_nlinv_data_stubs
    print("\nTesting generate_nlinv_data_stubs...")
    nlinv_phantom, nlinv_smaps, nlinv_acs, nlinv_uskspace, nlinv_mask = generate_nlinv_data_stubs(
        image_size=(32, 32), acs_region_shape=(10,10), num_coils=4, 
        device=test_device, undersampling_factor=2.5
    )
    print(f"  NLINV Phantom shape: {nlinv_phantom.shape}, dtype: {nlinv_phantom.dtype}")
    print(f"  NLINV SMaps shape: {nlinv_smaps.shape}, dtype: {nlinv_smaps.dtype}")
    print(f"  NLINV ACS K-space shape: {nlinv_acs.shape}, dtype: {nlinv_acs.dtype}")
    print(f"  NLINV Undersampled K-space shape: {nlinv_uskspace.shape}, dtype: {nlinv_uskspace.dtype}")
    print(f"  NLINV Undersampling Mask shape: {nlinv_mask.shape}, dtype: {nlinv_mask.dtype}, "
          f"Sampled points: {nlinv_mask.sum()}")
    
    assert nlinv_phantom.shape == (32, 32)
    assert nlinv_smaps.shape == (4, 32, 32)
    assert nlinv_acs.shape == (4, 10, 10)
    assert nlinv_uskspace.shape == (4, 32, 32)
    assert nlinv_mask.shape == (32, 32)
    assert nlinv_mask[16,16] == True # Center of ACS should be True
    # Check if some points in undersampled k-space are zero due to mask
    assert (nlinv_uskspace.abs().sum() < (torch.abs(nlinv_smaps).sum() * nlinv_phantom.abs().sum() * 100)), \
           "Undersampled k-space seems fully sampled or too dense." # Heuristic check

    print("\nAll toy dataset generation tests completed.")
