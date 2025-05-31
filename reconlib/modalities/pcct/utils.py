import torch
import numpy as np
from typing import List, Optional
import torch.nn.functional as F # For conv1d
import traceback # For printing exceptions in __main__

def generate_pcct_phantom_material_maps(
    image_shape: tuple[int, int],
    material_attenuations: dict[str, float], # E.g. {'water': 0.2, 'bone': 0.5} at reference energy
    num_features_per_material: int = 1,
    device='cpu'
) -> dict[str, torch.Tensor]:
    """
    Generates a phantom with multiple material types, each having a base attenuation.
    Outputs separate maps for each material's concentration/presence (0 or 1).
    """
    material_maps = {}
    all_occupied_mask = torch.zeros(image_shape, dtype=torch.bool, device=device)

    for mat_name, _ in material_attenuations.items():
        material_map = torch.zeros(image_shape, dtype=torch.float32, device=device)
        for _ in range(num_features_per_material):
            while True:
                temp_mask = torch.zeros(image_shape, dtype=torch.bool, device=device)
                is_rect = np.random.rand() > 0.3
                if is_rect:
                    w = np.random.randint(image_shape[1] // 8, image_shape[1] // 3)
                    h = np.random.randint(image_shape[0] // 8, image_shape[0] // 3)
                    x0 = np.random.randint(0, image_shape[1] - w)
                    y0 = np.random.randint(0, image_shape[0] - h)
                    temp_mask[y0:y0+h, x0:x0+w] = True
                else: # Circle
                    radius = np.random.randint(min(image_shape) // 10, min(image_shape) // 4)
                    cx = np.random.randint(radius, image_shape[1] - radius)
                    cy = np.random.randint(radius, image_shape[0] - radius)
                    yy, xx = torch.meshgrid(torch.arange(image_shape[0], device=device),
                                            torch.arange(image_shape[1], device=device), indexing='ij')
                    mask_circle = (xx - cx)**2 + (yy - cy)**2 < radius**2
                    temp_mask[mask_circle] = True
                if not torch.any(temp_mask & all_occupied_mask):
                    material_map[temp_mask] = 1.0
                    all_occupied_mask |= temp_mask
                    break
        material_maps[mat_name] = material_map
    return material_maps

def combine_material_maps_to_mu_ref(
    material_maps: dict[str, torch.Tensor],
    material_attenuations_ref: dict[str, float],
    device='cpu'
) -> torch.Tensor:
    if not material_maps:
        raise ValueError("Material maps dictionary is empty.")
    first_map_shape = next(iter(material_maps.values())).shape
    mu_ref_map = torch.zeros(first_map_shape, dtype=torch.float32, device=device)
    for mat_name, presence_map in material_maps.items():
        if mat_name not in material_attenuations_ref:
            raise ValueError(f"Attenuation for material '{mat_name}' not found in reference dictionary.")
        mu_ref_map += presence_map * material_attenuations_ref[mat_name]
    return mu_ref_map

def plot_pcct_results(mu_ref_true, measured_counts_stack, mu_ref_recon, energy_bin_idx_to_show=0):
    print("plot_pcct_results: Placeholder - Plotting not implemented.")
    if mu_ref_true is not None: print(f"  True mu_ref_map shape: {mu_ref_true.shape}")
    if measured_counts_stack is not None:
        print(f"  Measured counts stack shape: {measured_counts_stack.shape}")
        print(f"  (Showing sinogram for energy bin index: {energy_bin_idx_to_show})")
    if mu_ref_recon is not None: print(f"  Recon mu_ref_map shape: {mu_ref_recon.shape}")

def get_pcct_energy_scaling_factors(
    energy_bins_keV: list[tuple[float,float]],
    material_attenuation_model: str = 'water_simplified',
    reference_energy_keV: float = 60.0,
    device='cpu',
    dtype=torch.float32
) -> torch.Tensor:
    num_bins = len(energy_bins_keV)
    scaling_factors = torch.ones(num_bins, device=device, dtype=dtype)
    if material_attenuation_model == 'water_simplified':
        for i, (low_e, high_e) in enumerate(energy_bins_keV):
            mean_bin_energy = (low_e + high_e) / 2.0
            if mean_bin_energy > 0 and reference_energy_keV > 0:
                factor = (reference_energy_keV / mean_bin_energy)**3
                scaling_factors[i] = torch.clamp(torch.tensor(factor, device=device, dtype=dtype), 0.1, 10.0)
            else: scaling_factors[i] = 1.0
    elif material_attenuation_model == 'constant': pass
    else: print(f"Warning: Unknown material_attenuation_model '{material_attenuation_model}'. Using constant scaling factors (1.0).")
    return scaling_factors

def estimate_scatter_sinogram_kernel_based(
    primary_sinogram_stack: torch.Tensor,
    scatter_kernels: List[torch.Tensor],
    scatter_fraction_estimates: Optional[List[float]] = None
) -> torch.Tensor:
    if not isinstance(primary_sinogram_stack, torch.Tensor):
        raise TypeError("primary_sinogram_stack must be a PyTorch Tensor.")
    if not all(isinstance(k, torch.Tensor) for k in scatter_kernels):
        raise TypeError("All elements in scatter_kernels must be PyTorch Tensors.")

    num_bins, num_angles, num_detector_pixels = primary_sinogram_stack.shape

    if len(scatter_kernels) != num_bins:
        raise ValueError(f"Length of scatter_kernels ({len(scatter_kernels)}) must match num_bins ({num_bins}).")

    if scatter_fraction_estimates is not None:
        if len(scatter_fraction_estimates) != num_bins:
            raise ValueError(f"Length of scatter_fraction_estimates ({len(scatter_fraction_estimates)}) must match num_bins ({num_bins}).")

    scatter_estimate_stack = torch.zeros_like(primary_sinogram_stack)

    for b in range(num_bins):
        sinogram_bin = primary_sinogram_stack[b, :, :]
        kernel_1d = scatter_kernels[b].to(sinogram_bin.device, dtype=sinogram_bin.dtype)

        if kernel_1d.ndim != 1 or kernel_1d.numel() == 0:
            raise ValueError(f"Scatter kernel for bin {b} must be a 1D tensor with at least one element. Got shape {kernel_1d.shape}")

        kernel_1d_norm = kernel_1d / (torch.sum(kernel_1d) + 1e-9)
        input_for_conv = sinogram_bin.unsqueeze(1)
        kernel_for_conv = kernel_1d_norm.view(1, 1, -1)
        padding = (kernel_for_conv.shape[2] - 1) // 2
        convolved_sinogram = F.conv1d(input_for_conv, kernel_for_conv, padding=padding)
        convolved_sinogram = convolved_sinogram.squeeze(1)

        scale_factor = 1.0
        if scatter_fraction_estimates is not None:
            scale_factor = scatter_fraction_estimates[b]
            scatter_estimate_stack[b, :, :] = convolved_sinogram * scale_factor
        else:
            scatter_estimate_stack[b, :, :] = convolved_sinogram
    return scatter_estimate_stack

if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_utils = (64,64)

    materials_ref_mu = {'water': 0.2, 'bone': 0.5, 'contrast': 1.0}
    material_component_maps = generate_pcct_phantom_material_maps(
        img_s_utils, materials_ref_mu, num_features_per_material=1, device=dev_utils
    )
    print(f"Generated material maps for: {list(material_component_maps.keys())}")
    for name, map_tensor in material_component_maps.items():
        assert map_tensor.shape == img_s_utils
        print(f"  Map '{name}' shape: {map_tensor.shape}")

    mu_ref_combined = combine_material_maps_to_mu_ref(material_component_maps, materials_ref_mu, device=dev_utils)
    assert mu_ref_combined.shape == img_s_utils
    print(f"Combined mu_ref map generated: {mu_ref_combined.shape}")

    dummy_counts = torch.randn((3, 100, 128), device=dev_utils)
    plot_pcct_results(mu_ref_combined, dummy_counts, mu_ref_combined*0.8)

    print("\nTesting get_pcct_energy_scaling_factors...")
    ebins = [(20.,50.), (50.,80.), (80.,120.)]
    scales_water = get_pcct_energy_scaling_factors(ebins, 'water_simplified', device=dev_utils)
    print(f"Water-like scales for {ebins} (ref 60keV): {scales_water}")
    scales_const = get_pcct_energy_scaling_factors(ebins, 'constant', device=dev_utils)
    print(f"Constant scales for {ebins}: {scales_const}")
    print("get_pcct_energy_scaling_factors tests completed.")

    print("\n--- Testing estimate_scatter_sinogram_kernel_based ---")
    num_bins_scat = 2
    num_angles_scat = 32
    num_dets_scat = 64
    primary_sino = torch.zeros(num_bins_scat, num_angles_scat, num_dets_scat, device=dev_utils, dtype=torch.float32)
    primary_sino[:, :, num_dets_scat//4 : num_dets_scat*3//4] = 100.0
    kernel1 = torch.tensor([0.2, 0.6, 0.2], device=dev_utils, dtype=torch.float32)
    kernel2 = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=dev_utils, dtype=torch.float32)
    kernels_list = [kernel1, kernel2]
    spr_values = [0.1, 0.15]

    try:
        scatter_estimate_spr = estimate_scatter_sinogram_kernel_based(primary_sino, kernels_list, spr_values)
        assert scatter_estimate_spr.shape == primary_sino.shape, \
            f"SPR mode: Shape mismatch. Expected {primary_sino.shape}, Got {scatter_estimate_spr.shape}"
        for b in range(num_bins_scat):
            expected_scatter_sum = torch.sum(primary_sino[b]) * spr_values[b]
            actual_scatter_sum = torch.sum(scatter_estimate_spr[b])
            print(f"Bin {b} (SPR mode): Sum primary: {torch.sum(primary_sino[b]).item():.2f}, Sum estimated scatter: {actual_scatter_sum.item():.2f}, Expected scatter sum (approx): {expected_scatter_sum.item():.2f}")
            assert torch.allclose(actual_scatter_sum, expected_scatter_sum, rtol=0.1), \
                f"Bin {b} SPR mode: Scatter sum mismatch. Actual: {actual_scatter_sum.item()}, Expected: {expected_scatter_sum.item()}"
        print("  SPR mode test passed.")

        scatter_estimate_no_frac = estimate_scatter_sinogram_kernel_based(
            primary_sino.narrow(0,0,1), [kernel1], None
        )
        assert scatter_estimate_no_frac.shape == primary_sino.narrow(0,0,1).shape, \
            "No-fraction mode: Shape mismatch."
        expected_sum_no_frac = torch.sum(primary_sino[0])
        actual_sum_no_frac = torch.sum(scatter_estimate_no_frac[0])
        print(f"Bin 0 (no fraction mode): Sum primary: {torch.sum(primary_sino[0]).item():.2f}, Sum convolved output: {actual_sum_no_frac.item():.2f}, Expected sum (approx): {expected_sum_no_frac.item():.2f}")
        assert torch.allclose(actual_sum_no_frac, expected_sum_no_frac, rtol=0.1), \
             f"No-fraction mode: Sum mismatch. Actual: {actual_sum_no_frac.item()}, Expected: {expected_sum_no_frac.item()}"
        print("  No-fraction mode test passed.")
        print("estimate_scatter_sinogram_kernel_based tests passed.")
    except Exception as e:
        print(f"Error during estimate_scatter_sinogram_kernel_based tests: {e}")
        traceback.print_exc()

    print("\nAll PCCT utils checks completed.")
