import torch
import numpy as np
from typing import List, Optional, Dict # Added Dict
import torch.nn.functional as F # For conv1d
import traceback # For printing exceptions in __main__

from reconlib.modalities.pcct.operators import PCCTProjectorOperator # Added for new function

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

# --- New Function Definition ---
def simulate_flat_field_for_spectral_calibration(
    pcct_projector_config: Dict,
    reference_attenuation_value: Optional[float] = None,
    num_realizations: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Simulates flat-field measurements for PCCT spectral calibration purposes.
    This function runs the PCCTProjectorOperator with an ideal detector model
    (no spectral broadening, pile-up, charge sharing, or k-escape) using either
    an air scan or a uniform phantom.

    The output can be compared to real experimental flat-field data (which includes
    all detector non-idealities) to help infer the real detector's characteristics,
    or to validate source spectrum models if the detector is well-characterized.

    Args:
        pcct_projector_config (Dict): Configuration dictionary for PCCTProjectorOperator.
            Must include: 'image_shape', 'num_angles', 'num_detector_pixels',
            'energy_bins_keV', 'source_photons_per_bin', 'device'.
            Optional: 'energy_scaling_factors', 'add_poisson_noise'.
        reference_attenuation_value (Optional[float], optional): If provided, simulates a flat phantom
            with this uniform attenuation value (e.g., cm^-1). If None (default),
            simulates an air scan (zero attenuation).
        num_realizations (int, optional): Number of noisy simulation runs to perform and average.
            If num_realizations > 1, Poisson noise will be forcibly enabled for the simulations.
            If num_realizations == 1, the 'add_poisson_noise' setting from pcct_projector_config
            (or its default False) is used. Defaults to 1.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            'mean_counts_per_bin_spatial_avg': Tensor of shape (num_bins,), average counts per bin
                                               over all angles and detector pixels.
            'mean_sinogram_stack_per_bin': Tensor of shape (num_bins, num_angles, num_detector_pixels),
                                           the (averaged) full sinogram stack.
            'simulated_phantom_attenuation_value': The attenuation value used for the phantom (float).
    """
    required_keys = ['image_shape', 'num_angles', 'num_detector_pixels',
                     'energy_bins_keV', 'source_photons_per_bin', 'device']
    for key in required_keys:
        if key not in pcct_projector_config:
            raise ValueError(f"Missing required key in pcct_projector_config: {key}")

    projector_config_for_sim = pcct_projector_config.copy()

    # Ensure ideal detector for this calibration simulation
    projector_config_for_sim['spectral_resolution_keV'] = None
    projector_config_for_sim['pileup_parameters'] = None
    projector_config_for_sim['charge_sharing_kernel'] = None
    projector_config_for_sim['k_escape_probabilities'] = None # Assuming this is the correct param name

    # Handle Poisson noise based on num_realizations
    if num_realizations > 1:
        projector_config_for_sim['add_poisson_noise'] = True
    elif 'add_poisson_noise' not in projector_config_for_sim: # num_realizations == 1
        projector_config_for_sim['add_poisson_noise'] = False


    calib_projector = PCCTProjectorOperator(**projector_config_for_sim)

    img_shape = calib_projector.image_shape
    phantom_val = 0.0
    if reference_attenuation_value is not None:
        mu_phantom = torch.full(img_shape, reference_attenuation_value,
                                device=calib_projector.device, dtype=torch.float32)
        phantom_val = reference_attenuation_value
    else:
        mu_phantom = torch.zeros(img_shape, device=calib_projector.device, dtype=torch.float32)

    accumulated_counts_stack = torch.zeros(calib_projector.measurement_shape,
                                           device=calib_projector.device,
                                           dtype=torch.float32)

    for _ in range(num_realizations):
        current_counts_stack = calib_projector.op(mu_phantom)
        accumulated_counts_stack += current_counts_stack

    mean_sinogram_stack = accumulated_counts_stack / num_realizations

    mean_counts_per_bin_spatial_avg = torch.mean(mean_sinogram_stack, dim=(1, 2))

    return {
        'mean_counts_per_bin_spatial_avg': mean_counts_per_bin_spatial_avg,
        'mean_sinogram_stack_per_bin': mean_sinogram_stack,
        'simulated_phantom_attenuation_value': phantom_val
    }

# --- New Function Definition for Flux Scan ---
def simulate_flux_scan_for_pileup_calibration(
    base_pcct_projector_config: Dict,
    flux_levels: List[float],
    base_source_photons_per_bin: torch.Tensor,
    phantom: torch.Tensor,
    pileup_params_to_test: Optional[Dict] = None,
    num_realizations: int = 1
) -> List[Dict]:
    """
    Simulates PCCT measurements at various flux levels for pile-up calibration.

    This function repeatedly runs the PCCTProjectorOperator, scaling the source flux
    (source_photons_per_bin) by factors specified in `flux_levels`.
    It can simulate with or without specified pile-up parameters. Other advanced
    detector effects (spectral broadening, charge sharing, k-escape) are disabled
    to isolate flux-dependent effects like pile-up.

    The output data can be used to plot measured counts vs. incident flux (or a proxy)
    and fit pile-up model parameters by comparing simulated data (with pile-up model)
    to experimental measurements.

    Args:
        base_pcct_projector_config (Dict): Base configuration for PCCTProjectorOperator.
            Must include: 'image_shape', 'num_angles', 'num_detector_pixels',
            'energy_bins_keV', 'device'.
            Optional: 'energy_scaling_factors', 'add_poisson_noise'.
            `source_photons_per_bin` will be overridden by this function.
        flux_levels (List[float]): A list of scaling factors to apply to
                                   `base_source_photons_per_bin`.
        base_source_photons_per_bin (torch.Tensor): The baseline source photon counts
                                                    per bin before flux scaling.
        phantom (torch.Tensor): The phantom attenuation map (mu values).
        pileup_params_to_test (Optional[Dict], optional): Pile-up parameters to use for the simulation.
            If None (default), pile-up effect is disabled. If provided, should be a dict
            compatible with PCCTProjectorOperator's `pileup_parameters` (e.g.,
            {'method': 'paralyzable', 'dead_time_s': 200e-9, 'acquisition_time_s': 1e-3}).
        num_realizations (int, optional): Number of noisy simulation runs to perform and average
            for each flux level. If > 1, Poisson noise is forcibly enabled. Defaults to 1.

    Returns:
        List[Dict]: A list of dictionaries, one for each flux level. Each dictionary contains:
            'flux_scale': The flux scaling factor used.
            'simulated_source_photons_per_bin': Tensor of actual source photons per bin for this level (on CPU).
            'mean_sinogram_stack_per_bin': Tensor of shape (num_bins, num_angles, num_detector_pixels),
                                           the (averaged) full sinogram stack for this flux level (on CPU).
            'applied_pileup_params': The pile-up parameters used for this simulation.
    """
    results_per_flux_level = []

    required_keys = ['image_shape', 'num_angles', 'num_detector_pixels',
                     'energy_bins_keV', 'device']
    for key in required_keys:
        if key not in base_pcct_projector_config:
            raise ValueError(f"Missing required key in base_pcct_projector_config: {key}")

    for flux_scale in flux_levels:
        current_config = base_pcct_projector_config.copy()

        current_I0 = base_source_photons_per_bin.to(current_config['device']) * flux_scale
        current_config['source_photons_per_bin'] = current_I0

        # Ensure other advanced effects are disabled for focusing on pile-up vs flux
        current_config['spectral_resolution_keV'] = None
        current_config['charge_sharing_kernel'] = None
        current_config['k_escape_probabilities'] = None # Assuming param name

        current_config['pileup_parameters'] = pileup_params_to_test

        # Handle Poisson noise based on num_realizations
        if num_realizations > 1:
            current_config['add_poisson_noise'] = True
        else: # num_realizations == 1
            current_config['add_poisson_noise'] = base_pcct_projector_config.get('add_poisson_noise', False)

        projector = PCCTProjectorOperator(**current_config)
        phantom_dev = phantom.to(projector.device)

        accumulated_counts_stack = torch.zeros(projector.measurement_shape,
                                               device=projector.device,
                                               dtype=torch.float32)
        for _ in range(num_realizations):
            accumulated_counts_stack += projector.op(phantom_dev)

        mean_sinogram_stack = accumulated_counts_stack / num_realizations

        results_per_flux_level.append({
            'flux_scale': flux_scale,
            'simulated_source_photons_per_bin': current_I0.cpu(),
            'mean_sinogram_stack_per_bin': mean_sinogram_stack.cpu(),
            'applied_pileup_params': pileup_params_to_test
        })

    return results_per_flux_level


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

    print("\n--- Testing simulate_flat_field_for_spectral_calibration ---")
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Ensure dev_utils is defined

    calib_energy_bins = [(40,45), (45,50), (50,55), (55,60), (60,65)]
    num_calib_bins = len(calib_energy_bins)

    # Source photons primarily in the 3rd bin (index 2)
    source_photons_calib = torch.zeros(num_calib_bins, device=dev_utils, dtype=torch.float32)
    source_photons_calib[2] = 1e6

    base_calib_config = {
        'image_shape': (32, 32),
        'num_angles': 10,
        'num_detector_pixels': 30,
        'energy_bins_keV': calib_energy_bins,
        'source_photons_per_bin': source_photons_calib,
        'device': dev_utils,
        # 'add_poisson_noise' will be handled by the function or tested explicitly
    }

    try:
        # Test 1: Air scan, 1 realization (effectively no noise if add_poisson_noise=False)
        print("\nTest 1: Air Scan (1 realization, no explicit noise in config)")
        config_air_test = base_calib_config.copy()
        config_air_test['add_poisson_noise'] = False
        air_scan_results = simulate_flat_field_for_spectral_calibration(
            config_air_test,
            reference_attenuation_value=None,
            num_realizations=1
        )
        assert 'mean_counts_per_bin_spatial_avg' in air_scan_results
        assert air_scan_results['mean_counts_per_bin_spatial_avg'].shape == (num_calib_bins,), \
            f"Air scan spatial avg shape mismatch. Expected {(num_calib_bins,)}, Got {air_scan_results['mean_counts_per_bin_spatial_avg'].shape}"
        assert air_scan_results['mean_sinogram_stack_per_bin'].shape == (num_calib_bins, config_air_test['num_angles'], config_air_test['num_detector_pixels']), \
            "Air scan sinogram stack shape mismatch."

        counts_air = air_scan_results['mean_counts_per_bin_spatial_avg']
        print(f"  Air scan - spatially averaged counts per bin: {counts_air.cpu().numpy()}")
        # For an ideal detector air scan, mean counts should directly reflect source_photons_per_bin
        expected_counts_air = config_air_test['source_photons_per_bin']
        assert torch.allclose(counts_air, expected_counts_air, rtol=1e-5, atol=1e-5), \
             f"Air scan counts mismatch. Expected {expected_counts_air.cpu().numpy()}, Got {counts_air.cpu().numpy()}"
        assert counts_air[2] > counts_air[0], "Central bin should have more counts in air scan."
        print("  Air scan test passed.")

        # Test 2: Uniform phantom scan, multiple realizations (noise forced on)
        print("\nTest 2: Uniform Phantom Scan (3 realizations)")
        config_phantom_test = base_calib_config.copy()
        # 'add_poisson_noise' in config_phantom_test doesn't matter here as num_realizations > 1 will force it true.
        # However, to be explicit for a config that *would* be noisy if num_realizations=1:
        config_phantom_test['add_poisson_noise'] = True

        phantom_scan_results = simulate_flat_field_for_spectral_calibration(
            config_phantom_test,
            reference_attenuation_value=0.02,
            num_realizations=3
        )
        assert 'mean_counts_per_bin_spatial_avg' in phantom_scan_results
        assert phantom_scan_results['mean_counts_per_bin_spatial_avg'].shape == (num_calib_bins,)
        counts_phantom = phantom_scan_results['mean_counts_per_bin_spatial_avg']
        print(f"  Phantom scan - spatially averaged counts per bin: {counts_phantom.cpu().numpy()}")

        # Counts should be lower than air scan due to attenuation (for bins with initial counts)
        # And also lower than the original source photons due to attenuation
        attenuated_expected_counts = expected_counts_air * np.exp(-0.02) # Simplified: assumes scaling factor is 1
                                                                    # PCCTProjectorOperator applies energy_scaling_factors
                                                                    # which defaults to ones if not in config.

        # Check that counts in the main bin (bin 2) are reduced compared to air scan's main bin
        assert counts_phantom[2] < counts_air[2], \
            f"Phantom scan counts in main bin ({counts_phantom[2]}) not lower than air scan ({counts_air[2]})."
        # Check that other bins remain low (they had no source photons)
        assert torch.allclose(counts_phantom[0:2], torch.zeros_like(counts_phantom[0:2]), atol=1e-5)
        assert torch.allclose(counts_phantom[3:], torch.zeros_like(counts_phantom[3:]), atol=1e-5)
        print("  Uniform phantom scan test passed.")

        print("\nsimulate_flat_field_for_spectral_calibration tests passed.")

    except Exception as e:
        print(f"Error during simulate_flat_field_for_spectral_calibration tests: {e}")
        traceback.print_exc()

    print("\n--- Testing simulate_flux_scan_for_pileup_calibration ---")
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Re-ensure dev_utils

    base_pileup_calib_config = {
        'image_shape': (16, 16),
        'num_angles': 10,
        'num_detector_pixels': 30, # Keep consistent with previous test if needed
        'energy_bins_keV': [(40, 60), (60, 80)], # 2 bins
        'device': dev_utils
        # `source_photons_per_bin` and `add_poisson_noise` are set by the function or test case
    }
    num_flux_test_bins = len(base_pileup_calib_config['energy_bins_keV'])

    test_flux_levels = [0.01, 0.1, 1.0] # Relative flux scales
    test_base_I0 = torch.tensor([1e6] * num_flux_test_bins, device=dev_utils, dtype=torch.float32)

    calib_phantom = torch.zeros(base_pileup_calib_config['image_shape'], device=dev_utils, dtype=torch.float32)
    # Create a highly transmitting central region (near zero attenuation)
    center_y, center_x = calib_phantom.shape[0] // 2, calib_phantom.shape[1] // 2
    calib_phantom[center_y - 2 : center_y + 2, center_x - 2 : center_x + 2] = 0.0001 # Very low mu

    # Define ROI for checking counts based on detector pixels and phantom feature
    # For a (16,16) image and 30 detectors, central feature should be around central detector pixels
    # Example: if num_detector_pixels is 30, center is ~15. ROI could be 13-17.
    # This needs to be robust or known from projector geometry.
    # For simple_radon_transform, detector pixels span -D/2 to D/2.
    # A feature at image center (0,0) projects to detector center (det_idx ~ num_detector_pixels/2)
    # Let's pick a slice based on num_detector_pixels.
    num_dets_pileup_test = base_pileup_calib_config['num_detector_pixels']
    roi_slice = slice(num_dets_pileup_test // 2 - 3, num_dets_pileup_test // 2 + 3) # Central 6 pixels

    try:
        print("\nTest 1: Ideal detector (no pile-up)")
        ideal_results = simulate_flux_scan_for_pileup_calibration(
            base_pileup_calib_config,
            test_flux_levels,
            test_base_I0,
            calib_phantom,
            pileup_params_to_test=None,
            num_realizations=1 # No noise for ideal check of linearity
        )
        assert len(ideal_results) == len(test_flux_levels)

        # Check linearity in high-transmission ROI for the first energy bin
        # Extract max count in the ROI for each flux level
        max_counts_ideal = []
        for res in ideal_results:
            # Max over angles, then over ROI pixels
            # Assuming the feature is always visible, max over angles should be fine.
            # Taking mean over angles for more stability
            mean_sino_bin0_roi = res['mean_sinogram_stack_per_bin'][0, :, roi_slice].mean(dim=0) # Avg over angles
            max_counts_ideal.append(torch.max(mean_sino_bin0_roi)) # Max over ROI pixels

        print(f"  Ideal detector max counts in ROI for bin 0: {[f'{c.item():.2e}' for c in max_counts_ideal]}")
        # Check scaling relative to the first flux level
        base_max_count = max_counts_ideal[0]
        for i in range(1, len(test_flux_levels)):
            expected_scale = test_flux_levels[i] / test_flux_levels[0]
            actual_scale = max_counts_ideal[i] / base_max_count
            assert torch.isclose(actual_scale, torch.tensor(expected_scale, dtype=actual_scale.dtype), rtol=0.1), \
                f"Ideal detector linearity failed. Expected scale {expected_scale:.2f}, got {actual_scale.item():.2f}"
        print("  Ideal detector linearity test passed.")

        print("\nTest 2: Detector with paralyzable pile-up")
        # acquisition_time_s needs to be in pileup_params for the projector
        test_pileup_params = {'method': 'paralyzable', 'dead_time_s': 500e-9, 'acquisition_time_s': 1e-3}

        pileup_results = simulate_flux_scan_for_pileup_calibration(
            base_pileup_calib_config,
            test_flux_levels,
            test_base_I0,
            calib_phantom,
            pileup_params_to_test=test_pileup_params,
            num_realizations=1 # No noise for clear pile-up effect check
        )
        assert len(pileup_results) == len(test_flux_levels)

        max_counts_pileup = []
        for res in pileup_results:
            mean_sino_bin0_roi = res['mean_sinogram_stack_per_bin'][0, :, roi_slice].mean(dim=0)
            max_counts_pileup.append(torch.max(mean_sino_bin0_roi))

        print(f"  Pile-up detector max counts in ROI for bin 0: {[f'{c.item():.2e}' for c in max_counts_pileup]}")

        # Assert that counts at high flux with pile-up are lower than ideal detector at same high flux
        assert max_counts_pileup[-1] < max_counts_ideal[-1], \
            f"Pile-up counts at high flux ({max_counts_pileup[-1].item():.2e}) not lower than ideal ({max_counts_ideal[-1].item():.2e}). Pile-up effect may not be strong enough with chosen parameters."

        # Assert that pile-up counts do not scale linearly like ideal counts
        # (i.e., ratio of (pileup_high_flux/pileup_low_flux) should be less than (ideal_high_flux/ideal_low_flux))
        if len(test_flux_levels) > 1 and max_counts_pileup[0] > 1e-9 : # Avoid division by zero if low flux count is zero
            pileup_scaling = max_counts_pileup[-1] / max_counts_pileup[0]
            ideal_scaling = max_counts_ideal[-1] / max_counts_ideal[0]
            assert pileup_scaling < ideal_scaling, \
                f"Pile-up scaling ({pileup_scaling.item():.2f}) not less than ideal scaling ({ideal_scaling.item():.2f}). Indicates weak pile-up."
        print("  Pile-up effect tests passed (counts lower than ideal at high flux and non-linear scaling).")

        print("\nsimulate_flux_scan_for_pileup_calibration tests passed.")
    except Exception as e:
        print(f"Error during simulate_flux_scan_for_pileup_calibration tests: {e}")
        traceback.print_exc()

    print("\nAll PCCT utils checks completed.")
