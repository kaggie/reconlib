import torch
import numpy as np

def generate_pcct_phantom_material_maps(
    image_shape: tuple[int, int],
    material_attenuations: dict[str, float], # E.g. {'water': 0.2, 'bone': 0.5} at reference energy
    num_features_per_material: int = 1,
    device='cpu'
) -> dict[str, torch.Tensor]:
    """
    Generates a phantom with multiple material types, each having a base attenuation.
    Outputs separate maps for each material's concentration/presence (0 or 1).
    The 'image' to be reconstructed by some PCCT algorithms might be these material maps,
    or a single reference mu_map derived from them.

    Args:
        image_shape (tuple): Shape of the image (Ny, Nx).
        material_attenuations (dict): Dict of material names to their reference attenuation values.
        num_features_per_material (int): How many distinct regions for each material.
        device (str or torch.device): Device for the tensor.

    Returns:
        dict[str, torch.Tensor]: A dictionary where keys are material names and values
                                 are 2D tensors (0 or 1) indicating presence of material.
    """
    material_maps = {}
    all_occupied_mask = torch.zeros(image_shape, dtype=torch.bool, device=device)

    for mat_name, _ in material_attenuations.items():
        material_map = torch.zeros(image_shape, dtype=torch.float32, device=device)
        for _ in range(num_features_per_material):
            # Simple rectangular or circular regions
            while True: # Ensure we find a non-overlapping spot (highly simplified)
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

                # Check overlap with already assigned regions from *other* materials
                if not torch.any(temp_mask & all_occupied_mask):
                    material_map[temp_mask] = 1.0 # Binary presence of material
                    all_occupied_mask |= temp_mask # Add to overall occupied mask
                    break
                # If it overlaps, try generating a new region for this feature (very basic)
        material_maps[mat_name] = material_map

    return material_maps

def combine_material_maps_to_mu_ref(
    material_maps: dict[str, torch.Tensor],
    material_attenuations_ref: dict[str, float], # Attenuation at a reference energy
    device='cpu'
) -> torch.Tensor:
    """ Combines material maps into a single reference attenuation map. """
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
    """ Placeholder to plot PCCT results. """
    print("plot_pcct_results: Placeholder - Plotting not implemented.")
    if mu_ref_true is not None:
        print(f"  True mu_ref_map shape: {mu_ref_true.shape}")
    if measured_counts_stack is not None:
        print(f"  Measured counts stack shape: {measured_counts_stack.shape}")
        print(f"  (Showing sinogram for energy bin index: {energy_bin_idx_to_show})")
    if mu_ref_recon is not None:
        print(f"  Recon mu_ref_map shape: {mu_ref_recon.shape}")

    # TODO: Implement plotting using matplotlib.
    # Example:
    # import matplotlib.pyplot as plt
    # num_plots = sum(x is not None for x in [mu_ref_true, measured_counts_stack, mu_ref_recon])
    # fig, axes = plt.subplots(1, num_plots, figsize=(num_plots*5, 5))
    # current_ax = 0
    # if mu_ref_true is not None: ...
    # if measured_counts_stack is not None: axes[current_ax].imshow(measured_counts_stack[energy_bin_idx_to_show].cpu().numpy(), aspect='auto'); ...
    # if mu_ref_recon is not None: ...
    # plt.show()


def get_pcct_energy_scaling_factors(
    energy_bins_keV: list[tuple[float,float]],
    material_attenuation_model: str = 'water_simplified',
    reference_energy_keV: float = 60.0,
    device='cpu',
    dtype=torch.float32
) -> torch.Tensor:
    """
    Placeholder function to return simple energy scaling factors for mu per bin,
    relative to a reference energy.

    A real implementation would use material data (e.g., NIST) and average
    over the bin weighted by the source spectrum. This is highly simplified.

    Args:
        energy_bins_keV (list): List of (low_keV, high_keV) tuples for each bin.
        material_attenuation_model (str): Placeholder for future material models.
                                          Currently supports 'water_simplified' or 'constant'.
        reference_energy_keV (float): The reference energy at which the input mu_map is defined.
        device (str or torch.device): PyTorch device.
        dtype (torch.dtype): PyTorch dtype.

    Returns:
        torch.Tensor: Scaling factors, one for each energy bin.
    """
    num_bins = len(energy_bins_keV)
    scaling_factors = torch.ones(num_bins, device=device, dtype=dtype)

    if material_attenuation_model == 'water_simplified':
        # Very crude approximation: 1/E^3 behavior for photoelectric part of water, dominant at lower E.
        # This is NOT physically accurate across wide energy ranges or for other materials.
        # It's just to show some energy dependence.
        for i, (low_e, high_e) in enumerate(energy_bins_keV):
            mean_bin_energy = (low_e + high_e) / 2.0
            # Scale relative to reference energy. Factor = (ref_E / bin_E)^3
            # If mean_bin_energy is 0, this will cause issues. Ensure energies are positive.
            if mean_bin_energy > 0 and reference_energy_keV > 0:
                factor = (reference_energy_keV / mean_bin_energy)**3
                # Let's limit the factor to avoid extreme values for this placeholder
                scaling_factors[i] = torch.clamp(torch.tensor(factor, device=device, dtype=dtype), 0.1, 10.0)
            else:
                scaling_factors[i] = 1.0 # Default if energies are problematic
    elif material_attenuation_model == 'constant':
        # Factors remain 1.0 (no energy dependence modeled for mu)
        pass
    else:
        print(f"Warning: Unknown material_attenuation_model '{material_attenuation_model}'. Using constant scaling factors (1.0).")

    return scaling_factors


if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_utils = (64,64)

    materials_ref_mu = {'water': 0.2, 'bone': 0.5, 'contrast': 1.0} # Example mu @ ref energy

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

    # Dummy data for plot call
    dummy_counts = torch.randn((3, 100, 128), device=dev_utils) # 3 bins, 100 angles, 128 detectors
    plot_pcct_results(mu_ref_combined, dummy_counts, mu_ref_combined*0.8)

    print("\nTesting get_pcct_energy_scaling_factors...")
    ebins = [(20.,50.), (50.,80.), (80.,120.)]
    scales_water = get_pcct_energy_scaling_factors(ebins, 'water_simplified', device=dev_utils)
    print(f"Water-like scales for {ebins} (ref 60keV): {scales_water}")
    scales_const = get_pcct_energy_scaling_factors(ebins, 'constant', device=dev_utils)
    print(f"Constant scales for {ebins}: {scales_const}")
    print("PCCT utils checks completed.")
