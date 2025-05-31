import torch
import numpy as np

def generate_irt_phantom(
    image_shape: tuple[int, int] | tuple[int, int, int],
    num_defects: int = 2,
    defect_type: str = 'rect_inclusion',
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Generates a simple phantom for Infrared Thermography, representing subsurface
    defects or thermal anomalies.

    Args:
        image_shape (tuple): Shape of the subsurface image (Ny, Nx) or (Nz, Ny, Nx).
        num_defects (int): Number of defects to generate.
        defect_type (str): Type of defect. Options:
                           'rect_inclusion' (rectangular area with different thermal property)
                           'hotspot' (localized heat source)
        device (str | torch.device): Device for the tensor.

    Returns:
        torch.Tensor: A tensor representing the subsurface phantom.
                      For 'rect_inclusion', values might represent thermal resistance change (e.g., 0.5 to 1.5).
                      For 'hotspot', values represent heat intensity.
    """
    is_3d = len(image_shape) == 3
    phantom = torch.zeros(image_shape, dtype=torch.float32, device=device)
    if defect_type == 'rect_inclusion' and not is_3d: # Base material property is 1.0
        phantom += 1.0

    for _ in range(num_defects):
        if defect_type == 'rect_inclusion':
            # Represents a region with altered thermal properties
            defect_value = np.random.rand() * 1.0 + 0.5 # e.g., 0.5 (less resistive) to 1.5 (more resistive)
            if is_3d:
                Nz, Ny, Nx = image_shape
                dw = np.random.randint(Nx // 8, Nx // 3)
                dh = np.random.randint(Ny // 8, Ny // 3)
                dd = np.random.randint(Nz // 8, Nz // 3)
                x0 = np.random.randint(0, Nx - dw)
                y0 = np.random.randint(0, Ny - dh)
                z0 = np.random.randint(0, Nz - dd)
                phantom[z0:z0+dd, y0:y0+dh, x0:x0+dw] = defect_value # Assign, not add
            else:
                Ny, Nx = image_shape
                dw = np.random.randint(Nx // 8, Nx // 3)
                dh = np.random.randint(Ny // 8, Ny // 3)
                x0 = np.random.randint(0, Nx - dw)
                y0 = np.random.randint(0, Ny - dh)
                phantom[y0:y0+dh, x0:x0+dw] = defect_value # Assign

        elif defect_type == 'hotspot':
            # Represents a localized heat source
            source_intensity = np.random.rand() * 0.8 + 0.2
            if is_3d:
                Nz, Ny, Nx = image_shape
                # Smaller "spot" for hotspot
                dw = np.random.randint(Nx // 10, Nx // 5)
                dh = np.random.randint(Ny // 10, Ny // 5)
                dd = np.random.randint(Nz // 10, Nz // 5)
                x0 = np.random.randint(0, Nx - dw)
                y0 = np.random.randint(0, Ny - dh)
                z0 = np.random.randint(0, Nz - dd)
                phantom[z0:z0+dd, y0:y0+dh, x0:x0+dw] += source_intensity
            else:
                Ny, Nx = image_shape
                dw = np.random.randint(Nx // 10, Nx // 5)
                dh = np.random.randint(Ny // 10, Ny // 5)
                x0 = np.random.randint(0, Nx - dw)
                y0 = np.random.randint(0, Ny - dh)
                phantom[y0:y0+dh, x0:x0+dw] += source_intensity
        else:
            raise ValueError(f"Unknown defect_type: {defect_type}")

    if defect_type == 'hotspot': # Clamp hotspots if they overlap
         phantom = torch.clamp(phantom, 0, phantom.max() if phantom.max() > 0 else 1.0)

    return phantom

def plot_irt_results(
    subsurface_map_true: torch.Tensor | None = None,
    subsurface_map_recon: torch.Tensor | None = None,
    surface_temp_sequence: torch.Tensor | None = None, # (time, Ny, Nx)
    time_slice_to_display: int = 0
    ):
    """
    Placeholder for plotting Infrared Thermography results.

    Args:
        subsurface_map_true: Ground truth subsurface property map.
        subsurface_map_recon: Reconstructed subsurface property map.
        surface_temp_sequence: Recorded surface temperature data over time.
        time_slice_to_display: Which time frame of the surface temperature to display.
    """
    print("plot_irt_results: Placeholder - Plotting not implemented.")
    # Determine if maps are 2D or 3D (subsurface_map_true or _recon)
    # For 3D maps, might need to select a slice for display or use 3D plotter.

    is_3d_map = False
    if subsurface_map_true is not None and subsurface_map_true.ndim == 3:
        is_3d_map = True
        print(f"  True subsurface map (3D): shape {subsurface_map_true.shape}. Displaying slice 0 from Z.")
    elif subsurface_map_true is not None:
        print(f"  True subsurface map (2D): shape {subsurface_map_true.shape}")

    if subsurface_map_recon is not None and subsurface_map_recon.ndim == 3:
        is_3d_map = True # Could be different from true_map if one is None
        print(f"  Recon subsurface map (3D): shape {subsurface_map_recon.shape}. Displaying slice 0 from Z.")
    elif subsurface_map_recon is not None:
        print(f"  Recon subsurface map (2D): shape {subsurface_map_recon.shape}")

    if surface_temp_sequence is not None:
        print(f"  Surface temperature sequence: shape {surface_temp_sequence.shape}. Displaying time frame {time_slice_to_display}.")

    # TODO: Implement actual plotting using matplotlib.
    # Example:
    # import matplotlib.pyplot as plt
    # num_plots = sum(x is not None for x in [subsurface_map_true, subsurface_map_recon, surface_temp_sequence])
    # if num_plots == 0: return
    # fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    # current_ax = 0
    # def get_display_slice(data_map):
    #     if data_map is None: return None
    #     if data_map.ndim == 3: return data_map[data_map.shape[0]//2, ...].cpu().numpy() # Middle slice
    #     return data_map.cpu().numpy()

    # if subsurface_map_true is not None:
    #     ax = axes[current_ax] if num_plots > 1 else axes
    #     im = ax.imshow(get_display_slice(subsurface_map_true))
    #     ax.set_title("True Subsurface Map")
    #     plt.colorbar(im, ax=ax); current_ax +=1
    # # ... etc. for recon and surface_temp_sequence[time_slice_to_display]
    # plt.show()


if __name__ == '__main__':
    print("Running basic Infrared Thermography utils checks...")
    device_utils_irt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    phantom_2d_incl = generate_irt_phantom(image_shape=(64, 64), num_defects=2, defect_type='rect_inclusion', device=device_utils_irt)
    print(f"Generated 2D inclusion phantom: shape {phantom_2d_incl.shape}")
    assert phantom_2d_incl.shape == (64,64)

    phantom_2d_spot = generate_irt_phantom(image_shape=(64, 64), num_defects=3, defect_type='hotspot', device=device_utils_irt)
    print(f"Generated 2D hotspot phantom: shape {phantom_2d_spot.shape}")
    assert phantom_2d_spot.shape == (64,64)

    phantom_3d_incl = generate_irt_phantom(image_shape=(32,32,32), num_defects=2, defect_type='rect_inclusion', device=device_utils_irt)
    print(f"Generated 3D inclusion phantom: shape {phantom_3d_incl.shape}")
    assert phantom_3d_incl.shape == (32,32,32)

    phantom_3d_spot = generate_irt_phantom(image_shape=(32,32,32), num_defects=3, defect_type='hotspot', device=device_utils_irt)
    print(f"Generated 3D hotspot phantom: shape {phantom_3d_spot.shape}")
    assert phantom_3d_spot.shape == (32,32,32)

    dummy_surface_temps = torch.randn(5, 64, 64, device=device_utils_irt) # 5 time steps
    plot_irt_results(
        subsurface_map_true=phantom_2d_incl,
        subsurface_map_recon=phantom_2d_spot,
        surface_temp_sequence=dummy_surface_temps,
        time_slice_to_display=2
    )
    plot_irt_results(subsurface_map_true=phantom_3d_incl) # Test 3D plotting message

    print("Infrared Thermography utils checks completed.")
