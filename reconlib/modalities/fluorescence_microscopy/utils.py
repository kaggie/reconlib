import torch
import numpy as np

def generate_fluorescence_phantom(
    image_shape: tuple[int, int] | tuple[int, int, int],
    num_structures: int = 5,
    structure_type: str = 'cells', # 'cells', 'filaments'
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Generates a simple phantom for Fluorescence Microscopy.
    Creates an image with structures like cells (blobs) or filaments.

    Args:
        image_shape (tuple): Shape of the image (Ny, Nx) or (Nz, Ny, Nx).
        num_structures (int): Number of structures to generate.
        structure_type (str): Type of structure: 'cells' or 'filaments'.
        device (str | torch.device): Device for the tensor.

    Returns:
        torch.Tensor: A float32 tensor representing the fluorescence phantom.
    """
    is_3d = len(image_shape) == 3
    phantom = torch.zeros(image_shape, dtype=torch.float32, device=device)

    for _ in range(num_structures):
        intensity = np.random.rand() * 0.7 + 0.3 # Intensity between 0.3 and 1.0
        if structure_type == 'cells':
            if is_3d:
                Nz, Ny, Nx = image_shape
                # Ellipsoidal cell-like structures
                rad_z = np.random.randint(Nz // 10, Nz // 4) + 1
                rad_y = np.random.randint(Ny // 10, Ny // 4) + 1
                rad_x = np.random.randint(Nx // 10, Nx // 4) + 1
                cz = np.random.randint(rad_z, Nz - rad_z) if Nz > 2*rad_z else Nz//2
                cy = np.random.randint(rad_y, Ny - rad_y) if Ny > 2*rad_y else Ny//2
                cx = np.random.randint(rad_x, Nx - rad_x) if Nx > 2*rad_x else Nx//2

                zz, yy, xx = torch.meshgrid(
                    torch.arange(Nz, device=device),
                    torch.arange(Ny, device=device),
                    torch.arange(Nx, device=device),
                    indexing='ij'
                )
                mask = ((zz - cz)/rad_z)**2 + ((yy - cy)/rad_y)**2 + ((xx - cx)/rad_x)**2 < 1
                phantom[mask] += intensity # Allow overlap to sum
            else: # 2D cells
                Ny, Nx = image_shape
                rad_y = np.random.randint(Ny // 10, Ny // 3) + 1
                rad_x = np.random.randint(Nx // 10, Nx // 3) + 1
                cy = np.random.randint(rad_y, Ny - rad_y) if Ny > 2*rad_y else Ny//2
                cx = np.random.randint(rad_x, Nx - rad_x) if Nx > 2*rad_x else Nx//2
                yy, xx = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')
                mask = ((yy - cy)/rad_y)**2 + ((xx - cx)/rad_x)**2 < 1
                phantom[mask] += intensity

        elif structure_type == 'filaments':
            # Simple linear filaments for now
            if is_3d:
                Nz, Ny, Nx = image_shape
                len_frac = np.random.rand() * 0.3 + 0.2 # 20-50% of max dimension

                # Random start point
                z0,y0,x0 = (np.random.randint(0,s) for s in image_shape)
                # Random end point (can make this more directional later)
                z1 = np.clip(z0 + int((np.random.rand()-0.5)*Nz*len_frac), 0, Nz-1)
                y1 = np.clip(y0 + int((np.random.rand()-0.5)*Ny*len_frac), 0, Ny-1)
                x1 = np.clip(x0 + int((np.random.rand()-0.5)*Nx*len_frac), 0, Nx-1)

                # Draw a line (simple digital line for now)
                num_steps = int(max(abs(z1-z0), abs(y1-y0), abs(x1-x0))) + 1
                z_line = torch.linspace(z0, z1, num_steps, device=device).round().long()
                y_line = torch.linspace(y0, y1, num_steps, device=device).round().long()
                x_line = torch.linspace(x0, x1, num_steps, device=device).round().long()
                phantom[z_line, y_line, x_line] += intensity

            else: # 2D filaments
                Ny, Nx = image_shape
                len_frac = np.random.rand() * 0.4 + 0.3 # 30-70% of max dimension
                y0,x0 = (np.random.randint(0,s) for s in image_shape)
                y1 = np.clip(y0 + int((np.random.rand()-0.5)*Ny*len_frac), 0, Ny-1)
                x1 = np.clip(x0 + int((np.random.rand()-0.5)*Nx*len_frac), 0, Nx-1)
                num_steps = int(max(abs(y1-y0), abs(x1-x0))) + 1
                y_line = torch.linspace(y0, y1, num_steps, device=device).round().long()
                x_line = torch.linspace(x0, x1, num_steps, device=device).round().long()
                phantom[y_line, x_line] += intensity
        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")

    phantom = torch.clamp(phantom, 0, phantom.max() if phantom.max() > 0 else 1.0) # Clamp if overlaps are too strong
    return phantom

def plot_fm_results(
    true_map: torch.Tensor | None = None,
    observed_map: torch.Tensor | None = None, # Blurred
    deconvolved_map: torch.Tensor | None = None,
    slice_idx: int | None = None # For 3D data, which Z slice to show
    ):
    """
    Placeholder for plotting Fluorescence Microscopy results.

    Args:
        true_map: Ground truth fluorescence distribution.
        observed_map: Observed (blurred) image.
        deconvolved_map: Deconvolved fluorescence map.
        slice_idx (int | None): For 3D data, index of Z-slice to display.
                                If None, middle slice is shown.
    """
    print("plot_fm_results: Placeholder - Plotting not implemented.")

    def _get_display_slice(data_map_full, is_3d_map, slice_idx_val):
        if data_map_full is None: return None, "N/A"
        if is_3d_map:
            s_idx = slice_idx_val if slice_idx_val is not None else data_map_full.shape[0] // 2
            s_idx = min(max(s_idx, 0), data_map_full.shape[0]-1) # Ensure valid slice
            return data_map_full[s_idx, ...].cpu().numpy(), f"Slice {s_idx}"
        return data_map_full.cpu().numpy(), "2D Image"

    is_3d = true_map.ndim == 3 if true_map is not None else             (observed_map.ndim == 3 if observed_map is not None else              (deconvolved_map.ndim == 3 if deconvolved_map is not None else False))

    true_slice_info, true_label = _get_display_slice(true_map, is_3d, slice_idx)
    obs_slice_info, obs_label = _get_display_slice(observed_map, is_3d, slice_idx)
    decon_slice_info, decon_label = _get_display_slice(deconvolved_map, is_3d, slice_idx)

    if true_map is not None: print(f"  True Map ({true_label}): shape {true_map.shape}")
    if observed_map is not None: print(f"  Observed Map ({obs_label}): shape {observed_map.shape}")
    if deconvolved_map is not None: print(f"  Deconvolved Map ({decon_label}): shape {deconvolved_map.shape}")

    # TODO: Implement actual plotting using matplotlib.
    # Example:
    # import matplotlib.pyplot as plt
    # num_plots = sum(x is not None for x in [true_slice_info, obs_slice_info, decon_slice_info])
    # if num_plots == 0: return
    # fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    # current_ax = 0
    # titles = [f"True ({true_label})", f"Observed ({obs_label})", f"Deconvolved ({decon_label})"]
    # data_to_plot = [true_slice_info, obs_slice_info, decon_slice_info]
    # for i, data_slice in enumerate(data_to_plot):
    #     if data_slice is not None:
    #         ax = axes[current_ax] if num_plots > 1 else axes
    #         im = ax.imshow(data_slice, cmap='viridis') # Or 'gray'
    #         ax.set_title(titles[i])
    #         plt.colorbar(im, ax=ax); current_ax +=1
    # plt.show()

if __name__ == '__main__':
    print("Running basic Fluorescence Microscopy utils checks...")
    device_utils_fm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    phantom_2d_cells = generate_fluorescence_phantom(image_shape=(64,64), num_structures=5, structure_type='cells', device=device_utils_fm)
    print(f"Generated 2D cells phantom: shape {phantom_2d_cells.shape}")
    assert phantom_2d_cells.shape == (64,64)

    phantom_2d_filaments = generate_fluorescence_phantom(image_shape=(64,64), num_structures=3, structure_type='filaments', device=device_utils_fm)
    print(f"Generated 2D filaments phantom: shape {phantom_2d_filaments.shape}")

    phantom_3d_cells = generate_fluorescence_phantom(image_shape=(32,32,16), num_structures=4, structure_type='cells', device=device_utils_fm)
    print(f"Generated 3D cells phantom: shape {phantom_3d_cells.shape}")
    assert phantom_3d_cells.shape == (32,32,16)

    phantom_3d_filaments = generate_fluorescence_phantom(image_shape=(32,32,16), num_structures=2, structure_type='filaments', device=device_utils_fm)
    print(f"Generated 3D filaments phantom: shape {phantom_3d_filaments.shape}")

    plot_fm_results(
        true_map=phantom_2d_cells,
        observed_map=phantom_2d_cells*0.7, # Dummy observed
        deconvolved_map=phantom_2d_cells*0.9 # Dummy deconvolved
    )
    plot_fm_results(true_map=phantom_3d_cells, slice_idx=phantom_3d_cells.shape[0]//2)

    print("Fluorescence Microscopy utils checks completed.")
