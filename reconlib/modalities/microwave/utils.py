import torch
import numpy as np

def generate_mwi_phantom(
    image_shape: tuple[int, int] | tuple[int, int, int],
    num_objects: int = 2,
    object_type: str = 'circle_contrast',
    background_permittivity: complex = 1.0+0.0j, # Complex background (e.g., water, air)
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Generates a simple phantom for Microwave Imaging, representing dielectric
    property contrast (complex permittivity) relative to a background.

    Args:
        image_shape (tuple): Shape of the image (Ny, Nx) or (Nz, Ny, Nx).
        num_objects (int): Number of contrasting objects to generate.
        object_type (str): Type of object. Options: 'circle_contrast'.
        background_permittivity (complex): Complex permittivity of the background.
                                           The phantom will store the *contrast* relative to this.
        device (str | torch.device): Device for the tensor.

    Returns:
        torch.Tensor: A complex-valued tensor representing the dielectric contrast map
                      (object_permittivity - background_permittivity).
    """
    is_3d = len(image_shape) == 3
    # Phantom stores the DELTA / CONTRAST from the background.
    # So, it's initialized to zeros (zero contrast).
    phantom_contrast = torch.zeros(image_shape, dtype=torch.complex64, device=device)

    for _ in range(num_objects):
        # Define random permittivity for the object (absolute value)
        # Permittivity of inclusions often higher than background in medical MWI (e.g. tumor in breast)
        obj_real_part = background_permittivity.real + np.random.rand() * 20.0 + 5.0 # e.g., 5-25 higher
        obj_imag_part = background_permittivity.imag + np.random.rand() * 1.0 + 0.1  # e.g., 0.1-1.1 higher loss
        object_permittivity = obj_real_part + obj_imag_part * 1j

        contrast_value = object_permittivity - background_permittivity

        if object_type == 'circle_contrast':
            if is_3d:
                Nz, Ny, Nx = image_shape
                radius_z = np.random.randint(Nz // 8, Nz // 4)
                radius_y = np.random.randint(Ny // 8, Ny // 4)
                radius_x = np.random.randint(Nx // 8, Nx // 4)
                cz = np.random.randint(radius_z, Nz - radius_z)
                cy = np.random.randint(radius_y, Ny - radius_y)
                cx = np.random.randint(radius_x, Nx - radius_x)

                z_coords, y_coords, x_coords = torch.meshgrid(
                    torch.arange(Nz, device=device),
                    torch.arange(Ny, device=device),
                    torch.arange(Nx, device=device),
                    indexing='ij'
                )
                # Ellipsoidal region
                dist_sq = ((z_coords - cz)/radius_z)**2 +                           ((y_coords - cy)/radius_y)**2 +                           ((x_coords - cx)/radius_x)**2
                phantom_contrast[dist_sq < 1] = contrast_value # Assign contrast
            else:
                Ny, Nx = image_shape
                radius = np.random.randint(min(Ny, Nx) // 8, min(Ny, Nx) // 3)
                cx = np.random.randint(radius, Nx - radius)
                cy = np.random.randint(radius, Ny - radius)
                y_coords, x_coords = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')
                dist_sq = (y_coords - cy)**2 + (x_coords - cx)**2
                phantom_contrast[dist_sq < radius**2] = contrast_value # Assign contrast
        else:
            raise ValueError(f"Unknown object_type: {object_type}")

    return phantom_contrast

def plot_mwi_results(
    true_contrast_map: torch.Tensor | None = None, # Complex
    reconstructed_contrast_map: torch.Tensor | None = None, # Complex
    scattered_data: torch.Tensor | None = None, # Complex
    slice_idx: int | None = None # For 3D data
    ):
    """
    Placeholder for plotting Microwave Imaging results.
    Displays real and imaginary parts or magnitude and phase of complex maps.

    Args:
        true_contrast_map: Ground truth complex dielectric contrast map.
        reconstructed_contrast_map: Reconstructed complex dielectric contrast map.
        scattered_data: Recorded microwave scattered field data.
        slice_idx: Index of 2D slice to display for 3D data.
    """
    print("plot_mwi_results: Placeholder - Plotting not implemented.")

    def _plot_complex_map(data_map, title_prefix, is_3d_map, slice_idx_val):
        if data_map is None: return

        display_map = data_map
        map_dim_str = "2D"
        if is_3d_map :
            slice_to_show = slice_idx_val if slice_idx_val is not None else data_map.shape[0]//2
            display_map = data_map[slice_to_show]
            map_dim_str = f"3D (slice {slice_to_show})"

        print(f"  {title_prefix} ({map_dim_str}): shape {data_map.shape}, dtype {data_map.dtype}")
        # TODO: Implement plotting of real/imag or mag/phase using matplotlib
        # Example:
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # real_part = display_map.real.cpu().numpy()
        # imag_part = display_map.imag.cpu().numpy()
        # im1 = axes[0].imshow(real_part)
        # axes[0].set_title(f"{title_prefix} - Real Part")
        # plt.colorbar(im1, ax=axes[0])
        # im2 = axes[1].imshow(imag_part)
        # axes[1].set_title(f"{title_prefix} - Imaginary Part")
        # plt.colorbar(im2, ax=axes[1])
        # plt.suptitle(f"{title_prefix} ({map_dim_str})")
        # plt.show()


    is_3d_true = true_contrast_map.ndim == 3 if true_contrast_map is not None else False
    is_3d_recon = reconstructed_contrast_map.ndim == 3 if reconstructed_contrast_map is not None else False

    _plot_complex_map(true_contrast_map, "True Contrast Map", is_3d_true, slice_idx)
    _plot_complex_map(reconstructed_contrast_map, "Recon. Contrast Map", is_3d_recon, slice_idx)

    if scattered_data is not None:
        print(f"  Scattered data: shape {scattered_data.shape}, dtype {scattered_data.dtype}")
        # TODO: Plot scattered data (e.g., real/imag parts or magnitude vs. measurement index)


if __name__ == '__main__':
    print("Running basic Microwave Imaging utils checks...")
    device_utils_mwi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bg_eps = 40.0 + 2.0j # Example background (e.g. moderately lossy tissue)

    phantom_2d = generate_mwi_phantom(
        image_shape=(64, 64), num_objects=2,
        background_permittivity=bg_eps, device=device_utils_mwi
    )
    print(f"Generated 2D MWI phantom (contrast): shape {phantom_2d.shape}, dtype {phantom_2d.dtype}")
    assert phantom_2d.shape == (64,64)
    assert phantom_2d.is_complex()

    phantom_3d = generate_mwi_phantom(
        image_shape=(32,32,32), num_objects=1,
        background_permittivity=bg_eps, device=device_utils_mwi
    )
    print(f"Generated 3D MWI phantom (contrast): shape {phantom_3d.shape}, dtype {phantom_3d.dtype}")
    assert phantom_3d.shape == (32,32,32)
    assert phantom_3d.is_complex()

    dummy_scatter = torch.randn(100, dtype=torch.complex64, device=device_utils_mwi)
    plot_mwi_results(
        true_contrast_map=phantom_2d,
        reconstructed_contrast_map=phantom_2d * (0.5+0.5j), # Dummy recon
        scattered_data=dummy_scatter
    )
    plot_mwi_results(true_contrast_map=phantom_3d, slice_idx=15)

    print("Microwave Imaging utils checks completed.")
