import torch
import numpy as np

def generate_hsi_phantom(
    image_shape: tuple[int, int, int], # (Ny, Nx, N_bands)
    num_features: int = 2,
    feature_type: str = 'simple_spectra', # 'simple_spectra' or 'mixed_spectra' (for unmixing later)
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Generates a simple phantom for Hyperspectral Imaging (HSI).
    Creates a data cube with spatial features having distinct spectral signatures.

    Args:
        image_shape (tuple): Shape of the HSI cube (Ny, Nx, N_bands).
        num_features (int): Number of distinct spatial features to generate.
        feature_type (str):
            'simple_spectra': Each feature has one simple, distinct spectrum.
            'mixed_spectra': Not implemented yet, placeholder for future unmixing phantoms.
        device (str | torch.device): Device for the tensor.

    Returns:
        torch.Tensor: A float32 tensor representing the HSI data cube.
    """
    Ny, Nx, N_bands = image_shape
    hsi_cube = torch.zeros(image_shape, dtype=torch.float32, device=device)

    # Define some base spectra
    spectra_list = []
    for i in range(num_features + 1): # +1 for a possible background variation if needed
        # Create diverse simple spectra (e.g., sine, cosine, Gaussian-like peak, linear slope)
        if i % 4 == 0:
            spec = torch.sin(torch.linspace(0, (i+1)*np.pi/2, N_bands, device=device)) * 0.5 + 0.5
        elif i % 4 == 1:
            spec = torch.cos(torch.linspace(0, (i+1)*np.pi/2, N_bands, device=device)) * 0.5 + 0.5
        elif i % 4 == 2:
            peak_pos = N_bands * (0.25 + 0.5 * (i / (num_features+1)))
            spec_range = torch.arange(N_bands, device=device)
            spec = torch.exp(-((spec_range - peak_pos)**2) / (2 * (N_bands/8)**2))
        else:
            spec = torch.linspace(0.1 + 0.1*i, 0.5 + 0.1*i, N_bands, device=device)
        spectra_list.append(torch.clamp(spec, 0.01, 1.0)) # Ensure positive and bounded

    if feature_type == 'simple_spectra':
        for i in range(num_features):
            # Create a spatial region (e.g., rectangle or circle)
            is_rect = np.random.rand() > 0.5
            current_spectrum = spectra_list[i]

            if is_rect:
                w = np.random.randint(Nx // 8, Nx // 3)
                h = np.random.randint(Ny // 8, Ny // 3)
                x0 = np.random.randint(0, Nx - w)
                y0 = np.random.randint(0, Ny - h)
                hsi_cube[y0:y0+h, x0:x0+w, :] = current_spectrum.unsqueeze(0).unsqueeze(0) # Broadcast spectrum
            else: # Circle
                radius = np.random.randint(min(Ny, Nx) // 10, min(Ny, Nx) // 4)
                cx = np.random.randint(radius, Nx - radius)
                cy = np.random.randint(radius, Ny - radius)
                yy_coords, xx_coords = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')
                mask = (xx_coords - cx)**2 + (yy_coords - cy)**2 < radius**2
                hsi_cube[mask, :] = current_spectrum.unsqueeze(0) # Broadcast spectrum
    elif feature_type == 'mixed_spectra':
        print("Warning: 'mixed_spectra' phantom type not fully implemented yet. Using simple spectra.")
        # Placeholder: would involve creating abundance maps for endmembers.
        # For now, just fall back to simple_spectra behavior for testing.
        for i in range(num_features): # Fallback
            is_rect = np.random.rand() > 0.5; current_spectrum = spectra_list[i]
            if is_rect: w,h,x0,y0 = Nx//5,Ny//5,np.random.randint(0,Nx-Nx//5),np.random.randint(0,Ny-Ny//5); hsi_cube[y0:y0+h,x0:x0+w,:]=current_spectrum.view(1,1,-1)
            else: radius,cx,cy=min(Ny,Nx)//6,np.random.randint(min(Ny,Nx)//6,Nx-min(Ny,Nx)//6),np.random.randint(min(Ny,Nx)//6,Ny-min(Ny,Nx)//6); yy,xx=torch.meshgrid(torch.arange(Ny,device=device),torch.arange(Nx,device=device),indexing='ij');mask=(xx-cx)**2+(yy-cy)**2<radius**2;hsi_cube[mask,:]=current_spectrum.view(1,-1)

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return hsi_cube

def plot_hsi_results(
    true_hsi_cube: torch.Tensor | None = None,      # (Ny, Nx, N_bands)
    recon_hsi_cube: torch.Tensor | None = None,     # (Ny, Nx, N_bands)
    measurement_data: torch.Tensor | None = None, # (num_measurements,)
    rgb_bands: tuple[int, int, int] | None = None # e.g., (2,1,0) for specific bands for R,G,B
    ):
    """
    Placeholder for plotting Hyperspectral Imaging results.
    Can display RGB composites or selected band images.

    Args:
        true_hsi_cube: Ground truth HSI data cube.
        recon_hsi_cube: Reconstructed HSI data cube.
        measurement_data: Recorded sensor/measurement data.
        rgb_bands (tuple[int,int,int]): Indices of three bands to use for an RGB display.
                                        If None, might show grayscale of first band or mean.
    """
    print("plot_hsi_results: Placeholder - Plotting not implemented.")

    def _get_rgb_composite(cube, bands):
        if cube is None: return None
        if bands is None or not (isinstance(bands, (list, tuple)) and len(bands) == 3):
            # Default: show first band as grayscale, or mean over bands
            # return cube[..., 0].cpu().numpy()
            return torch.mean(cube, dim=-1).cpu().numpy()


        # Ensure bands are within range
        valid_bands = [min(max(b, 0), cube.shape[-1]-1) for b in bands]
        img_r = cube[..., valid_bands[0]]
        img_g = cube[..., valid_bands[1]]
        img_b = cube[..., valid_bands[2]]

        # Normalize each channel to 0-1 for display
        img_r = (img_r - img_r.min()) / (img_r.max() - img_r.min() + 1e-6)
        img_g = (img_g - img_g.min()) / (img_g.max() - img_g.min() + 1e-6)
        img_b = (img_b - img_b.min()) / (img_b.max() - img_b.min() + 1e-6)

        return torch.stack([img_r, img_g, img_b], dim=-1).cpu().numpy()

    if true_hsi_cube is not None:
        print(f"  True HSI Cube: shape {true_hsi_cube.shape}")
        # true_display_img = _get_rgb_composite(true_hsi_cube, rgb_bands)
        # TODO: plt.imshow(true_display_img); plt.title("True HSI (RGB or Band X)")

    if recon_hsi_cube is not None:
        print(f"  Recon HSI Cube: shape {recon_hsi_cube.shape}")
        # recon_display_img = _get_rgb_composite(recon_hsi_cube, rgb_bands)
        # TODO: plt.imshow(recon_display_img); plt.title("Recon HSI (RGB or Band X)")

    if measurement_data is not None:
        print(f"  Measurement data: shape {measurement_data.shape}")
        # TODO: Plot measurement data if useful (e.g. plt.plot(measurement_data.cpu().numpy()))

if __name__ == '__main__':
    print("Running basic Hyperspectral Imaging utils checks...")
    device_utils_hsi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hsi_shape = (32, 32, 16) # Ny, Nx, N_bands
    phantom_hsi = generate_hsi_phantom(image_shape=hsi_shape, num_features=3, device=device_utils_hsi)
    print(f"Generated HSI phantom: shape {phantom_hsi.shape}, dtype {phantom_hsi.dtype}")
    assert phantom_hsi.shape == hsi_shape

    # Example RGB bands (indices for R, G, B - ensure they are within N_bands)
    example_rgb = (hsi_shape[2]-1, hsi_shape[2]//2, 0)

    plot_hsi_results(
        true_hsi_cube=phantom_hsi,
        recon_hsi_cube=phantom_hsi * 0.8, # Dummy recon
        rgb_bands=example_rgb
    )

    print("Hyperspectral Imaging utils checks completed.")
