import numpy as np
import matplotlib.pyplot as plt

from reconlibs.modality.epr.reconstruction import (
    preprocess_cw_epr_data,
    ARTReconstructor,
    # For synthetic data generation, we might need lineshape functions if we convolve
    gaussian_lineshape
)
# No direct import from reconlibs.modality.epr.continuous_wave for this demo to keep it focused on ARTReconstructor


def generate_simple_phantom(grid_size=(64, 64)):
    """Generates a simple phantom image."""
    phantom = np.zeros(grid_size)
    # Add a few shapes
    # Circle 1
    center_x1, center_y1 = grid_size[1] // 4, grid_size[0] // 4
    radius1 = grid_size[0] // 8
    y, x = np.ogrid[:grid_size[0], :grid_size[1]]
    dist_from_center1 = np.sqrt((x - center_x1)**2 + (y - center_y1)**2)
    phantom[dist_from_center1 <= radius1] = 1.0

    # Square 1
    sq_size = grid_size[0] // 4
    phantom[grid_size[0] // 2 : grid_size[0] // 2 + sq_size,
            grid_size[1] // 2 : grid_size[1] // 2 + sq_size] = 0.75

    return phantom

def simplified_forward_projection(phantom, angles_deg, num_bins, lineshape_fwhm_bins=None):
    """
    Generates projection data from a phantom using a simplified model.
    Uses nearest neighbor projection and optionally applies a lineshape.
    """
    grid_size_y, grid_size_x = phantom.shape
    projections = np.zeros((len(angles_deg), num_bins))

    # For lineshape convolution if specified
    lineshape_kernel = None
    if lineshape_fwhm_bins is not None and lineshape_fwhm_bins > 0:
        # Create a kernel for lineshape centered at 0
        # Spread should be enough to capture the lineshape
        spread = int(np.ceil(lineshape_fwhm_bins * 2.5))
        if spread == 0: spread = 1
        kernel_x = np.arange(-spread, spread + 1)
        lineshape_kernel = gaussian_lineshape(kernel_x, 0, lineshape_fwhm_bins)

    # Projection parameters (matching ARTReconstructor's _initialize_system_matrix defaults)
    max_img_dim = max(grid_size_x, grid_size_y)
    p_min = -max_img_dim / 2.0
    projection_axis_length = max_img_dim
    bin_width = projection_axis_length / num_bins

    image_center_x = grid_size_x / 2.0
    image_center_y = grid_size_y / 2.0

    for i, angle_deg in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        current_projection_raw = np.zeros(num_bins)

        for r_idx in range(grid_size_y):
            for c_idx in range(grid_size_x):
                if phantom[r_idx, c_idx] > 0:
                    # Pixel center relative to image center
                    center_x_rel = (c_idx + 0.5) - image_center_x
                    center_y_rel = (r_idx + 0.5) - image_center_y

                    # Project pixel center
                    p = center_x_rel * cos_theta + center_y_rel * sin_theta

                    # Map to bin index
                    p_shifted = p - p_min
                    bin_idx_center = int(np.floor(p_shifted / bin_width))

                    if 0 <= bin_idx_center < num_bins:
                        if lineshape_kernel is not None:
                            for k_offset, l_val in enumerate(lineshape_kernel):
                                target_bin = bin_idx_center + (k_offset - (len(lineshape_kernel) // 2))
                                if 0 <= target_bin < num_bins:
                                    current_projection_raw[target_bin] += phantom[r_idx, c_idx] * l_val
                        else:
                             current_projection_raw[bin_idx_center] += phantom[r_idx, c_idx]
        projections[i, :] = current_projection_raw

    return projections


def main():
    # --- 1. Setup Parameters ---
    grid_dim = 32 # Using a smaller grid for faster demo
    grid_size = (grid_dim, grid_dim)
    angles_deg = np.linspace(0, 180, grid_dim, endpoint=False) # e.g., 32 angles
    num_projection_bins = int(np.ceil(np.sqrt(2) * grid_dim)) # Approx diagonal length

    art_iterations = 50
    art_relaxation = 0.15

    # --- 2. Generate Synthetic Data ---
    print("Generating synthetic phantom and projection data...")
    phantom_true = generate_simple_phantom(grid_size)

    # Generate "clean" projections (can add lineshape here if desired for true data)
    # For this demo, let's make the "true" data with a slight lineshape broadening
    # to make the lineshape reconstruction more meaningful.
    projections_clean = simplified_forward_projection(phantom_true, angles_deg, num_projection_bins, lineshape_fwhm_bins=1.5)

    # Add noise and baseline to projections
    noise_level = 0.05 * np.max(projections_clean)
    projections_noisy = projections_clean + np.random.normal(0, noise_level, projections_clean.shape)

    # Add a simple baseline (e.g., a slight tilt)
    for i in range(projections_noisy.shape[0]):
        baseline = np.linspace(0, 0.1 * np.max(projections_noisy[i,:]), num_projection_bins)
        projections_noisy[i, :] += baseline

    raw_sinogram = projections_noisy.copy() # Keep a copy for visualization

    # --- 3. Preprocessing Demonstration ---
    print("Preprocessing projection data...")
    preprocess_params = {
        'baseline_correct_method': 'als',
        'als_lambda': 1e5,
        'als_p_asymmetry': 0.005,
        'denoise_method': 'wavelet',
        'wavelet_type': 'db4',
        'wavelet_level': 3,
        'wavelet_threshold_sigma_multiplier': 2.0,
        'normalize_method': 'max' # Normalize after other steps
    }
    projections_processed = preprocess_cw_epr_data(projections_noisy, preprocess_params)
    preprocessed_sinogram = projections_processed.copy()

    # --- 4. Reconstruction Demonstrations ---

    # Scenario 1: Basic ART (Nearest, No Lineshape)
    print("Running ART Reconstruction: Scenario 1 (Nearest, No Lineshape)...")
    art_basic = ARTReconstructor(
        projection_data=projections_processed.copy(), # Use a copy for each recon
        gradient_angles=angles_deg,
        grid_size=grid_size,
        num_iterations=art_iterations,
        relaxation_param=art_relaxation,
        projector_type='nearest',
        lineshape_model=None
    )
    recon_basic = art_basic.reconstruct()

    # Scenario 2: Siddon-like Projector
    print("Running ART Reconstruction: Scenario 2 (Siddon-like, No Lineshape)...")
    art_siddon = ARTReconstructor(
        projection_data=projections_processed.copy(),
        gradient_angles=angles_deg,
        grid_size=grid_size,
        num_iterations=art_iterations,
        relaxation_param=art_relaxation,
        projector_type='siddon_like',
        lineshape_model=None
    )
    recon_siddon = art_siddon.reconstruct()

    # Scenario 3: Lineshape Model (e.g., Gaussian with Nearest Projector)
    print("Running ART Reconstruction: Scenario 3 (Nearest, Gaussian Lineshape)...")
    # FWHM for lineshape in system matrix (in terms of projection bins)
    # This should ideally match or be related to the lineshape used for generating the data,
    # or an estimate of the inherent lineshape if trying to deconvolve.
    lineshape_fwhm_for_recon = 1.5
    art_lineshape = ARTReconstructor(
        projection_data=projections_processed.copy(),
        gradient_angles=angles_deg,
        grid_size=grid_size,
        num_iterations=art_iterations,
        relaxation_param=art_relaxation,
        projector_type='nearest', # Could also be 'siddon_like'
        lineshape_model='gaussian',
        lineshape_params={'fwhm': lineshape_fwhm_for_recon}
    )
    recon_lineshape = art_lineshape.reconstruct()

    # --- 5. Visualization ---
    print("Displaying results...")
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(phantom_true, cmap='viridis', origin='lower')
    plt.title("Original Phantom")
    plt.colorbar()

    plt.subplot(2, 4, 2)
    plt.imshow(raw_sinogram.T, cmap='viridis', aspect='auto', origin='lower')
    plt.title("Raw Sinogram (Noisy)")
    plt.xlabel("Angle Index")
    plt.ylabel("Projection Bin")
    plt.colorbar()

    plt.subplot(2, 4, 3)
    plt.imshow(preprocessed_sinogram.T, cmap='viridis', aspect='auto', origin='lower')
    plt.title("Preprocessed Sinogram")
    plt.xlabel("Angle Index")
    plt.ylabel("Projection Bin")
    plt.colorbar()

    # Reconstructions
    plt.subplot(2, 4, 5)
    plt.imshow(recon_basic, cmap='viridis', origin='lower')
    plt.title("Recon: Nearest, No Lineshape")
    plt.colorbar()

    plt.subplot(2, 4, 6)
    plt.imshow(recon_siddon, cmap='viridis', origin='lower')
    plt.title("Recon: Siddon-like, No Lineshape")
    plt.colorbar()

    plt.subplot(2, 4, 7)
    plt.imshow(recon_lineshape, cmap='viridis', origin='lower')
    plt.title(f"Recon: Nearest, Gaussian Lineshape (FWHM={lineshape_fwhm_for_recon})")
    plt.colorbar()

    plt.subplot(2, 4, 8) # Placeholder for a profile or comparison
    # Example: Plot a central row from phantom and recons
    center_row_idx = grid_size[0] // 2
    plt.plot(phantom_true[center_row_idx, :], label='Phantom True')
    plt.plot(recon_basic[center_row_idx, :], label='Recon Basic', alpha=0.7)
    plt.plot(recon_siddon[center_row_idx, :], label='Recon Siddon', alpha=0.7)
    plt.plot(recon_lineshape[center_row_idx, :], label='Recon Lineshape', alpha=0.7)
    plt.title(f"Profile of Center Row ({center_row_idx})")
    plt.xlabel("Pixel Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
    print("Example script finished.")
