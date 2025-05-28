import numpy as np
import scipy.ndimage
import scipy.fft

EPSILON = 1e-9

# Placeholder functions for operations not directly implemented yet
def regrid_kspace(kspace_data, trajectory, grid_size, density_weights=None):
    """Placeholder for k-space regridding operation."""
    raise NotImplementedError("regrid_kspace not yet implemented.")

def filter_low_frequencies(kspace_data, trajectory):
    """Placeholder for filtering low k-space frequencies."""
    raise NotImplementedError("filter_low_frequencies not yet implemented.")

def extract_calibration_region(kspace_data, trajectory):
    """Placeholder for extracting calibration region for ESPIRiT."""
    raise NotImplementedError("extract_calibration_region not yet implemented.")

def compute_espirit_kernel(calibration_data):
    """Placeholder for computing ESPIRiT kernel."""
    raise NotImplementedError("compute_espirit_kernel not yet implemented.")

def apply_espirit_kernel(kernel, grid_size):
    """Placeholder for applying ESPIRiT kernel."""
    raise NotImplementedError("apply_espirit_kernel not yet implemented.")

def compute_voronoi_tessellation(coords):
    """Placeholder for Voronoi tessellation."""
    raise NotImplementedError("compute_voronoi_tessellation not yet implemented. Consider using scipy.spatial.Voronoi if available and appropriate.")

def compute_polygon_area(vertices):
    """Placeholder for computing polygon area."""
    raise NotImplementedError("compute_polygon_area not yet implemented.")


def coil_combination_with_phase(coil_images, method="sos", sensitivity_maps=None, phase_maps=None):
    """
    Combine images from multiple coils with phase correction.

    Args:
        coil_images (np.ndarray): Array of reconstructed coil images, shape (num_coils, Nx, Ny[, Nz]).
        method (str): "sos" (sum-of-squares), "roemer" (optimal), or "sos_sensitivity" (SoS with sensitivity).
        sensitivity_maps (np.ndarray, optional): Coil sensitivity maps, shape (num_coils, Nx, Ny[, Nz]).
        phase_maps (np.ndarray, optional): Coil phase maps, shape (num_coils, Nx, Ny[, Nz]).

    Returns:
        np.ndarray: Combined image, shape (Nx, Ny[, Nz]).
    """
    if not isinstance(coil_images, np.ndarray):
        raise TypeError("coil_images must be a NumPy array.")
    if coil_images.ndim < 3:
        raise ValueError("coil_images must have at least 3 dimensions (num_coils, Nx, Ny).")

    num_coils = coil_images.shape[0]
    image_shape = coil_images.shape[1:]

    # Step 1: Estimate phase maps if not provided
    if phase_maps is None:
        phase_maps = estimate_phase_maps(coil_images, method="lowres")

    if phase_maps.shape != coil_images.shape:
        raise ValueError(f"phase_maps shape {phase_maps.shape} must match coil_images shape {coil_images.shape}.")

    # Step 2: Apply phase correction to coil images
    corrected_images = np.empty_like(coil_images, dtype=np.complex128)
    for coil in range(num_coils):
        corrected_images[coil] = coil_images[coil] * np.exp(-1j * phase_maps[coil])

    # Step 3: Perform coil combination
    if method == "sos":
        combined_image = np.sum(np.abs(corrected_images) ** 2, axis=0)
        combined_image = np.sqrt(combined_image)
    elif method == "roemer":
        if sensitivity_maps is None:
            raise ValueError("Sensitivity maps required for Roemer combination.")
        if sensitivity_maps.shape != coil_images.shape:
            raise ValueError(f"sensitivity_maps shape {sensitivity_maps.shape} must match coil_images shape {coil_images.shape}.")
        
        # Reshape for vectorized operations: (num_coils, N_pixels)
        # N_pixels = np.prod(image_shape)
        # corrected_images_flat = corrected_images.reshape(num_coils, N_pixels)
        # sensitivity_maps_flat = sensitivity_maps.reshape(num_coils, N_pixels)
        
        # combined_image_flat = np.zeros(N_pixels, dtype=np.complex128)
        
        # norm_sens_sq = np.sum(np.abs(sensitivity_maps_flat)**2, axis=0) # Norm squared for each pixel
        # weights_numerator = np.conj(sensitivity_maps_flat)
        
        # # Avoid division by zero where norm_sens_sq is close to zero
        # valid_pixels = norm_sens_sq > EPSILON
        
        # weights = np.zeros_like(sensitivity_maps_flat, dtype=np.complex128)
        # weights[:, valid_pixels] = weights_numerator[:, valid_pixels] / norm_sens_sq[valid_pixels]
        
        # combined_image_flat[valid_pixels] = np.sum(corrected_images_flat[:, valid_pixels] * weights[:, valid_pixels], axis=0)
        # combined_image = np.abs(combined_image_flat.reshape(image_shape))

        # Using broadcasting and avoiding explicit pixel loop
        # Denominator: sum(|S_i|^2) over coils for each pixel
        sensitivity_norm_sq = np.sum(np.abs(sensitivity_maps)**2, axis=0) # Shape: (Nx, Ny[, Nz])
        
        # Weights: S_i* / sum(|S_j|^2)
        # Add EPSILON to avoid division by zero
        weights_numerator = np.conj(sensitivity_maps) # Shape: (num_coils, Nx, Ny[, Nz])
        
        # Ensure sensitivity_norm_sq has same dimensions as weights_numerator for broadcasting
        # This is usually fine if sensitivity_norm_sq is (Nx, Ny) and weights_numerator is (num_coils, Nx, Ny)
        # but if sensitivity_norm_sq is (Nx,Ny,Nz), it matches weights_numerator's trailing dims.
        
        weights = weights_numerator / (sensitivity_norm_sq[np.newaxis, ...] + EPSILON) # Add newaxis for broadcasting over coils
        
        # Combined image: sum(C_i * W_i) over coils
        combined_image = np.sum(corrected_images * weights, axis=0)
        combined_image = np.abs(combined_image)

    elif method == "sos_sensitivity":
        if sensitivity_maps is None:
            raise ValueError("Sensitivity maps required for SoS with sensitivity.")
        if sensitivity_maps.shape != coil_images.shape:
            raise ValueError(f"sensitivity_maps shape {sensitivity_maps.shape} must match coil_images shape {coil_images.shape}.")
        
        weighted_images_sq = (np.abs(corrected_images) * np.abs(sensitivity_maps)) ** 2
        combined_image = np.sqrt(np.sum(weighted_images_sq, axis=0))
    else:
        raise ValueError(f"Unknown coil combination method: {method}")

    # Step 4: Normalize output
    max_val = np.max(combined_image)
    if max_val > 0:
        combined_image = combined_image / max_val

    return combined_image


def estimate_phase_maps(coil_images, method="lowres", reference_coil=0):
    """
    Estimate phase maps for each coil to align phases.

    Args:
        coil_images (np.ndarray): Array of reconstructed coil images, shape (num_coils, Nx, Ny[, Nz]).
        method (str): "lowres" (low-resolution phase estimation) or "reference" (relative to a reference coil).
        reference_coil (int): Index of reference coil for "reference" method.

    Returns:
        np.ndarray: Phase maps, shape (num_coils, Nx, Ny[, Nz]).
    """
    if not isinstance(coil_images, np.ndarray):
        raise TypeError("coil_images must be a NumPy array.")
    if coil_images.ndim < 3:
        raise ValueError("coil_images must have at least 3 dimensions (num_coils, Nx, Ny).")

    num_coils = coil_images.shape[0]
    image_shape = coil_images.shape[1:]
    phase_maps = np.empty_like(coil_images, dtype=float) # Phase is real

    if method == "lowres":
        # Determine sigma for Gaussian filter based on image size (heuristic)
        # For a cutoff of 0.1, this means features smaller than 10% of image size are smoothed.
        # Sigma is related to cutoff frequency. A common heuristic: sigma = 1 / (2 * pi * cutoff_norm)
        # cutoff_norm = cutoff_freq / sampling_freq. Here, sampling_freq is related to image dimension.
        # Let's use a fraction of the image dimension for sigma.
        # For example, sigma = 0.05 * min(image_shape)
        sigmas = [0.05 * s for s in image_shape] # Separate sigma for each dimension
        if len(sigmas) == 2: # 2D case
             sigmas_for_filter = [0] + sigmas # Add 0 for coil dimension
        else: # 3D case
             sigmas_for_filter = [0] + sigmas # Add 0 for coil dimension

        for coil in range(num_coils):
            # Apply low-pass filter to complex image, then extract phase
            # It's generally better to filter the complex image then take the angle,
            # rather than filtering the phase directly, which can have wrapping issues.
            # However, pseudocode suggests filtering the image and then taking angle.
            # To match pseudocode "LOW_PASS_FILTER(coil_images[coil], cutoff=0.1)"
            # we filter the magnitude and phase separately if we interpret it strictly,
            # or more commonly, filter the complex image.
            # Let's filter the complex image by filtering real and imaginary parts.
            lowres_real = scipy.ndimage.gaussian_filter(coil_images[coil].real, sigma=sigmas, mode='wrap')
            lowres_imag = scipy.ndimage.gaussian_filter(coil_images[coil].imag, sigma=sigmas, mode='wrap')
            lowres_image = lowres_real + 1j * lowres_imag
            phase_maps[coil] = np.angle(lowres_image)
    elif method == "reference":
        if not (0 <= reference_coil < num_coils):
            raise ValueError(f"reference_coil index {reference_coil} is out of bounds for {num_coils} coils.")
        ref_phase = np.angle(coil_images[reference_coil])
        for coil in range(num_coils):
            phase_maps[coil] = np.angle(coil_images[coil]) - ref_phase
            # Unwrap phase differences? Pseudocode doesn't specify, but could be useful.
            # phase_maps[coil] = np.unwrap(phase_maps[coil]) # Might be too aggressive
    else:
        raise ValueError(f"Unknown phase estimation method: {method}")

    # Smooth phase maps to reduce noise
    # Kernel size 5 suggests a sigma of around 1 to 1.5 for Gaussian filter
    # For multiple dimensions, kernel_size=5 might mean (5,5) or (5,5,5)
    # Let's use a fixed sigma for smoothing.
    smoothing_sigma = [1.5] * len(image_shape) # Sigma for each spatial dimension
    for coil in range(num_coils):
        phase_maps[coil] = scipy.ndimage.gaussian_filter(phase_maps[coil], sigma=smoothing_sigma, mode='wrap')
        # Note: 'wrap' mode for phase data is important.

    return phase_maps


def estimate_sensitivity_maps(kspace_data, trajectory, grid_size, method="lowres"):
    """
    Estimate coil sensitivity maps from k-space data.

    Args:
        kspace_data (np.ndarray): Complex k-space data, shape (num_coils, num_arms, num_samples) or (num_coils, N_kpoints).
        trajectory (np.ndarray): K-space trajectory, shape (num_arms, num_samples, 2 or 3) or (N_kpoints, 2 or 3).
        grid_size (list or tuple): Size of output image grid (e.g., [Nx, Ny]).
        method (str): "lowres" (low-resolution image) or "espirit".

    Returns:
        np.ndarray: Complex array, shape (num_coils, Nx, Ny) or (num_coils, Nx, Ny, Nz).
    """
    if not isinstance(kspace_data, np.ndarray) or not np.iscomplexobj(kspace_data):
        raise TypeError("kspace_data must be a complex NumPy array.")
    if kspace_data.ndim < 2: # (num_coils, N_kpoints)
        raise ValueError("kspace_data must have at least 2 dimensions.")
    
    num_coils = kspace_data.shape[0]
    
    # Ensure grid_size is a tuple for consistency
    if not isinstance(grid_size, tuple):
        grid_size = tuple(grid_size)
        
    sensitivity_maps_shape = (num_coils,) + grid_size
    sensitivity_maps = np.empty(sensitivity_maps_shape, dtype=np.complex128)

    if method == "lowres":
        # This implementation assumes regrid_kspace and filter_low_frequencies are available.
        # The division by 2 for grid_size in regrid_kspace call is a bit arbitrary,
        # depends on how "low resolution" is defined. Let's assume a fixed low-res grid.
        # For example, grid_size // 4 or a fixed size like (32,32).
        # Pseudocode has grid_size / 2 for regrid, then upsample to grid_size.
        
        low_res_grid_size = tuple(s // 2 for s in grid_size)
        if any(s == 0 for s in low_res_grid_size): # Ensure grid dims are not zero
            low_res_grid_size = tuple(max(1,s) for s in low_res_grid_size)

        for coil in range(num_coils):
            # 1. Filter low frequencies (conceptual step, depends on definition)
            # Assuming kspace_data[coil] might be (num_arms, num_samples)
            # and trajectory might be (num_arms, num_samples, dims)
            # For simplicity, we'll assume filter_low_frequencies returns k-space data
            # that can be directly used by regrid_kspace.
            # This might involve selecting k-space points near the center.
            lowres_kspace_filtered = filter_low_frequencies(kspace_data[coil], trajectory) # Placeholder
            
            # 2. Regrid to a low-resolution Cartesian grid
            # The trajectory passed to regrid_kspace here should correspond to lowres_kspace_filtered
            lowres_cartesian_kspace = regrid_kspace(lowres_kspace_filtered, trajectory, low_res_grid_size) # Placeholder

            # 3. Inverse FFT
            if len(grid_size) == 2:
                lowres_image = scipy.fft.ifft2(scipy.fft.ifftshift(lowres_cartesian_kspace))
            elif len(grid_size) == 3:
                lowres_image = scipy.fft.ifftn(scipy.fft.ifftshift(lowres_cartesian_kspace))
            else:
                raise ValueError(f"Unsupported grid dimension: {len(grid_size)}")
            
            # 4. Upsample to original grid_size
            zoom_factors = [gs_orig / gs_low for gs_orig, gs_low in zip(grid_size, lowres_image.shape)]
            # scipy.ndimage.zoom expects zoom factors for each dimension.
            # If lowres_image is complex, zoom real and imaginary parts separately.
            s_map_real = scipy.ndimage.zoom(lowres_image.real, zoom_factors, order=1) # Bilinear interpolation
            s_map_imag = scipy.ndimage.zoom(lowres_image.imag, zoom_factors, order=1)
            sensitivity_maps[coil] = s_map_real + 1j * s_map_imag
        
        # Normalize by Sum-of-Squares of all coil sensitivities
        sos = np.sqrt(np.sum(np.abs(sensitivity_maps) ** 2, axis=0))
        
        # Add EPSILON for stability, expand sos dims to match sensitivity_maps for broadcasting
        sensitivity_maps = sensitivity_maps / (sos[np.newaxis, ...] + EPSILON)

    elif method == "espirit":
        # calibration_data = extract_calibration_region(kspace_data, trajectory) # Placeholder
        # for coil in range(num_coils):
        #     kernel = compute_espirit_kernel(calibration_data[coil]) # Placeholder
        #     sensitivity_maps[coil] = apply_espirit_kernel(kernel, grid_size) # Placeholder
        # sos = np.sqrt(np.sum(np.abs(sensitivity_maps) ** 2, axis=0))
        # sensitivity_maps = sensitivity_maps / (sos[np.newaxis, ...] + EPSILON)
        raise NotImplementedError("ESPIRiT method not yet implemented in this translation.")
    else:
        raise ValueError(f"Unknown sensitivity estimation method: {method}")

    return sensitivity_maps


def reconstruct_coil_images(kspace_data, trajectory, grid_size, density_weights=None):
    """
    Reconstruct individual coil images from non-Cartesian k-space data.

    Args:
        kspace_data (np.ndarray): Complex k-space data, shape (num_coils, num_arms, num_samples) or (num_coils, N_kpoints).
        trajectory (np.ndarray): K-space trajectory, shape (num_arms, num_samples, 2 or 3) or (N_kpoints, 2 or 3).
        grid_size (list or tuple): Size of output image grid (e.g., [Nx, Ny]).
        density_weights (np.ndarray, optional): Density compensation weights. Shape should match trajectory samples.

    Returns:
        np.ndarray: Complex array of coil images, shape (num_coils, Nx, Ny) or (num_coils, Nx, Ny, Nz).
    """
    if not isinstance(kspace_data, np.ndarray) or not np.iscomplexobj(kspace_data):
        raise TypeError("kspace_data must be a complex NumPy array.")
    if kspace_data.ndim < 2:
        raise ValueError("kspace_data must have at least 2 dimensions.")

    num_coils = kspace_data.shape[0]
    if not isinstance(grid_size, tuple):
        grid_size = tuple(grid_size)
        
    coil_images_shape = (num_coils,) + grid_size
    coil_images = np.empty(coil_images_shape, dtype=np.complex128)

    if density_weights is None:
        # Assuming trajectory might be (num_arms, num_samples, dims) or (N_kpoints, dims)
        # compute_density_compensation needs to handle these shapes.
        density_weights = compute_density_compensation(trajectory, method="pipe") # Default to pipe if Voronoi is complex

    # Validate shape of density_weights against kspace_data[coil] and trajectory
    # If kspace_data is (num_coils, num_arms, num_samples), density_weights could be (num_arms, num_samples)
    # If kspace_data is (num_coils, N_kpoints), density_weights could be (N_kpoints,)
    # This needs to be handled inside regrid_kspace or by reshaping density_weights.
    # For now, assume density_weights has a compatible shape for regrid_kspace.

    for coil in range(num_coils):
        # The regrid_kspace function is assumed to handle non-Cartesian data and apply density compensation.
        # kspace_data[coil] would be (num_arms, num_samples) or (N_kpoints)
        # trajectory would be (num_arms, num_samples, dims) or (N_kpoints, dims)
        gridded_k_space = regrid_kspace(kspace_data[coil], trajectory, grid_size, density_weights) # Placeholder
        
        # Perform Inverse FFT (centered)
        # Use ifftshift before IFFT to handle k-space center, and fftshift after for image center (if needed by convention)
        # Standard is ifftshift -> ifft -> fftshift
        shifted_gridded_k_space = scipy.fft.ifftshift(gridded_k_space)
        if len(grid_size) == 2:
            img = scipy.fft.ifft2(shifted_gridded_k_space)
        elif len(grid_size) == 3:
            img = scipy.fft.ifftn(shifted_gridded_k_space)
        else:
            raise ValueError(f"Unsupported grid dimension: {len(grid_size)}")
        coil_images[coil] = scipy.fft.fftshift(img) # Center the image

    return coil_images


def compute_density_compensation(trajectory, method="voronoi"):
    """
    Compute density compensation weights for non-Cartesian trajectory.

    Args:
        trajectory (np.ndarray): K-space trajectory.
                       Expected shapes: (num_total_samples, dims) for real coordinates,
                       or (num_total_samples,) for complex coordinates (dim inferred as 2).
                       `dims` is typically 2 or 3.
        method (str): "voronoi" or "pipe".

    Returns:
        np.ndarray: Density compensation weights. Shape will be (num_total_samples,).
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("trajectory must be a NumPy array.")

    # Determine shape of trajectory and ensure it's 2D (N_points, N_dims) or 1D (N_points, complex)
    if trajectory.ndim == 3: # (num_arms, num_samples, dims)
        num_arms, num_samples, dims = trajectory.shape
        trajectory_flat = trajectory.reshape(-1, dims) # (N_total_samples, dims)
        original_shape = (num_arms, num_samples)
    elif trajectory.ndim == 2: # (N_total_samples, dims) or (num_arms, num_samples) if complex
        if np.iscomplexobj(trajectory): # (num_arms, num_samples) complex, treat as (N_total_samples,) complex
            trajectory_flat = trajectory.flatten() # (N_total_samples,) complex
            original_shape = trajectory.shape
        else: # (N_total_samples, dims) real
            trajectory_flat = trajectory
            original_shape = (trajectory.shape[0],) # Output will be 1D
    elif trajectory.ndim == 1: # (N_total_samples,) complex or real (1D trajectory?)
        trajectory_flat = trajectory # Assume (N_total_samples,)
        original_shape = trajectory.shape
        if not np.iscomplexobj(trajectory_flat) and method != "pipe": # Pipe method can work with 1D real (radius)
             # For Voronoi, 1D real trajectory is ambiguous without more context.
             # Assuming it might be radial distance for pipe, or needs to be paired for Voronoi.
             pass # Allow pipe method, others might fail or need specific handling.
    else:
        raise ValueError(f"Unsupported trajectory shape: {trajectory.shape}")


    if method == "voronoi":
        # kx = REAL(trajectory)
        # ky = IMAG(trajectory)
        # coords = STACK(kx, ky, axis=2) // Shape: (num_arms, num_samples, 2)
        # ...
        # For simplicity, if Voronoi is too complex for direct translation now,
        # raise a NotImplementedError or return a simple default.
        raise NotImplementedError("Voronoi method for density compensation not yet implemented. Consider 'pipe' or a simpler method.")

    elif method == "pipe":
        # Determine if trajectory is complex or real coordinates
        if np.iscomplexobj(trajectory_flat): # (N_points,) complex
            radius = np.abs(trajectory_flat)
        elif trajectory_flat.ndim == 2 : # (N_points, dims) real
            radius = np.sqrt(np.sum(trajectory_flat**2, axis=-1))
        elif trajectory_flat.ndim == 1 and not np.iscomplexobj(trajectory_flat): # (N_points,) real, assume it's already radius
            radius = trajectory_flat
        else:
            raise ValueError(f"Unsupported trajectory format for pipe method: shape {trajectory_flat.shape}, dtype {trajectory_flat.dtype}")
        
        density_weights = radius
        if len(original_shape) > 1 and density_weights.ndim == 1: # Reshape if original was e.g. (num_arms, num_samples)
            density_weights = density_weights.reshape(original_shape)

    else:
        raise ValueError(f"Unknown density compensation method: {method}")

    # Optional normalization (pseudocode mentions it)
    # if np.max(density_weights) > 0:
    #    density_weights = density_weights / np.max(density_weights)
    return density_weights

# Example usage (commented out, for testing/illustration if run directly)
# if __name__ == '__main__':
    # Create some dummy data
    # num_coils, Nx, Ny, Nz = 4, 64, 64, 32
    # coil_imgs_3d = np.random.rand(num_coils, Nx, Ny, Nz) + 1j * np.random.rand(num_coils, Nx, Ny, Nz)
    # coil_imgs_2d = np.random.rand(num_coils, Nx, Ny) + 1j * np.random.rand(num_coils, Nx, Ny)
    
    # sens_maps_3d = np.random.rand(num_coils, Nx, Ny, Nz) + 1j * np.random.rand(num_coils, Nx, Ny, Nz)
    # sens_maps_2d = np.random.rand(num_coils, Nx, Ny) + 1j * np.random.rand(num_coils, Nx, Ny)
    
    # # Normalize sensitivity maps (as they typically are)
    # sos_3d = np.sqrt(np.sum(np.abs(sens_maps_3d)**2, axis=0))
    # sens_maps_3d = sens_maps_3d / (sos_3d[np.newaxis,...] + EPSILON)
    # sos_2d = np.sqrt(np.sum(np.abs(sens_maps_2d)**2, axis=0))
    # sens_maps_2d = sens_maps_2d / (sos_2d[np.newaxis,...] + EPSILON)

    # print("Testing 2D coil combination:")
    # try:
    #     combined_sos_2d = coil_combination_with_phase(coil_imgs_2d, method="sos")
    #     print(f"SOS 2D output shape: {combined_sos_2d.shape}")
    #     combined_roemer_2d = coil_combination_with_phase(coil_imgs_2d, method="roemer", sensitivity_maps=sens_maps_2d)
    #     print(f"Roemer 2D output shape: {combined_roemer_2d.shape}")
    #     combined_sos_sens_2d = coil_combination_with_phase(coil_imgs_2d, method="sos_sensitivity", sensitivity_maps=sens_maps_2d)
    #     print(f"SOS_Sensitivity 2D output shape: {combined_sos_sens_2d.shape}")
    # except Exception as e:
    #     print(f"Error in 2D combination: {e}")

    # print("\nTesting 3D coil combination:")
    # try:
    #     combined_sos_3d = coil_combination_with_phase(coil_imgs_3d, method="sos")
    #     print(f"SOS 3D output shape: {combined_sos_3d.shape}")
    #     combined_roemer_3d = coil_combination_with_phase(coil_imgs_3d, method="roemer", sensitivity_maps=sens_maps_3d)
    #     print(f"Roemer 3D output shape: {combined_roemer_3d.shape}")
    # except Exception as e:
    #     print(f"Error in 3D combination: {e}")

    # print("\nTesting phase estimation (2D):")
    # try:
    #     phase_maps_lowres_2d = estimate_phase_maps(coil_imgs_2d, method="lowres")
    #     print(f"Phase maps lowres 2D shape: {phase_maps_lowres_2d.shape}")
    #     phase_maps_ref_2d = estimate_phase_maps(coil_imgs_2d, method="reference", reference_coil=0)
    #     print(f"Phase maps reference 2D shape: {phase_maps_ref_2d.shape}")
    # except Exception as e:
    #     print(f"Error in phase estimation: {e}")

    # print("\nTesting density compensation:")
    # try:
        # traj_complex_1d = np.random.rand(100) + 1j*np.random.rand(100) # (N_points,)
        # traj_real_2d = np.random.rand(100, 2) # (N_points, 2)
        # traj_real_3d_arms = np.random.rand(10, 50, 2) # (num_arms, num_samples, 2)

        # dcw_pipe_1 = compute_density_compensation(traj_complex_1d, method="pipe")
        # print(f"DCW Pipe (complex 1D input) shape: {dcw_pipe_1.shape}")
        # dcw_pipe_2 = compute_density_compensation(traj_real_2d, method="pipe")
        # print(f"DCW Pipe (real 2D input) shape: {dcw_pipe_2.shape}")
        # dcw_pipe_3 = compute_density_compensation(traj_real_3d_arms, method="pipe")
        # print(f"DCW Pipe (real 3D input) shape: {dcw_pipe_3.shape}, expected {(10,50)}")

        # try:
        #     compute_density_compensation(traj_real_2d, method="voronoi")
        # except NotImplementedError as nie:
        #     print(f"Voronoi DCW: {nie}")
            
    # except Exception as e:
    #     print(f"Error in density compensation: {e}")

    # print("\nTesting sensitivity map estimation (placeholders):")
    # try:
    #     num_coils_k, num_arms_k, num_samples_k = 4, 16, 128
    #     k_data = np.random.randn(num_coils_k, num_arms_k, num_samples_k) + 1j * np.random.randn(num_coils_k, num_arms_k, num_samples_k)
    #     traj = np.random.randn(num_arms_k, num_samples_k, 2)
    #     grid_sz = (64,64)
    #     estimate_sensitivity_maps(k_data, traj, grid_sz, method="lowres")
    # except NotImplementedError as nie:
    #     print(f"Sensitivity maps (lowres): {nie}") # Expected due to placeholders
    # except Exception as e:
    #     print(f"Error in sensitivity map estimation: {e}")

    # print("\nTesting reconstruct_coil_images (placeholders):")
    # try:
    #     reconstruct_coil_images(k_data, traj, grid_sz)
    # except NotImplementedError as nie:
    #     print(f"Reconstruct coil images: {nie}") # Expected due to placeholders
    # except Exception as e:
    #     print(f"Error in reconstruct_coil_images: {e}")
