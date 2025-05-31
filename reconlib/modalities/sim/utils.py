import torch
import numpy as np

def generate_sim_phantom_hr(image_shape: tuple[int, int], num_details: int = 10, detail_type: str = 'mixed', device='cpu') -> torch.Tensor:
    """
    Generates a high-resolution phantom with fine details for SIM.

    Args:
        image_shape (tuple[int, int]): Shape of the high-resolution image (Ny, Nx).
        num_details (int): Number of detailed structures to add.
        detail_type (str): 'lines', 'points', or 'mixed'.
        device (str or torch.device): Device for the tensor.
    """
    phantom = torch.zeros(image_shape, dtype=torch.float32, device=device)
    Ny, Nx = image_shape

    for _ in range(num_details):
        current_detail_type = detail_type
        if detail_type == 'mixed':
            current_detail_type = 'line' if np.random.rand() > 0.5 else 'point'

        val = np.random.rand()*0.5 + 0.5 # Intensity 0.5 to 1.0

        if current_detail_type == 'line':
            is_horiz = np.random.rand() > 0.5
            if is_horiz:
                row = np.random.randint(0, Ny)
                col_start = np.random.randint(0, Nx // 2)
                # Make lines relatively fine, e.g., 1-2 pixels thick if possible, or longer
                line_length = np.random.randint(Nx // 20, Nx // 5)
                line_thickness = 1 # np.random.randint(1,3) # For thicker lines
                col_end = col_start + line_length
                phantom[row:min(row+line_thickness, Ny), col_start:min(col_end, Nx)] = val
            else: # Vertical line
                col = np.random.randint(0, Nx)
                row_start = np.random.randint(0, Ny // 2)
                line_length = np.random.randint(Ny // 20, Ny // 5)
                line_thickness = 1 # np.random.randint(1,3)
                row_end = row_start + line_length
                phantom[row_start:min(row_end, Ny), col:min(col+line_thickness, Nx)] = val

        elif current_detail_type == 'point': # Point-like detail (e.g., 1x1 or 2x2 pixels)
            r = np.random.randint(0, Ny -1) # -1 to allow for 2x2
            c = np.random.randint(0, Nx -1)
            point_size = np.random.randint(1,3) # 1x1 or 2x2
            phantom[r:min(r+point_size, Ny), c:min(c+point_size, Nx)] = val + 0.2 # Make points slightly brighter

    return torch.clamp(phantom, 0, 1.0)


def generate_sim_patterns(
    hr_image_shape: tuple[int,int],
    num_angles: int,
    num_phases: int,
    k_vector_max_rel: float = 0.8, # Pattern spatial frequency relative to OTF cutoff (approx Nyquist/2 for HR image)
    modulation_depth: float = 1.0, # Modulation depth 'm' of the pattern
    mean_intensity: float = 0.5,   # Mean intensity 'I0' of the pattern
    device='cpu'
) -> torch.Tensor:
    """
    Generates sinusoidal illumination patterns for SIM: I(r) = I0 * (1 + m * cos(k.r + phase)).
    Patterns are normalized to be positive.

    Args:
        hr_image_shape (tuple[int,int]): Shape of the high-resolution image (Ny, Nx).
        num_angles (int): Number of pattern orientations (e.g., 3 for standard 2D SIM).
        num_phases (int): Number of phases for each orientation (e.g., 3 or 5).
        k_vector_max_rel (float): Spatial frequency of the pattern, defined as a fraction
                                 of the HR grid's Nyquist frequency (0.5 cycles/pixel).
                                 E.g., 0.8 means 0.8 * 0.5 = 0.4 cycles/pixel.
        modulation_depth (float): Modulation depth 'm' (0 to 1.0).
        mean_intensity (float): This argument is noted but the formula is fixed to produce
                                patterns that vary around 0.5 if modulation_depth is 1.
                                The output is effectively (1 + m*cos(...))/2.
    Returns:
        torch.Tensor: Stack of patterns (num_angles*num_phases, Ny, Nx), values approx [0,1].
    """
    Ny, Nx = hr_image_shape
    patterns = torch.zeros((num_angles * num_phases, Ny, Nx), device=device, dtype=torch.float32)

    # k_val is cycles over the Field of View (along the shortest dimension direction)
    k_val_cycles_per_fov = k_vector_max_rel * (min(Ny, Nx) / 2.0)

    yy, xx = torch.meshgrid(torch.arange(Ny, dtype=torch.float32, device=device),
                            torch.arange(Nx, dtype=torch.float32, device=device), indexing='ij')

    idx = 0
    for i_angle in range(num_angles):
        # Orientations from 0 to pi (exclusive for last angle if it's same as first)
        angle_rad = i_angle * (np.pi / num_angles)

        # k components in cycles per pixel: (k_cycles_fov / N_pixels_axis)
        # These are then multiplied by pixel coordinates (xx, yy) and 2*pi for the cosine argument.
        kx_cpp = (k_val_cycles_per_fov * np.cos(angle_rad)) / Nx
        ky_cpp = (k_val_cycles_per_fov * np.sin(angle_rad)) / Ny

        for i_phase in range(num_phases):
            phase_rad = i_phase * (2 * np.pi / num_phases)

            # Pattern: I_final = 0.5 * (1 + m * cos(2*pi*(kx_cpp*x_pix + ky_cpp*y_pix) + phase_rad) )
            cos_term = torch.cos(2 * np.pi * (kx_cpp * xx + ky_cpp * yy) + phase_rad)
            illumination = 0.5 * (1 + modulation_depth * cos_term)
            patterns[idx, ...] = torch.clamp(illumination, 0.0, 1.0)
            idx += 1
    return patterns


def plot_sim_results(
    true_hr_map: torch.Tensor | None = None,
    raw_sim_images: torch.Tensor | None = None, # Stack (num_patterns, Ny, Nx)
    reconstructed_hr_map: torch.Tensor | None = None,
    detection_psf: torch.Tensor | None = None,
    sim_patterns: torch.Tensor | None = None, # Stack (num_patterns, Ny, Nx)
    num_raw_to_show: int = 3,
    num_patterns_to_show: int = 3
    ):
    """ Placeholder to plot SIM results. """
    print("plot_sim_results: Details for plotting SIM results:")

    if true_hr_map is not None:
        print(f"  - True HR map provided: shape {true_hr_map.shape}, dtype {true_hr_map.dtype}")
    if detection_psf is not None:
        print(f"  - Detection PSF provided: shape {detection_psf.shape}, dtype {detection_psf.dtype}")
    if sim_patterns is not None:
        print(f"  - SIM Patterns stack provided: shape {sim_patterns.shape}, dtype {sim_patterns.dtype}")
        print(f"    (Will attempt to show first {min(num_patterns_to_show, sim_patterns.shape[0])} patterns)")
    if raw_sim_images is not None:
        print(f"  - Raw SIM images stack provided: shape {raw_sim_images.shape}, dtype {raw_sim_images.dtype}")
        print(f"    (Will attempt to show first {min(num_raw_to_show, raw_sim_images.shape[0])} raw images)")
    if reconstructed_hr_map is not None:
        print(f"  - Reconstructed HR map provided: shape {reconstructed_hr_map.shape}, dtype {reconstructed_hr_map.dtype}")

    # TODO: Implement actual plotting using matplotlib.
    # Example structure:
    # import matplotlib.pyplot as plt
    # num_cols = max(1, num_raw_to_show if raw_sim_images is not None else 0, num_patterns_to_show if sim_patterns is not None else 0)
    # num_rows = sum(x is not None for x in [true_hr_map, reconstructed_hr_map]) +     #            (1 if raw_sim_images is not None else 0) +     #            (1 if sim_patterns is not None else 0) +     #            (1 if detection_psf is not None else 0)
    # if num_rows == 0: print("No data to plot."); return
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3)) # Adjust figsize
    # # ... logic to populate axes ...
    # plt.tight_layout()
    # plt.show()
    print("--- End of plot_sim_results (plotting not implemented) ---")


if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hr_s_utils = (64,64)

    print("\nTesting generate_sim_phantom_hr...")
    phantom = generate_sim_phantom_hr(hr_s_utils, num_details=15, detail_type='mixed', device=dev_utils)
    assert phantom.shape == hr_s_utils
    print(f"SIM HR phantom generated: {phantom.shape}")

    print("\nTesting generate_sim_patterns...")
    patterns_sim = generate_sim_patterns(
        hr_s_utils, num_angles=3, num_phases=3,
        k_vector_max_rel=0.8, modulation_depth=0.9, device=dev_utils
    )
    assert patterns_sim.shape == (9, *hr_s_utils)
    print(f"SIM patterns generated: {patterns_sim.shape}, min: {patterns_sim.min()}, max: {patterns_sim.max()}")

    # import matplotlib.pyplot as plt # For visual check
    # fig, axes = plt.subplots(1,3, figsize=(9,3))
    # axes[0].imshow(patterns_sim[0].cpu()); axes[0].set_title("Pat 0")
    # axes[1].imshow(patterns_sim[patterns_sim.shape[0]//2].cpu()); axes[1].set_title("Pat mid")
    # axes[2].imshow(patterns_sim[-1].cpu()); axes[2].set_title("Pat last")
    # plt.show()
    # fig, ax = plt.subplots(1,1); ax.imshow(phantom.cpu()); ax.set_title("Phantom"); plt.show()


    plot_sim_results(
        true_hr_map=phantom,
        raw_sim_images=patterns_sim, # Just to pass something
        reconstructed_hr_map=phantom*0.8,
        detection_psf=torch.rand(5,5,device=dev_utils),
        sim_patterns=patterns_sim
    )
    print("SIM utils checks completed.")
