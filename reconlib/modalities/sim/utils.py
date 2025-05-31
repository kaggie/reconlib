import torch
import numpy as np

def generate_sim_phantom_hr(image_shape: tuple[int, int], num_details: int = 10, device='cpu') -> torch.Tensor:
    """ Generates a high-resolution phantom with fine details for SIM. """
    phantom = torch.zeros(image_shape, dtype=torch.float32, device=device)
    # Add some fine lines / points
    for _ in range(num_details):
        is_line = np.random.rand() > 0.5
        val = np.random.rand()*0.5 + 0.5
        if is_line:
            is_horiz = np.random.rand() > 0.5
            if is_horiz:
                row = np.random.randint(0, image_shape[0])
                col_start = np.random.randint(0, image_shape[1]//2)
                col_end = col_start + np.random.randint(image_shape[1]//10, image_shape[1]//3)
                phantom[row, col_start:min(col_end, image_shape[1])] = val
            else:
                col = np.random.randint(0, image_shape[1])
                row_start = np.random.randint(0, image_shape[0]//2)
                row_end = row_start + np.random.randint(image_shape[0]//10, image_shape[0]//3)
                phantom[row_start:min(row_end, image_shape[0]), col] = val
        else: # Point-like detail
            r, c = np.random.randint(0, image_shape[0]), np.random.randint(0, image_shape[1])
            phantom[r,c] = val + 0.2 # Make points brighter
    return torch.clamp(phantom,0,1)

def generate_sim_patterns(hr_image_shape: tuple[int,int], num_angles: int, num_phases: int,
                          spatial_frequency_factor: float = 0.2, # Fraction of Nyquist limit
                          device='cpu') -> torch.Tensor:
    """
    Generates sinusoidal illumination patterns for SIM.
    spatial_frequency_factor: k / (Nyquist = 0.5 * min_dim_pixels)
    """
    Ny, Nx = hr_image_shape
    patterns = torch.zeros((num_angles * num_phases, Ny, Nx), device=device)

    # Determine k_max (related to image dimensions for pixel units)
    # Max spatial frequency that can be represented is 0.5 cycles/pixel.
    # Let's define k as cycles per image width/height.
    # k_abs = spatial_frequency_factor * (min(Ny,Nx) / 2.0)
    # For simplicity, let k be a fraction of the image dimension
    k_val = spatial_frequency_factor * min(Ny,Nx)


    idx = 0
    for i_angle in range(num_angles):
        angle = i_angle * (np.pi / num_angles) # Orientations over 0 to pi
        kx = k_val * np.cos(angle)
        ky = k_val * np.sin(angle)

        yy, xx = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')

        for i_phase in range(num_phases):
            phase = i_phase * (2 * np.pi / num_phases)
            # Pattern: 0.5 * (1 + cos(2*pi*(kx*x + ky*y)/N + phase)) normalized to [0,1]
            # Using pixel coordinates directly:
            patterns[idx, ...] = (torch.cos( (xx * ky + yy * kx) * (2*np.pi/min(Ny,Nx)) + phase ) + 1) / 2.0
            idx += 1
    return patterns


def plot_sim_results(true_hr_map, raw_sim_images, reconstructed_hr_map, num_raw_to_show=3):
    """ Placeholder to plot SIM results. """
    print("plot_sim_results: Placeholder - Plotting not implemented.")
    print(f"  True HR map shape: {true_hr_map.shape if true_hr_map is not None else 'N/A'}")
    print(f"  Raw SIM images stack shape: {raw_sim_images.shape if raw_sim_images is not None else 'N/A'}")
    print(f"  Reconstructed HR map shape: {reconstructed_hr_map.shape if reconstructed_hr_map is not None else 'N/A'}")
    # TODO: Implement plotting (e.g., true, a few raw images, reconstruction)
    # import matplotlib.pyplot as plt
    # ...

if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hr_s_utils = (64,64)
    phantom = generate_sim_phantom_hr(hr_s_utils, device=dev_utils)
    assert phantom.shape == hr_s_utils
    print(f"SIM HR phantom generated: {phantom.shape}")

    patterns_sim = generate_sim_patterns(hr_s_utils, num_angles=3, num_phases=3, device=dev_utils)
    assert patterns_sim.shape == (9, *hr_s_utils)
    print(f"SIM patterns generated: {patterns_sim.shape}")

    plot_sim_results(phantom, patterns_sim, phantom*0.8) # Dummy data for plot call
    print("SIM utils checks completed.")
