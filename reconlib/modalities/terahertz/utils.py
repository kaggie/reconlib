import torch
import numpy as np

def generate_thz_phantom(image_shape: tuple[int, int], num_shapes: int = 2, shape_type: str = 'rect', device: str | torch.device = 'cpu') -> torch.Tensor:
    """
    Generates a simple Terahertz phantom with random rectangles or circles.
    THz images often represent variations in refractive index or absorption.

    Args:
        image_shape (tuple[int, int]): The shape of the image (Ny, Nx).
        num_shapes (int, optional): Number of shapes to generate. Defaults to 2.
        shape_type (str, optional): Type of shapes ('rect', 'circle'). Defaults to 'rect'.
        device (str | torch.device, optional): Device for the tensor. Defaults to 'cpu'.

    Returns:
        torch.Tensor: A 2D tensor representing the phantom. Values typically float.
    """
    Ny, Nx = image_shape
    phantom = torch.zeros(image_shape, dtype=torch.float32, device=device)

    for _ in range(num_shapes):
        val = np.random.rand() * 0.8 + 0.2 # Intensity between 0.2 and 1.0
        if shape_type == 'rect':
            w = np.random.randint(Nx // 10, Nx // 3)
            h = np.random.randint(Ny // 10, Ny // 3)
            x0 = np.random.randint(0, Nx - w)
            y0 = np.random.randint(0, Ny - h)
            phantom[y0:y0+h, x0:x0+w] += val
        elif shape_type == 'circle':
            radius = np.random.randint(min(Ny, Nx) // 10, min(Ny, Nx) // 4)
            cx = np.random.randint(radius, Nx - radius)
            cy = np.random.randint(radius, Ny - radius)
            y_coords, x_coords = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')
            dist_sq = (y_coords - cy)**2 + (x_coords - cx)**2
            phantom[dist_sq < radius**2] += val
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}. Choose 'rect' or 'circle'.")

    phantom = torch.clamp(phantom, 0, phantom.max() if phantom.max() > 0 else 1.0) # Normalize if needed, or just clip
    return phantom

def plot_thz_results(
    true_image: torch.Tensor | None = None,
    reconstructed_image: torch.Tensor | None = None,
    measurement_data: torch.Tensor | None = None,
    slice_idx: int | None = None # For 3D data
    ):
    """
    Placeholder for plotting Terahertz Imaging results.

    Args:
        true_image (torch.Tensor | None): Ground truth THz image (e.g., refractive index map).
        reconstructed_image (torch.Tensor | None): Reconstructed THz image.
        measurement_data (torch.Tensor | None): Recorded THz sensor/measurement data.
        slice_idx (int | None): Index of 2D slice to display for 3D data.
    """
    print("plot_thz_results: Placeholder - Plotting not implemented.")
    print("Available data for plotting:")
    if true_image is not None:
        print(f"  True THz image: shape {true_image.shape}")
    if reconstructed_image is not None:
        print(f"  Reconstructed THz image: shape {reconstructed_image.shape}")
    if measurement_data is not None:
        print(f"  Measurement data: shape {measurement_data.shape}")

    # TODO: Implement actual plotting using matplotlib or similar
    # Example:
    # import matplotlib.pyplot as plt
    # num_plots = sum(x is not None for x in [true_image, reconstructed_image, measurement_data])
    # if num_plots == 0: return
    # fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    # current_ax = 0
    # if true_image is not None:
    #     ax = axes[current_ax] if num_plots > 1 else axes
    #     im = ax.imshow(true_image.cpu().numpy())
    #     ax.set_title("True THz Image")
    #     plt.colorbar(im, ax=ax)
    #     current_ax += 1
    # # ... and so on for other images/data
    # plt.show()

if __name__ == '__main__':
    print("Running basic Terahertz utils checks...")
    device_utils_thz = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_phantom_rect = generate_thz_phantom(image_shape=(64, 64), num_shapes=3, shape_type='rect', device=device_utils_thz)
    print(f"Generated rect phantom shape: {test_phantom_rect.shape}, dtype: {test_phantom_rect.dtype}")
    assert test_phantom_rect.shape == (64, 64)

    test_phantom_circle = generate_thz_phantom(image_shape=(64, 64), num_shapes=2, shape_type='circle', device=device_utils_thz)
    print(f"Generated circle phantom shape: {test_phantom_circle.shape}, dtype: {test_phantom_circle.dtype}")
    assert test_phantom_circle.shape == (64, 64)

    plot_thz_results(true_image=test_phantom_rect, reconstructed_image=test_phantom_circle*0.8)
    print("Terahertz utils checks completed.")
