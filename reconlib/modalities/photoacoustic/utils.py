import torch
import numpy as np

def generate_pat_phantom(image_shape: tuple[int, int], num_circles: int = 3, device: str | torch.device = 'cpu') -> torch.Tensor:
    """
    Generates a simple photoacoustic phantom with random circles.

    Args:
        image_shape (tuple[int, int]): The shape of the image (Ny, Nx).
        num_circles (int, optional): Number of circles to generate. Defaults to 3.
        device (str | torch.device, optional): Device for the tensor. Defaults to 'cpu'.

    Returns:
        torch.Tensor: A 2D tensor representing the phantom.
    """
    Ny, Nx = image_shape
    phantom = torch.zeros(image_shape, device=device)

    for _ in range(num_circles):
        radius = np.random.randint(min(Ny, Nx) // 10, min(Ny, Nx) // 4)
        cx = np.random.randint(radius, Nx - radius)
        cy = np.random.randint(radius, Ny - radius)
        intensity = np.random.rand() * 0.5 + 0.5 # Intensity between 0.5 and 1.0

        y_coords, x_coords = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')
        dist_sq = (y_coords - cy)**2 + (x_coords - cx)**2
        phantom[dist_sq < radius**2] += intensity # Use += to allow overlaps to sum up

    phantom = torch.clamp(phantom, 0, 1.0) # Ensure values are within [0,1] if overlaps are strong
    return phantom

def plot_pat_results(
    initial_pressure_map: torch.Tensor | None = None,
    reconstructed_map: torch.Tensor | None = None,
    sensor_data: torch.Tensor | None = None,
    sensor_positions: torch.Tensor | None = None,
    slice_idx: int | None = None # For 3D data if applicable
    ):
    """
    Placeholder for plotting Photoacoustic Tomography results.

    Args:
        initial_pressure_map (torch.Tensor | None): Ground truth initial pressure.
        reconstructed_map (torch.Tensor | None): Reconstructed initial pressure.
        sensor_data (torch.Tensor | None): Recorded sensor data (sinogram-like).
        sensor_positions (torch.Tensor | None): Positions of the sensors.
        slice_idx (int | None): Index of 2D slice to display for 3D data.
    """
    print("plot_pat_results: Placeholder - Plotting not implemented.")
    print("Available data for plotting:")
    if initial_pressure_map is not None:
        print(f"  Initial pressure map: shape {initial_pressure_map.shape}")
    if reconstructed_map is not None:
        print(f"  Reconstructed map: shape {reconstructed_map.shape}")
    if sensor_data is not None:
        print(f"  Sensor data: shape {sensor_data.shape}")
    if sensor_positions is not None:
        print(f"  Sensor positions: shape {sensor_positions.shape}")

    # TODO: Implement actual plotting using matplotlib or similar
    # Example:
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 3 if sensor_data is not None else 2, figsize=(15, 5))
    # if initial_pressure_map is not None:
    #     axes[0].imshow(initial_pressure_map.cpu().numpy())
    #     axes[0].set_title("Ground Truth Pressure")
    # if reconstructed_map is not None:
    #     axes[1].imshow(reconstructed_map.cpu().numpy())
    #     axes[1].set_title("Reconstructed Pressure")
    # if sensor_data is not None:
    #     axes[2].imshow(sensor_data.cpu().numpy(), aspect='auto')
    #     axes[2].set_title("Sensor Data")
    # plt.show()

if __name__ == '__main__':
    print("Running basic Photoacoustic utils checks...")
    device_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_phantom = generate_pat_phantom(image_shape=(128, 128), num_circles=5, device=device_utils)
    print(f"Generated phantom shape: {test_phantom.shape}, dtype: {test_phantom.dtype}, device: {test_phantom.device}")
    assert test_phantom.shape == (128, 128)
    assert test_phantom.max() <= 1.0 and test_phantom.min() >= 0.0

    plot_pat_results(initial_pressure_map=test_phantom, reconstructed_map=test_phantom*0.8)
    print("Photoacoustic utils checks completed.")
