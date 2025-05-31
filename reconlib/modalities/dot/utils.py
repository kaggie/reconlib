import torch
import numpy as np

def generate_dot_phantom_delta_mu(image_shape: tuple[int, int], num_anomalies: int = 1, device='cpu') -> torch.Tensor:
    """
    Generates a phantom representing changes in an optical property (e.g., delta_mu_a)
    from a baseline. Values represent the change.
    """
    delta_mu_map = torch.zeros(image_shape, dtype=torch.float32, device=device)
    for _ in range(num_anomalies):
        # Optical property change (e.g., for mu_a, typically positive for an absorber)
        # Units depend on what J matrix would represent. For mu_a, often in mm^-1 or cm^-1.
        # Placeholder values are unitless here.
        delta_val = (np.random.rand() * 0.05 + 0.005) * (1 if np.random.rand() > 0.3 else -1) # e.g., +0.005 to +0.055, or negative

        # Simple circular anomalies
        radius = np.random.randint(min(image_shape) // 6, min(image_shape) // 3)
        cx = np.random.randint(radius, image_shape[1] - radius)
        cy = np.random.randint(radius, image_shape[0] - radius)
        yy, xx = torch.meshgrid(torch.arange(image_shape[0], device=device),
                                torch.arange(image_shape[1], device=device), indexing='ij')
        mask = (xx - cx)**2 + (yy - cy)**2 < radius**2
        delta_mu_map[mask] += delta_val # Add change

    return delta_mu_map


def plot_dot_results(delta_mu_true, delta_y_measured, delta_mu_recon):
    """ Placeholder to plot DOT results. """
    print("plot_dot_results: Placeholder - Plotting not implemented.")
    print(f"  True delta_mu shape: {delta_mu_true.shape if delta_mu_true is not None else 'N/A'}")
    print(f"  Measured delta_y shape: {delta_y_measured.shape if delta_y_measured is not None else 'N/A'}")
    print(f"  Recon delta_mu shape: {delta_mu_recon.shape if delta_mu_recon is not None else 'N/A'}")
    # TODO: Implement plotting
    # import matplotlib.pyplot as plt
    # ...

if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_utils = (64,64)
    phantom_dm = generate_dot_phantom_delta_mu(img_s_utils, num_anomalies=2, device=dev_utils)
    assert phantom_dm.shape == img_s_utils
    print(f"DOT delta_mu phantom generated: {phantom_dm.shape}")

    dummy_dy = torch.randn(100, device=dev_utils)
    plot_dot_results(phantom_dm, dummy_dy, phantom_dm*0.3)
    print("DOT utils checks completed.")
