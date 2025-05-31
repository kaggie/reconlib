import torch
import numpy as np

def generate_eit_phantom_delta_sigma(image_shape: tuple[int, int], num_regions: int = 2, device='cpu') -> torch.Tensor:
    """
    Generates a phantom representing changes in conductivity (delta_sigma)
    from a baseline. Positive values mean higher conductivity, negative lower.
    """
    delta_sigma_map = torch.zeros(image_shape, dtype=torch.float32, device=device)
    for _ in range(num_regions):
        # Conductivity change value (can be positive or negative)
        delta_val = (np.random.rand() - 0.5) * 0.2 # e.g., changes from -0.1 to +0.1 S/m

        # Simple rectangular regions
        w = np.random.randint(image_shape[1] // 5, image_shape[1] // 2)
        h = np.random.randint(image_shape[0] // 5, image_shape[0] // 2)
        x0 = np.random.randint(0, image_shape[1] - w)
        y0 = np.random.randint(0, image_shape[0] - h)
        delta_sigma_map[y0:y0+h, x0:x0+w] += delta_val # Add change, allow overlap effects

    return delta_sigma_map


def plot_eit_results(delta_sigma_true, delta_v_measured, delta_sigma_recon):
    """ Placeholder to plot EIT results. """
    print("plot_eit_results: Placeholder - Plotting not implemented.")
    print(f"  True delta_sigma shape: {delta_sigma_true.shape if delta_sigma_true is not None else 'N/A'}")
    print(f"  Measured delta_v shape: {delta_v_measured.shape if delta_v_measured is not None else 'N/A'}")
    print(f"  Recon delta_sigma shape: {delta_sigma_recon.shape if delta_sigma_recon is not None else 'N/A'}")
    # TODO: Implement plotting (true delta_sigma, recon delta_sigma, maybe delta_v signal)
    # import matplotlib.pyplot as plt
    # ...

if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_utils = (64,64)
    phantom_ds = generate_eit_phantom_delta_sigma(img_s_utils, num_regions=3, device=dev_utils)
    assert phantom_ds.shape == img_s_utils
    print(f"EIT delta_sigma phantom generated: {phantom_ds.shape}")

    dummy_dv = torch.randn(100, device=dev_utils) # Dummy measurements
    plot_eit_results(phantom_ds, dummy_dv, phantom_ds*0.5) # Dummy recon
    print("EIT utils checks completed.")
