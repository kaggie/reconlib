import torch
import numpy as np

def generate_xrd_phantom(image_shape: tuple[int, int], num_features: int = 1, feature_type: str = 'crystal', device='cpu') -> torch.Tensor:
    """ Generates a simple phantom for X-ray Diffraction (e.g., a crystal shape). """
    phantom = torch.zeros(image_shape, dtype=torch.float32, device=device)
    if feature_type == 'crystal': # Simple geometric shapes
        for _ in range(num_features):
            val = np.random.rand()*0.5 + 0.5
            is_rect = np.random.rand() > 0.2 # More likely rect for "crystal"
            if is_rect:
                w = np.random.randint(image_shape[1] // 6, image_shape[1] // 2)
                h = np.random.randint(image_shape[0] // 6, image_shape[0] // 2)
                x0 = np.random.randint(0, image_shape[1] - w)
                y0 = np.random.randint(0, image_shape[0] - h)
                phantom[y0:y0+h, x0:x0+w] = val
            else: # Circle/Blob
                radius = np.random.randint(min(image_shape) // 8, min(image_shape) // 3)
                cx = np.random.randint(radius, image_shape[1] - radius)
                cy = np.random.randint(radius, image_shape[0] - radius)
                yy, xx = torch.meshgrid(torch.arange(image_shape[0], device=device),
                                        torch.arange(image_shape[1], device=device), indexing='ij')
                mask = (xx - cx)**2 + (yy - cy)**2 < radius**2
                phantom[mask] = val
    else:
        phantom[image_shape[0]//4:image_shape[0]*3//4, image_shape[1]//4:image_shape[1]*3//4] = 1.0 # Default square

    return torch.clamp(phantom,0,1)


def plot_xrd_results(object_map_true, diffraction_magnitudes, object_map_recon, log_magnitudes=True):
    """ Placeholder to plot X-ray Diffraction results. """
    print("plot_xrd_results: Placeholder - Plotting not implemented.")
    print(f"  True Object map shape: {object_map_true.shape if object_map_true is not None else 'N/A'}")
    print(f"  Diffraction Magnitudes shape: {diffraction_magnitudes.shape if diffraction_magnitudes is not None else 'N/A'}")
    print(f"  Reconstructed Object map shape: {object_map_recon.shape if object_map_recon is not None else 'N/A'}")

    # TODO: Implement plotting (e.g., true object, |FT| or log(|FT|) of true, recon object)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,3, figsize=(15,5))
    # if object_map_true is not None: axes[0].imshow(object_map_true.cpu().numpy(), cmap='gray'); axes[0].set_title("True Object")
    # if diffraction_magnitudes is not None:
    #     display_mags = diffraction_magnitudes.cpu().numpy()
    #     if log_magnitudes: display_mags = np.log1p(display_mags)
    #     axes[1].imshow(np.fft.fftshift(display_mags), cmap='viridis'); axes[1].set_title(f"{'Log ' if log_magnitudes else ''}Magnitudes (fftshifted)")
    # if object_map_recon is not None: axes[2].imshow(object_map_recon.cpu().numpy(), cmap='gray'); axes[2].set_title("Reconstructed Object")
    # plt.show()


if __name__ == '__main__':
    dev_utils = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_utils = (64,64)
    phantom = generate_xrd_phantom(img_s_utils, num_features=2, device=dev_utils)
    assert phantom.shape == img_s_utils
    print(f"XRD phantom generated: {phantom.shape}")

    # Dummy data for plot call
    dummy_mags = torch.abs(torch.fft.fft2(phantom))
    plot_xrd_results(phantom, dummy_mags, phantom*0.8)
    print("XRD utils checks completed.")
