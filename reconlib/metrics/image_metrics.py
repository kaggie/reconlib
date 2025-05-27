import torch
import torch.nn.functional as F
import math

def mse(image_true: torch.Tensor, image_test: torch.Tensor) -> torch.Tensor:
    """Computes the Mean Squared Error (MSE) between two images."""
    if image_true.shape != image_test.shape:
        raise ValueError(f"Input images must have the same shape. Got {image_true.shape} and {image_test.shape}")
    return torch.mean((image_true - image_test) ** 2)

def psnr(image_true: torch.Tensor, image_test: torch.Tensor, data_range: float | None = None) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
    PSNR = 20 * log10(data_range / sqrt(MSE)).
    """
    if data_range is None:
        data_range = torch.max(image_true) - torch.min(image_true)
        if data_range == 0: # Handle case of constant image
            # If images are identical and flat, PSNR is infinite (or MSE is 0)
            # If different and flat, this will lead to div by zero in MSE if not handled
            if mse(image_true, image_test) < 1e-12: # Effectively zero MSE
                 return torch.tensor(float('inf'), device=image_true.device)
            else: # Non-zero MSE on flat data means data_range was 0, this is tricky.
                  # Defaulting to a common behavior: if data_range is 0 and MSE is not, PSNR is 0 or -inf.
                  # Let's return 0 for this edge case.
                 return torch.tensor(0.0, device=image_true.device)


    mse_val = mse(image_true, image_test)
    if mse_val < 1e-12: # Effectively zero MSE
        return torch.tensor(float('inf'), device=image_true.device)
    
    return 20 * torch.log10(data_range / torch.sqrt(mse_val))

def _ssim_gaussian_kernel(window_size: int, sigma: float, num_channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Creates a 1D Gaussian kernel for SSIM."""
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum() # Normalize
    # Reshape to (1, 1, 1, window_size) or (1, 1, window_size) for conv1d/2d
    # For conv2d, it's (out_channels, in_channels/groups, kH, kW)
    # We want (num_channels, 1, window_size, 1) and (num_channels, 1, 1, window_size)
    # For now, make it (1,1,window_size) for 1D conv, then create 2D window from it.
    return g.reshape(1, 1, window_size)


def ssim(
    image_true: torch.Tensor, 
    image_test: torch.Tensor, 
    data_range: float | None = None,
    window_size: int = 11, 
    sigma: float = 1.5,
    k1: float = 0.01, 
    k2: float = 0.03
) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    This is a simplified PyTorch implementation, primarily for 2D single-channel images.
    Assumes image_true and image_test are (N, C, H, W) or (C, H, W) or (H, W).
    Handles grayscale (C=1) for now.
    """
    if image_true.shape != image_test.shape:
        raise ValueError(f"Input images must have the same shape. Got {image_true.shape} and {image_test.shape}")

    if data_range is None:
        data_range = image_true.max() - image_true.min()
        if data_range == 0: # If flat image
            return torch.tensor(1.0 if torch.allclose(image_true, image_test) else 0.0, device=image_true.device)


    # Ensure 4D input (N, C, H, W) for convolutions
    orig_ndim = image_true.ndim
    if orig_ndim == 2: # (H, W)
        image_true = image_true.unsqueeze(0).unsqueeze(0)
        image_test = image_test.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3: # (C, H, W)
        image_true = image_true.unsqueeze(0)
        image_test = image_test.unsqueeze(0)
    
    if image_true.shape[1] != 1:
        # For multi-channel, SSIM is often averaged over channels.
        # This simple version will process first channel if C > 1, or needs extension.
        print("Warning: SSIM function currently processes grayscale (first channel if C>1). For proper multichannel SSIM, average results per channel.")
        image_true = image_true[:, 0:1, :, :]
        image_test = image_test[:, 0:1, :, :]

    num_channels = image_true.shape[1]

    # Gaussian window
    kernel_1d = _ssim_gaussian_kernel(window_size, sigma, num_channels, image_true.device, image_true.dtype)
    # Correct way to create 2D window from 1D for SSIM:
    # Create a (window_size, 1) kernel and multiply by its transpose (1, window_size)
    # kernel_x = kernel_1d.reshape(window_size, 1)
    # kernel_y = kernel_1d.reshape(1, window_size)
    # window = kernel_x * kernel_y # This is (window_size, window_size)
    # window = window.unsqueeze(0).unsqueeze(0) # Shape (1, 1, W_size, W_size)
    # window = window.expand(num_channels, 1, window_size, window_size)
    
    # A simpler way to get a 2D Gaussian window, if sigma is isotropic:
    coords_h = torch.arange(window_size, device=image_true.device, dtype=image_true.dtype) - window_size // 2
    coords_w = torch.arange(window_size, device=image_true.device, dtype=image_true.dtype) - window_size // 2
    g_h = torch.exp(-(coords_h ** 2) / (2 * sigma ** 2))
    g_w = torch.exp(-(coords_w ** 2) / (2 * sigma ** 2))
    g_h /= g_h.sum()
    g_w /= g_w.sum()
    window = torch.outer(g_h, g_w) # Shape (window_size, window_size)
    window = window.unsqueeze(0).unsqueeze(0) # Shape (1,1,H,W) for conv2d
    window = window.expand(num_channels, 1, window_size, window_size) # Shape (C,1,H,W) for grouped conv


    # Constants
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Mean
    mu1 = F.conv2d(image_true, window, padding=window_size//2, groups=num_channels)
    mu2 = F.conv2d(image_test, window, padding=window_size//2, groups=num_channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Variance
    sigma1_sq = F.conv2d(image_true * image_true, window, padding=window_size//2, groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(image_test * image_test, window, padding=window_size//2, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(image_true * image_test, window, padding=window_size//2, groups=num_channels) - mu1_mu2

    # SSIM formula
    ssim_map_num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ssim_map_den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = ssim_map_num / ssim_map_den
    
    ssim_val = ssim_map.mean()
    
    return ssim_val


if __name__ == '__main__':
    print("Testing image_metrics...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy images
    img1 = torch.rand(64, 64, device=device)
    img2 = img1 * 0.75 + 0.25 * torch.rand(64, 64, device=device)
    img_zeros = torch.zeros(64,64, device=device)
    img_ones = torch.ones(64,64, device=device)


    # Test MSE
    mse_val = mse(img1, img2)
    print(f"MSE(img1, img2): {mse_val.item()}")
    mse_identity = mse(img1, img1)
    print(f"MSE(img1, img1): {mse_identity.item()} (expected close to 0)")
    assert mse_identity < 1e-9

    # Test PSNR
    psnr_val = psnr(img1, img2) # data_range calculated internally
    print(f"PSNR(img1, img2): {psnr_val.item()}")
    psnr_identity = psnr(img1, img1)
    print(f"PSNR(img1, img1): {psnr_identity.item()} (expected inf)")
    assert psnr_identity == float('inf')
    psnr_zeros = psnr(img_zeros, img_ones, data_range=1.0) # Should be 0 if MSE is 1 and data_range is 1
    print(f"PSNR(zeros, ones, dr=1): {psnr_zeros.item()} (expected 0)")
    assert torch.isclose(psnr_zeros, torch.tensor(0.0))


    # Test SSIM (simple 2D single channel)
    ssim_val = ssim(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)) # Add batch and channel
    print(f"SSIM(img1, img2): {ssim_val.item()}")
    ssim_identity = ssim(img1.unsqueeze(0).unsqueeze(0), img1.unsqueeze(0).unsqueeze(0))
    print(f"SSIM(img1, img1): {ssim_identity.item()} (expected close to 1)")
    assert torch.isclose(ssim_identity, torch.tensor(1.0), atol=1e-4) # Might not be exactly 1 due to conv precision

    # Test with 3D data (will process first channel/slice effectively, or needs proper 3D SSIM)
    img3d_1 = torch.rand(5, 64, 64, device=device) # (C, H, W) or (D, H, W)
    img3d_2 = img3d_1 * 0.8
    # ssim_3d_val = ssim(img3d_1, img3d_2) # Current SSIM is 2D focused
    # print(f"SSIM(img3d_1, img3d_2) (channel-wise or first channel): {ssim_3d_val.item()}")


    print("image_metrics basic tests completed.")
