import torch
from scipy.special import i0

def kaiser_bessel_kernel(r, width, beta):
    mask = r < (width / 2)
    z = torch.sqrt(1 - (2 * r[mask] / width) ** 2)
    kb = torch.zeros_like(r)
    kb[mask] = torch.from_numpy(i0(beta * z.cpu().numpy())).to(r.device) / i0(beta)
    return kb

def estimate_density_compensation(kx, ky):
    r = torch.sqrt(kx ** 2 + ky ** 2)
    dcf = r + 1e-3  # avoid divide by zero
    return dcf / dcf.max()

def regrid_to_cartesian(kx, ky, data, image_shape, oversamp=2.0, width=4, beta=13.9085):
    """
    Regrid non-Cartesian k-space data onto Cartesian grid using Kaiser-Bessel interpolation.

    Args:
        kx, ky: 1D tensors of k-space coords in [-0.5, 0.5]
        data: 1D complex tensor of k-space samples
        image_shape: tuple (Nx, Ny)
        oversamp: oversampling factor (e.g. 2.0)
        width: kernel width (e.g. 4)
        beta: Kaiser-Bessel beta (e.g. 13.9085)

    Returns:
        cartesian_kspace: complex-valued 2D grid (oversampled)
    """
    device = kx.device
    Nx, Ny = image_shape
    Nx_os = int(Nx * oversamp)
    Ny_os = int(Ny * oversamp)

    # Scale coords to oversampled grid
    kx_scaled = (kx + 0.5) * Nx_os
    ky_scaled = (ky + 0.5) * Ny_os

    # Density compensation
    dcf = estimate_density_compensation(kx, ky).to(device)
    data = data * dcf

    # Initialize output grid
    grid = torch.zeros((Nx_os, Ny_os), dtype=torch.complex64, device=device)
    weight = torch.zeros((Nx_os, Ny_os), dtype=torch.float32, device=device)

    half_width = width // 2

    # Regrid using Kaiser-Bessel interpolation
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx = torch.floor(kx_scaled + dx).long() % Nx_os
            y_idx = torch.floor(ky_scaled + dy).long() % Ny_os

            x_dist = kx_scaled - x_idx.float()
            y_dist = ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist ** 2 + y_dist ** 2)
            w = kaiser_bessel_kernel(r, width, beta)

            for i in range(data.shape[0]):
                grid[x_idx[i], y_idx[i]] += data[i] * w[i]
                weight[x_idx[i], y_idx[i]] += w[i]

    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    grid /= weight

    return grid
