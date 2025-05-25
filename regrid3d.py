import torch
from scipy.special import i0

class NUFFTRegridding3D:
    def __init__(self, image_shape, oversamp=2.0, width=4, beta=13.9085, device='cuda'):
        self.image_shape = image_shape  # (Nx, Ny, Nz)
        self.oversamp = oversamp
        self.width = width
        self.beta = beta
        self.device = device

        self.Nx_os = int(image_shape[0] * oversamp)
        self.Ny_os = int(image_shape[1] * oversamp)
        self.Nz_os = int(image_shape[2] * oversamp)

    def kaiser_bessel_kernel(self, r):
        mask = r < (self.width / 2)
        z = torch.sqrt(1 - (2 * r[mask] / self.width) ** 2)
        kb = torch.zeros_like(r)
        kb[mask] = torch.from_numpy(i0(self.beta * z.cpu().numpy())).to(r.device) / i0(self.beta)
        return kb

    def estimate_density_compensation(self, kx, ky, kz):
        r = torch.sqrt(kx**2 + ky**2 + kz**2)
        dcf = r + 1e-3
        return dcf / dcf.max()

    def regrid(self, kx, ky, kz, data):
        """
        Args:
            kx, ky, kz: [N] tensors of non-Cartesian coords in [-0.5, 0.5]
            data: [C, N] complex tensor (multi-coil k-space)
        Returns:
            cartesian_kspace: [C, Nx_os, Ny_os, Nz_os] complex tensor
        """
        C, N = data.shape
        kx = kx.to(self.device)
        ky = ky.to(self.device)
        kz = kz.to(self.device)
        data = data.to(self.device)

        # Scale coords
        kx_scaled = (kx + 0.5) * self.Nx_os
        ky_scaled = (ky + 0.5) * self.Ny_os
        kz_scaled = (kz + 0.5) * self.Nz_os

        # DCF
        dcf = self.estimate_density_compensation(kx, ky, kz)
        data = data * dcf.unsqueeze(0)

        # Init grid
        grid = torch.zeros((C, self.Nx_os, self.Ny_os, self.Nz_os), dtype=torch.complex64, device=self.device)
        weight = torch.zeros((self.Nx_os, self.Ny_os, self.Nz_os), dtype=torch.float32, device=self.device)

        hw = self.width // 2

        for dx in range(-hw, hw + 1):
            for dy in range(-hw, hw + 1):
                for dz in range(-hw, hw + 1):
                    x_idx = (torch.floor(kx_scaled + dx).long()) % self.Nx_os
                    y_idx = (torch.floor(ky_scaled + dy).long()) % self.Ny_os
                    z_idx = (torch.floor(kz_scaled + dz).long()) % self.Nz_os

                    x_dist = kx_scaled - x_idx.float()
                    y_dist = ky_scaled - y_idx.float()
                    z_dist = kz_scaled - z_idx.float()
                    r = torch.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
                    w = self.kaiser_bessel_kernel(r)

                    for c in range(C):
                        for i in range(N):
                            grid[c, x_idx[i], y_idx[i], z_idx[i]] += data[c, i] * w[i]
                            if c == 0:
                                weight[x_idx[i], y_idx[i], z_idx[i]] += w[i]

        weight = torch.where(weight == 0, torch.ones_like(weight), weight)
        grid /= weight.unsqueeze(0)
        return grid
