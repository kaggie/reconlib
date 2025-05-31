import torch
import numpy as np # For np.pi, np.arange if used
from reconlib.operators import Operator

class AstronomicalInterferometryOperator(Operator):
    """
    Forward and Adjoint Operator for Astronomical Interferometry.

    Models data acquisition in radio astronomy (and other interferometric methods)
    as sampling the 2D Fourier transform of the sky brightness map at
    specified (u,v) spatial frequency coordinates.

    Args:
        image_shape (tuple[int, int]): Shape of the input sky brightness map (Height/Dec, Width/RA).
                                       Typically (Ny, Nx).
        uv_coordinates (torch.Tensor): Tensor of (u,v) spatial frequency coordinates where the
                                       Fourier transform of the sky image is sampled. These (u,v)
                                       points are determined by the telescope baselines.
                                       Shape (num_visibilities, 2).
                                       'u' corresponds to kx (frequency along width/RA).
                                       'v' corresponds to ky (frequency along height/Dec).
                                       Assumed to be scaled appropriately to index a zero-centered,
                                       fftshifted 2D FFT grid of the image after an offset.
                                       For an image (Ny, Nx), u: [-Nx/2, Nx/2-1], v: [-Ny/2, Ny/2-1].
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx)
                 uv_coordinates: torch.Tensor, # (num_visibilities, 2)
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape # (Ny, Nx)
        self.device = torch.device(device)

        if not isinstance(uv_coordinates, torch.Tensor):
            uv_coordinates = torch.tensor(uv_coordinates, dtype=torch.float32)
        self.uv_coordinates = uv_coordinates.to(self.device)

        if self.uv_coordinates.ndim != 2 or self.uv_coordinates.shape[1] != 2:
            raise ValueError("uv_coordinates must be a 2D tensor of shape (num_visibilities, 2).")

        self.Ny, self.Nx = self.image_shape

        # Optional: Validate uv_coordinates range (similar to SAR operator)
        u_min, u_max = -self.Nx // 2, (self.Nx - 1) // 2
        v_min, v_max = -self.Ny // 2, (self.Ny - 1) // 2
        if not (torch.all(self.uv_coordinates[:, 0] >= u_min) and \
                torch.all(self.uv_coordinates[:, 0] <= u_max)):
            print(f"Warning: Astronomical u-coordinates may be outside typical FFT range [{u_min}, {u_max}] "
                  f"for image width {self.Nx}.")
        if not (torch.all(self.uv_coordinates[:, 1] >= v_min) and \
                torch.all(self.uv_coordinates[:, 1] <= v_max)):
            print(f"Warning: Astronomical v-coordinates may be outside typical FFT range [{v_min}, {v_max}] "
                  f"for image height {self.Ny}.")


    def op(self, x_sky_brightness_map: torch.Tensor) -> torch.Tensor:
        """
        Forward Astronomical Interferometry operation: Sky brightness to visibilities.
        x_sky_brightness_map: (Ny, Nx) sky brightness distribution.
        Returns: (num_visibilities,) complex tensor of visibilities.
        """
        if x_sky_brightness_map.shape != self.image_shape:
            raise ValueError(f"Input x_sky_brightness_map shape {x_sky_brightness_map.shape} must match {self.image_shape}.")
        if x_sky_brightness_map.device != self.device:
            x_sky_brightness_map = x_sky_brightness_map.to(self.device)
        # Sky brightness is real and non-negative, but operator can handle complex input for generality.
        if not torch.is_complex(x_sky_brightness_map):
             x_sky_brightness_map = x_sky_brightness_map.to(torch.complex64)

        # 1. Perform 2D FFT of the sky map
        k_space_full = torch.fft.fft2(x_sky_brightness_map, norm='ortho')

        # 2. Shift zero-frequency component to the center
        k_space_shifted = torch.fft.fftshift(k_space_full) # (Ny, Nx)

        # 3. Sample from k_space_shifted at uv_coordinates
        # Map (u,v) from [-N/2, N/2-1] to array indices [0, N-1]
        u_coords_shifted = self.uv_coordinates[:, 0] + self.Nx / 2.0
        v_coords_shifted = self.uv_coordinates[:, 1] + self.Ny / 2.0

        # Nearest neighbor lookup (rounding and clamping)
        u_indices = torch.round(u_coords_shifted).long().clamp(0, self.Nx - 1)
        v_indices = torch.round(v_coords_shifted).long().clamp(0, self.Ny - 1)

        visibilities = k_space_shifted[v_indices, u_indices]

        return visibilities # Shape: (num_visibilities,)

    def op_adj(self, y_visibilities: torch.Tensor) -> torch.Tensor:
        """
        Adjoint Astronomical Interferometry: Visibilities to "dirty image".
        y_visibilities: (num_visibilities,) complex tensor of visibilities.
        Returns: (Ny, Nx) complex tensor (dirty image of the sky).
        """
        if y_visibilities.ndim != 1 or y_visibilities.shape[0] != self.uv_coordinates.shape[0]:
            raise ValueError(f"Input y_visibilities shape {y_visibilities.shape} is invalid. "
                             f"Expected 1D tensor of length {self.uv_coordinates.shape[0]}.")
        if y_visibilities.device != self.device:
            y_visibilities = y_visibilities.to(self.device)
        if not torch.is_complex(y_visibilities):
             y_visibilities = y_visibilities.to(torch.complex64)

        # 1. Create an empty k-space grid (for fftshifted data)
        k_space_gridded_shifted = torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)

        # 2. Map (u,v) to array indices and place/grid visibilities
        u_coords_shifted = self.uv_coordinates[:, 0] + self.Nx / 2.0
        v_coords_shifted = self.uv_coordinates[:, 1] + self.Ny / 2.0

        u_indices = torch.round(u_coords_shifted).long().clamp(0, self.Nx - 1)
        v_indices = torch.round(v_coords_shifted).long().clamp(0, self.Ny - 1)

        # Use index_add_ for proper adjoint accumulation if multiple (u,v) map to same grid cell
        k_space_gridded_shifted.index_put_((v_indices, u_indices), y_visibilities, accumulate=True)

        # 3. Inverse shift zero-frequency from center to corner
        k_space_gridded_ifftshifted = torch.fft.ifftshift(k_space_gridded_shifted)

        # 4. Perform Inverse 2D FFT to get the dirty image
        dirty_image = torch.fft.ifft2(k_space_gridded_ifftshifted, norm='ortho')

        return dirty_image

if __name__ == '__main__':
    print("Running basic AstronomicalInterferometryOperator checks...")
    device = torch.device('cpu')
    img_shape_astro = (64, 64) # Ny, Nx

    num_vis_astro = 150
    uv_coords_astro = torch.stack([
        torch.randint(-img_shape_astro[1]//2, img_shape_astro[1]//2, (num_vis_astro,), device=device),
        torch.randint(-img_shape_astro[0]//2, img_shape_astro[0]//2, (num_vis_astro,), device=device)
    ], dim=1).float()

    try:
        astro_op_test = AstronomicalInterferometryOperator(
            image_shape=img_shape_astro,
            uv_coordinates=uv_coords_astro,
            device=device
        )
        print("AstronomicalInterferometryOperator instantiated.")

        phantom_sky = torch.randn(img_shape_astro, dtype=torch.complex64, device=device)
        # Add a bright source
        phantom_sky[img_shape_astro[0]//3, img_shape_astro[1]//3] = 5.0

        visibilities_sim_astro = astro_op_test.op(phantom_sky)
        print(f"Forward op output shape (visibilities): {visibilities_sim_astro.shape}")
        assert visibilities_sim_astro.shape == (num_vis_astro,)

        dirty_img_astro = astro_op_test.op_adj(visibilities_sim_astro)
        print(f"Adjoint op output shape (dirty image): {dirty_img_astro.shape}")
        assert dirty_img_astro.shape == img_shape_astro

        # Basic dot product test
        x_dp_astro = torch.randn_like(phantom_sky)
        y_dp_rand_astro = torch.randn_like(visibilities_sim_astro)
        Ax_astro = astro_op_test.op(x_dp_astro)
        Aty_astro = astro_op_test.op_adj(y_dp_rand_astro)
        lhs_astro = torch.vdot(Ax_astro.flatten(), y_dp_rand_astro.flatten())
        rhs_astro = torch.vdot(x_dp_astro.flatten(), Aty_astro.flatten())
        print(f"Astro Dot product test: LHS={lhs_astro.item():.4f}, RHS={rhs_astro.item():.4f}")
        if not np.isclose(lhs_astro.real.item(), rhs_astro.real.item(), rtol=1e-3) or \
           not np.isclose(lhs_astro.imag.item(), rhs_astro.imag.item(), rtol=1e-3):
           print("Warning: Astro Dot product components differ. Check gridding/sampling.")

        print("AstronomicalInterferometryOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in AstronomicalInterferometryOperator __main__ checks: {e}")
