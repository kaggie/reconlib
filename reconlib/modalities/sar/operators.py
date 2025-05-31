import torch
import numpy as np # For np.pi, np.arange if used
from reconlib.operators import Operator
# Potentially, if uv_coordinates are non-integer & a proper NUFFT is desired:
# from reconlib.nufft import NUFFT2D # Assuming it's available and suitable

class SARForwardOperator(Operator):
    """
    Forward and Adjoint Operator for Synthetic Aperture Radar (SAR).

    Models SAR data acquisition as sampling the 2D Fourier transform of the
    target reflectivity map at specified (u,v) coordinates.

    Args:
        image_shape (tuple[int, int]): Shape of the input reflectivity image (height, width).
        uv_coordinates (torch.Tensor): Tensor of (u,v) k-space coordinates where the
                                       Fourier transform of the image is sampled.
                                       Shape (num_visibilities, 2).
                                       'u' corresponds to kx (frequency along width/columns).
                                       'v' corresponds to ky (frequency along height/rows).
                                       These coordinates are assumed to be scaled such that they
                                       can directly index a zero-centered, fftshifted 2D FFT grid
                                       of the image after appropriate offset. For an image (Ny, Nx),
                                       u: [-Nx/2, Nx/2-1], v: [-Ny/2, Ny/2-1].
        center_freq (float, optional): Radar center frequency in Hz. Not directly used in this
                                     simplified FFT model but important for scaling uv_coords
                                     from physical baselines. Here, uv_coords are assumed pre-scaled.
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx)
                 uv_coordinates: torch.Tensor, # (num_visibilities, 2)
                 center_freq: float = 10e9, # Example: 10 GHz, for context
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape # (Ny, Nx)
        self.device = torch.device(device)

        if not isinstance(uv_coordinates, torch.Tensor):
            uv_coordinates = torch.tensor(uv_coordinates, dtype=torch.float32)
        self.uv_coordinates = uv_coordinates.to(self.device)

        if self.uv_coordinates.ndim != 2 or self.uv_coordinates.shape[1] != 2:
            raise ValueError("uv_coordinates must be a 2D tensor of shape (num_visibilities, 2).")

        # Store Ny (height, rows, dim 0) and Nx (width, columns, dim 1)
        self.Ny, self.Nx = self.image_shape

        # Validate uv_coordinates against image_shape (assuming fftshift indexing)
        # u (horizontal freq) corresponds to Nx, v (vertical freq) to Ny
        u_min, u_max = -self.Nx // 2, (self.Nx - 1) // 2
        v_min, v_max = -self.Ny // 2, (self.Ny - 1) // 2

        # Basic check, can be made more robust if needed
        if not (torch.all(self.uv_coordinates[:, 0] >= u_min) and \
                torch.all(self.uv_coordinates[:, 0] <= u_max)):
            print(f"Warning: Some u-coordinates may be outside the typical range [{u_min}, {u_max}] "
                  f"for an image width of {self.Nx}. Found min {self.uv_coordinates[:,0].min()}, max {self.uv_coordinates[:,0].max()}. "
                  f"Behavior depends on sampling strategy (e.g., clamping or error).")

        if not (torch.all(self.uv_coordinates[:, 1] >= v_min) and \
                torch.all(self.uv_coordinates[:, 1] <= v_max)):
            print(f"Warning: Some v-coordinates may be outside the typical range [{v_min}, {v_max}] "
                  f"for an image height of {self.Ny}. Found min {self.uv_coordinates[:,1].min()}, max {self.uv_coordinates[:,1].max()}.")

        self.center_freq = center_freq # Stored for context, not used in this simplified model

    def op(self, x_reflectivity: torch.Tensor) -> torch.Tensor:
        """
        Forward SAR operation: Transforms reflectivity image to k-space visibilities.
        x_reflectivity: (Ny, Nx) target reflectivity map.
        Returns: (num_visibilities,) complex tensor of visibilities.
        """
        if x_reflectivity.shape != self.image_shape:
            raise ValueError(f"Input x_reflectivity shape {x_reflectivity.shape} must match {self.image_shape}.")
        if x_reflectivity.device != self.device:
            x_reflectivity = x_reflectivity.to(self.device)
        if not torch.is_complex(x_reflectivity):
             x_reflectivity = x_reflectivity.to(torch.complex64)

        # 1. Perform 2D FFT of the image
        #    norm='ortho' ensures Parseval's theorem holds, good for adjoint consistency.
        k_space_full = torch.fft.fft2(x_reflectivity, norm='ortho')

        # 2. Shift zero-frequency component to the center for easier indexing with (u,v)
        k_space_shifted = torch.fft.fftshift(k_space_full) # (Ny, Nx)

        # 3. Sample (interpolate) from k_space_shifted at uv_coordinates
        #    uv_coordinates are (u,v) where u is horizontal (rel to Nx), v is vertical (rel to Ny).
        #    Map (u,v) from [-N/2, N/2-1] to array indices [0, N-1]
        #    u_indices = u + Nx/2
        #    v_indices = v + Ny/2

        u_coords_shifted = self.uv_coordinates[:, 0] + self.Nx / 2.0
        v_coords_shifted = self.uv_coordinates[:, 1] + self.Ny / 2.0

        # For now, use nearest neighbor lookup. For higher accuracy, interpolation (e.g., bilinear) would be needed.
        # This requires uv_coordinates to be effectively integer indices after shifting if not interpolating.
        # Or, if we assume uv_coordinates were already scaled to be integer indices for fftshifted grid:
        u_indices = torch.round(u_coords_shifted).long()
        v_indices = torch.round(v_coords_shifted).long()

        # Clamp indices to be within the valid range of the k_space_shifted grid
        u_indices = torch.clamp(u_indices, 0, self.Nx - 1)
        v_indices = torch.clamp(v_indices, 0, self.Ny - 1)

        visibilities = k_space_shifted[v_indices, u_indices]

        return visibilities # Shape: (num_visibilities,)

    def op_adj(self, y_visibilities: torch.Tensor) -> torch.Tensor:
        """
        Adjoint SAR operation: Transforms k-space visibilities to a 'dirty image'.
        y_visibilities: (num_visibilities,) complex tensor of visibilities.
        Returns: (Ny, Nx) complex tensor (dirty image).
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

        # 2. Map (u,v) to array indices and place visibilities onto the grid
        #    This is the "gridding" step.
        u_coords_shifted = self.uv_coordinates[:, 0] + self.Nx / 2.0
        v_coords_shifted = self.uv_coordinates[:, 1] + self.Ny / 2.0

        u_indices = torch.round(u_coords_shifted).long()
        v_indices = torch.round(v_coords_shifted).long()

        # Clamp indices (important if original uv_coords were outside standard range)
        u_indices = torch.clamp(u_indices, 0, self.Nx - 1)
        v_indices = torch.clamp(v_indices, 0, self.Ny - 1)

        # Use index_add_ for proper adjoint if multiple (u,v) points map to the same grid cell
        # (though with rounding, this is less likely unless uv_coordinates are dense or identical)
        # A simple assignment is `k_space_gridded_shifted[v_indices, u_indices] = y_visibilities`
        # For true adjoint of nearest neighbor sampling, it's a bit more complex.
        # If forward was proper interpolation, adjoint would be transpose of that interpolation.
        # For now, simple gridding by assignment (or index_put for safety if using non-rounded indices).
        # If using rounded indices, index_add_ is more robust for potential overlaps.
        k_space_gridded_shifted.index_put_((v_indices, u_indices), y_visibilities, accumulate=True)

        # 3. Inverse shift zero-frequency from center to corner
        k_space_gridded_ifftshifted = torch.fft.ifftshift(k_space_gridded_shifted)

        # 4. Perform Inverse 2D FFT to get the dirty image
        dirty_image = torch.fft.ifft2(k_space_gridded_ifftshifted, norm='ortho')

        return dirty_image

if __name__ == '__main__':
    # This block will not be executed by this subtask script.
    # It's kept here for users who might run the file directly in a stable environment.
    print("Running basic SARForwardOperator checks (if run directly)...")
    device = torch.device('cpu') # Force CPU for subtask __main__
    img_shape_sar = (64, 64) # Ny, Nx

    # Simulate some (u,v) coordinates (integers for simple FFT indexing)
    num_vis = 100
    # u ranges approx -Nx/2 to Nx/2-1. v ranges approx -Ny/2 to Ny/2-1
    uv_coords = torch.stack([
        torch.randint(-img_shape_sar[1]//2, img_shape_sar[1]//2, (num_vis,), device=device),
        torch.randint(-img_shape_sar[0]//2, img_shape_sar[0]//2, (num_vis,), device=device)
    ], dim=1).float() # Ensure float for operator, will be rounded to long for indexing

    try:
        sar_op_test = SARForwardOperator(
            image_shape=img_shape_sar,
            uv_coordinates=uv_coords,
            device=device
        )
        print("SARForwardOperator instantiated.")

        phantom_sar = torch.randn(img_shape_sar, dtype=torch.complex64, device=device)
        phantom_sar[img_shape_sar[0]//2 - 2 : img_shape_sar[0]//2 + 2,
                    img_shape_sar[1]//2 - 2 : img_shape_sar[1]//2 + 2] = 3.0 # Add a bright spot

        visibilities_sim = sar_op_test.op(phantom_sar)
        print(f"Forward op output shape (visibilities): {visibilities_sim.shape}")
        assert visibilities_sim.shape == (num_vis,)

        dirty_img_recon = sar_op_test.op_adj(visibilities_sim)
        print(f"Adjoint op output shape (dirty image): {dirty_img_recon.shape}")
        assert dirty_img_recon.shape == img_shape_sar

        # Basic dot product test
        x_dp_sar = torch.randn_like(phantom_sar)
        y_dp_rand_sar = torch.randn_like(visibilities_sim)
        Ax_sar = sar_op_test.op(x_dp_sar)
        Aty_sar = sar_op_test.op_adj(y_dp_rand_sar)
        lhs_sar = torch.vdot(Ax_sar.flatten(), y_dp_rand_sar.flatten())
        rhs_sar = torch.vdot(x_dp_sar.flatten(), Aty_sar.flatten())
        print(f"SAR Dot product test: LHS={lhs_sar.item():.4f}, RHS={rhs_sar.item():.4f}")
        if not np.isclose(lhs_sar.real.item(), rhs_sar.real.item(), rtol=1e-3) or \
           not np.isclose(lhs_sar.imag.item(), rhs_sar.imag.item(), rtol=1e-3):
           print("Warning: SAR Dot product components differ. This might be due to rounding in indexing or simple gridding.")

        print("SARForwardOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in SARForwardOperator __main__ checks: {e}")
