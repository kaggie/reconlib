import torch
from reconlib.operators import Operator
import numpy as np

class TerahertzOperator(Operator):
    """
    Forward and Adjoint Operator for Terahertz (THz) Imaging.

    Models THz wave interaction with a sample, specifically simulating a
    Fourier sampling scenario. This could represent simplified cases of
    THz holography or systems where k-space data is acquired.

    The forward operator performs a 2D FFT of the input image (material property map)
    and then samples this k-space at specified locations.
    The adjoint operator performs the conjugate operation: gridding the k-space
    samples and performing an inverse 2D FFT.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx)
                 k_space_locations: torch.Tensor, # (num_measurements, 2) specifying (kx, ky) coordinates
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape # (Ny, Nx)
        self.Ny, self.Nx = self.image_shape
        self.device = torch.device(device)

        if not isinstance(k_space_locations, torch.Tensor):
            k_space_locations = torch.tensor(k_space_locations, dtype=torch.float32)
        self.k_space_locations = k_space_locations.to(self.device) # (num_measurements, 2)

        if self.k_space_locations.ndim != 2 or self.k_space_locations.shape[1] != 2:
            raise ValueError("k_space_locations must be a 2D tensor of shape (num_measurements, 2).")

        self.num_measurements = self.k_space_locations.shape[0]

        # Validate k-space coordinates: should be in range [-N/2, N/2-1] for corresponding dimension
        # kx corresponds to Nx (width), ky to Ny (height)
        # u (kx) range: [-Nx/2, Nx/2 -1], v (ky) range: [-Ny/2, Ny/2 -1]
        u_min, u_max = -self.Nx // 2, (self.Nx - 1) // 2
        v_min, v_max = -self.Ny // 2, (self.Ny - 1) // 2

        if not (torch.all(self.k_space_locations[:, 0] >= u_min) and \
                torch.all(self.k_space_locations[:, 0] <= u_max)):
            print(f"Warning: kx coordinates may be outside typical FFT range [{u_min}, {u_max}] for image width {self.Nx}.")
        if not (torch.all(self.k_space_locations[:, 1] >= v_min) and \
                torch.all(self.k_space_locations[:, 1] <= v_max)):
            print(f"Warning: ky coordinates may be outside typical FFT range [{v_min}, {v_max}] for image height {self.Ny}.")

        print(f"TerahertzOperator (Fourier Sampling) initialized for image shape {self.image_shape}.")
        print(f"  {self.num_measurements} k-space sampling locations provided.")


    def op(self, image_estimate: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Image estimate to k-space measurement data.
        Performs 2D FFT on image_estimate and samples at self.k_space_locations.

        Args:
            image_estimate (torch.Tensor): The material property map (e.g., refractive index, absorption).
                                           Shape: self.image_shape (Ny, Nx). Assumed real.
        Returns:
            torch.Tensor: Simulated k-space measurement data (complex-valued).
                          Shape: (num_measurements,).
        """
        if image_estimate.shape != self.image_shape:
            raise ValueError(f"Input image_estimate shape {image_estimate.shape} must match {self.image_shape}.")
        if image_estimate.device != self.device:
            image_estimate = image_estimate.to(self.device)

        # Image is typically real, FFT output is complex
        if image_estimate.is_complex():
            # This model assumes a real input image leading to Hermitian k-space
            print("Warning: Input image_estimate is complex. Taking real part for FFT-based THz model.")
            image_estimate = image_estimate.real

        # 1. Perform 2D FFT of the image
        k_space_full = torch.fft.fft2(image_estimate, norm='ortho') # Output is complex

        # 2. Shift zero-frequency component to the center for easier indexing with k_space_locations
        k_space_shifted = torch.fft.fftshift(k_space_full) # (Ny, Nx)

        # 3. Sample from k_space_shifted at k_space_locations
        # Map kx, ky from [-N/2, N/2-1] to array indices [0, N-1]
        # u_coords = kx, v_coords = ky
        # u_indices = kx + Nx/2
        # v_indices = ky + Ny/2
        u_indices = torch.round(self.k_space_locations[:, 0] + self.Nx / 2.0).long()
        v_indices = torch.round(self.k_space_locations[:, 1] + self.Ny / 2.0).long()

        # Clamp indices to be within valid range [0, N-1]
        u_indices = torch.clamp(u_indices, 0, self.Nx - 1)
        v_indices = torch.clamp(v_indices, 0, self.Ny - 1)

        measurement_data = k_space_shifted[v_indices, u_indices] # Complex valued

        return measurement_data

    def op_adj(self, measurement_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: k-space measurement data to image domain.
        Grids the k-space samples and performs inverse 2D FFT.

        Args:
            measurement_data (torch.Tensor): THz k-space measurement data (complex-valued).
                                             Shape: (num_measurements,).
        Returns:
            torch.Tensor: Image reconstructed by adjoint operation.
                          Shape: self.image_shape. Output is real (should be if k-space data was Hermitian).
        """
        if not measurement_data.is_complex():
            # This could happen if op somehow produced real data or if input is wrong
            print("Warning: op_adj received real measurement_data. K-space data should be complex.")
            measurement_data = measurement_data.to(torch.complex64) # Ensure complex

        if measurement_data.ndim != 1 or measurement_data.shape[0] != self.num_measurements:
            raise ValueError(f"Input data has invalid shape {measurement_data.shape}. Expected ({self.num_measurements},).")
        measurement_data = measurement_data.to(self.device)

        # 1. Create an empty k-space grid (for fftshifted data)
        k_space_gridded_shifted = torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)

        # 2. Map kx, ky to array indices and place/grid measurements
        u_indices = torch.round(self.k_space_locations[:, 0] + self.Nx / 2.0).long()
        v_indices = torch.round(self.k_space_locations[:, 1] + self.Ny / 2.0).long()

        u_indices = torch.clamp(u_indices, 0, self.Nx - 1)
        v_indices = torch.clamp(v_indices, 0, self.Ny - 1)

        # Place measurements into the k-space grid.
        # Using index_put_ with accumulate=True for proper adjoint if multiple k-space locations map to the same grid cell (due to rounding).
        k_space_gridded_shifted.index_put_((v_indices, u_indices), measurement_data, accumulate=True)

        # 3. Inverse shift zero-frequency from center to corner
        k_space_gridded_ifftshifted = torch.fft.ifftshift(k_space_gridded_shifted)

        # 4. Perform Inverse 2D FFT to get the image
        # Output should be real if the original image was real and k-space sampling was Hermitian (or handled correctly)
        reconstructed_image = torch.fft.ifft2(k_space_gridded_ifftshifted, norm='ortho')

        # Since the input image to op() is assumed real, the output of op_adj() should also be real.
        # The k-space data generated from a real image is Hermitian.
        # If op_adj() is correct, ifft2 of such gridded data should result in a primarily real image.
        return reconstructed_image.real


if __name__ == '__main__':
    print("\nRunning basic TerahertzOperator (Fourier Sampling) checks...")
    device_thz_op = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_s_thz = (32, 32) # Ny, Nx

    # Define some k-space sampling locations (e.g., random sparse pattern)
    num_measurements_thz = img_s_thz[0] * img_s_thz[1] // 4 # 1/4 sampling
    kx_coords = torch.randint(-img_s_thz[1]//2, img_s_thz[1]//2, (num_measurements_thz,), device=device_thz_op).float()
    ky_coords = torch.randint(-img_s_thz[0]//2, img_s_thz[0]//2, (num_measurements_thz,), device=device_thz_op).float()
    k_locs = torch.stack([kx_coords, ky_coords], dim=1)

    try:
        thz_op = TerahertzOperator(
            image_shape=img_s_thz,
            k_space_locations=k_locs,
            device=device_thz_op
        )
        print("TerahertzOperator (Fourier Sampling) instantiated.")

        # Create a simple real phantom image
        phantom_thz_image = torch.zeros(img_s_thz, device=device_thz_op)
        phantom_thz_image[img_s_thz[0]//4:img_s_thz[0]*3//4, img_s_thz[1]//4:img_s_thz[1]*3//4] = 1.0
        phantom_thz_image[img_s_thz[0]//3:img_s_thz[0]*2//3, img_s_thz[1]//3:img_s_thz[1]*2//3] = 2.0


        simulated_k_data = thz_op.op(phantom_thz_image)
        print(f"Forward op output shape (k-space data): {simulated_k_data.shape}")
        assert simulated_k_data.shape == (num_measurements_thz,)
        assert simulated_k_data.is_complex()

        reconstructed_img_adj = thz_op.op_adj(simulated_k_data)
        print(f"Adjoint op output shape (reconstructed image): {reconstructed_img_adj.shape}")
        assert reconstructed_img_adj.shape == img_s_thz
        assert not reconstructed_img_adj.is_complex() # Should be real

        # Basic dot product test
        # x_dp should be real, y_dp_rand (k-space) should be complex
        x_dp_thz = torch.randn_like(phantom_thz_image)
        y_dp_rand_thz = torch.randn(num_measurements_thz, dtype=torch.complex64, device=device_thz_op)

        Ax_thz = thz_op.op(x_dp_thz) # Ax is complex
        Aty_thz = thz_op.op_adj(y_dp_rand_thz) # Aty is real

        # LHS: <Ax, y> (complex dot product)
        # Ax is complex, y_dp_rand_thz is complex
        lhs_thz = torch.vdot(Ax_thz.flatten(), y_dp_rand_thz.flatten())

        # RHS: <x, A^H y> (real dot product, as x_dp_thz and Aty_thz are real)
        # x_dp_thz is real, Aty_thz is real
        rhs_thz = torch.dot(x_dp_thz.flatten(), Aty_thz.flatten())

        # For <Ax,y> = <x, A^H y> where x is real, y is complex:
        # LHS is sum( (Ax)_i * conj(y_i) )
        # RHS is sum( x_i * conj((A^H y)_i) )
        # Since (A^H y) is real for our operator, conj((A^H y)_i) = (A^H y)_i.
        # So RHS becomes sum(x_i * (A^H y)_i), which is a real value.
        # Thus, the imaginary part of LHS should be zero.
        print(f"THz Fourier Op Dot product test: LHS={lhs_thz.item():.6f} (complex), RHS={rhs_thz.item():.6f} (real)")
        assert np.isclose(lhs_thz.real.item(), rhs_thz.item(), rtol=1e-3), "Real parts of dot product differ."
        assert np.isclose(lhs_thz.imag.item(), 0.0, atol=1e-4), "Imaginary part of LHS should be near zero."


        print("TerahertzOperator (Fourier Sampling) __main__ checks completed.")

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1,3, figsize=(12,4))
        # axes[0].imshow(phantom_thz_image.cpu().numpy()); axes[0].set_title("Phantom")
        # #axes[1].plot(np.abs(simulated_k_data.cpu().numpy())); axes[1].set_title("K-space Data Mag")
        # k_grid_vis = torch.zeros(img_s_thz, dtype=torch.complex64, device=device_thz_op)
        # u_indices_vis = torch.round(k_locs[:, 0] + img_s_thz[1] / 2.0).long().clamp(0, img_s_thz[1] - 1)
        # v_indices_vis = torch.round(k_locs[:, 1] + img_s_thz[0] / 2.0).long().clamp(0, img_s_thz[0] - 1)
        # k_grid_vis[v_indices_vis, u_indices_vis] = simulated_k_data
        # axes[1].imshow(torch.log(torch.abs(k_grid_vis)+1e-9).cpu().numpy()); axes[1].set_title("Gridded K-space (log abs)")
        # axes[2].imshow(reconstructed_img_adj.cpu().numpy()); axes[2].set_title("Adjoint Recon (iFFT)")
        # plt.show()

    except Exception as e:
        print(f"Error in TerahertzOperator (Fourier Sampling) __main__ checks: {e}")
        import traceback
        traceback.print_exc()
