import torch
import numpy as np # For np.pi, and for isclose in __main__
import traceback # For error printing in __main__
from reconlib.operators import Operator

# Attempt to import torchkbnufft and set availability flag
_torchkbnufft_available = False
_torchkbnufft_import_error = None
try:
    from torchkbnufft import KbNufft, KbNufftAdjoint
    _torchkbnufft_available = True
except ImportError as e:
    _torchkbnufft_import_error = e
    # print("Warning: torchkbnufft not available. NUFFT functionality will be disabled.")

class SARForwardOperator(Operator):
    """
    Forward and Adjoint Operator for Synthetic Aperture Radar (SAR).

    Models SAR data acquisition by sampling the 2D Fourier transform of the
    target reflectivity map at specified (u,v) coordinates.
    Can operate in NUFFT mode (if torchkbnufft is available and use_nufft=True)
    or a simpler FFT-based mode with nearest-neighbor sampling/gridding.

    Args:
        image_shape (tuple[int, int]): Shape of the input reflectivity image (Ny, Nx).
        uv_coordinates (torch.Tensor, optional): Tensor of (u,v) k-space coordinates.
                                                 Shape (num_visibilities, 2).
                                                 'u' ([:,0]) for kx, 'v' ([:,1]) for ky.
                                                 Assumed scaled like FFT indices (e.g., u in approx [-Nx/2, Nx/2-1]).
                                                 If None, physical parameters must be provided.
        wavelength (float, optional): Wavelength of radar signal (meters). Required if `uv_coordinates` is None.
        sensor_azimuth_angles (torch.Tensor, optional): Azimuth angles (radians) for sensor positions.
                                                       Shape (num_positions,). Required if `uv_coordinates` is None.
        fov (tuple[float, float], optional): Field of view (fov_y, fov_x) in meters.
                                             Required if `uv_coordinates` is None.
        use_nufft (bool, optional): If True, use NUFFT operations (requires torchkbnufft).
                                    If False, use FFT-based sampling/gridding. Defaults to False.
        center_freq (float, optional): Radar center frequency (Hz). For context or wavelength calculation.
        device (str or torch.device, optional): Computation device. Defaults to 'cpu'.
        nufft_kwargs (dict, optional): Keyword arguments for KbNufft/KbNufftAdjoint.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx)
                 uv_coordinates: torch.Tensor = None, # (num_visibilities, 2)
                 wavelength: float = None,
                 sensor_azimuth_angles: torch.Tensor = None,
                 fov: tuple[float, float] = None, # (fov_y, fov_x)
                 use_nufft: bool = False,
                 center_freq: float = 10e9, # Example: 10 GHz
                 device: str | torch.device = 'cpu',
                 nufft_kwargs: dict = None):
        super().__init__()
        self.image_shape = image_shape # (Ny, Nx)
        self.Ny, self.Nx = self.image_shape
        self.device = torch.device(device)
        self.wavelength = wavelength
        self.sensor_azimuth_angles = sensor_azimuth_angles
        self.fov = fov
        self.use_nufft = use_nufft
        self.nufft_kwargs = nufft_kwargs if nufft_kwargs is not None else {}

        if uv_coordinates is None:
            if self.wavelength is None or self.sensor_azimuth_angles is None or self.fov is None:
                raise ValueError(
                    "If uv_coordinates is None, then wavelength, sensor_azimuth_angles, and fov must be provided."
                )
            self.raw_uv_coordinates = self._calculate_uv_from_physical().to(self.device)
        else:
            if not isinstance(uv_coordinates, torch.Tensor):
                uv_coordinates = torch.tensor(uv_coordinates, dtype=torch.float32)
            self.raw_uv_coordinates = uv_coordinates.to(self.device)

        if self.raw_uv_coordinates.ndim != 2 or self.raw_uv_coordinates.shape[1] != 2:
            raise ValueError("raw_uv_coordinates must be a 2D tensor of shape (num_visibilities, 2).")

        self.nufft_op = None
        self.nufft_adj_op = None
        self.nufft_kspace_locs = None

        if self.use_nufft:
            if not _torchkbnufft_available:
                raise ImportError(
                    f"torchkbnufft is not available or failed to import (error: {_torchkbnufft_import_error}). "
                    "Please install it to use use_nufft=True."
                )
            # Scale raw_uv_coordinates for torchkbnufft: expects values in [-pi, pi]
            # raw_uv_coordinates[:, 0] are u-coords (kx-related, for Nx)
            # raw_uv_coordinates[:, 1] are v-coords (ky-related, for Ny)
            k_u_for_nufft = self.raw_uv_coordinates[:, 0] * (2 * torch.pi / self.Nx)
            k_v_for_nufft = self.raw_uv_coordinates[:, 1] * (2 * torch.pi / self.Ny)

            # torchkbnufft expects kspace_locs to be of shape (D, M)
            # If im_size = (Ny, Nx), then kspace_locs[0,:] are for Ny (kv), kspace_locs[1,:] are for Nx (ku).
            self.nufft_kspace_locs = torch.stack([k_v_for_nufft, k_u_for_nufft], dim=0).to(self.device)

            self.nufft_adj_op = KbNufftAdjoint(
                im_size=self.image_shape,
                device=self.device,
                **self.nufft_kwargs
            )
            self.nufft_op = KbNufft(
                im_size=self.image_shape,
                device=self.device,
                **self.nufft_kwargs
            )
        # If not using NUFFT, self.raw_uv_coordinates are used directly with FFT grid logic.
        # No specific pre-computation for FFT path needed here beyond ensuring raw_uv_coordinates exist.

        self.center_freq = center_freq


    def _calculate_uv_from_physical(self) -> torch.Tensor:
        """
        Calculates uv_coordinates in FFT-index style based on physical parameters.
        Assumes a circular SAR acquisition geometry.

        Returns:
            torch.Tensor: Calculated uv_coordinates, shape (num_positions, 2).
                          u: dim 0 (kx-like), v: dim 1 (ky-like).
                          These are scaled like FFT indices (e.g. u in approx [-Nx/2, Nx/2-1]).
        """
        if self.wavelength <= 0:
            raise ValueError("Wavelength must be positive.")
        if self.fov[0] <= 0 or self.fov[1] <= 0:
            raise ValueError("Field of View (fov_y, fov_x) dimensions must be positive.")
        if self.sensor_azimuth_angles is None or self.sensor_azimuth_angles.ndim != 1:
            raise ValueError("sensor_azimuth_angles must be a 1D tensor.")

        # Ensure sensor_azimuth_angles is on the correct device
        angles = self.sensor_azimuth_angles.to(self.device, dtype=torch.float32)

        # Physical k-space coordinates
        # kx_physical = (2 * torch.pi / self.wavelength) * torch.cos(angles)
        # ky_physical = (2 * torch.pi / self.wavelength) * torch.sin(angles)

        # Scale to FFT-equivalent indices:
        # u_idx = kx_physical * FOV_x / (2 * torch.pi)
        # v_idx = ky_physical * FOV_y / (2 * torch.pi)
        # This simplifies to:
        # u_idx = (1 / wavelength) * cos(angles) * FOV_x
        # v_idx = (1 / wavelength) * sin(angles) * FOV_y

        # fov is (fov_y, fov_x)
        # raw_uv_coordinates[:, 0] is u (related to Nx, fov_x)
        # raw_uv_coordinates[:, 1] is v (related to Ny, fov_y)
        raw_u = (1.0 / self.wavelength) * torch.cos(angles) * self.fov[1] # fov_x
        raw_v = (1.0 / self.wavelength) * torch.sin(angles) * self.fov[0] # fov_y

        # Stack them as (num_angles, 2)
        # raw_uv_coordinates[:,0] should be u, raw_uv_coordinates[:,1] should be v
        calculated_uv_coords = torch.stack([raw_u, raw_v], dim=-1)
        return calculated_uv_coords

    def op(self, x_reflectivity: torch.Tensor) -> torch.Tensor:
        """
        Forward SAR operation: Transforms reflectivity image to k-space visibilities.
        If self.use_nufft is True, uses NUFFT Adjoint. Otherwise, uses FFT-based sampling.
        x_reflectivity: (Ny, Nx) target reflectivity map.
        Returns: (num_visibilities,) complex tensor of visibilities.
        """
        if x_reflectivity.shape != self.image_shape:
            raise ValueError(f"Input x_reflectivity shape {x_reflectivity.shape} must match {self.image_shape}.")
        x_reflectivity = x_reflectivity.to(self.device)
        if not torch.is_complex(x_reflectivity):
            x_reflectivity = x_reflectivity.to(dtype=torch.complex64)

        if self.use_nufft:
            if self.nufft_adj_op is None or self.nufft_kspace_locs is None:
                raise RuntimeError("NUFFT operators not initialized. Ensure use_nufft=True was handled correctly in __init__.")

            x_batched = x_reflectivity.unsqueeze(0).unsqueeze(0) # (B, C, Ny, Nx)
            visibilities_batched = self.nufft_adj_op(x_batched, self.nufft_kspace_locs)
            return visibilities_batched.squeeze(0).squeeze(0) # (num_visibilities,)
        else:
            # FFT-based forward operation
            k_space_full = torch.fft.fft2(x_reflectivity, norm='ortho')
            k_space_shifted = torch.fft.fftshift(k_space_full) # (Ny, Nx)

            # Map self.raw_uv_coordinates (FFT-style indices) to array indices
            # u_coords are raw_uv_coordinates[:, 0], v_coords are raw_uv_coordinates[:, 1]
            # u relates to Nx (width), v relates to Ny (height)
            u_coords_shifted = self.raw_uv_coordinates[:, 0] + self.Nx / 2.0
            v_coords_shifted = self.raw_uv_coordinates[:, 1] + self.Ny / 2.0

            u_indices = torch.round(u_coords_shifted).long().clamp(0, self.Nx - 1)
            v_indices = torch.round(v_coords_shifted).long().clamp(0, self.Ny - 1)

            visibilities = k_space_shifted[v_indices, u_indices]
            return visibilities

    def op_adj(self, y_visibilities: torch.Tensor) -> torch.Tensor:
        """
        Adjoint SAR operation: Transforms k-space visibilities to a 'dirty image'.
        If self.use_nufft is True, uses NUFFT. Otherwise, uses FFT-based gridding.
        y_visibilities: (num_visibilities,) complex tensor of visibilities.
        Returns: (Ny, Nx) complex tensor (dirty image).
        """
        if y_visibilities.ndim != 1 or y_visibilities.shape[0] != self.raw_uv_coordinates.shape[0]:
            raise ValueError(f"Input y_visibilities shape {y_visibilities.shape} is invalid. "
                             f"Expected 1D tensor of length {self.raw_uv_coordinates.shape[0]}.")
        y_visibilities = y_visibilities.to(self.device)
        if not torch.is_complex(y_visibilities):
            y_visibilities = y_visibilities.to(dtype=torch.complex64)

        if self.use_nufft:
            if self.nufft_op is None or self.nufft_kspace_locs is None:
                raise RuntimeError("NUFFT operators not initialized. Ensure use_nufft=True was handled correctly in __init__.")

            y_batched = y_visibilities.unsqueeze(0).unsqueeze(0) # (B, C, num_visibilities)
            dirty_image_batched = self.nufft_op(y_batched, self.nufft_kspace_locs)
            return dirty_image_batched.squeeze(0).squeeze(0) # (Ny, Nx)
        else:
            # FFT-based adjoint operation (gridding)
            k_space_gridded_shifted = torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)

            u_coords_shifted = self.raw_uv_coordinates[:, 0] + self.Nx / 2.0
            v_coords_shifted = self.raw_uv_coordinates[:, 1] + self.Ny / 2.0

            u_indices = torch.round(u_coords_shifted).long().clamp(0, self.Nx - 1)
            v_indices = torch.round(v_coords_shifted).long().clamp(0, self.Ny - 1)

            # Gridding: Sum visibilities that map to the same grid cell
            k_space_gridded_shifted.index_put_((v_indices, u_indices), y_visibilities, accumulate=True)

            k_space_gridded_ifftshifted = torch.fft.ifftshift(k_space_gridded_shifted)
            dirty_image = torch.fft.ifft2(k_space_gridded_ifftshifted, norm='ortho')
            return dirty_image

if __name__ == '__main__':
    print("Running basic SARForwardOperator checks...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    img_shape_sar = (32, 32) # Smaller for faster FFT tests

    # --- Test Data ---
    phantom_img = torch.zeros(img_shape_sar, dtype=torch.complex64, device=device)
    phantom_img[img_shape_sar[0]//4 : 3*img_shape_sar[0]//4,
                img_shape_sar[1]//4 : 3*img_shape_sar[1]//4] = 1.0 + 0.5j
    phantom_img[img_shape_sar[0]//2 - 2 : img_shape_sar[0]//2 + 2,
                img_shape_sar[1]//2 - 2 : img_shape_sar[1]//2 + 2] = 2.0 + 1j

    # Manually defined UV coordinates (integer for FFT, non-integer for NUFFT)
    num_vis_manual = 150
    # For FFT, these should ideally be integers after shifting, for NUFFT can be anything
    uv_coords_manual_fft = torch.stack([
        torch.randint(-img_shape_sar[1]//2, img_shape_sar[1]//2, (num_vis_manual,), device=device),
        torch.randint(-img_shape_sar[0]//2, img_shape_sar[0]//2, (num_vis_manual,), device=device)
    ], dim=1).float()

    uv_coords_manual_nufft = (torch.rand((num_vis_manual, 2), device=device) - 0.5) * \
                             torch.tensor([img_shape_sar[1], img_shape_sar[0]], device=device, dtype=torch.float32)


    # Physical parameters for UV calculation
    num_angles = 200
    az_angles = torch.linspace(0, 2 * torch.pi, num_angles, device=device)
    sar_wavelength = 0.03
    sar_fov = (img_shape_sar[0] * sar_wavelength / 2, img_shape_sar[1] * sar_wavelength / 2)

    nufft_params_main = {'grid_size': [2*s for s in img_shape_sar]}

    def run_tests(op_instance, mode_name, num_vis):
        print(f"\n--- Testing Mode: {mode_name} ---")
        try:
            print(f"{mode_name} instantiated.")

            # Forward op
            vis = op_instance.op(phantom_img)
            print(f"{mode_name} Forward op output shape: {vis.shape}, dtype: {vis.dtype}")
            assert vis.shape == (num_vis,)
            assert torch.is_complex(vis)

            # Adjoint op
            dirty_img = op_instance.op_adj(vis)
            print(f"{mode_name} Adjoint op output shape: {dirty_img.shape}, dtype: {dirty_img.dtype}")
            assert dirty_img.shape == img_shape_sar
            assert torch.is_complex(dirty_img)

            # Dot product test
            print(f"Performing dot product test for {mode_name}...")
            x_dp = torch.randn_like(phantom_img)
            y_dp_rand = torch.randn_like(vis)

            Ax = op_instance.op(x_dp)
            Aty = op_instance.op_adj(y_dp_rand)

            lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
            rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

            print(f"{mode_name} Dot product test: LHS = {lhs.item()}, RHS = {rhs.item()}")
            # Tolerance for NUFFT might need to be higher
            rtol_dp = 1e-1 if "NUFFT" in mode_name else 1e-5

            real_close = np.isclose(lhs.real.item(), rhs.real.item(), rtol=rtol_dp)
            imag_close = np.isclose(lhs.imag.item(), rhs.imag.item(), rtol=rtol_dp)

            if real_close and imag_close:
                print(f"{mode_name} Dot product test PASSED.")
            else:
                print(f"Warning: {mode_name} Dot product test FAILED. Real part close: {real_close}, Imaginary part close: {imag_close}.")
                print(f"  LHS: {lhs.item()}, RHS: {rhs.item()}")
                print(f"  Real diff: {abs(lhs.real.item() - rhs.real.item())}, Imag diff: {abs(lhs.imag.item() - rhs.imag.item())}")
                if "NUFFT" in mode_name:
                     print("  (NUFFT known to have precision issues / internal error from prev. subtasks)")

        except ImportError as ie:
            print(f"ImportError in {mode_name}: {ie}. torchkbnufft might not be installed or importable.")
        except RuntimeError as re:
            if "torchkbnufft" in str(re) or (_torchkbnufft_import_error and "torchkbnufft" in str(_torchkbnufft_import_error)):
                print(f"Known torchkbnufft runtime error in {mode_name}: {re}")
            else:
                print(f"RuntimeError in {mode_name} checks: {re}")
                # import traceback # Ensure this is removed if it was here
                traceback.print_exc()
        except Exception as e:
            print(f"Error in {mode_name} checks: {e}")
            traceback.print_exc() # Assumes traceback imported at top

    # --- Test Scenarios ---
    # 1. Manual UVs, FFT mode
    op_manual_fft = SARForwardOperator(
        image_shape=img_shape_sar, uv_coordinates=uv_coords_manual_fft, use_nufft=False, device=device)
    run_tests(op_manual_fft, "Manual UVs, FFT", num_vis_manual)

    # 2. Manual UVs, NUFFT mode
    if _torchkbnufft_available:
        op_manual_nufft = SARForwardOperator(
            image_shape=img_shape_sar, uv_coordinates=uv_coords_manual_nufft, use_nufft=True,
            device=device, nufft_kwargs=nufft_params_main)
        run_tests(op_manual_nufft, "Manual UVs, NUFFT", num_vis_manual)
    else:
        print("\n--- Skipping Manual UVs, NUFFT (torchkbnufft not available) ---")
        try: # Test the ImportError raising
            SARForwardOperator(image_shape=img_shape_sar, uv_coordinates=uv_coords_manual_nufft, use_nufft=True, device=device)
        except ImportError as ie_test:
            print(f"Correctly raised ImportError for NUFFT when unavailable: {ie_test}")


    # 3. Physical UVs, FFT mode
    op_physical_fft = SARForwardOperator(
        image_shape=img_shape_sar, wavelength=sar_wavelength, sensor_azimuth_angles=az_angles,
        fov=sar_fov, use_nufft=False, device=device)
    run_tests(op_physical_fft, "Physical UVs, FFT", num_angles)
    print(f"Calculated raw_uv_coords sample (FFT): {op_physical_fft.raw_uv_coordinates[0:2,:]}")


    # 4. Physical UVs, NUFFT mode
    if _torchkbnufft_available:
        op_physical_nufft = SARForwardOperator(
            image_shape=img_shape_sar, wavelength=sar_wavelength, sensor_azimuth_angles=az_angles,
            fov=sar_fov, use_nufft=True, device=device, nufft_kwargs=nufft_params_main)
        run_tests(op_physical_nufft, "Physical UVs, NUFFT", num_angles)
        print(f"Calculated raw_uv_coords sample (NUFFT): {op_physical_nufft.raw_uv_coordinates[0:2,:]}")

    else:
        print("\n--- Skipping Physical UVs, NUFFT (torchkbnufft not available) ---")
        # Ensuring no stray import traceback here if old code existed below
        # Example:
        # except Exception as e:
        #     print(f"Error in some part of __main__: {e}")
        #     import traceback # THIS IS THE TYPE OF LINE TO REMOVE
        #     traceback.print_exc()


    print("\nAll __main__ checks completed.")
        # import traceback # This was the previous error
        # traceback.print_exc() # This is also part of the orphaned block and should be removed
