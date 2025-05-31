import torch
from reconlib.operators import Operator
import numpy as np

# --- Basic Radon Transform Utilities (Placeholder) ---
# These are simplified and assume parallel beam.
# A more robust Radon transform might be needed from elsewhere in reconlib or a dependency.

def simple_radon_transform(image: torch.Tensor, num_angles: int,
                           num_detector_pixels: int | None = None,
                           device='cpu') -> torch.Tensor:
    """
    Simplified parallel-beam Radon transform (placeholder).
    Assumes image is square if num_detector_pixels is None.
    Output sinogram shape: (num_angles, num_detector_pixels)
    """
    Ny, Nx = image.shape
    if num_detector_pixels is None:
        num_detector_pixels = max(Ny, Nx) # Simple assumption

    image = image.to(device)
    angles = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device)
    sinogram = torch.zeros((num_angles, num_detector_pixels), device=device, dtype=image.dtype)

    # Create a grid of coordinates for the image
    x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Detector pixel coordinates (simplified: covers -D/2 to D/2 where D is image diagonal)
    # For this placeholder, assume detector pixels roughly correspond to image pixel size projection
    # A more robust version would define detector geometry explicitly.
    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1,
                                     num_detector_pixels, device=device)

    for i, angle in enumerate(angles):
        # Rotated coordinates: t = x*cos(theta) + y*sin(theta)
        rot_coords = grid_x * torch.cos(angle) + grid_y * torch.sin(angle) # These are 't' values for each pixel

        # For each detector pixel, sum up image pixels that project onto it
        for j, det_pos in enumerate(detector_coords):
            # Simple projection: find pixels where rot_coords is close to det_pos
            # This is a very crude nearest-neighbor projection.
            # A proper radon involves line integrals (e.g. sum along lines).
            # This placeholder will be slow and not very accurate.
            # Let's use a slightly better approach: sum pixels whose rotated coordinate falls into the detector bin
            pixel_width_on_detector = 1.0 # Assume detector pixel width matches rotated image pixel width
            mask = (rot_coords >= det_pos - pixel_width_on_detector/2) & \
                   (rot_coords < det_pos + pixel_width_on_detector/2)
            sinogram[i, j] = torch.sum(image[mask])

    return sinogram

def simple_back_projection(sinogram: torch.Tensor, image_shape: tuple[int,int],
                           device='cpu') -> torch.Tensor:
    """
    Simplified parallel-beam back-projection (placeholder - adjoint of simple_radon_transform).
    Output image shape: image_shape
    """
    num_angles, num_detector_pixels = sinogram.shape
    Ny, Nx = image_shape
    sinogram = sinogram.to(device)

    reconstructed_image = torch.zeros(image_shape, device=device, dtype=sinogram.dtype)
    angles = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device)

    x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1,
                                     num_detector_pixels, device=device)

    for i, angle in enumerate(angles):
        # Rotated coordinates for each pixel in the image to be reconstructed
        rot_coords_pixel = grid_x * torch.cos(angle) + grid_y * torch.sin(angle) # (Ny, Nx)

        # For each pixel, find which detector bin it corresponds to at this angle
        # and add the sinogram value from that bin.
        # This is nearest neighbor interpolation for back-projection.

        # Find nearest detector bin for each pixel's rotated coordinate
        # rot_coords_pixel.unsqueeze(-1) -> (Ny, Nx, 1)
        # detector_coords.view(1,1,-1) -> (1,1,num_detector_pixels)
        diffs = torch.abs(rot_coords_pixel.unsqueeze(-1) - detector_coords.view(1,1,-1))
        nearest_det_indices = torch.argmin(diffs, dim=2) # (Ny, Nx)

        # Add corresponding sinogram value
        # sinogram[i, nearest_det_indices] will be (Ny, Nx)
        reconstructed_image += sinogram[i, nearest_det_indices]

    return reconstructed_image / num_angles # Average over angles

# --- End of Basic Radon Transform Utilities ---

class PCCTProjectorOperator(Operator):
    """
    Basic Forward and Adjoint Operator for Photon Counting CT (PCCT).
    Includes option for adding Poisson noise to simulated photon counts.
    ... (rest of docstring as before) ...
    """
    def __init__(self,
                 image_shape: tuple[int, int],
                 num_angles: int,
                 num_detector_pixels: int,
                 energy_bins_keV: list[tuple[float,float]],
                 source_photons_per_bin: torch.Tensor | list[float],
                 energy_scaling_factors: torch.Tensor | list[float] | None = None,
                 add_poisson_noise: bool = False, # New parameter
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.Ny, self.Nx = self.image_shape
        self.num_angles = num_angles
        self.num_detector_pixels = num_detector_pixels
        self.device = torch.device(device)

        self.energy_bins_keV = energy_bins_keV
        self.num_bins = len(energy_bins_keV)

        if isinstance(source_photons_per_bin, list):
            source_photons_per_bin = torch.tensor(source_photons_per_bin, device=self.device, dtype=torch.float32)
        self.source_photons_per_bin = source_photons_per_bin.to(self.device)
        if self.source_photons_per_bin.shape[0] != self.num_bins:
            raise ValueError("source_photons_per_bin must have one entry per energy bin.")

        if energy_scaling_factors is None:
            self.energy_scaling_factors = torch.ones(self.num_bins, device=self.device, dtype=torch.float32)
        else:
            if isinstance(energy_scaling_factors, list):
                energy_scaling_factors = torch.tensor(energy_scaling_factors, device=self.device, dtype=torch.float32)
            self.energy_scaling_factors = energy_scaling_factors.to(self.device)
            if self.energy_scaling_factors.shape[0] != self.num_bins:
                raise ValueError("energy_scaling_factors must have one entry per energy bin.")

        self.add_poisson_noise = add_poisson_noise # Store noise flag
        self.sinogram_shape = (self.num_angles, self.num_detector_pixels)
        self.measurement_shape = (self.num_bins, self.num_angles, self.num_detector_pixels)

        print(f"PCCTProjectorOperator (Basic Attenuation, Noise: {self.add_poisson_noise}) initialized.")
        # ... (rest of print statements as before)
        print(f"  Image (mu_ref) Shape: {self.image_shape}")
        print(f"  Sinogram Shape per bin: {self.sinogram_shape}")
        print(f"  Energy Bins: {self.energy_bins_keV} (Num bins: {self.num_bins})")
        print(f"  Output (photon counts) Shape: {self.measurement_shape}")


    def op(self, mu_reference_map: torch.Tensor) -> torch.Tensor:
        """
        Forward: Reference attenuation map to stack of photon count sinograms per bin.
        Can include Poisson noise if self.add_poisson_noise is True.
        """
        if mu_reference_map.shape != self.image_shape:
            raise ValueError(f"Input mu_reference_map shape {mu_reference_map.shape} must match {self.image_shape}.")
        mu_reference_map = mu_reference_map.to(self.device)

        output_sinograms_counts = torch.zeros(self.measurement_shape, device=self.device, dtype=torch.float32)

        for i in range(self.num_bins):
            mu_effective_bin = mu_reference_map * self.energy_scaling_factors[i]

            sinogram_mu_eff_bin = simple_radon_transform(
                mu_effective_bin, self.num_angles, self.num_detector_pixels, self.device
            )

            counts_bin_mean = self.source_photons_per_bin[i] * torch.exp(-sinogram_mu_eff_bin)

            if self.add_poisson_noise:
                # torch.poisson expects rate parameter (mean counts)
                # Ensure counts_bin_mean is non-negative for poisson
                counts_bin = torch.poisson(torch.clamp(counts_bin_mean, min=0.0))
            else:
                counts_bin = counts_bin_mean
            output_sinograms_counts[i, ...] = counts_bin

        return output_sinograms_counts

    def op_adj(self, measured_counts_stack: torch.Tensor) -> torch.Tensor:
        # Adjoint op remains the same, as it operates on whatever counts are provided.
        # The non-linearity and noise are part of the forward process being inverted.
        if measured_counts_stack.shape != self.measurement_shape:
            raise ValueError(f"Input counts_stack shape {measured_counts_stack.shape} must match {self.measurement_shape}.")
        measured_counts_stack = measured_counts_stack.to(self.device)

        accumulated_mu_adj = torch.zeros(self.image_shape, device=self.device, dtype=torch.float32)
        epsilon = 1e-9

        for i in range(self.num_bins):
            I0_bin = self.source_photons_per_bin[i]
            counts_bin_measured = measured_counts_stack[i, ...]

            transmission = counts_bin_measured / (I0_bin + epsilon)
            transmission = torch.clamp(transmission, epsilon, 1.0 - epsilon)
            line_integrals_bin = -torch.log(transmission)

            mu_reconstructed_bin_adj = simple_back_projection(
                line_integrals_bin, self.image_shape, self.device
            )

            mu_input_adj_contribution = mu_reconstructed_bin_adj / (self.energy_scaling_factors[i] + epsilon)
            accumulated_mu_adj += mu_input_adj_contribution

        return accumulated_mu_adj

if __name__ == '__main__':
    print("\nRunning basic PCCTProjectorOperator checks (with noise option)...")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s = (16, 16)
    n_angles = 10
    n_dets = 20
    energy_bins = [(20,50), (50,80)]
    n_bins = len(energy_bins)
    I0_per_bin = torch.tensor([10000.0, 10000.0], device=dev)
    energy_scales = torch.tensor([1.0, 0.8], device=dev)

    mu_phantom = torch.zeros(img_s, device=dev, dtype=torch.float32)
    mu_phantom[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 0.02

    # Test without noise
    try:
        pcct_op_no_noise = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_per_bin,
            energy_scaling_factors=energy_scales, add_poisson_noise=False, device=dev
        )
        print("PCCTProjectorOperator (no noise) instantiated.")
        counts_no_noise = pcct_op_no_noise.op(mu_phantom)
        assert counts_no_noise.shape == (n_bins, n_angles, n_dets)
        adj_recon_no_noise = pcct_op_no_noise.op_adj(counts_no_noise)
        assert adj_recon_no_noise.shape == img_s
        print("Op and Op_adj (no noise) ran successfully.")

    except Exception as e:
        print(f"Error in PCCTProjectorOperator (no noise) checks: {e}")
        import traceback; traceback.print_exc()

    # Test with noise
    try:
        pcct_op_with_noise = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_per_bin,
            energy_scaling_factors=energy_scales, add_poisson_noise=True, device=dev
        )
        print("PCCTProjectorOperator (with noise) instantiated.")
        counts_with_noise = pcct_op_with_noise.op(mu_phantom)
        assert counts_with_noise.shape == (n_bins, n_angles, n_dets)
        # Check if noise introduced some difference from mean (statistical check)
        # This is hard to assert strictly, but counts should not be identical to counts_no_noise
        # if I0 is reasonably high for Poisson to be different from mean.
        # A simple check: if any element is different.
        # For very low counts, poisson(mean) can be equal to mean if mean is integer.
        # For higher counts, it's unlikely they are identical.
        mean_counts_for_noise_test = pcct_op_no_noise.op(mu_phantom) # Re-calc for direct comparison
        self_test_noise_added = not torch.allclose(counts_with_noise, mean_counts_for_noise_test, atol=1e-1) # Check if they are NOT all close
        print(f"Poisson noise added and values differ from mean (expected): {self_test_noise_added}")


        adj_recon_with_noise = pcct_op_with_noise.op_adj(counts_with_noise)
        assert adj_recon_with_noise.shape == img_s
        print("Op and Op_adj (with noise) ran successfully.")

    except Exception as e:
        print(f"Error in PCCTProjectorOperator (with noise) checks: {e}")
        import traceback; traceback.print_exc()

    # Radon part test (remains the same)
    print("\nTesting Radon/Back-projection part for adjointness (conceptual)...")
    class SimpleRadonLinearOperator(Operator): # Copied from previous test
        def __init__(self, image_shape, num_angles, num_detector_pixels, device):
            super().__init__(); self.image_shape = image_shape; self.num_angles = num_angles;
            self.num_detector_pixels = num_detector_pixels; self.device = device
        def op(self, image): return simple_radon_transform(image, self.num_angles, self.num_detector_pixels, self.device)
        def op_adj(self, sinogram): return simple_back_projection(sinogram, self.image_shape, self.device)

    radon_op_test = SimpleRadonLinearOperator(img_s, n_angles, n_dets, dev)
    x_radon_dp = torch.randn(img_s, device=dev, dtype=torch.float32)
    y_radon_dp = torch.randn((n_angles, n_dets), device=dev, dtype=torch.float32)
    Ax_radon = radon_op_test.op(x_radon_dp); Aty_radon = radon_op_test.op_adj(y_radon_dp)
    lhs_radon = torch.dot(Ax_radon.flatten(), y_radon_dp.flatten())
    rhs_radon = torch.dot(x_radon_dp.flatten(), Aty_radon.flatten())
    print(f"  Radon part Dot Test: LHS={lhs_radon.item():.4f}, RHS={rhs_radon.item():.4f}")
    if not np.isclose(lhs_radon.item(), rhs_radon.item(), rtol=0.2):
         print("  WARNING: Radon dot product test has a notable difference.")
    else: print("  Radon dot product test passed (with loose tolerance).")
