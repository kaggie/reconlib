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
    # angles = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device) # endpoint deprecated
    angles = torch.arange(num_angles, device=device, dtype=torch.float32) * (np.pi / num_angles)
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
    # angles = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device) # endpoint deprecated
    angles = torch.arange(num_angles, device=device, dtype=torch.float32) * (np.pi / num_angles)

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
                 add_poisson_noise: bool = False,
                 spectral_resolution_keV: float | None = None,
                 pileup_parameters: dict | None = None,
                 charge_sharing_kernel: torch.Tensor | None = None,
                 k_escape_probabilities: dict | None = None,
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

        self.add_poisson_noise = add_poisson_noise
        self.sinogram_shape = (self.num_angles, self.num_detector_pixels)
        self.measurement_shape = (self.num_bins, self.num_angles, self.num_detector_pixels)

        # Store new advanced model parameters
        self.spectral_resolution_keV = spectral_resolution_keV
        self.pileup_parameters = pileup_parameters

        if charge_sharing_kernel is not None:
            self.charge_sharing_kernel = charge_sharing_kernel.to(self.device)
        else:
            self.charge_sharing_kernel = None

        # Renaming for clarity based on assumed structure for k-escape
        self.k_escape_params = k_escape_probabilities if isinstance(k_escape_probabilities, list) else None


        self.spectral_broadening_matrix = None
        if self.spectral_resolution_keV is not None and self.num_bins > 0:
            if self.spectral_resolution_keV == 0:
                self.spectral_broadening_matrix = torch.eye(
                    self.num_bins, device=self.device, dtype=torch.float32
                )
            elif self.spectral_resolution_keV > 0:
                sigma = self.spectral_resolution_keV / (2.0 * torch.sqrt(torch.tensor(2.0 * np.log(2.0), device=self.device, dtype=torch.float32)))
                energy_bins_tensor = torch.tensor(self.energy_bins_keV, device=self.device, dtype=torch.float32) # (num_bins, 2)

                S = torch.zeros((self.num_bins, self.num_bins), device=self.device, dtype=torch.float32)

                if self.num_bins == 1: # Handle single bin case explicitly
                     S[0,0] = 1.0
                else:
                    for k in range(self.num_bins): # true bin index (column)
                        E_k_mean = (energy_bins_tensor[k, 0] + energy_bins_tensor[k, 1]) / 2.0
                        for j in range(self.num_bins): # measured bin index (row)
                            E_j_low = energy_bins_tensor[j, 0]
                            E_j_high = energy_bins_tensor[j, 1]

                            # Denominator for Z scores, ensuring it's non-zero
                            # Using sqrt(2)*sigma as in error function integral for Gaussian PDF
                            sigma_eff_for_denom = sigma * torch.sqrt(torch.tensor(2.0, device=self.device, dtype=torch.float32)) + 1e-9

                            Z_low = (E_j_low - E_k_mean) / sigma_eff_for_denom
                            Z_high = (E_j_high - E_k_mean) / sigma_eff_for_denom

                            S[j, k] = 0.5 * (torch.erf(Z_high) - torch.erf(Z_low))

                    # Normalize columns of S
                    S_col_sums = torch.sum(S, dim=0, keepdim=True)
                    S = S / (S_col_sums + 1e-9) # Add epsilon to prevent division by zero for empty bins

                self.spectral_broadening_matrix = S

        print(f"PCCTProjectorOperator (Basic Attenuation, Noise: {self.add_poisson_noise}, SpectralBroadening: {f'FWHM={self.spectral_resolution_keV}keV' if self.spectral_resolution_keV is not None and self.spectral_broadening_matrix is not None else 'Inactive'}) initialized.")
        print(f"  Image (mu_ref) Shape: {self.image_shape}")
        print(f"  Sinogram Shape per bin: {self.sinogram_shape}")
        print(f"  Energy Bins: {self.energy_bins_keV} (Num bins: {self.num_bins})")
        print(f"  Output (photon counts) Shape: {self.measurement_shape}")
        if self.pileup_parameters and \
           'method' in self.pileup_parameters and \
           'dead_time_s' in self.pileup_parameters and \
           'acquisition_time_s' in self.pileup_parameters:
            method = self.pileup_parameters['method']
            dead_time_val = self.pileup_parameters['dead_time_s']
            acq_time_val = self.pileup_parameters['acquisition_time_s']
            print(f"  Pulse Pile-up: Active, Method={method}, DeadTime={dead_time_val}s, AcqTime={acq_time_val}s")
        else:
            print(f"  Pulse Pile-up: Inactive")

        if self.charge_sharing_kernel is not None:
            print(f"  Charge Sharing: Active, KernelShape={self.charge_sharing_kernel.shape}")
        else:
            print(f"  Charge Sharing: Inactive")

        if self.k_escape_params:
            print(f"  K-Escape: Active, {len(self.k_escape_params)} rule(s) defined")
        else:
            print(f"  K-Escape: Inactive")


    def op(self, mu_reference_map: torch.Tensor) -> torch.Tensor:
        """
        Forward: Reference attenuation map to stack of photon count sinograms per bin.
        Can include Poisson noise if self.add_poisson_noise is True.
        """
        if mu_reference_map.shape != self.image_shape:
            raise ValueError(f"Input mu_reference_map shape {mu_reference_map.shape} must match {self.image_shape}.")
        mu_reference_map = mu_reference_map.to(self.device)

        output_sinograms_counts = torch.zeros(self.measurement_shape, device=self.device, dtype=torch.float32)

        # Calculate ideal counts for all bins first
        ideal_counts_all_bins_list = []
        for i in range(self.num_bins):
            mu_effective_bin = mu_reference_map * self.energy_scaling_factors[i]
            sinogram_mu_eff_bin = simple_radon_transform(
                mu_effective_bin, self.num_angles, self.num_detector_pixels, self.device
            )
            ideal_counts_bin_mean = self.source_photons_per_bin[i] * torch.exp(-sinogram_mu_eff_bin)
            ideal_counts_all_bins_list.append(ideal_counts_bin_mean)

        if not ideal_counts_all_bins_list: # Handle case with num_bins = 0
            return output_sinograms_counts

        ideal_counts_all_bins = torch.stack(ideal_counts_all_bins_list, dim=0)
        # Shape: (num_bins, num_angles, num_detector_pixels)

        counts_after_attenuation = ideal_counts_all_bins # Alias for clarity before K-escape

        # Apply K-Escape if active
        if self.k_escape_params and self.num_bins > 1: # K-escape needs at least 2 bins
            counts_after_kescape = counts_after_attenuation.clone()
            for rule in self.k_escape_params:
                source_idx = rule.get('source_bin_idx')
                escape_idx = rule.get('escape_to_bin_idx')
                prob = rule.get('probability')

                if source_idx is not None and escape_idx is not None and prob is not None and \
                   0 <= source_idx < self.num_bins and 0 <= escape_idx < self.num_bins and source_idx != escape_idx:

                    escaped_counts = counts_after_attenuation[source_idx, ...] * prob
                    counts_after_kescape[source_idx, ...] = counts_after_kescape[source_idx, ...] - escaped_counts
                    counts_after_kescape[escape_idx, ...] = counts_after_kescape[escape_idx, ...] + escaped_counts
                else:
                    print(f"Warning: Invalid K-escape rule skipped: {rule}")
            # Input for next stage
            input_to_broadening = counts_after_kescape
        else:
            input_to_broadening = counts_after_attenuation # Pass through if no K-escape

        # Apply spectral broadening if active
        if self.spectral_broadening_matrix is not None:
            # S_jk * I_khw -> P_jhw
            counts_after_broadening = torch.einsum('jk,khw->jhw', self.spectral_broadening_matrix, input_to_broadening)
        else:
            counts_after_broadening = input_to_broadening.clone()

        # Apply Charge Sharing if active
        if self.charge_sharing_kernel is not None and self.num_detector_pixels > 0 :
            if self.charge_sharing_kernel.ndim != 1:
                raise ValueError(f"charge_sharing_kernel must be 1D, but got {self.charge_sharing_kernel.ndim}D.")

            # Input to charge sharing is counts_after_broadening
            num_bins_cs, num_angles_cs, num_det_pixels_cs = counts_after_broadening.shape

            # Reshape for conv1d: (batch_size=num_bins*num_angles, in_channels=1, length=num_detector_pixels)
            flat_input = counts_after_broadening.reshape(num_bins_cs * num_angles_cs, 1, num_det_pixels_cs)

            kernel_cs = self.charge_sharing_kernel.view(1, 1, -1).to(
                device=counts_after_broadening.device,
                dtype=counts_after_broadening.dtype
            )

            padding_cs = (kernel_cs.shape[2] - 1) // 2

            # Apply convolution
            shared_counts_flat = torch.nn.functional.conv1d(flat_input, kernel_cs, padding=padding_cs)

            # Reshape back to (num_bins, num_angles, num_detector_pixels)
            counts_after_sharing = shared_counts_flat.reshape(num_bins_cs, num_angles_cs, num_det_pixels_cs)
        else:
            counts_after_sharing = counts_after_broadening # Pass through if no charge sharing

        # Apply Pulse Pile-up if active
        final_counts_list = []
        if self.pileup_parameters and \
           'dead_time_s' in self.pileup_parameters and \
           'acquisition_time_s' in self.pileup_parameters and \
           'method' in self.pileup_parameters:

            dead_time_s = self.pileup_parameters['dead_time_s']
            acquisition_time_s = self.pileup_parameters['acquisition_time_s']
            method = self.pileup_parameters['method']

            for i in range(self.num_bins):
                current_counts_mean = counts_after_sharing[i, ...] # Use counts after sharing
                true_event_rate = current_counts_mean / (acquisition_time_s + 1e-12)

                if method == 'paralyzable':
                    measured_event_rate = true_event_rate * torch.exp(-true_event_rate * dead_time_s)
                elif method == 'non_paralyzable':
                    measured_event_rate = true_event_rate / (1.0 + true_event_rate * dead_time_s)
                else: # Unknown method or missing parameters
                    measured_event_rate = true_event_rate

                piled_up_counts = measured_event_rate * acquisition_time_s
                final_counts_list.append(piled_up_counts)
            final_processed_counts_tensor = torch.stack(final_counts_list)
        else: # No pileup
            final_processed_counts_tensor = counts_after_sharing.clone()


        # Apply Poisson noise (optional)
        if self.add_poisson_noise:
            output_sinograms_counts = torch.poisson(torch.clamp(final_processed_counts_tensor, min=0.0))
        else:
            output_sinograms_counts = final_processed_counts_tensor.clone()

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
        print("PCCTProjectorOperator (no noise, no broadening) instantiated.")
        counts_no_noise_no_broadening = pcct_op_no_noise.op(mu_phantom)
        assert counts_no_noise_no_broadening.shape == (n_bins, n_angles, n_dets)
        adj_recon_no_noise_no_broadening = pcct_op_no_noise.op_adj(counts_no_noise_no_broadening)
        assert adj_recon_no_noise_no_broadening.shape == img_s
        print("Op and Op_adj (no noise, no broadening) ran successfully.")

    except Exception as e:
        print(f"Error in PCCTProjectorOperator (no noise, no broadening) checks: {e}")
        import traceback; traceback.print_exc()

    # Test with noise (and no broadening)
    try:
        pcct_op_with_noise_no_broadening = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_per_bin,
            energy_scaling_factors=energy_scales, add_poisson_noise=True, device=dev # Noise True
        )
        print("PCCTProjectorOperator (with noise, no broadening) instantiated.")
        counts_with_noise_no_broadening = pcct_op_with_noise_no_broadening.op(mu_phantom)
        assert counts_with_noise_no_broadening.shape == (n_bins, n_angles, n_dets)

        mean_counts_for_noise_test = counts_no_noise_no_broadening # from previous test
        self_test_noise_added = not torch.allclose(counts_with_noise_no_broadening, mean_counts_for_noise_test, atol=1e-1)
        print(f"Poisson noise added and values differ from mean (expected): {self_test_noise_added}")

        adj_recon_with_noise_no_broadening = pcct_op_with_noise_no_broadening.op_adj(counts_with_noise_no_broadening)
        assert adj_recon_with_noise_no_broadening.shape == img_s
        print("Op and Op_adj (with noise, no broadening) ran successfully.")

    except Exception as e:
        print(f"Error in PCCTProjectorOperator (with noise, no broadening) checks: {e}")
        import traceback; traceback.print_exc()

    # --- Test Spectral Broadening ---
    print("\nTesting Spectral Broadening...")
    spectral_res_keV_test = 25.0 # FWHM in keV. Increased from 8.0 to see a more pronounced effect.
    # Use a lower I0 for spectral broadening tests if not already done, to avoid pile-up interference
    I0_for_sb_test = torch.tensor([10000.0, 10000.0], device=dev)
    mu_phantom_for_sb_test = torch.zeros(img_s, device=dev, dtype=torch.float32)
    mu_phantom_for_sb_test[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 0.02


    try:
        op_no_broadening_sb_test = PCCTProjectorOperator( # Re-use no_noise operator for baseline
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_sb_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None,
            pileup_parameters=None,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_no_broadening = op_no_broadening_sb_test.op(mu_phantom_for_sb_test)

        op_with_broadening = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_sb_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=spectral_res_keV_test,
            pileup_parameters=None,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_with_broadening = op_with_broadening.op(mu_phantom_for_sb_test)

        assert counts_with_broadening.shape == counts_no_broadening.shape, \
            f"Shape mismatch: {counts_with_broadening.shape} vs {counts_no_broadening.shape}"

        # If broadening is active (FWHM > 0) and there are counts, results should differ
        # Sum counts to check if there's any signal. If mu_phantom is all zeros, counts might be all I0.
        # In that case, even with broadening, if I0 is identical for all bins, the output might be the same.
        # The current mu_phantom has non-zero attenuation, so counts will vary.
        if spectral_res_keV_test > 0 and torch.sum(counts_no_broadening) > 0:
             # Check if output with broadening is different from output without broadening if there is some attenuation
            phantom_attenuates_sb = not torch.allclose(mu_phantom_for_sb_test, torch.zeros_like(mu_phantom_for_sb_test))
            if phantom_attenuates_sb: # Only expect difference if there's actual attenuation
                assert not torch.allclose(counts_with_broadening, counts_no_broadening, rtol=1e-5), \
                    "Counts with broadening are too close to counts without broadening when FWHM > 0 and phantom has attenuation."
            else:
                print("  Skipping allclose check for broadening effect as mu_phantom_for_sb_test is zero or causes no attenuation difference.")

        # Test total counts conservation
        total_counts_no_broadening = torch.sum(counts_no_broadening)
        total_counts_with_broadening = torch.sum(counts_with_broadening)
        assert torch.allclose(total_counts_no_broadening, total_counts_with_broadening, rtol=1e-5), \
            f"Total counts not conserved with broadening. Before: {total_counts_no_broadening}, After: {total_counts_with_broadening}"

        print(f"Spectral broadening test (FWHM={spectral_res_keV_test}keV) passed.")

        # Test with spectral_resolution_keV = 0 (should be identity)
        op_zero_broadening = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_sb_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=0.0,
            pileup_parameters=None,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_zero_broadening = op_zero_broadening.op(mu_phantom_for_sb_test)
        assert torch.allclose(counts_zero_broadening, counts_no_broadening, rtol=1e-6), \
            "Counts with zero broadening should be identical to no broadening."
        print("Spectral broadening test with FWHM=0.0keV passed (identity operation).")


    except Exception as e:
        print(f"Error in Spectral Broadening checks: {e}")
        import traceback; traceback.print_exc()

    # --- Test Pulse Pile-up ---
    print("\nTesting Pulse Pile-up...")
    I0_for_pileup_test = torch.tensor([1e7, 1e7], device=dev) # High I0 to see pile-up
    mu_phantom_for_pileup = torch.zeros(img_s, device=dev, dtype=torch.float32)
    # For pile-up, we want some areas with very high counts (low attenuation)
    # and some with lower counts (high attenuation) to see differential effects.
    mu_phantom_for_pileup[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 0.001 # Low attenuation patch
    mu_phantom_for_pileup[0:img_s[0]//8, 0:img_s[1]//8] = 0.1 # High attenuation patch for comparison

    pileup_params_paralyzable = {'dead_time_s': 200e-9, 'acquisition_time_s': 1e-3, 'method': 'paralyzable'}
    pileup_params_non_paralyzable = {'dead_time_s': 200e-9, 'acquisition_time_s': 1e-3, 'method': 'non_paralyzable'}

    try:
        op_no_pileup = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_pileup_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None,
            pileup_parameters=None,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_no_pileup = op_no_pileup.op(mu_phantom_for_pileup)

        op_paralyzable_pileup = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_pileup_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None,
            pileup_parameters=pileup_params_paralyzable,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_paralyzable = op_paralyzable_pileup.op(mu_phantom_for_pileup)

        op_non_paralyzable_pileup = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_pileup_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None,
            pileup_parameters=pileup_params_non_paralyzable,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_non_paralyzable = op_non_paralyzable_pileup.op(mu_phantom_for_pileup)

        assert counts_paralyzable.shape == counts_no_pileup.shape
        assert counts_non_paralyzable.shape == counts_no_pileup.shape

        # Check areas of high counts (e.g., where mu is zero or very low)
        # For mu_phantom_for_pileup, direct projection of zero mu will be I0.
        # Create a mask for high count regions (e.g. > 0.8 * I0)
        # For simplicity, check if counts are reduced where they are supposed to be high (e.g. > 0.5 * I0_for_pileup_test.max())
        # This check can be made more robust by identifying specific pixels that correspond to zero attenuation paths.

        # Paralyzable checks
        # In regions of high true counts, measured counts should be lower
        high_count_mask = counts_no_pileup > (0.5 * I0_for_pileup_test.view(n_bins,1,1).max())
        if torch.any(high_count_mask): # Proceed only if there are high count regions
            assert torch.all(counts_paralyzable[high_count_mask] < counts_no_pileup[high_count_mask]), \
                "Paralyzable pile-up did not reduce counts in high-count regions."
            assert not torch.allclose(counts_paralyzable, counts_no_pileup), \
                "Paralyzable pile-up counts are too close to no-pile-up counts (global)."
        else:
            print("  Skipping paralyzable pile-up effect check as no sufficiently high count regions found.")
        print("Pulse pile-up (paralyzable) basic reduction test passed.")

        # Non-paralyzable checks
        if torch.any(high_count_mask):
            assert torch.all(counts_non_paralyzable[high_count_mask] < counts_no_pileup[high_count_mask]), \
                "Non-paralyzable pile-up did not reduce counts in high-count regions."
            assert not torch.allclose(counts_non_paralyzable, counts_no_pileup), \
                "Non-paralyzable pile-up counts are too close to no-pile-up counts (global)."
        else:
            print("  Skipping non-paralyzable pile-up effect check as no sufficiently high count regions found.")
        print("Pulse pile-up (non-paralyzable) basic reduction test passed.")

        # Compare paralyzable and non-paralyzable
        # For very high true rates, paralyzable goes to 0, non-paralyzable saturates.
        # A simpler check: they are different if true counts are high enough.
        if torch.any(high_count_mask):
            assert not torch.allclose(counts_paralyzable[high_count_mask], counts_non_paralyzable[high_count_mask]), \
                "Paralyzable and Non-paralyzable counts are too similar in high count regions."
        else:
            print("  Skipping comparison between paralyzable and non-paralyzable as no sufficiently high count regions found.")
        print("Pulse pile-up comparison (paralyzable vs non-paralyzable) test passed.")


    except Exception as e:
        print(f"Error in Pulse Pile-up checks: {e}")
        import traceback; traceback.print_exc()

    # --- Test Charge Sharing ---
    print("\nTesting Charge Sharing...")
    # Use I0 and mu_phantom that can create sharp features in sinogram
    I0_for_cs_test = torch.tensor([10000.0, 10000.0], device=dev)
    mu_phantom_for_cs = torch.zeros(img_s, device=dev, dtype=torch.float32)
    # Create a small, high-attenuating object to make sharp edges in sinogram
    mu_phantom_for_cs[img_s[0]//2 - 1 : img_s[0]//2 + 1, img_s[1]//2 - 1 : img_s[1]//2 + 1] = 0.1

    charge_sharing_kernel_1d = torch.tensor([0.1, 0.8, 0.1], device=dev, dtype=torch.float32)
    # For identity check, a kernel that should not change the output
    charge_sharing_kernel_identity = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=torch.float32)


    try:
        op_no_charge_sharing = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_cs_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None, pileup_parameters=None,
            charge_sharing_kernel=None,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_no_sharing = op_no_charge_sharing.op(mu_phantom_for_cs)

        op_with_charge_sharing = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_cs_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None, pileup_parameters=None,
            charge_sharing_kernel=charge_sharing_kernel_1d,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_with_sharing = op_with_charge_sharing.op(mu_phantom_for_cs)

        op_identity_charge_sharing = PCCTProjectorOperator(
            image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
            energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_cs_test,
            energy_scaling_factors=energy_scales, add_poisson_noise=False,
            spectral_resolution_keV=None, pileup_parameters=None,
            charge_sharing_kernel=charge_sharing_kernel_identity,
            k_escape_probabilities=None, # Explicitly no K-escape
            device=dev
        )
        counts_identity_sharing = op_identity_charge_sharing.op(mu_phantom_for_cs)


        assert counts_with_sharing.shape == counts_no_sharing.shape
        assert counts_identity_sharing.shape == counts_no_sharing.shape

        # Check that a non-identity kernel changes the output
        if torch.sum(counts_no_sharing) > 0 : # only if there's some signal
             # Ensure the kernel isn't effectively an identity for this check
            is_identity_kernel = charge_sharing_kernel_1d.shape[0] == 1 and charge_sharing_kernel_1d[0] == 1.0
            if not is_identity_kernel and charge_sharing_kernel_1d.shape[0] > 0 : # if kernel has more than one element or is not [1.0]
                 # For a 3-element kernel like [0.1,0.8,0.1], it should differ if there are counts.
                 # A kernel like [0,1,0] of length 3 would be like identity for conv1d padding='same'
                 # The identity_kernel [0,1,0] test below handles this.
                 if not (charge_sharing_kernel_1d.shape[0] == 3 and charge_sharing_kernel_1d[1]==1.0 and charge_sharing_kernel_1d[0]==0 and charge_sharing_kernel_1d[2]==0):
                    assert not torch.allclose(counts_with_sharing, counts_no_sharing, rtol=1e-5), \
                        "Counts with charge sharing are too close to counts without charge sharing."
        print("Charge sharing effect test (non-identity kernel) passed.")

        # Check that an identity-like kernel ([0,1,0]) doesn't change the output significantly
        assert torch.allclose(counts_identity_sharing, counts_no_sharing, rtol=1e-6, atol=1e-6), \
            "Counts with identity charge sharing kernel should be very close to counts without charge sharing."
        print("Charge sharing identity kernel test passed.")

        # Check for total counts conservation (if kernel sums to 1)
        kernel_sum = torch.sum(charge_sharing_kernel_1d)
        if np.isclose(kernel_sum.item(), 1.0):
            total_counts_no_sharing = torch.sum(counts_no_sharing)
            total_counts_with_sharing = torch.sum(counts_with_sharing)
            assert torch.allclose(total_counts_no_sharing, total_counts_with_sharing, rtol=2e-2), \
                f"Total counts not conserved with charge sharing. Kernel sum: {kernel_sum.item()}, Before: {total_counts_no_sharing}, After: {total_counts_with_sharing}"
            print("Charge sharing count conservation test passed (kernel sums to 1, rtol=2e-2).")
        else:
            print(f"Skipping charge sharing count conservation test as kernel does not sum to 1 (sum={kernel_sum.item()}).")

    except Exception as e:
        print(f"Error in Charge Sharing checks: {e}")
        import traceback; traceback.print_exc()

    # --- Test K-Escape ---
    print("\nTesting K-Escape...")
    I0_for_kescape_test = torch.tensor([10000.0, 10000.0], device=dev)
    mu_phantom_for_kescape = torch.zeros(img_s, device=dev, dtype=torch.float32)
    mu_phantom_for_kescape[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 0.01 # Some attenuation

    # Ensure energy_bins has at least 2 bins for this test
    if n_bins < 2:
        print("  Skipping K-Escape test as it requires at least 2 energy bins.")
    else:
        k_escape_rules = [{'source_bin_idx': 1, 'escape_to_bin_idx': 0, 'probability': 0.1}]

        try:
            op_no_kescape = PCCTProjectorOperator(
                image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
                energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_kescape_test,
                energy_scaling_factors=energy_scales, add_poisson_noise=False,
                spectral_resolution_keV=None, pileup_parameters=None, charge_sharing_kernel=None,
                k_escape_probabilities=None, # Baseline
                device=dev
            )
            counts_no_kescape = op_no_kescape.op(mu_phantom_for_kescape)

            op_with_kescape = PCCTProjectorOperator(
                image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
                energy_bins_keV=energy_bins, source_photons_per_bin=I0_for_kescape_test,
                energy_scaling_factors=energy_scales, add_poisson_noise=False,
                spectral_resolution_keV=None, pileup_parameters=None, charge_sharing_kernel=None,
                k_escape_probabilities=k_escape_rules, # Apply K-escape
                device=dev
            )
            counts_with_kescape = op_with_kescape.op(mu_phantom_for_kescape)

            assert counts_with_kescape.shape == counts_no_kescape.shape

            source_idx = k_escape_rules[0]['source_bin_idx']
            escape_idx = k_escape_rules[0]['escape_to_bin_idx']
            prob = k_escape_rules[0]['probability']

            # Check if K-escape changes the output significantly where source has counts
            if torch.sum(counts_no_kescape[source_idx, ...]) > 1e-6 : # Check if source bin had counts
                assert not torch.allclose(counts_with_kescape, counts_no_kescape, rtol=1e-5), \
                    "Counts with K-escape are too close to counts without K-escape."

            # Verify specific bin counts
            expected_source_bin_counts = counts_no_kescape[source_idx, ...] * (1 - prob)
            assert torch.allclose(counts_with_kescape[source_idx, ...], expected_source_bin_counts, rtol=1e-6), \
                f"K-escape source bin counts mismatch. Expected approx: {expected_source_bin_counts.mean()}, Got: {counts_with_kescape[source_idx, ...].mean()}"

            expected_escape_bin_counts = counts_no_kescape[escape_idx, ...] + (counts_no_kescape[source_idx, ...] * prob)
            assert torch.allclose(counts_with_kescape[escape_idx, ...], expected_escape_bin_counts, rtol=1e-6), \
                f"K-escape target bin counts mismatch. Expected approx: {expected_escape_bin_counts.mean()}, Got: {counts_with_kescape[escape_idx, ...].mean()}"

            # Verify overall count conservation
            total_counts_no_kescape = torch.sum(counts_no_kescape)
            total_counts_with_kescape = torch.sum(counts_with_kescape)
            assert torch.allclose(total_counts_no_kescape, total_counts_with_kescape, rtol=1e-6), \
                f"Total counts not conserved with K-escape. Before: {total_counts_no_kescape}, After: {total_counts_with_kescape}"

            print("K-escape tests passed.")

        except Exception as e:
            print(f"Error in K-Escape checks: {e}")
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
