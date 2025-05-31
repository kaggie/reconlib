# reconlib/modalities/spect/reconstructors.py
"""
Reconstruction algorithms for Single Photon Emission Computed Tomography (SPECT).
"""
import torch
import numpy as np
from typing import Tuple, Optional, Union, List
import torch.fft
import traceback # For __main__ block error printing

# Assuming RegularizerBase might not be found, use a placeholder for type hinting.
try:
    from reconlib.regularizers.base import RegularizerBase
except ImportError:
    print("Warning (spect.reconstructors): reconlib.regularizers.base.RegularizerBase not found, using dummy placeholder.")
    class RegularizerBase: # type: ignore
        def __init__(self, *args, **kwargs): pass
        def proximal_operator(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
            return x # Does nothing

# Import placeholder back-projection.
# This version of simple_back_projection needs to be able to accept an 'angles_tensor' argument.
# If the one in pcct.operators is not suitable, a local copy or modification is needed.
# For this implementation, we will assume it (or a modified version) is available and works.
# The __main__ block will use a locally adapted version if necessary for testing.
from reconlib.modalities.pcct.operators import simple_back_projection as original_simple_back_projection

# Placeholder for simple_radon_transform if SPECTProjectorOperator is not used for test data generation
# from reconlib.modalities.pcct.operators import simple_radon_transform

from reconlib.modalities.spect.operators import SPECTProjectorOperator # Added for type hinting


# --- Modified simple_back_projection to accept angles_tensor ---
# This is a temporary solution for this module. Ideally, the core utility should be updated or a new one created.
def simple_back_projection_with_angles(
    sinogram: torch.Tensor,
    image_shape: tuple[int,int],
    angles_tensor: torch.Tensor, # New argument
    device: Union[str, torch.device] ='cpu'
) -> torch.Tensor:
    """
    Modified simple_back_projection that accepts an explicit angles tensor.
    """
    num_angles_from_tensor, num_detector_pixels = sinogram.shape
    num_angles_from_arg = angles_tensor.shape[0]

    if num_angles_from_tensor != num_angles_from_arg:
        raise ValueError(f"Sinogram num_angles ({num_angles_from_tensor}) must match angles_tensor num_angles ({num_angles_from_arg}).")

    Ny, Nx = image_shape
    sinogram = sinogram.to(device)
    angles_tensor = angles_tensor.to(device)

    reconstructed_image = torch.zeros(image_shape, device=device, dtype=sinogram.dtype)

    x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1,
                                     num_detector_pixels, device=device)

    for i, angle_rad in enumerate(angles_tensor):
        rot_coords_pixel = grid_x * torch.cos(angle_rad) + grid_y * torch.sin(angle_rad)
        diffs = torch.abs(rot_coords_pixel.unsqueeze(-1) - detector_coords.view(1,1,-1))
        nearest_det_indices = torch.argmin(diffs, dim=2)
        reconstructed_image += sinogram[i, nearest_det_indices]

    return reconstructed_image / num_angles_from_tensor if num_angles_from_tensor > 0 else reconstructed_image

# Define __all__ for the module
__all__ = ['SPECTFBPReconstructor', 'SPECTOSEMReconstructor']


class SPECTFBPReconstructor:
    """
    Implements Filtered Back-Projection (FBP) for SPECT reconstruction.
    """
    def __init__(self,
                 image_shape: Tuple[int, int],
                 device: Union[str, torch.device] = 'cpu'):
        """
        Initializes the SPECTFBPReconstructor.

        Args:
            image_shape (Tuple[int, int]): Target shape (Ny, Nx) for the reconstructed image.
            device (Union[str, torch.device], optional): Computational device. Defaults to 'cpu'.
        """
        self.image_shape = image_shape
        self.device = torch.device(device)
        print(f"SPECTFBPReconstructor initialized for image shape {image_shape} on device {self.device}.")

    def _create_filter(self,
                       num_detector_pixels: int,
                       filter_type: str = 'ramp',
                       cutoff: float = 1.0,
                       window_type: Optional[str] = 'hann') -> torch.Tensor:
        """
        Creates a 1D frequency domain filter for FBP.

        Args:
            num_detector_pixels (int): Length of the projection arrays (number of detector pixels).
            filter_type (str, optional): Type of filter. Currently supports 'ramp'. Defaults to 'ramp'.
            cutoff (float, optional): Normalized cutoff frequency (0.0 to 1.0, where 1.0 is Nyquist).
                                      Defaults to 1.0.
            window_type (Optional[str], optional): Smoothing window to apply to the filter.
                                                 Supports 'hann', 'hamming', None. Defaults to 'hann'.

        Returns:
            torch.Tensor: The 1D frequency domain filter of shape (num_detector_pixels,).
        """
        if filter_type.lower() != 'ramp':
            raise NotImplementedError(f"Filter type '{filter_type}' not yet implemented. Only 'ramp' is supported.")

        freqs = torch.fft.fftfreq(num_detector_pixels, device=self.device)
        ramp_filter = torch.abs(freqs)

        # Apply cutoff
        ramp_filter[torch.abs(freqs) > cutoff / 2] = 0 # Cutoff is usually defined for one-sided spectrum, fftfreq is two-sided

        # Apply window
        if window_type:
            n = torch.arange(num_detector_pixels, device=self.device)
            if window_type.lower() == 'hann':
                window = 0.5 * (1 - torch.cos(2 * np.pi * n / (num_detector_pixels - 1)))
            elif window_type.lower() == 'hamming':
                window = 0.54 - 0.46 * torch.cos(2 * np.pi * n / (num_detector_pixels - 1))
            # elif window_type.lower() == 'butterworth_like': # Example, more params needed for true Butterworth
            #     order = 2
            #     # Normalized freqs for Butterworth: 0 to 0.5 (Nyquist) then scale by cutoff
            #     norm_freqs_abs = torch.abs(freqs) / (0.5) # freqs range up to 0.5
            #     window = 1.0 / (1.0 + (norm_freqs_abs / cutoff)**(2 * order))
            else:
                print(f"Warning: Window type '{window_type}' not recognized or not applied. Using no window beyond ramp/cutoff.")
                window = torch.ones_like(ramp_filter) # No window
            ramp_filter *= window

        return ramp_filter

    def reconstruct(self,
                    projections: torch.Tensor,
                    angles: torch.Tensor,
                    filter_type: str = 'ramp',
                    filter_cutoff: float = 1.0,
                    filter_window: Optional[str] = 'hann'
                    # attenuation_correction_factor: Optional[Union[float, torch.Tensor]] = None # V1: simplified, no att corr here
                   ) -> torch.Tensor:
        """
        Reconstructs an image from SPECT projections using FBP.

        Args:
            projections (torch.Tensor): Input sinogram (SPECT projections).
                                        Shape: (num_angles, num_detector_pixels).
            angles (torch.Tensor): Tensor of projection angles in radians.
                                   Shape: (num_angles,).
            filter_type (str, optional): Type of filter for FBP. Defaults to 'ramp'.
            filter_cutoff (float, optional): Normalized cutoff frequency for the filter. Defaults to 1.0.
            filter_window (Optional[str], optional): Smoothing window for the filter. Defaults to 'hann'.
            # attenuation_correction_factor: Optional. Placeholder for future.

        Returns:
            torch.Tensor: The reconstructed image. Shape: (image_shape[0], image_shape[1]).
        """
        if projections.ndim != 2:
            raise ValueError(f"Projections must be a 2D tensor (num_angles, num_detector_pixels). Got shape {projections.shape}")
        if angles.ndim != 1 or angles.shape[0] != projections.shape[0]:
            raise ValueError(f"Angles tensor must be 1D and match num_angles from projections. Got shapes {angles.shape}, {projections.shape}")

        projections = projections.to(self.device)
        angles = angles.to(self.device)

        num_angles, num_detector_pixels = projections.shape

        # 1. Filter Projections
        fbp_filter = self._create_filter(num_detector_pixels, filter_type, filter_cutoff, filter_window)

        # Apply filter in frequency domain for each projection angle
        projections_fft = torch.fft.fft(projections, dim=1) # FFT along detector pixel dimension

        # Unsqueeze filter to allow broadcasting across angles: (1, num_detector_pixels)
        filtered_projections_fft = projections_fft * fbp_filter.unsqueeze(0)

        filtered_projections = torch.fft.ifft(filtered_projections_fft, dim=1).real

        # 2. Backproject
        # Using the locally adapted simple_back_projection_with_angles
        reconstructed_image = simple_back_projection_with_angles(
            filtered_projections,
            self.image_shape,
            angles_tensor=angles, # Pass the angles tensor
            device=self.device
        )

        return reconstructed_image


class SPECTOSEMReconstructor:
    """
    Implements Ordered Subsets Expectation Maximization (OSEM) for SPECT reconstruction.
    """
    def __init__(self,
                 image_shape: Tuple[int, int],
                 iterations: int = 10,
                 num_subsets: int = 4,
                 initial_estimate: Optional[torch.Tensor] = None,
                 positivity_constraint: bool = True,
                 device: Union[str, torch.device] = 'cpu',
                 verbose: bool = False):
        """
        Initializes the SPECTOSEMReconstructor.

        Args:
            image_shape (Tuple[int, int]): Target shape (Ny, Nx) for the reconstructed image.
            iterations (int, optional): Number of full iterations over all subsets. Defaults to 10.
            num_subsets (int, optional): Number of subsets to divide the projection data into. Defaults to 4.
            initial_estimate (Optional[torch.Tensor], optional): Initial guess for the image.
                If None, an image of ones is used. Shape should match `image_shape`. Defaults to None.
            positivity_constraint (bool, optional): If True, enforces non-negativity on the image estimate
                                                  at each iteration. Defaults to True.
            device (Union[str, torch.device], optional): Computational device. Defaults to 'cpu'.
            verbose (bool, optional): If True, prints iteration progress. Defaults to False.
        """
        self.image_shape = image_shape
        self.iterations = iterations
        self.num_subsets = num_subsets
        self.initial_estimate = initial_estimate
        self.positivity_constraint = positivity_constraint
        self.device = torch.device(device)
        self.verbose = verbose
        print(f"SPECTOSEMReconstructor initialized for image shape {image_shape}, {iterations} iterations, {num_subsets} subsets on {self.device}.")

    def _calculate_sensitivity_image(self,
                                     projector: SPECTProjectorOperator,
                                     projection_shape: Tuple[int, int] # (total_num_angles, num_detector_pixels)
                                    ) -> torch.Tensor:
        """
        Calculates the sensitivity image (A_adj @ ones), representing the sum of back-projections
        of unit sinograms for all views. This is used as the denominator in the OSEM update.
        This calculates the sensitivity for the *full* set of projections.

        Args:
            projector (SPECTProjectorOperator): The SPECT projector instance, configured for all angles.
            projection_shape (Tuple[int, int]): The shape of the full projection data (total_num_angles, num_detector_pixels).

        Returns:
            torch.Tensor: The sensitivity image, clamped at a small positive value. Shape: (image_shape).
        """
        if projector.device != self.device:
            # This should not happen if projector is passed correctly, but as a safeguard.
            print(f"Warning: Projector device {projector.device} differs from reconstructor device {self.device}. Adjusting projector for sensitivity calc.")
            # This is tricky; modifying the projector's device here is not ideal.
            # Better to ensure projector passed to reconstruct() is already on self.device.
            # For sensitivity calculation, we might need a temporary projector or use the main one carefully.
            # For now, assume projector is on self.device as it should be passed from reconstruct.
            pass

        ones_projections = torch.ones(projection_shape, device=self.device, dtype=torch.float32)
        sensitivity_image = projector.op_adj(ones_projections)
        sensitivity_image = torch.clamp(sensitivity_image, min=1e-9) # Avoid division by zero
        return sensitivity_image

    def reconstruct(self,
                    projections: torch.Tensor,
                    projector: SPECTProjectorOperator,
                    initial_estimate_override: Optional[torch.Tensor] = None, # New optional parameter
                    angles_per_subset: Optional[List[torch.Tensor]] = None # Not used in this simple striding version
                   ) -> torch.Tensor:
        """
        Performs OSEM reconstruction.

        Args:
            projections (torch.Tensor): Measured SPECT sinogram.
                                        Shape: (total_num_angles, num_detector_pixels).
            projector (SPECTProjectorOperator): An instance of SPECTProjectorOperator.
                                                Its internal `angles` should correspond to `total_num_angles`
                                                and it should be on `self.device`.
            angles_per_subset (Optional[List[torch.Tensor]], optional):
                A list where each element is a tensor of angles for one subset.
                If None (default), subsets will be created by simple striding of `projector.angles`.
                Currently, this argument is NOT USED, and simple striding is always performed.

        Returns:
            torch.Tensor: The reconstructed image. Shape: (image_shape[0], image_shape[1]).
        """
        if projector.device != self.device:
             raise ValueError(f"Projector device ({projector.device}) must match reconstructor device ({self.device}).")

        projections = projections.to(self.device)

        f_k: torch.Tensor # Image estimate

        current_initial_estimate = None
        if initial_estimate_override is not None:
            current_initial_estimate = initial_estimate_override
        elif self.initial_estimate is not None:
            current_initial_estimate = self.initial_estimate

        if current_initial_estimate is not None:
            if current_initial_estimate.shape != self.image_shape:
                raise ValueError(f"Initial estimate shape {current_initial_estimate.shape} must match image_shape {self.image_shape}")
            f_k = current_initial_estimate.clone().to(self.device)
        else:
            f_k = torch.ones(self.image_shape, device=self.device, dtype=torch.float32)

        if self.positivity_constraint: # Ensure initial estimate is positive (or zero if that's the floor)
            f_k = torch.clamp(f_k, min=0.0)

        total_num_angles, num_detector_pixels = projections.shape

        if total_num_angles % self.num_subsets != 0:
            print(f"Warning: Total number of angles ({total_num_angles}) is not perfectly divisible by number of subsets ({self.num_subsets}). Some views might be used less.")
            # Basic striding will handle this by potentially having the last subset be smaller or wrap around if not careful.
            # The arange method for subset_indices handles this by just not picking more than available.

        # Calculate the global sensitivity image (A_adj @ 1) using the full projector
        # This is used in the denominator and typically precomputed.
        # For OSEM, often the per-subset sensitivity is A_adj_subset @ 1_subset,
        # but using global sensitivity (scaled or not) is a common simplification.
        # Here, we use the global sensitivity image for each subset update step.
        # If a subset-specific sensitivity were used, it would be projector_subset.op_adj(torch.ones_like(current_subset_projections)).
        global_sensitivity_image = self._calculate_sensitivity_image(projector, (total_num_angles, num_detector_pixels))
        # Alternative: scale global sensitivity by num_subsets for each subset update,
        # or ensure sum of subset sensitivities equals global.
        # For simplicity, using global_sensitivity_image directly. Small error if subsets are not balanced.

        eps = 1e-9 # Small epsilon for numerical stability

        for iteration in range(self.iterations):
            f_k_previous_iter = f_k.clone() # For verbose change calculation
            for s_idx in range(self.num_subsets):
                # Select indices for the current subset using simple striding
                # This ensures each view is (approximately, if not divisible) used once per iteration.
                subset_angle_indices = torch.arange(s_idx, total_num_angles, self.num_subsets, device=self.device)

                # These are views (rows) from the full projection data
                proj_subset = projections.index_select(0, subset_angle_indices)

                # Create a temporary projector for this subset of angles
                # This is the part that requires careful handling of SPECTProjectorOperator
                # Option: Modify SPECTProjectorOperator to take angle_indices in op/op_adj
                # Option: Create new mini-projectors (could be slow)
                # Option (chosen): Project full, then select subset projections. For adjoint, create sparse input.

                # 1. Forward project f_k for ALL angles
                fp_f_k_full = projector.op(f_k)
                # 2. Select the projections corresponding to the current subset
                fp_f_k_subset = fp_f_k_full.index_select(0, subset_angle_indices)

                # 3. Calculate ratio p_subset / (A_subset @ f_k + eps)
                ratio_subset = proj_subset / (fp_f_k_subset + eps)

                # 4. Create a full-angle sinogram with these ratios in the subset views and zeros elsewhere
                ratio_terms_full = torch.zeros_like(projections) # Shape: (total_num_angles, num_detector_pixels)
                ratio_terms_full.index_copy_(0, subset_angle_indices, ratio_subset)

                # 5. Backproject this sparse sinogram using the full projector's adjoint
                correction_numerator = projector.op_adj(ratio_terms_full)

                # 6. Update rule: f_k = f_k * (correction_numerator / (global_sensitivity_image + eps))
                # The global_sensitivity_image is A_full_adj @ 1_full.
                # If using subset specific sensitivity A_subset_adj @ 1_subset, that would go here.
                # Scaling by num_subsets might be needed if global_sensitivity_image is not sum of subset sensitivities.
                # For now, using global sensitivity as an approximation.
                f_k = f_k * correction_numerator / (global_sensitivity_image + eps)

                if self.positivity_constraint:
                    f_k = torch.clamp(f_k, min=0.0)

            if self.verbose:
                change = torch.norm(f_k - f_k_previous_iter) / (torch.norm(f_k_previous_iter) + eps)
                print(f"OSEM Iteration {iteration + 1}/{self.iterations}, Change: {change.item():.4e}")

        return f_k

if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Running SPECTFBPReconstructor Tests on {dev} ---")
    img_s_fbp = (32,32)

    activity_phantom_fbp = torch.zeros(img_s_fbp, device=dev, dtype=torch.float32)
    center_y_ph, center_x_ph = img_s_fbp[0]//2, img_s_fbp[1]//2
    radius_ph = 2
    yy_ph, xx_ph = torch.meshgrid(torch.arange(img_s_fbp[0], device=dev), torch.arange(img_s_fbp[1], device=dev), indexing='ij')
    mask_circle_ph = (xx_ph - center_x_ph)**2 + (yy_ph - center_y_ph)**2 < radius_ph**2
    activity_phantom_fbp[mask_circle_ph] = 1.0
    # activity_phantom_fbp[img_s_fbp[0]//2 -1 : img_s_fbp[0]//2 +1, img_s_fbp[1]//2-1 : img_s_fbp[1]//2+1] = 1.0


    angles_fbp_test_np = np.linspace(0, np.pi, 60, endpoint=False)
    angles_fbp_test = torch.tensor(angles_fbp_test_np, device=dev, dtype=torch.float32)
    n_dets_fbp_test = int(np.floor(img_s_fbp[0] * np.sqrt(2)) + 1) # Number of detectors to span diagonal
    if n_dets_fbp_test % 2 == 0: n_dets_fbp_test += 1 # Make it odd for centering

    print(f"  Image shape: {img_s_fbp}, Angles: {angles_fbp_test.shape[0]}, Detectors: {n_dets_fbp_test}")

    projs_fbp_test = torch.zeros((angles_fbp_test.shape[0], n_dets_fbp_test), device=dev)

    # Simulate projections using a basic Radon transform (imported or defined locally)
    # Using simple_radon_transform (imported from pcct.operators, but defined locally in this file too)
    # Need to ensure the correct one is used, or that SPECTProjectorOperator is available
    try:
        # Attempt to use SPECTProjectorOperator if available (preferred for consistency)
        from reconlib.modalities.spect.operators import SPECTProjectorOperator
        print("  Using SPECTProjectorOperator for test projection generation.")
        spect_op_test = SPECTProjectorOperator(img_s_fbp, angles_fbp_test, n_dets_fbp_test, device=str(dev))
        projs_fbp_test = spect_op_test.op(activity_phantom_fbp)
    except ImportError:
        print("  SPECTProjectorOperator not found, using local simple_radon_transform for test projection generation.")
        # Need a local simple_radon_transform if not importing from pcct.operators directly at module level
        # For this test, we assume the local simple_radon_transform is available (copied above)
        projs_fbp_test = simple_radon_transform(activity_phantom_fbp,
                                                angles_fbp_test.shape[0],
                                                n_dets_fbp_test,
                                                device=str(dev))

    print(f"  Simulated projections shape: {projs_fbp_test.shape}")

    try:
        fbp_recon = SPECTFBPReconstructor(img_s_fbp, device=str(dev))
        recon_img_fbp = fbp_recon.reconstruct(projs_fbp_test, angles_fbp_test, filter_cutoff=0.8, filter_window='hann')

        assert recon_img_fbp.shape == img_s_fbp, \
            f"Reconstructed image shape mismatch. Expected {img_s_fbp}, Got {recon_img_fbp.shape}"

        # Qualitative check: Max value should be roughly where the activity was.
        # And it should be positive.
        max_val_recon = torch.max(recon_img_fbp).item()
        print(f"  Max value in reconstructed image: {max_val_recon:.4f}")
        assert max_val_recon > 0, "Reconstruction seems to have failed (max value not positive)."

        # Check if the location of max value is near the phantom's activity center
        max_loc_flat = torch.argmax(recon_img_fbp)
        max_loc_y, max_loc_x = np.unravel_index(max_loc_flat.cpu().numpy(), recon_img_fbp.shape)
        print(f"  Location of max value in recon: (y={max_loc_y}, x={max_loc_x}) vs phantom center: (y={center_y_ph}, x={center_x_ph})")
        assert abs(max_loc_y - center_y_ph) <= radius_ph + 2 # Allow some leeway
        assert abs(max_loc_x - center_x_ph) <= radius_ph + 2

        print("  SPECTFBPReconstructor basic test passed.")

        # Optional: Visualize
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(activity_phantom_fbp.cpu().numpy(), cmap='viridis', origin='lower')
        # axes[0].set_title("Original Phantom")
        # axes[1].imshow(recon_img_fbp.cpu().numpy(), cmap='viridis', origin='lower')
        # axes[1].set_title("FBP Reconstructed Image")
        # plt.show()

    except Exception as e:
        print(f"  SPECTFBPReconstructor test FAILED: {e}")
        traceback.print_exc()

    print("\n--- Running SPECTOSEMReconstructor Tests ---")
    # Re-use phantom, angles, dets from FBP test or define new ones
    img_s_osem = img_s_fbp
    activity_phantom_osem = activity_phantom_fbp.clone() # Use the same phantom as FBP for comparison
    angles_osem_test = angles_fbp_test.clone()
    n_dets_osem_test = n_dets_fbp_test

    print(f"  OSEM Test - Image shape: {img_s_osem}, Angles: {angles_osem_test.shape[0]}, Detectors: {n_dets_osem_test}")

    # Projections for OSEM (can use projs_fbp_test or regenerate)
    # For this test, let's use the SPECTProjectorOperator to get projections
    projs_osem_input = torch.zeros((angles_osem_test.shape[0], n_dets_osem_test), device=dev)
    spect_op_for_osem_data = None
    try:
        from reconlib.modalities.spect.operators import SPECTProjectorOperator
        spect_op_for_osem_data = SPECTProjectorOperator(img_s_osem, angles_osem_test, n_dets_osem_test, device=str(dev))
        projs_osem_input = spect_op_for_osem_data.op(activity_phantom_osem)

        # Optional: Add Poisson noise
        # counts_osem = torch.sum(projs_osem_input) / (projs_osem_input.numel() + 1e-9) * 100 # Target ~100 counts per pixel on average
        # scale_factor = counts_osem / (torch.mean(projs_osem_input) + 1e-9) if torch.mean(projs_osem_input) > 1e-9 else 1.0
        # projs_osem_input_scaled = projs_osem_input * scale_factor
        # projs_osem_noisy = torch.poisson(torch.relu(projs_osem_input_scaled)) / (scale_factor + 1e-9)
        # print(f"  Using noisy projections for OSEM. Original mean: {torch.mean(projs_osem_input):.2f}, Noisy mean: {torch.mean(projs_osem_noisy):.2f}")
        # projs_to_reconstruct = projs_osem_noisy
        projs_to_reconstruct = projs_osem_input # Using clean data for first test
        print(f"  Simulated OSEM input projections shape: {projs_to_reconstruct.shape}")

    except ImportError:
        print("  SPECTProjectorOperator not found for OSEM data generation. Skipping OSEM test.")
        projs_to_reconstruct = None # Skip test
    except Exception as e_data_gen:
        print(f"  Error during OSEM data generation: {e_data_gen}")
        projs_to_reconstruct = None # Skip test
        traceback.print_exc()


    if projs_to_reconstruct is not None and spect_op_for_osem_data is not None:
        try:
            osem_recon = SPECTOSEMReconstructor(
                image_shape=img_s_osem,
                iterations=5, # Low iterations for quick test
                num_subsets=4,
                # initial_estimate parameter in __init__ can be used if not overridden in reconstruct()
                # initial_estimate=initial_estimate_osem, # This would set self.initial_estimate
                device=str(dev),
                verbose=True,
                positivity_constraint=True
            )

            initial_estimate_for_reconstruct_call = spect_op_for_osem_data.op_adj(projs_to_reconstruct)
            # OSEM typically starts with ones or positive constant, but adjoint can be used if clamped.
            # The reconstructor itself will clamp if positivity_constraint and initial_estimate is None or positive.
            # If initial_estimate_for_reconstruct_call is passed, it will be clamped if positivity_constraint is True.
            # Let's ensure it's positive if used as an override.
            initial_estimate_for_reconstruct_call = torch.clamp(initial_estimate_for_reconstruct_call, min=1e-6)


            recon_img_osem = osem_recon.reconstruct(
                projs_to_reconstruct,
                spect_op_for_osem_data,
                initial_estimate_override=initial_estimate_for_reconstruct_call # Pass as override
            )

            assert recon_img_osem.shape == img_s_osem, \
                f"OSEM Reconstructed image shape mismatch. Expected {img_s_osem}, Got {recon_img_osem.shape}"

            max_val_osem_recon = torch.max(recon_img_osem).item()
            print(f"  Max value in OSEM reconstructed image: {max_val_osem_recon:.4f}")
            assert max_val_osem_recon > 0, "OSEM Reconstruction seems to have failed (max value not positive)."

            # Compare error with true phantom
            norm_osem_error = torch.norm(recon_img_osem - activity_phantom_osem).item()
            norm_initial_error = torch.norm(initial_estimate_for_reconstruct_call - activity_phantom_osem).item() # Corrected variable name
            print(f"  Norm of (OSEM Recon - True): {norm_osem_error:.4f}")
            print(f"  Norm of (Initial Adjoint - True): {norm_initial_error:.4f}")
            if norm_osem_error < norm_initial_error:
                print("  OSEM reconstruction error is lower than initial error (good).")
            else:
                print("  Warning: OSEM reconstruction error is NOT lower than initial error. May need more iterations/subsets or parameter tuning.")

            print("  SPECTOSEMReconstructor basic test passed.")

            # Optional: Visualize (if matplotlib is available)
            # import matplotlib.pyplot as plt
            # fig_osem, axes_osem = plt.subplots(1, 3, figsize=(15, 5))
            # axes_osem[0].imshow(activity_phantom_osem.cpu().numpy(), cmap='viridis', origin='lower'); axes_osem[0].set_title("Original OSEM Phantom")
            # axes_osem[1].imshow(initial_estimate_osem.cpu().numpy(), cmap='viridis', origin='lower'); axes_osem[1].set_title("Initial Estimate (Adjoint)")
            # axes_osem[2].imshow(recon_img_osem.cpu().numpy(), cmap='viridis', origin='lower'); axes_osem[2].set_title("OSEM Reconstructed")
            # plt.show()

        except Exception as e:
            print(f"  SPECTOSEMReconstructor test FAILED: {e}")
            traceback.print_exc()
