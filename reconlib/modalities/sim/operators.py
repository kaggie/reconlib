import torch
from reconlib.operators import Operator
import numpy as np
from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf # Re-use for PSF
from .utils import generate_sim_patterns # Import from local utils

class SIMOperator(Operator):
    """
    Refined Forward and Adjoint Operator for Structured Illumination Microscopy (SIM).

    Models SIM acquisition:
    1. True high-res image X_hr.
    2. For each illumination pattern P_i (generated if not provided):
       - Modulated image: M_i = X_hr * P_i
       - Observed raw image: Y_i = PSF_det * M_i (convolution with detection PSF)
    The operator outputs a stack of Y_i.
    """
    def __init__(self,
                 hr_image_shape: tuple[int, int],
                 psf_detection: torch.Tensor,
                 num_angles: int = 3, # Used if patterns are not provided
                 num_phases: int = 3, # Used if patterns are not provided
                 patterns: torch.Tensor | None = None,
                 pattern_k_max_rel: float = 0.8, # Used for internal pattern generation
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.hr_image_shape = hr_image_shape
        self.psf_detection = psf_detection.to(torch.device(device))
        self.device = torch.device(device)

        if self.psf_detection.ndim != 2: # Assuming 2D SIM for now
            raise ValueError("Detection PSF must be 2D for this SIM operator.")

        # Manual padding for 'same' convolution with detection PSF
        self.psf_padding = [(s - 1) // 2 for s in self.psf_detection.shape] # (pad_H, pad_W)

        if patterns is not None:
            self.patterns = patterns.to(self.device)
            self.num_patterns = self.patterns.shape[0]
            if self.patterns.shape[1:] != self.hr_image_shape:
                raise ValueError(f"Provided patterns spatial shape {self.patterns.shape[1:]} "
                                 f"does not match hr_image_shape {self.hr_image_shape}.")
        else:
            print("SIMOperator: Patterns not provided, generating internally.")
            self.num_patterns = num_angles * num_phases
            self.patterns = generate_sim_patterns(
                hr_image_shape=self.hr_image_shape,
                num_angles=num_angles,
                num_phases=num_phases,
                k_vector_max_rel=pattern_k_max_rel,
                device=self.device
            )
            if self.patterns.shape[0] != self.num_patterns: # Should match by construction
                 raise ValueError("Internal pattern generation failed to produce num_patterns.")


        self.measurement_shape = (self.num_patterns, *self.hr_image_shape)

        print(f"SIMOperator (Refined) initialized.")
        print(f"  HR Image Shape: {self.hr_image_shape}, Num Patterns: {self.num_patterns}")
        print(f"  Detection PSF shape: {self.psf_detection.shape}, Padding: {self.psf_padding}")

    def op(self, hr_image: torch.Tensor) -> torch.Tensor:
        if hr_image.shape != self.hr_image_shape:
            raise ValueError(f"Input hr_image shape {hr_image.shape} must match {self.hr_image_shape}.")
        hr_image = hr_image.to(self.device)
        if hr_image.dtype != self.patterns.dtype: # Ensure dtype consistency for multiplication
            hr_image = hr_image.to(self.patterns.dtype)


        raw_sim_images = torch.zeros(self.measurement_shape, device=self.device, dtype=hr_image.dtype)
        psf_exp = self.psf_detection.unsqueeze(0).unsqueeze(0).to(hr_image.dtype) # (1,1,kH,kW)

        for i in range(self.num_patterns):
            pattern_i = self.patterns[i, ...]
            modulated_image = hr_image * pattern_i
            modulated_image_exp = modulated_image.unsqueeze(0).unsqueeze(0)

            blurred_modulated_image = torch.nn.functional.conv2d(
                modulated_image_exp, psf_exp, padding=self.psf_padding
            )
            raw_sim_images[i, ...] = blurred_modulated_image.squeeze()

        return raw_sim_images

    def op_adj(self, raw_sim_images_stack: torch.Tensor) -> torch.Tensor:
        if raw_sim_images_stack.shape != self.measurement_shape:
            raise ValueError(f"Input stack shape {raw_sim_images_stack.shape} must match {self.measurement_shape}.")
        raw_sim_images_stack = raw_sim_images_stack.to(self.device)
        if raw_sim_images_stack.dtype != self.patterns.dtype:
            raw_sim_images_stack = raw_sim_images_stack.to(self.patterns.dtype)


        hr_image_estimate = torch.zeros(self.hr_image_shape, device=self.device, dtype=raw_sim_images_stack.dtype)
        psf_flipped_exp = torch.flip(self.psf_detection, dims=[0,1]).unsqueeze(0).unsqueeze(0).to(raw_sim_images_stack.dtype)

        for i in range(self.num_patterns):
            raw_image_i_exp = raw_sim_images_stack[i, ...].unsqueeze(0).unsqueeze(0)

            adj_conv_raw_image = torch.nn.functional.conv2d(
                raw_image_i_exp, psf_flipped_exp, padding=self.psf_padding
            ).squeeze()

            pattern_i = self.patterns[i, ...]
            hr_image_estimate += adj_conv_raw_image * pattern_i

        # The division by num_patterns in the previous placeholder was a simple normalization.
        # A true adjoint for sum_i A_i x (where A_i is pattern_i * conv_psf)
        # would be sum_i A_i^T y_i. Each A_i^T involves pattern_i * conv_psf_adj.
        # So the sum is correct. Normalization can be handled outside if needed.
        return hr_image_estimate

if __name__ == '__main__':
    print("\nRunning basic SIMOperator (Refined) checks...")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hr_shape = (32, 32) # Smaller for faster test
    n_angles_test = 2
    n_phases_test = 2
    n_pats_test = n_angles_test * n_phases_test

    psf_det_sim_test = generate_gaussian_psf(shape=(5,5), sigma=1.0, device=dev).to(torch.float32)

    # Test case 1: Operator generates patterns internally
    print("\n--- Test Case: Internal Pattern Generation ---")
    try:
        sim_op_internal_pats = SIMOperator(
            hr_image_shape=hr_shape,
            psf_detection=psf_det_sim_test,
            num_angles=n_angles_test,
            num_phases=n_phases_test,
            patterns=None, # Force internal generation
            pattern_k_max_rel=0.7,
            device=dev
        )
        print("SIMOperator (internal patterns) instantiated.")

        phantom_hr_test = torch.randn(hr_shape, device=dev, dtype=torch.float32)
        raw_images_test = sim_op_internal_pats.op(phantom_hr_test)
        assert raw_images_test.shape == (n_pats_test, *hr_shape)

        adj_recon_test = sim_op_internal_pats.op_adj(raw_images_test)
        assert adj_recon_test.shape == hr_shape

        x_dp_i = torch.randn_like(phantom_hr_test)
        y_dp_rand_i = torch.randn_like(raw_images_test)
        Ax_i = sim_op_internal_pats.op(x_dp_i)
        Aty_i = sim_op_internal_pats.op_adj(y_dp_rand_i)
        lhs_i = torch.dot(Ax_i.flatten(), y_dp_rand_i.flatten())
        rhs_i = torch.dot(x_dp_i.flatten(), Aty_i.flatten())
        print(f"SIM Internal Pats Dot Test: LHS={lhs_i.item():.6f}, RHS={rhs_i.item():.6f}")
        assert np.isclose(lhs_i.item(), rhs_i.item(), rtol=1e-3, atol=1e-5), "Dot product test failed (internal patterns)."
        print("SIMOperator (internal patterns) dot product test passed.")

    except Exception as e:
        print(f"Error in SIMOperator (internal patterns) checks: {e}")
        import traceback; traceback.print_exc()

    # Test case 2: External patterns provided
    print("\n--- Test Case: External Pattern Generation ---")
    external_patterns_test = generate_sim_patterns(
        hr_shape, n_angles_test, n_phases_test, k_vector_max_rel=0.7, device=dev
    ).to(torch.float32)
    try:
        sim_op_external_pats = SIMOperator(
            hr_image_shape=hr_shape,
            psf_detection=psf_det_sim_test,
            patterns=external_patterns_test, # Provide patterns
            device=dev
        )
        print("SIMOperator (external patterns) instantiated.")
        # ... (rest of tests similar to above, using sim_op_external_pats) ...
        phantom_hr_test_e = torch.randn(hr_shape, device=dev, dtype=torch.float32)
        raw_images_test_e = sim_op_external_pats.op(phantom_hr_test_e)
        adj_recon_test_e = sim_op_external_pats.op_adj(raw_images_test_e)

        x_dp_e = torch.randn_like(phantom_hr_test_e)
        y_dp_rand_e = torch.randn_like(raw_images_test_e)
        Ax_e = sim_op_external_pats.op(x_dp_e)
        Aty_e = sim_op_external_pats.op_adj(y_dp_rand_e)
        lhs_e = torch.dot(Ax_e.flatten(), y_dp_rand_e.flatten())
        rhs_e = torch.dot(x_dp_e.flatten(), Aty_e.flatten())
        print(f"SIM External Pats Dot Test: LHS={lhs_e.item():.6f}, RHS={rhs_e.item():.6f}")
        assert np.isclose(lhs_e.item(), rhs_e.item(), rtol=1e-3, atol=1e-5), "Dot product test failed (external patterns)."
        print("SIMOperator (external patterns) dot product test passed.")

        print("\nSIMOperator __main__ checks completed for both cases.")
    except Exception as e:
        print(f"Error in SIMOperator (external patterns) checks: {e}")
        import traceback; traceback.print_exc()
