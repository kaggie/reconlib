import torch
import numpy as np
import unittest

from reconlib.b0_mapping import (
    calculate_b0_map_dual_echo,
    calculate_b0_map_multi_echo_linear_fit
)
from reconlib.phase_unwrapping import unwrap_phase_3d_quality_guided

# Helper to wrap phase consistently for test assertions if needed
def _wrap_phase_torch(phase_tensor: torch.Tensor) -> torch.Tensor:
    """Wraps phase values to the interval [-pi, pi) using PyTorch operations."""
    pi = getattr(torch, 'pi', np.pi)
    return (phase_tensor + pi) % (2 * pi) - pi

def _create_spatial_wrap_pattern(shape_spatial, device, max_val_factor=1.5):
    """
    Creates a 3D spatial pattern that can induce phase wrapping.
    The pattern is a sum of linear ramps along each spatial dimension.
    max_val_factor determines how many times pi the ramp reaches.
    """
    pi = getattr(torch, 'pi', np.pi)
    max_val = max_val_factor * pi
    
    dim_ramps = []
    for i, dim_size in enumerate(shape_spatial):
        ramp_1d = torch.linspace(0, max_val, dim_size, device=device)
        view_shape = [1] * len(shape_spatial)
        view_shape[i] = dim_size
        dim_ramps.append(ramp_1d.view(view_shape))
    
    pattern = torch.zeros(shape_spatial, device=device)
    for r in dim_ramps:
        pattern += r # Summing ramps from each dimension
    return pattern

def _generate_synthetic_b0_data(shape=(8, 16, 16), tes_list=[0.002, 0.004, 0.006], max_b0_hz=30.0, device='cpu', apply_mask_to_b0=True, add_spatial_wraps_to_echoes=False, spatial_wrap_factor=1.5):
    """
    Generates synthetic 3D multi-echo phase data and the true B0 map.
    """
    d, h, w = shape
    
    # Create a simple 3D B0 map: linear gradient along x, scaled by z
    b0_map_true = torch.zeros(shape, dtype=torch.float32, device=device)
    x_ramp = torch.linspace(-max_b0_hz, max_b0_hz, w, device=device)
    for z_idx in range(d):
        z_scale = (z_idx + 1) / d # Scale gradient by z slice
        b0_map_true[z_idx, :, :] = x_ramp.view(1, -1) * z_scale
        
    if apply_mask_to_b0:
        # Create a simple cylindrical mask to make it more realistic (B0 only exists within object)
        magnitude_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        cylinder_mask_2d = ((y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2)
        for z_idx in range(d):
            magnitude_mask[z_idx, :, :] = cylinder_mask_2d
        b0_map_true[~magnitude_mask] = 0.0 # Apply B0 only within the object
    
    echo_times_torch = torch.tensor(tes_list, dtype=torch.float32, device=device)
    num_echoes = len(tes_list)
    
    phase_images_torch = torch.zeros((num_echoes,) + shape, dtype=torch.float32, device=device)
    pi = getattr(torch, 'pi', np.pi)
    
    spatial_wrapping_field = None
    if add_spatial_wraps_to_echoes:
        spatial_wrapping_field = _create_spatial_wrap_pattern(shape, device, max_val_factor=spatial_wrap_factor)

    for i in range(num_echoes):
        base_phase = 2 * pi * b0_map_true * echo_times_torch[i]
        if add_spatial_wraps_to_echoes and spatial_wrapping_field is not None:
            # Add spatial wraps and then re-wrap the result
            phase_images_torch[i, ...] = _wrap_phase_torch(base_phase + spatial_wrapping_field)
        else:
            # Store the base phase (which might be > pi or < -pi due to B0*TE)
            # The dual_echo function's unwrap_method_fn handles the *difference* map.
            # The multi_echo's spatial_unwrap_fn handles each echo.
            # If no unwrapper, multi-echo expects "smooth" phase, dual-echo difference is used raw.
            # For simplicity, we store the raw calculated phase, which might implicitly wrap if B0*TE is large enough.
            # Let's ensure input to functions is explicitly wrapped to test unwrappers.
            phase_images_torch[i, ...] = _wrap_phase_torch(base_phase) 
            
    return phase_images_torch, echo_times_torch, b0_map_true

class TestB0MappingPyTorch(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_shape_3d = (4, 8, 10) # Smaller 3D shape for faster tests: D, H, W
        self.tes_dual_echo = [0.0025, 0.0050] # For dual-echo, delta_TE = 0.0025s
        self.tes_multi_echo = [0.0020, 0.0040, 0.0060, 0.0080] # For multi-echo
        
        self.low_b0_max_hz = 40.0  # delta_TE * B0_max = 0.0025 * 40 = 0.1. Phase diff = 2*pi*0.1 (no wrap for diff)
        self.high_b0_max_hz = 220.0 # delta_TE * B0_max = 0.0025 * 220 = 0.55. Phase diff = 2*pi*0.55 (wraps for diff)

        # Data for low B0 (no wrapping in phase_diff for dual echo)
        self.phases_low_b0, self.tes_low_b0, self.b0_true_low_b0 = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_dual_echo, max_b0_hz=self.low_b0_max_hz, device=self.device, add_spatial_wraps_to_echoes=False
        )
        
        # Data for high B0 (wrapping in phase_diff for dual echo)
        self.phases_high_b0, self.tes_high_b0, self.b0_true_high_b0 = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_dual_echo, max_b0_hz=self.high_b0_max_hz, device=self.device, add_spatial_wraps_to_echoes=False
        )

        # Data for multi-echo fit (spatially smooth echoes, low B0 for true field)
        self.phases_multi_smooth, self.tes_multi_smooth, self.b0_true_multi_smooth = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_multi_echo, max_b0_hz=self.low_b0_max_hz, device=self.device, add_spatial_wraps_to_echoes=False
        )
        
        # Data for multi-echo fit with SPATIALLY WRAPPED individual echoes (low B0 for true field)
        self.phases_spatially_wrapped, self.tes_spatially_wrapped, self.b0_true_spatially_wrapped = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_multi_echo, max_b0_hz=self.low_b0_max_hz, device=self.device, add_spatial_wraps_to_echoes=True, spatial_wrap_factor=2.0 # Ensure >1pi wraps
        )

        # Generic mask (all true for these tests, assuming B0 is non-zero everywhere in test data after helper)
        self.mask_all_true = torch.ones(self.test_shape_3d, dtype=torch.bool, device=self.device)

    def _assert_b0_map_correctness(self, b0_calculated, b0_true, mask, tolerance_hz=0.5, msg_prefix=""):
        self.assertEqual(b0_calculated.shape, b0_true.shape, f"{msg_prefix}Shape mismatch.")
        self.assertTrue(torch.is_tensor(b0_calculated), f"{msg_prefix}Output is not a tensor.")
        self.assertEqual(b0_calculated.device, b0_true.device, f"{msg_prefix}Device mismatch.")

        diff = (b0_calculated - b0_true)
        # Apply mask to diff before calculating mean error
        if mask is not None:
            diff_masked = diff[mask]
            if diff_masked.numel() == 0: # Avoid error if mask is all false
                 mean_abs_error = torch.tensor(0.0, device=diff.device)
            else:
                 mean_abs_error = torch.mean(torch.abs(diff_masked))
        else:
            mean_abs_error = torch.mean(torch.abs(diff))
        
        self.assertLess(mean_abs_error.item(), tolerance_hz,
                        f"{msg_prefix}Mean absolute error {mean_abs_error.item():.4f} Hz exceeds tolerance {tolerance_hz:.4f} Hz.")

    def test_dual_echo_no_unwrapping_low_b0(self):
        """Test dual-echo with low B0 (no phase_diff wrapping), no unwrapper function."""
        b0_calc = calculate_b0_map_dual_echo(
            self.phases_low_b0, 
            self.tes_low_b0, 
            mask=self.mask_all_true,
            unwrap_method_fn=None 
        )
        self._assert_b0_map_correctness(b0_calc, self.b0_true_low_b0, self.mask_all_true, 
                                        tolerance_hz=1e-2, msg_prefix="Low B0, No Unwrap: ")

    def test_dual_echo_no_unwrapping_high_b0_aliased(self):
        """Test dual-echo with high B0 (phase_diff wraps), no unwrapper. Expect aliased result."""
        b0_calc_aliased = calculate_b0_map_dual_echo(
            self.phases_high_b0, 
            self.tes_high_b0, 
            mask=self.mask_all_true,
            unwrap_method_fn=None
        )
        # The error should be large when compared to the true high B0 map
        # because the result is aliased.
        with self.assertRaises(AssertionError, msg="Error should be high for aliased B0 map compared to true B0."):
            self._assert_b0_map_correctness(b0_calc_aliased, self.b0_true_high_b0, self.mask_all_true, 
                                            tolerance_hz=10.0, msg_prefix="High B0, No Unwrap (Aliased): ") 
                                            # Using a loose tolerance that it should still fail.
        
        # Optional: Check if it's close to an expected *aliased* B0 map.
        # This requires calculating the expected aliased map.
        # True phase diff: true_phase_diff = 2 * pi * self.b0_true_high_b0 * (self.tes_high_b0[1] - self.tes_high_b0[0])
        # Wrapped phase diff: wrapped_true_phase_diff = _wrap_phase_torch(true_phase_diff)
        # Expected aliased B0: expected_aliased_b0 = wrapped_true_phase_diff / (2 * pi * (self.tes_high_b0[1] - self.tes_high_b0[0]))
        # self._assert_b0_map_correctness(b0_calc_aliased, expected_aliased_b0, self.mask_all_true, 
        #                                 tolerance_hz=1e-2, msg_prefix="High B0, No Unwrap (vs Expected Alias): ")


    def test_dual_echo_with_quality_guided_unwrapping_high_b0(self):
        """Test dual-echo with high B0, using quality-guided unwrapper."""
        b0_calc_unwrapped = calculate_b0_map_dual_echo(
            self.phases_high_b0, 
            self.tes_high_b0, 
            mask=self.mask_all_true,
            unwrap_method_fn=unwrap_phase_3d_quality_guided # Pass the unwrapping function
        )
        # Quality-guided unwrapper might not be perfect, tolerance is a bit higher.
        self._assert_b0_map_correctness(b0_calc_unwrapped, self.b0_true_high_b0, self.mask_all_true, 
                                        tolerance_hz=5.0, msg_prefix="High B0, QualityGuided Unwrap: ")


    def test_dual_echo_with_quality_guided_unwrapping_low_b0(self):
        """Test dual-echo with low B0, using quality-guided unwrapper (should not harm)."""
        b0_calc_unwrapped = calculate_b0_map_dual_echo(
            self.phases_low_b0, 
            self.tes_low_b0, 
            mask=self.mask_all_true,
            unwrap_method_fn=unwrap_phase_3d_quality_guided 
        )
        self._assert_b0_map_correctness(b0_calc_unwrapped, self.b0_true_low_b0, self.mask_all_true, 
                                        tolerance_hz=1e-2, msg_prefix="Low B0, QualityGuided Unwrap: ")


    def test_calculate_b0_map_multi_echo_linear_fit_pytorch(self):
        b0_calc = calculate_b0_map_multi_echo_linear_fit(
            self.phases_multi, 
            self.tes_multi, 
            mask=self.mask
        )
        # Linear fit should be accurate for this clean, linear synthetic data.
        self._assert_b0_map_correctness(b0_calc, self.b0_true_multi, self.mask, tolerance_hz=1e-3)

if __name__ == '__main__':
    unittest.main()
