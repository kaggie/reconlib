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

def _generate_synthetic_b0_data(shape=(8, 16, 16), tes_list=[0.002, 0.004, 0.006], max_b0_hz=30.0, device='cpu', apply_mask_to_b0=True):
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
    
    for i in range(num_echoes):
        phase_images_torch[i, ...] = 2 * pi * b0_map_true * echo_times_torch[i]
        # Phase images are directly calculated, so they are "intrinsically unwrapped" 
        # relative to the B0*TE product. Wrapping occurs if this product itself is large.
        
    return phase_images_torch, echo_times_torch, b0_map_true

class TestB0MappingPyTorch(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_shape_3d = (4, 8, 10) # Smaller 3D shape for faster tests: D, H, W
        self.tes_dual_echo = [0.0025, 0.0050] # For dual-echo, delta_TE = 0.0025s
        self.tes_multi_echo = [0.0020, 0.0040, 0.0060, 0.0080] # For multi-echo
        
        self.low_b0_max_hz = 40.0  # delta_TE * B0_max = 0.0025 * 40 = 0.1. Phase diff = 2*pi*0.1 (no wrap)
        self.high_b0_max_hz = 220.0 # delta_TE * B0_max = 0.0025 * 220 = 0.55. Phase diff = 2*pi*0.55 (wraps)

        # Data for low B0 (no wrapping in phase_diff for dual echo)
        self.phases_low_b0, self.tes_low_b0, self.b0_true_low_b0 = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_dual_echo, max_b0_hz=self.low_b0_max_hz, device=self.device
        )
        
        # Data for high B0 (wrapping in phase_diff for dual echo)
        self.phases_high_b0, self.tes_high_b0, self.b0_true_high_b0 = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_dual_echo, max_b0_hz=self.high_b0_max_hz, device=self.device
        )

        # Data for multi-echo fit (can use low or high B0, let's use low for simplicity)
        self.phases_multi, self.tes_multi, self.b0_true_multi = _generate_synthetic_b0_data(
            shape=self.test_shape_3d, tes_list=self.tes_multi_echo, max_b0_hz=self.low_b0_max_hz, device=self.device
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
