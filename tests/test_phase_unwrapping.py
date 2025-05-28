import torch
import numpy as np
import unittest

from reconlib.phase_unwrapping import (
    unwrap_phase_3d_quality_guided,
    unwrap_phase_3d_least_squares,
    unwrap_phase_3d_goldstein
)

# Helper to wrap phase consistently in tests
def _wrap_phase_torch(phase_tensor: torch.Tensor) -> torch.Tensor:
    """Wraps phase values to the interval [-pi, pi) using PyTorch operations."""
    pi = getattr(torch, 'pi', np.pi)
    return (phase_tensor + pi) % (2 * pi) - pi

def _generate_synthetic_3d_phase(shape=(16, 32, 32), ramps_scale=(1.0, 1.5, 2.0), device='cpu'):
    """
    Generates synthetic 3D true and wrapped phase data.
    Creates a sum of 3D linear ramps.
    """
    d, h, w = shape
    z_coords = torch.linspace(-np.pi * ramps_scale[0], np.pi * ramps_scale[0], d, device=device)
    y_coords = torch.linspace(-np.pi * ramps_scale[1], np.pi * ramps_scale[1], h, device=device)
    x_coords = torch.linspace(-np.pi * ramps_scale[2], np.pi * ramps_scale[2], w, device=device)

    true_phase = z_coords.view(-1, 1, 1) + y_coords.view(1, -1, 1) + x_coords.view(1, 1, -1)
    
    # Ensure true_phase has the exact shape requested, in case linspace doesn't match perfectly
    true_phase = true_phase.expand(d,h,w) 

    wrapped_phase = _wrap_phase_torch(true_phase)
    return true_phase, wrapped_phase

class TestPhaseUnwrapping3D(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_shape = (8, 16, 16) # Smaller shape for faster tests
        self.true_phase, self.wrapped_phase = _generate_synthetic_3d_phase(
            shape=self.test_shape, 
            ramps_scale=(1.0, 1.2, 1.5), # Scales chosen to ensure multiple wraps
            device=self.device
        )

    def _assert_unwrapping_correctness(self, unwrapped_phase, true_phase, tolerance=0.1):
        self.assertEqual(unwrapped_phase.shape, true_phase.shape)
        self.assertTrue(torch.is_tensor(unwrapped_phase))
        self.assertEqual(unwrapped_phase.device, true_phase.device)

        # Difference after accounting for constant offset
        diff = unwrapped_phase - true_phase
        diff_centered = diff - torch.mean(diff)
        
        mean_abs_error = torch.mean(torch.abs(diff_centered))
        self.assertLess(mean_abs_error.item(), tolerance, 
                        f"Mean absolute error {mean_abs_error.item()} exceeds tolerance {tolerance}")

    def test_quality_guided_unwrap_3d(self):
        unwrapped = unwrap_phase_3d_quality_guided(self.wrapped_phase, sigma_blur=0.5)
        # Quality guided might be less accurate on pure ramps than LS, adjust tolerance.
        # It's also sensitive to start point.
        self._assert_unwrapping_correctness(unwrapped, self.true_phase, tolerance=0.2) 

    def test_least_squares_unwrap_3d(self):
        unwrapped = unwrap_phase_3d_least_squares(self.wrapped_phase)
        # Least squares should be quite accurate for this type of data (global solution)
        self._assert_unwrapping_correctness(unwrapped, self.true_phase, tolerance=1e-3)

    def test_goldstein_unwrap_3d(self):
        unwrapped = unwrap_phase_3d_goldstein(self.wrapped_phase, k_filter_strength=1.0)
        # Goldstein's method precision can vary, tolerance might need adjustment.
        # It's a filtering method, so global offset might be less consistent.
        self._assert_unwrapping_correctness(unwrapped, self.true_phase, tolerance=0.15)

    def test_goldstein_unwrap_3d_no_filter(self):
        # With k_filter_strength=0, it should ideally return something close to original wrapped phase's angle
        # after FFT/IFFT, but not necessarily perfectly unwrapped.
        # This test mainly checks if the path for k_filter_strength=0 works.
        unwrapped = unwrap_phase_3d_goldstein(self.wrapped_phase, k_filter_strength=0.0)
        self.assertEqual(unwrapped.shape, self.wrapped_phase.shape)
        self.assertTrue(torch.is_tensor(unwrapped))
        # Cannot assert unwrapping correctness here as it's not expected to unwrap with filter_strength=0.

if __name__ == '__main__':
    unittest.main()
