import torch
import numpy as np
import unittest

from reconlib.b0_mapping import (
    calculate_b0_map_dual_echo,
    calculate_b0_map_multi_echo_linear_fit
)

def _generate_synthetic_b0_data(shape=(8, 16, 16), tes_list=[0.002, 0.004, 0.006], max_b0_hz=30.0, device='cpu'):
    """
    Generates synthetic 3D multi-echo phase data and the true B0 map.
    """
    d, h, w = shape
    
    # Create a simple 3D B0 map: linear gradient along x, scaled by z
    b0_map_true_np = np.zeros(shape, dtype=np.float32)
    x_ramp = np.linspace(-max_b0_hz, max_b0_hz, w)
    for z_idx in range(d):
        z_scale = (z_idx + 1) / d # Scale gradient by z slice
        b0_map_true_np[z_idx, :, :] = x_ramp.reshape(1, -1) * z_scale
        
    b0_map_true_torch = torch.from_numpy(b0_map_true_np).to(device)
    
    echo_times_torch = torch.tensor(tes_list, dtype=torch.float32, device=device)
    num_echoes = len(tes_list)
    
    phase_images_torch = torch.zeros((num_echoes,) + shape, dtype=torch.float32, device=device)
    pi = getattr(torch, 'pi', np.pi)
    
    for i in range(num_echoes):
        phase_images_torch[i, ...] = 2 * pi * b0_map_true_torch * echo_times_torch[i]
        # Phase wrapping is not explicitly applied here, as B0 functions handle it or assume unwrapped.
        # The dual_echo function handles wrapped phase differences.
        # The multi_echo_linear_fit ideally expects unwrapped phase; for this synthetic data without noise,
        # the direct linear relationship should hold.
        
    return phase_images_torch, echo_times_torch, b0_map_true_torch

class TestB0MappingPyTorch(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_shape = (8, 12, 16) # Smaller shape for faster tests
        self.tes_list_dual = [0.0025, 0.0050] # For dual-echo
        self.tes_list_multi = [0.0020, 0.0040, 0.0060, 0.0080] # For multi-echo
        self.max_b0 = 40.0

        self.phases_dual, self.tes_dual, self.b0_true_dual = _generate_synthetic_b0_data(
            shape=self.test_shape, tes_list=self.tes_list_dual, max_b0_hz=self.max_b0, device=self.device
        )
        self.phases_multi, self.tes_multi, self.b0_true_multi = _generate_synthetic_b0_data(
            shape=self.test_shape, tes_list=self.tes_list_multi, max_b0_hz=self.max_b0, device=self.device
        )
        
        # Create a simple mask (e.g., non-zero B0 regions, or all ones for simplicity)
        # For these tests, an all-true mask is fine as data is clean.
        self.mask = torch.ones(self.test_shape, dtype=torch.bool, device=self.device)


    def _assert_b0_map_correctness(self, b0_calculated, b0_true, mask, tolerance_hz=0.5):
        self.assertEqual(b0_calculated.shape, b0_true.shape)
        self.assertTrue(torch.is_tensor(b0_calculated))
        self.assertEqual(b0_calculated.device, b0_true.device)

        # Calculate error only within the mask
        diff = (b0_calculated - b0_true)[mask]
        mean_abs_error = torch.mean(torch.abs(diff))
        
        self.assertLess(mean_abs_error.item(), tolerance_hz,
                        f"Mean absolute error {mean_abs_error.item()} Hz exceeds tolerance {tolerance_hz} Hz")

    def test_calculate_b0_map_dual_echo_pytorch(self):
        # Dual echo uses first two echoes from the 'dual' dataset
        b0_calc = calculate_b0_map_dual_echo(
            self.phases_dual.narrow(0, 0, 2), 
            self.tes_dual.narrow(0, 0, 2), 
            mask=self.mask
        )
        # Dual echo can be sensitive to phase wrapping and noise. Tolerance might need adjustment.
        self._assert_b0_map_correctness(b0_calc, self.b0_true_dual, self.mask, tolerance_hz=1e-2)

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
