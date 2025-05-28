import torch
import unittest
import numpy as np # For np.pi fallback if torch.pi not available

from reconlib.utils import combine_coils_complex_sum

class TestCombineCoils(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pi = getattr(torch, 'pi', np.pi)

    def test_combine_coils_complex_sum_basic(self):
        """Test complex sum coil combination with simple 2D spatial data."""
        coil1_data = torch.tensor([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=torch.complex64, device=self.device)
        coil2_data = torch.tensor([[1-1j, 0-0j], [1-1j, 0-0j]], dtype=torch.complex64, device=self.device)
        multi_coil_data = torch.stack([coil1_data, coil2_data], dim=0) # Shape: (2, 2, 2)

        expected_sum = torch.tensor([[2+0j, 2+2j], [4+2j, 4+4j]], dtype=torch.complex64, device=self.device)
        expected_phase = torch.angle(expected_sum)
        expected_magnitude = torch.abs(expected_sum)

        calc_phase, calc_mag = combine_coils_complex_sum(multi_coil_data)

        self.assertEqual(calc_phase.shape, expected_phase.shape)
        self.assertEqual(calc_mag.shape, expected_magnitude.shape)
        torch.testing.assert_close(calc_phase, expected_phase, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(calc_mag, expected_magnitude, rtol=1e-6, atol=1e-6)

    def test_combine_coils_complex_sum_with_mask(self):
        """Test complex sum coil combination with a mask."""
        coil1_data = torch.tensor([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=torch.complex64, device=self.device)
        coil2_data = torch.tensor([[1-1j, 0-0j], [1-1j, 0-0j]], dtype=torch.complex64, device=self.device)
        multi_coil_data = torch.stack([coil1_data, coil2_data], dim=0) 

        mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool, device=self.device)
        
        # Calculate expected sum and then apply mask
        sum_unmasked = torch.tensor([[2+0j, 2+2j], [4+2j, 4+4j]], dtype=torch.complex64, device=self.device)
        expected_sum_masked = sum_unmasked.clone()
        expected_sum_masked[~mask] = 0 + 0j
        
        expected_phase_masked = torch.angle(expected_sum_masked)
        expected_magnitude_masked = torch.abs(expected_sum_masked)

        calc_phase_masked, calc_mag_masked = combine_coils_complex_sum(multi_coil_data, mask=mask)

        self.assertEqual(calc_phase_masked.shape, expected_phase_masked.shape)
        self.assertEqual(calc_mag_masked.shape, expected_magnitude_masked.shape)
        torch.testing.assert_close(calc_phase_masked, expected_phase_masked, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(calc_mag_masked, expected_magnitude_masked, rtol=1e-6, atol=1e-6)

    def test_combine_coils_dimensions_3d_spatial(self):
        """Test complex sum with 3D spatial data (Coils, D, H, W)."""
        num_coils, D, H, W = 4, 3, 5, 6
        # Create random complex data
        real_part = torch.randn(num_coils, D, H, W, device=self.device)
        imag_part = torch.randn(num_coils, D, H, W, device=self.device)
        multi_coil_data_3d = torch.complex(real_part, imag_part)

        calc_phase_3d, calc_mag_3d = combine_coils_complex_sum(multi_coil_data_3d)

        expected_spatial_shape = (D, H, W)
        self.assertEqual(calc_phase_3d.shape, expected_spatial_shape)
        self.assertEqual(calc_mag_3d.shape, expected_spatial_shape)
        self.assertEqual(calc_phase_3d.device, self.device)
        self.assertEqual(calc_mag_3d.device, self.device)

    def test_combine_coils_input_validation(self):
        """Test input validation for combine_coils_complex_sum."""
        with self.assertRaisesRegex(TypeError, "must be a PyTorch tensor"):
            combine_coils_complex_sum(np.array([1+1j]))
        
        with self.assertRaisesRegex(ValueError, "must be a complex-valued tensor"):
            combine_coils_complex_sum(torch.randn(2,3,3, device=self.device))

        with self.assertRaisesRegex(ValueError, "must have 3 .* or 4 .* dimensions"):
            combine_coils_complex_sum(torch.complex(torch.randn(2,3), torch.randn(2,3)).to(self.device)) # 2D
        
        with self.assertRaisesRegex(ValueError, "must have 3 .* or 4 .* dimensions"):
            combine_coils_complex_sum(torch.complex(torch.randn(2,3,3,3,3), torch.randn(2,3,3,3,3)).to(self.device)) # 5D

        # Mask validation
        dummy_data = torch.complex(torch.randn(2,3,3), torch.randn(2,3,3)).to(self.device)
        with self.assertRaisesRegex(TypeError, "mask must be a PyTorch tensor"):
            combine_coils_complex_sum(dummy_data, mask=np.array([True]))
        
        with self.assertRaisesRegex(TypeError, "mask must be a boolean tensor"):
            combine_coils_complex_sum(dummy_data, mask=torch.randn(3,3).to(self.device))

        with self.assertRaisesRegex(ValueError, "Mask shape .* must match input data spatial shape"):
            combine_coils_complex_sum(dummy_data, mask=torch.tensor([True, False]).to(self.device))


if __name__ == '__main__':
    unittest.main()
