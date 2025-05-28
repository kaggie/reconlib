import torch
import unittest
import numpy as np # For np.pi fallback

from reconlib.pipeline_utils import preprocess_multi_coil_multi_echo_data
from reconlib.phase_unwrapping import unwrap_phase_3d_quality_guided # A real unwrapper for more thorough test
from reconlib.utils import combine_coils_complex_sum # For ground truth calculation
from reconlib.phase_unwrapping.reference_echo_unwrap import _wrap_to_pi # For ground truth calculation

class TestPipelineUtils(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pi = getattr(torch, 'pi', np.pi)

        # A mock spatial unwrapper that does minimal change, good for testing pipeline structure
        def mock_spatial_unwrapper_fn(phase_tensor, mask_tensor):
            # This mock should return the phase tensor, potentially masked.
            # For simplicity, we assume it handles the mask internally or returns unmasked.
            # If it must return masked, then: return phase_tensor * mask_tensor.float()
            return phase_tensor 
        self.mock_spatial_unwrapper = mock_spatial_unwrapper_fn

        # A real spatial unwrapper for more integration-style testing
        def real_spatial_unwrapper_fn(phase_tensor, mask_tensor):
            # unwrap_phase_3d_quality_guided doesn't take a mask in its main signature.
            # The `unwrap_multi_echo_masked_reference` applies the mask to the *output* of its
            # internal unwrapping calls if the unwrapper itself doesn't handle it.
            # Here, we simulate an unwrapper that might be used.
            # The mask is passed by `unwrap_multi_echo_masked_reference` to this function.
            # We can choose to use it or ignore it if the unwrapper (like quality_guided) doesn't accept it.
            # For this test, we'll call quality_guided and then apply the mask to its output,
            # mimicking how it might be used or how `unwrap_multi_echo_masked_reference` ensures masking.
            unwrapped = unwrap_phase_3d_quality_guided(phase_tensor, sigma_blur=0.5) 
            return unwrapped * mask_tensor.float() # Ensure output respects the mask
            
        self.real_spatial_unwrapper_for_test = real_spatial_unwrapper_fn


    def test_preprocess_shapes_types_3d_spatial(self):
        """Test output shapes and types for 3D spatial data."""
        num_echoes, num_coils, D, H, W = 2, 2, 4, 8, 10
        
        # Create random complex data
        real_part = torch.randn(num_echoes, num_coils, D, H, W, device=self.device)
        imag_part = torch.randn(num_echoes, num_coils, D, H, W, device=self.device)
        test_data = torch.complex(real_part, imag_part)
        snr_threshold = 0.1

        unwrapped_phases, mask, combined_magnitudes = preprocess_multi_coil_multi_echo_data(
            test_data, snr_threshold, self.mock_spatial_unwrapper
        )

        expected_spatial_shape = (D, H, W)
        expected_echo_spatial_shape = (num_echoes,) + expected_spatial_shape

        self.assertEqual(unwrapped_phases.shape, expected_echo_spatial_shape)
        self.assertEqual(mask.shape, expected_spatial_shape)
        self.assertEqual(combined_magnitudes.shape, expected_echo_spatial_shape)

        self.assertEqual(unwrapped_phases.dtype, torch.float32) # or specific float type of input
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(combined_magnitudes.dtype, torch.float32) # abs is float

        self.assertEqual(unwrapped_phases.device, self.device)
        self.assertEqual(mask.device, self.device)
        self.assertEqual(combined_magnitudes.device, self.device)
        
    def test_preprocess_shapes_types_2d_spatial(self):
        """Test output shapes and types for 2D spatial data."""
        num_echoes, num_coils, H, W = 2, 2, 8, 10
        
        real_part = torch.randn(num_echoes, num_coils, H, W, device=self.device)
        imag_part = torch.randn(num_echoes, num_coils, H, W, device=self.device)
        test_data = torch.complex(real_part, imag_part)
        snr_threshold = 0.1

        unwrapped_phases, mask, combined_magnitudes = preprocess_multi_coil_multi_echo_data(
            test_data, snr_threshold, self.mock_spatial_unwrapper
        )

        expected_spatial_shape = (H, W)
        expected_echo_spatial_shape = (num_echoes,) + expected_spatial_shape

        self.assertEqual(unwrapped_phases.shape, expected_echo_spatial_shape)
        self.assertEqual(mask.shape, expected_spatial_shape)
        self.assertEqual(combined_magnitudes.shape, expected_echo_spatial_shape)
        self.assertEqual(unwrapped_phases.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(combined_magnitudes.dtype, torch.float32)


    def test_preprocess_end_to_end_simple_case(self):
        """Test the pipeline orchestration with a simple, predictable case."""
        num_echoes, num_coils, D, H, W = 2, 1, 2, 2, 2 # Very small data
        snr_threshold = 0.05 

        # Create simple, predictable multi-coil data
        # Echo 1: Coil 1 has basic phase, Mag allows simple mask
        # Echo 2: Coil 1 phase is Echo 1 phase + small increment
        
        # True underlying combined data (what we want after coil sum and unwrapping)
        true_mag_echo1 = torch.tensor([[[0.0, 0.0], [0.0, 0.2]],  # D0
                                       [[0.2, 0.2], [0.2, 0.2]]], # D1
                                      dtype=torch.float32, device=self.device) 
        true_mag_echo2 = true_mag_echo1 * 0.9

        true_phase_echo1_unwrapped = torch.tensor([[[0.0, 0.0], [0.0, 0.1*self.pi]], # D0
                                                   [[0.2*self.pi, 0.3*self.pi], [0.4*self.pi, 0.5*self.pi]]], # D1
                                                  dtype=torch.float32, device=self.device)
        
        # Phase difference between echo2 and echo1 for unwrapped is a small constant 0.6*pi
        # This difference, when wrapped, will be 0.6*pi.
        # If spatial_unwrap_fn is mock, unwrapped_diff = wrapped_diff = 0.6*pi
        # So, true_phase_echo2_unwrapped = true_phase_echo1_unwrapped + 0.6*pi
        true_phase_echo2_unwrapped = true_phase_echo1_unwrapped + 0.6 * self.pi
        
        # Create multi-coil data that would sum to this (single coil for simplicity)
        mc_data_echo1 = (true_mag_echo1 * torch.exp(1j * _wrap_to_pi(true_phase_echo1_unwrapped))).unsqueeze(0) # Add coil dim
        mc_data_echo2 = (true_mag_echo2 * torch.exp(1j * _wrap_to_pi(true_phase_echo2_unwrapped))).unsqueeze(0) # Add coil dim
        
        multi_coil_multi_echo_input = torch.stack([mc_data_echo1, mc_data_echo2], dim=0) # Add echo dim
        
        # Expected outputs
        expected_unwrapped_phases = torch.stack([true_phase_echo1_unwrapped, true_phase_echo2_unwrapped], dim=0)
        expected_magnitudes = torch.stack([true_mag_echo1, true_mag_echo2], dim=0)
        expected_mask = true_mag_echo1 > snr_threshold

        # Run pipeline
        calc_unwrapped_phases, calc_mask, calc_magnitudes = preprocess_multi_coil_multi_echo_data(
            multi_coil_multi_echo_input, 
            snr_threshold, 
            self.mock_spatial_unwrapper # Mock unwrapper returns phase as is
        )
        
        # Assertions
        torch.testing.assert_close(calc_magnitudes, expected_magnitudes, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(calc_mask, expected_mask)
        
        # For phase, check within the mask and allow for mean offset
        for e in range(num_echoes):
            true_ph_masked = expected_unwrapped_phases[e][expected_mask]
            calc_ph_masked = calc_unwrapped_phases[e][expected_mask]
            
            if true_ph_masked.numel() > 0: # Avoid issues if mask is empty
                # Correct for potential global phase offset from unwrapping steps
                offset = torch.mean(calc_ph_masked - true_ph_masked)
                torch.testing.assert_close(calc_ph_masked - offset, true_ph_masked, rtol=1e-4, atol=1e-4, 
                                           msg=f"Phase mismatch for echo {e}")


if __name__ == '__main__':
    unittest.main()
