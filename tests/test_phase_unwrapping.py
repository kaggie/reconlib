import torch
import numpy as np
import unittest

from reconlib.phase_unwrapping import (
    unwrap_phase_3d_quality_guided,
    unwrap_phase_3d_least_squares,
    unwrap_phase_3d_goldstein,
    unwrap_multi_echo_masked_reference
)
from reconlib.utils import combine_coils_complex_sum # Added for new test structure

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

    def _generate_synthetic_multicoil_multiecho_data(
        self,
        spatial_shape=(8, 16, 16), # D, H, W
        num_echoes=3,
        num_coils=4,
        snr_thresh_for_mask_gen=0.2, # Used to define magnitude for mask generation
        base_ramp_scales=(1.0, 1.2, 1.5), # For spatial wraps in echo 1 (combined)
        diff_ramp_scales=(0.3, 0.4, 0.5)  # For evolving pattern in subsequent echoes (combined)
    ):
        """
        Generates synthetic multi-coil, multi-echo complex data.
        Outputs:
            - multi_coil_complex_images (num_echoes, num_coils, D, H, W)
            - true_unwrapped_coil_combined_phases (num_echoes, D, H, W)
            - true_coil_combined_magnitude_echo1 (D, H, W) - for mask verification
        """
        device = self.device
        pi = getattr(torch, 'pi', np.pi)

        # 1. Generate True Unwrapped Coil-Combined Phases and Magnitudes
        true_unwrapped_coil_combined_phases = torch.zeros((num_echoes,) + spatial_shape, device=device)
        true_coil_combined_magnitudes = torch.zeros((num_echoes,) + spatial_shape, device=device)

        # Echo 1: Phase has spatial wraps, Magnitude is structured for mask generation
        true_phase_echo1_combined, _ = _generate_synthetic_3d_phase(
            shape=spatial_shape, ramps_scale=base_ramp_scales, device=device
        )
        true_unwrapped_coil_combined_phases[0, ...] = true_phase_echo1_combined
        
        mag_echo1_combined = torch.full(spatial_shape, snr_thresh_for_mask_gen / 2, device=device) # Background
        d, h, w = spatial_shape
        slice_d, slice_h, slice_w = d//4, h//4, w//4
        mag_echo1_combined[slice_d:-slice_d, slice_h:-slice_h, slice_w:-slice_w] = snr_thresh_for_mask_gen * 2 # Foreground
        true_coil_combined_magnitudes[0, ...] = mag_echo1_combined
        true_combined_magnitude_echo1 = mag_echo1_combined # Save for mask verification

        # Subsequent echoes: Evolving phase and slightly decaying magnitude
        for i in range(1, num_echoes):
            current_diff_ramp_scales = tuple(s * (1 + 0.2*i) for s in diff_ramp_scales) 
            evolving_pattern, _ = _generate_synthetic_3d_phase(
                shape=spatial_shape, ramps_scale=current_diff_ramp_scales, device=device
            )
            evolving_pattern -= torch.mean(evolving_pattern) # Center it
            true_unwrapped_coil_combined_phases[i, ...] = true_unwrapped_coil_combined_phases[0, ...] + evolving_pattern
            true_coil_combined_magnitudes[i, ...] = true_coil_combined_magnitudes[0, ...] * (1 - 0.1 * i) # Slight decay

        # 2. Create Mock Coil Sensitivity Maps (CSMs)
        mock_csms = torch.zeros((num_coils,) + spatial_shape, dtype=torch.complex64, device=device)
        for c in range(num_coils):
            # Simple CSM: spatially varying phase and magnitude
            csm_phase_offset, _ = _generate_synthetic_3d_phase(shape=spatial_shape, ramps_scale=(0.1*c, 0.1*c, 0.1*c), device=device)
            csm_mag_profile, _ = _generate_synthetic_3d_phase(shape=spatial_shape, ramps_scale=(0.2, 0.2, 0.2), device=device)
            csm_mag_profile = (torch.cos(csm_mag_profile) + 1.5) / 2.5 # Ensure positive and varying
            mock_csms[c, ...] = csm_mag_profile * torch.exp(1j * csm_phase_offset)
        
        # Normalize CSMs (sum of squares = 1)
        rss_csms = torch.sqrt(torch.sum(torch.abs(mock_csms)**2, dim=0, keepdim=True)) + 1e-9
        mock_csms_normalized = mock_csms / rss_csms
        
        # 3. Synthesize Multi-Coil Complex Images
        multi_coil_complex_images = torch.zeros((num_echoes, num_coils) + spatial_shape, dtype=torch.complex64, device=device)
        for e in range(num_echoes):
            target_combined_complex_echo_e = true_coil_combined_magnitudes[e] * torch.exp(1j * true_unwrapped_coil_combined_phases[e])
            for c in range(num_coils):
                # Distribute combined signal to coils using CSMs
                # multi_coil = combined_image * CSM_conjugate
                # So, combined = sum(multi_coil_image_c * CSM_c) / sum(|CSM_c|^2) (if CSMs are not normalized)
                # Or, if CSMs are normalized (sum(|CSM_c|^2) = 1), then combined = sum(multi_coil_image_c * CSM_c)
                # For synthesis: multi_coil_image_c = combined_image * CSM_c (simplified, assumes CSMs are orthonormal for perfect sum)
                # A simpler synthesis: distribute magnitude equally, add phase from CSM
                # multi_coil_complex_images[e,c,...] = (true_coil_combined_magnitudes[e] / num_coils) * torch.exp(1j * (true_unwrapped_coil_combined_phases[e] + csm_phase_offset_for_coil_c))
                # Let's use the "target_combined * CSM_conj" then scale to approximate sum.
                # This is a very rough approximation for test data generation.
                multi_coil_complex_images[e, c, ...] = target_combined_complex_echo_e * mock_csms_normalized[c, ...]
                # The key is that combine_coils_complex_sum(multi_coil_complex_images[e]) should give something
                # whose phase is _wrap_phase_torch(true_unwrapped_coil_combined_phases[e])
                # and whose magnitude is close to true_coil_combined_magnitudes[e].
                # The sum of mock_csms_normalized[c, ...] * mock_csms_normalized[c, ...].conj() is 1.
                # So sum(target_combined_complex_echo_e * mock_csms_normalized[c, ...] * mock_csms_normalized[c, ...].conj())
                # = target_combined_complex_echo_e * sum(|mock_csms_normalized[c, ...}|^2) = target_combined_complex_echo_e

        return multi_coil_complex_images, true_unwrapped_coil_combined_phases, true_combined_magnitude_echo1


    def test_unwrap_multi_echo_masked_reference(self):
        num_echoes = 3
        num_coils = 4
        shape_spatial = self.test_shape # Use shape from setUp: (8,16,16)
        snr_threshold = 0.15 # Example threshold
        
        multi_coil_images, true_unwrapped_coil_combined_phases, true_combined_mag_echo1 = \
            self._generate_synthetic_multicoil_multiecho_data(
                spatial_shape=shape_spatial,
                num_echoes=num_echoes,
                num_coils=num_coils,
                snr_thresh_for_mask_gen=snr_threshold,
                base_ramp_scales=(1.0, 1.3, 1.6), # Ensure echo 1 combined phase has wraps
                diff_ramp_scales=(0.2, 0.25, 0.3) 
            )

        # Preprocessing Step: Coil Combination
        list_combined_phases = []
        list_combined_magnitudes = []
        for i in range(num_echoes):
            combined_phase, combined_mag = combine_coils_complex_sum(multi_coil_images[i])
            list_combined_phases.append(combined_phase)
            list_combined_magnitudes.append(combined_mag)
        
        input_magnitudes_for_unwrapper = torch.stack(list_combined_magnitudes, dim=0)
        input_wrapped_phases_for_unwrapper = torch.stack(list_combined_phases, dim=0)
        # input_wrapped_phases_for_unwrapper is already wrapped due to torch.angle in combine_coils_complex_sum

        def mock_spatial_unwrapper(phase_tensor, mask_tensor_arg):
            unwrapped = unwrap_phase_3d_least_squares(phase_tensor) 
            return unwrapped * mask_tensor_arg.float()

        spatial_unwrap_fn_to_test = mock_spatial_unwrapper

        unwrapped_phases_calc, generated_mask_calc = unwrap_multi_echo_masked_reference(
            input_magnitudes_for_unwrapper, 
            input_wrapped_phases_for_unwrapper, 
            snr_threshold, 
            spatial_unwrap_fn_to_test
        )

        # 1. Assert Mask Correctness
        self.assertTrue(torch.is_tensor(generated_mask_calc))
        self.assertEqual(generated_mask_calc.shape, true_combined_mag_echo1.shape) # Compare with true combined mag echo1 shape
        self.assertEqual(generated_mask_calc.dtype, torch.bool)
        # The mask is generated from input_magnitudes_for_unwrapper[0], which is derived from multi_coil_images.
        # This should be reasonably close to a mask derived from true_combined_mag_echo1 if coil combination is good.
        expected_mask = input_magnitudes_for_unwrapper[0] > torch.tensor(snr_threshold, device=self.device, dtype=input_magnitudes_for_unwrapper.dtype)
        torch.testing.assert_close(generated_mask_calc.float(), expected_mask.float(), rtol=0, atol=0)

        # 2. Assert Unwrapped Phases Correctness
        self.assertEqual(unwrapped_phases_calc.shape, true_unwrapped_coil_combined_phases.shape)
        
        for e in range(num_echoes):
            current_true_unwrapped = true_unwrapped_coil_combined_phases[e]
            current_calc_unwrapped = unwrapped_phases_calc[e]
            
            if generated_mask_calc.sum() == 0:
                print(f"Warning: Generated mask for echo {e} is all false. Skipping correctness assertion.")
                continue

            diff = current_calc_unwrapped - current_true_unwrapped
            masked_true_unwrapped = current_true_unwrapped[generated_mask_calc]
            masked_calc_unwrapped = current_calc_unwrapped[generated_mask_calc]
            masked_diff = diff[generated_mask_calc]
            
            if masked_diff.numel() > 0:
                offset_corrected_diff = masked_diff - torch.mean(masked_diff)
                mean_abs_err = torch.mean(torch.abs(offset_corrected_diff))
                tolerance = 0.1 
                if e == 0: tolerance = 1e-2 
                else: tolerance = 0.2 
                self.assertLess(mean_abs_err.item(), tolerance, 
                                f"Echo {e}: Mean absolute error {mean_abs_err.item()} (masked, offset-corrected) exceeds tolerance {tolerance}")

if __name__ == '__main__':
    unittest.main()
