import unittest
import numpy as np
import pywt # For wavelet preprocessing tests
from scipy.signal import find_peaks

# Modules to test
from reconlibs.modality.epr.reconstruction import (
    preprocess_cw_epr_data,
    _baseline_als, # For direct testing if needed, though usually tested via preprocess
    gaussian_lineshape,
    lorentzian_lineshape,
    voigt_lineshape,
    ARTReconstructor
)

# Global parameters for tests if needed, e.g., for ART
DEFAULT_GRID_SIZE = (10, 10)
DEFAULT_ANGLES_DEG = np.array([0, 45, 90])


class TestEPRPreprocessing(unittest.TestCase):
    def test_als_baseline_correction(self):
        x = np.linspace(-10, 10, 200)
        true_peak = gaussian_lineshape(x, center=0, fwhm=2)
        baseline = 0.5 + 0.1 * x
        spectrum_with_baseline = true_peak + baseline

        projection_data = spectrum_with_baseline.reshape(1, -1) # Needs 2D array

        params_als = {
            'baseline_correct_method': 'als',
            'als_lambda': 1e6,
            'als_p_asymmetry': 0.01,
            'als_niter': 10
        }

        corrected_data_als = preprocess_cw_epr_data(projection_data.copy(), params_als)

        # After correction, the mean should be closer to the mean of the true peak
        # or the baseline component should be significantly reduced.
        # A simple check: mean of the difference between corrected and true peak should be small.
        self.assertTrue(np.mean(np.abs(corrected_data_als[0] - true_peak)) < 0.1,
                        "ALS corrected data far from true peak")

        # Check that the corrected baseline (original - corrected data) is close to the known baseline
        estimated_baseline = projection_data[0] - corrected_data_als[0]
        np.testing.assert_allclose(estimated_baseline, baseline, atol=0.2,
                                   err_msg="ALS estimated baseline not close to true baseline")


    def test_wavelet_denoising(self):
        x = np.linspace(-5, 5, 256) # Length good for wavelets
        true_signal = gaussian_lineshape(x, center=0, fwhm=1.5)
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, len(x))
        noisy_signal = true_signal + noise

        projection_data = noisy_signal.reshape(1, -1)

        params_wavelet = {
            'denoise_method': 'wavelet',
            'wavelet_type': 'db4',
            'wavelet_level': 4,
            'wavelet_threshold_sigma_multiplier': 3
        }

        denoised_data = preprocess_cw_epr_data(projection_data.copy(), params_wavelet)

        original_rmse = np.sqrt(np.mean((noisy_signal - true_signal)**2))
        denoised_rmse = np.sqrt(np.mean((denoised_data[0] - true_signal)**2))

        self.assertTrue(denoised_rmse < original_rmse,
                        f"Wavelet denoising did not reduce RMSE: original {original_rmse}, denoised {denoised_rmse}")
        # Check if std dev of residual is smaller
        self.assertTrue(np.std(denoised_data[0] - true_signal) < noise_std,
                        "Wavelet denoising did not reduce noise standard deviation effectively.")


    def test_spectral_alignment_cross_correlation(self):
        x = np.arange(100)
        ref_signal = gaussian_lineshape(x, center=50, fwhm=10)
        shift = 5
        shifted_signal = gaussian_lineshape(x, center=50 + shift, fwhm=10)

        projection_data = np.vstack([ref_signal, shifted_signal])

        params_align_corr = {
            'align_spectra': True,
            'reference_projection_index': 0,
            'align_peak_prominence': None # Ensure cross-correlation is used
        }

        aligned_data = preprocess_cw_epr_data(projection_data.copy(), params_align_corr)

        # After alignment, the shifted spectrum should be very close to the reference
        np.testing.assert_allclose(aligned_data[1, :], ref_signal, atol=0.1,
                                   err_msg="Cross-correlation alignment failed to align shifted signal to reference.")
        # Check peak positions
        ref_peak_idx = np.argmax(aligned_data[0,:])
        aligned_peak_idx = np.argmax(aligned_data[1,:])
        self.assertAlmostEqual(ref_peak_idx, aligned_peak_idx, delta=1,
                               msg="Peak positions not aligned after cross-correlation alignment.")


    def test_spectral_alignment_peak_based(self):
        x = np.arange(100)
        ref_signal = gaussian_lineshape(x, center=40, fwhm=5) + 0.5 * gaussian_lineshape(x, center=60, fwhm=3) # Multi-peak
        shift = 7
        shifted_signal = gaussian_lineshape(x, center=40 + shift, fwhm=5) + 0.5 * gaussian_lineshape(x, center=60 + shift, fwhm=3)

        projection_data = np.vstack([ref_signal, shifted_signal])

        params_align_peak = {
            'align_spectra': True,
            'reference_projection_index': 0,
            'align_peak_prominence': 0.4 # Prominence to pick the first, larger peak
        }

        aligned_data = preprocess_cw_epr_data(projection_data.copy(), params_align_peak)

        # Check peak positions of the first major peak
        ref_peaks, _ = find_peaks(aligned_data[0,:], prominence=0.4)
        aligned_peaks, _ = find_peaks(aligned_data[1,:], prominence=0.4)

        self.assertTrue(len(ref_peaks) > 0, "No peak found in reference for peak-based alignment test.")
        self.assertTrue(len(aligned_peaks) > 0, "No peak found in aligned signal for peak-based alignment test.")

        # Assuming the most prominent peak is the one we targeted (center=40)
        self.assertAlmostEqual(ref_peaks[0], aligned_peaks[0], delta=1,
                               msg="Peak positions not aligned after peak-based alignment.")
        np.testing.assert_allclose(aligned_data[1, :], ref_signal, atol=0.15, # May not be perfect due to edge effects of roll
                                   err_msg="Peak-based alignment failed to align shifted signal to reference.")


    def test_normalization_methods(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0], # Zero row
                         [-1.0, -2.0, 1.0, 2.0, 0.0]])
                         # Mixed positive/negative

        # Test 'max' normalization
        params_max = {'normalize_method': 'max'}
        norm_max = preprocess_cw_epr_data(data.copy(), params_max)
        np.testing.assert_allclose(norm_max[0, :], data[0, :]/5.0)
        np.testing.assert_allclose(norm_max[1, :], data[1, :]) # Zero row should remain zero
        np.testing.assert_allclose(norm_max[2, :], data[2, :]/2.0) # Max is 2.0

        # Test 'area' normalization
        params_area = {'normalize_method': 'area'}
        norm_area = preprocess_cw_epr_data(data.copy(), params_area)
        np.testing.assert_allclose(norm_area[0, :], data[0, :]/np.sum(np.abs(data[0,:])))
        np.testing.assert_allclose(norm_area[1, :], data[1, :])
        np.testing.assert_allclose(norm_area[2, :], data[2, :]/np.sum(np.abs(data[2,:])))
        for i in range(norm_area.shape[0]):
            if not np.all(data[i,:] == 0): # skip zero rows for this check
                 self.assertAlmostEqual(np.sum(np.abs(norm_area[i,:])), 1.0,
                                        msg=f"Area normalization failed for row {i}")

        # Test 'intensity_sum' normalization
        params_sum = {'normalize_method': 'intensity_sum'}
        norm_sum = preprocess_cw_epr_data(data.copy(), params_sum)
        np.testing.assert_allclose(norm_sum[0, :], data[0, :]/np.sum(data[0,:]))
        np.testing.assert_allclose(norm_sum[1, :], data[1, :])
        # For row 2, sum is 0. So it should remain unchanged by current logic (division by zero protection)
        # Or result might be all zeros if that's how it's handled.
        # Current preprocess_cw_epr_data has `if row_sum != 0:`. So it should be unchanged.
        np.testing.assert_allclose(norm_sum[2, :], data[2, :])
        # Let's add a row where sum is not zero
        data_for_sum = np.array([[1.,2.,3.]])
        norm_sum_single = preprocess_cw_epr_data(data_for_sum.copy(), params_sum)
        self.assertAlmostEqual(np.sum(norm_sum_single[0,:]), 1.0,
                                msg="Intensity sum normalization failed for single positive row.")


class TestEPRLineshapes(unittest.TestCase):
    def _check_fwhm(self, x, y, expected_fwhm, tolerance=0.1):
        """ Helper to check FWHM of a peak centered at x=0. """
        peak_val = y[np.argmax(y)] # Should be close to 1.0
        half_max = peak_val / 2.0

        # Find indices where y is close to half_max
        # More robust: find where it crosses half_max
        above_half_max = np.where(y >= half_max)[0]
        if len(above_half_max) < 2 : # Not enough points above half max to determine width
             # This can happen if FWHM is very small relative to x-spacing
             # Try to find points closest to half_max on either side of peak center
            center_idx = np.argmax(y)
            left_idx = np.argmin(np.abs(y[:center_idx] - half_max))
            right_idx = center_idx + np.argmin(np.abs(y[center_idx:] - half_max))
            if left_idx == center_idx or right_idx == center_idx :
                 self.fail(f"Could not reliably determine FWHM; peak too narrow or x-axis too coarse. Max val: {peak_val}")

            actual_fwhm = x[right_idx] - x[left_idx]

        else:
            # Interpolate to find more accurate FWHM
            # Left side
            left_cross_idx = -1
            for i in range(np.argmax(y) - 1, -1, -1):
                if y[i] < half_max and y[i+1] >= half_max:
                    left_cross_idx = i
                    break
            if left_cross_idx == -1 and y[0] >= half_max : # Peak starts above half max
                self.fail("FWHM check failed: peak too wide or x starts within FWHM on left.")
                return

            x1, x2 = x[left_cross_idx], x[left_cross_idx+1]
            y1, y2 = y[left_cross_idx], y[left_cross_idx+1]
            left_x_at_half_max = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if (y2-y1) != 0 else x1

            # Right side
            right_cross_idx = -1
            for i in range(np.argmax(y), len(x) - 1):
                if y[i] >= half_max and y[i+1] < half_max:
                    right_cross_idx = i
                    break
            if right_cross_idx == -1 and y[-1] >= half_max:
                 self.fail("FWHM check failed: peak too wide or x ends within FWHM on right.")
                 return

            x1, x2 = x[right_cross_idx], x[right_cross_idx+1]
            y1, y2 = y[right_cross_idx], y[right_cross_idx+1]
            right_x_at_half_max = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if (y2-y1) !=0 else x1

            actual_fwhm = right_x_at_half_max - left_x_at_half_max

        self.assertAlmostEqual(actual_fwhm, expected_fwhm, delta=expected_fwhm * tolerance,
                               msg=f"FWHM check failed. Expected {expected_fwhm}, got {actual_fwhm}")


    def test_gaussian_lineshape(self):
        x = np.linspace(-10, 10, 500) # Fine grid for FWHM check
        center = 0.0
        fwhm = 2.0
        y = gaussian_lineshape(x, center, fwhm)

        self.assertAlmostEqual(y[np.where(x==center)[0][0]], 1.0, places=5, msg="Gaussian peak height not 1.0")
        self.assertEqual(np.argmax(y), np.where(x==center)[0][0], msg="Gaussian peak not at center")
        self._check_fwhm(x, y, fwhm, tolerance=0.05) # Tighter tolerance for fine grid

        with self.assertRaises(ValueError):
            gaussian_lineshape(x, center, fwhm=0)
        with self.assertRaises(ValueError):
            gaussian_lineshape(x, center, fwhm=-1)


    def test_lorentzian_lineshape(self):
        x = np.linspace(-10, 10, 500)
        center = 0.0
        fwhm = 2.0
        y = lorentzian_lineshape(x, center, fwhm)

        self.assertAlmostEqual(y[np.where(x==center)[0][0]], 1.0, places=5, msg="Lorentzian peak height not 1.0")
        self.assertEqual(np.argmax(y), np.where(x==center)[0][0], msg="Lorentzian peak not at center")
        self._check_fwhm(x, y, fwhm, tolerance=0.05)

        with self.assertRaises(ValueError):
            lorentzian_lineshape(x, center, fwhm=0)
        with self.assertRaises(ValueError):
            lorentzian_lineshape(x, center, fwhm=-1)


    def test_voigt_lineshape(self):
        x = np.linspace(-10, 10, 200)
        center = 0.0
        fwhm_g = 2.0
        fwhm_l = 1.0

        y_voigt = voigt_lineshape(x, center, fwhm_g, fwhm_l)
        self.assertAlmostEqual(np.max(y_voigt), 1.0, places=3, msg="Voigt peak height not close to 1.0") # Max might not be exactly at x=center due to discretization
        center_idx = np.argmin(np.abs(x - center))
        self.assertEqual(np.argmax(y_voigt), center_idx, msg="Voigt peak not at center (or x-grid too coarse)")

        # Test pure Gaussian case
        y_pure_g = voigt_lineshape(x, center, fwhm_g, 0.0)
        y_gauss_ref = gaussian_lineshape(x, center, fwhm_g)
        np.testing.assert_allclose(y_pure_g, y_gauss_ref, atol=1e-5, err_msg="Voigt (pure G) not matching Gaussian")

        # Test pure Lorentzian case
        y_pure_l = voigt_lineshape(x, center, 0.0, fwhm_l)
        y_lorentz_ref = lorentzian_lineshape(x, center, fwhm_l)
        np.testing.assert_allclose(y_pure_l, y_lorentz_ref, atol=1e-5, err_msg="Voigt (pure L) not matching Lorentzian")

        # Check that if both FWHM_G and FWHM_L are significant, it's different from pure G or L
        y_gauss_for_voigt_fwhm = gaussian_lineshape(x, center, fwhm_g + fwhm_l) # approx
        self.assertFalse(np.allclose(y_voigt, y_gauss_for_voigt_fwhm, atol=1e-2), "Voigt should not be simple Gaussian sum of FWHMs")

        with self.assertRaises(ValueError):
            voigt_lineshape(x, center, fwhm_g=-1, fwhm_l=1)
        with self.assertRaises(ValueError):
            voigt_lineshape(x, center, fwhm_g=1, fwhm_l=-1)

        # Test with very small FWHMs (should default to narrow Gaussian)
        y_both_zero = voigt_lineshape(x, center, 0.0, 0.0)
        # Expect a very narrow Gaussian-like shape if both are zero due to fallback in main code
        self.assertAlmostEqual(np.max(y_both_zero), 1.0, places=3)
        self.assertTrue(np.sum(y_both_zero > 0.1) < 10, "Voigt with zero FWHMs should be very narrow")


class TestARTSystemMatrix(unittest.TestCase):
    def test_basic_matrix_properties_nearest_no_lineshape(self):
        grid_size = (2, 2) # num_pixels_y, num_pixels_x
        num_pixels_y, num_pixels_x = grid_size
        num_pixels = num_pixels_y * num_pixels_x

        angles_deg = np.array([0.0]) # Single angle
        num_angles = len(angles_deg)
        num_bins = 3 # Number of projection bins

        # Dummy projection data (not used for system matrix init, but required by constructor)
        projection_data = np.zeros((num_angles, num_bins))

        art = ARTReconstructor(
            projection_data=projection_data,
            gradient_angles=angles_deg,
            grid_size=grid_size,
            num_iterations=1, # Not used for this test
            relaxation_param=0.1, # Not used
            projector_type='nearest',
            lineshape_model=None
        )

        A, A_row_norms_sq = art._initialize_system_matrix()

        self.assertEqual(A.shape, (num_angles * num_bins, num_pixels), "System matrix shape is incorrect.")

        # For a 2x2 grid and 0-degree projection:
        # Image center: (1.0, 1.0)
        # Pixels (coords relative to image center for projection):
        #   (0,0) -> center (-0.5, -0.5) -> p = -0.5 (assuming pixel width 1)
        #   (0,1) -> center ( 0.5, -0.5) -> p =  0.5
        #   (1,0) -> center (-0.5,  0.5) -> p = -0.5
        #   (1,1) -> center ( 0.5,  0.5) -> p =  0.5
        # Projection axis for 0 deg: x-axis.
        # p_min = -max(2,2)/2 = -1.0, p_max = 1.0. Length = 2.0.
        # bin_width = 2.0 / 3 bins = 0.666...
        # Bins:
        #   0: [-1.0, -0.333) -> center -0.666
        #   1: [-0.333, 0.333) -> center  0.0
        #   2: [ 0.333, 1.0]   -> center  0.666
        # Projected pixel centers:
        #   Pix 0 (0,0), flat_idx 0: p = -0.5 -> maps to bin 0. A[0,0]=1
        #   Pix 1 (0,1), flat_idx 1: p =  0.5 -> maps to bin 2. A[2,1]=1
        #   Pix 2 (1,0), flat_idx 2: p = -0.5 -> maps to bin 0. A[0,2]=1
        #   Pix 3 (1,1), flat_idx 3: p =  0.5 -> maps to bin 2. A[2,3]=1

        expected_A = np.zeros((num_bins * num_angles, num_pixels))
        # Angle 0:
        # Pixel 0 (0,0) projects to x = (0+0.5) - 1 = -0.5. Bin index floor((-0.5 - (-1))/0.666) = floor(0.5/0.666) = 0.
        expected_A[0, 0] = 1.0
        # Pixel 1 (0,1) projects to x = (1+0.5) - 1 = 0.5. Bin index floor((0.5 - (-1))/0.666) = floor(1.5/0.666) = 2.
        expected_A[2, 1] = 1.0
        # Pixel 2 (1,0) projects to x = (0+0.5) - 1 = -0.5. Bin index 0. (y coord does not matter for 0 deg)
        expected_A[0, 2] = 1.0
        # Pixel 3 (1,1) projects to x = (1+0.5) - 1 = 0.5. Bin index 2.
        expected_A[2, 3] = 1.0

        np.testing.assert_array_equal(A.toarray() if hasattr(A, "toarray") else A, expected_A,
                                      err_msg="System matrix for nearest, no lineshape is not as expected.")
        self.assertTrue(np.all(A_row_norms_sq >= 0), "Row norms squared should be non-negative.")
        expected_row_norms_sq = np.sum(expected_A**2, axis=1)
        np.testing.assert_allclose(A_row_norms_sq, expected_row_norms_sq, err_msg="Row norms squared calculation incorrect.")


    def test_lineshape_incorporation_nearest_gaussian(self):
        grid_size = (3, 3)
        num_pixels_y, num_pixels_x = grid_size
        num_pixels = num_pixels_y * num_pixels_x

        angles_deg = np.array([0.0])
        num_angles = len(angles_deg)
        num_bins = 11 # Enough bins to see the lineshape

        projection_data = np.zeros((num_angles, num_bins))

        # Single active pixel in the center of the 3x3 grid
        image_flat = np.zeros(num_pixels)
        center_pixel_x, center_pixel_y = num_pixels_x // 2, num_pixels_y // 2
        center_pixel_flat_idx = center_pixel_y * num_pixels_x + center_pixel_x
        image_flat[center_pixel_flat_idx] = 1.0

        fwhm_bins = 2.0
        art = ARTReconstructor(
            projection_data=projection_data,
            gradient_angles=angles_deg,
            grid_size=grid_size,
            num_iterations=1, relaxation_param=0.1,
            projector_type='nearest',
            lineshape_model='gaussian',
            lineshape_params={'fwhm': fwhm_bins}
        )

        A, _ = art._initialize_system_matrix()

        # For 0-degree projection, center pixel (1,1) in 3x3 grid.
        # Pixel center relative to image center (1.5,1.5):
        # ( (1+0.5) - 1.5, (1+0.5) - 1.5 ) = (0,0)
        # Projected p = 0.
        # p_min = -max(3,3)/2 = -1.5, p_max = 1.5. Length = 3.0.
        # bin_width = 3.0 / 11 bins approx 0.27
        # Center bin for p=0: floor((0 - (-1.5)) / (3.0/11)) = floor(1.5 * 11 / 3) = floor(5.5) = 5.
        expected_center_bin = num_bins // 2 # Should be bin 5 for 11 bins if p_min/p_max symmetric around 0 and p=0

        # The system matrix row for this pixel should be a Gaussian shape
        projection_for_center_pixel = A[:, center_pixel_flat_idx].toarray().flatten()

        # Check if the peak of the projection is at the expected center bin
        self.assertEqual(np.argmax(projection_for_center_pixel), expected_center_bin,
                         "Peak of lineshape not at projected center bin.")

        # Check if the shape is Gaussian (qualitatively by checking peak value and spread)
        self.assertAlmostEqual(projection_for_center_pixel[expected_center_bin], 1.0, places=3,
                               "Peak value of lineshape in system matrix should be 1.0 (peak norm).")

        # Verify FWHM roughly using the generated projection
        # Create an x-axis for this lineshape part of the matrix
        lineshape_x_axis = np.arange(num_bins) - expected_center_bin
        self._check_fwhm_for_A(lineshape_x_axis, projection_for_center_pixel, fwhm_bins, tolerance=0.2) # Higher tolerance for matrix

    def _check_fwhm_for_A(self, x_bins, y_values, expected_fwhm_bins, tolerance=0.2):
        """ Helper to check FWHM from system matrix columns/rows, x_bins is relative to center bin index"""
        peak_val = np.max(y_values)
        half_max = peak_val / 2.0

        above_half_max = np.where(y_values >= half_max)[0]
        if len(above_half_max) < 1 : # Should find at least the peak
             self.fail(f"Could not reliably determine FWHM from system matrix slice; peak too narrow. Max val: {peak_val}")

        # Find indices where y_values crosses half_max
        # This simplified version assumes x_bins is already centered (0 is peak)
        # and looks for width directly.
        left_cross_bins = x_bins[np.where((x_bins <= 0) & (y_values >= half_max))[0]]
        right_cross_bins = x_bins[np.where((x_bins >= 0) & (y_values >= half_max))[0]]

        if not left_cross_bins.size or not right_cross_bins.size:
            self.fail(f"FWHM check failed for system matrix slice: cannot find half-max points. Peak val {peak_val}")
            return

        # A simple estimate of FWHM from the spread of bins above half-max
        # This isn't as precise as the lineshape one but gives an indication.
        actual_fwhm_bins_est = (np.max(right_cross_bins) - np.min(left_cross_bins))

        # For a more direct check, we can use the logic from TestEPRLineshapes._check_fwhm
        # by passing x_bins and y_values.
        # However, the x_bins here are integer bin numbers. Interpolation might be less effective.
        # Let's use a simpler check on number of bins above half max as a proxy.
        num_bins_around_peak_above_half = len(above_half_max)
        # For a Gaussian, FWHM corresponds to points where exp(-4*ln2*(x/FWHM)^2) = 0.5
        # (x/FWHM)^2 = 0.25 => x = +/- FWHM/2. So width is FWHM.
        # Number of bins should be approx FWHM_bins.
        self.assertAlmostEqual(num_bins_around_peak_above_half, expected_fwhm_bins + 1, delta=2, # Allow some leeway (+1 for center bin, delta for spread)
                               msg=f"FWHM (est. by num bins {num_bins_around_peak_above_half}) not matching expected {expected_fwhm_bins} for system matrix.")


    def test_siddon_like_projector_no_lineshape(self):
        grid_size = (3, 3)
        angles_deg = np.array([45.0]) # Diagonal projection
        num_bins = 5
        projection_data = np.zeros((len(angles_deg), num_bins))

        art = ARTReconstructor(
            projection_data=projection_data,
            gradient_angles=angles_deg,
            grid_size=grid_size,
            num_iterations=1, relaxation_param=0.1,
            projector_type='siddon_like',
            lineshape_model=None
        )
        A, _ = art._initialize_system_matrix()

        # Center pixel (1,1) flat index is 1*3+1 = 4
        center_pixel_flat_idx = 4
        # For 45 deg projection, a ray passing through the center of the image
        # should intersect the center pixel.
        # p_val for center ray (k_bin_idx = num_bins // 2 = 2):
        # p_min = -max(3,3)/2 = -1.5, p_max = 1.5. Length = 3.0
        # bin_width = 3.0 / 5 = 0.6
        # p_val for bin 2 = -1.5 + (2+0.5)*0.6 = -1.5 + 2.5*0.6 = -1.5 + 1.5 = 0.0
        # So, the ray corresponding to p_val=0 (central ray) at 45 deg.
        central_ray_global_row_idx = 0 * num_bins + (num_bins // 2) # angle_idx = 0

        # Weight for center pixel and central ray should be non-zero
        weight_center_pixel_center_ray = A[central_ray_global_row_idx, center_pixel_flat_idx]
        self.assertTrue(weight_center_pixel_center_ray > 0,
                        "Center pixel should have non-zero weight for central ray in Siddon-like.")

        # Approximate expected length for a ray passing perfectly through center of 1x1 pixel at 45 deg is sqrt(2)
        # The calculated w_geom is t_max - t_min. Ray direction is (-sin45, cos45).
        # For a pixel from (0,0) to (1,1) and ray x=y (origin at 0,0, dir (1,1)/sqrt(2) )
        # Intersection with x=0 is t=0, with x=1 is t=1/cos(45), y=0 is t=0, y=1 is t=1/sin(45)
        # This needs careful check of the _calculate_ray_pixel_intersection_length logic.
        # Current check: just ensure it's positive and reasonable.
        # Pixel width/height is 1. Diagonal length is sqrt(2) approx 1.414.
        self.assertAlmostEqual(weight_center_pixel_center_ray, np.sqrt(2), delta=0.1,
                               msg=f"Intersection length for diagonal ray through center pixel incorrect. Got {weight_center_pixel_center_ray}")

        # Corner pixel (0,0) flat index 0. Should also be hit by some rays.
        # A ray slightly off center, e.g. bin (num_bins//2 - 1) might hit it.
        # Check a pixel guaranteed NOT to be hit by the central ray (e.g. (0,2) flat_idx 2)
        corner_pixel_flat_idx = 2 # Pixel (0,2)
        weight_corner_pixel_center_ray = A[central_ray_global_row_idx, corner_pixel_flat_idx]
        self.assertEqual(weight_corner_pixel_center_ray, 0.0,
                         "Corner pixel (0,2) should have zero weight for central ray at 45 deg.")


    def test_siddon_like_projector_with_lineshape(self):
        grid_size = (3, 3)
        angles_deg = np.array([0.0])
        num_bins = 11
        projection_data = np.zeros((len(angles_deg), num_bins))

        image_flat = np.zeros(grid_size[0]*grid_size[1])
        center_pixel_flat_idx = grid_size[0]*grid_size[1] // 2 # 4 for 3x3
        image_flat[center_pixel_flat_idx] = 1.0

        fwhm_bins = 2.0
        art = ARTReconstructor(
            projection_data=projection_data,
            gradient_angles=angles_deg,
            grid_size=grid_size,
            num_iterations=1, relaxation_param=0.1,
            projector_type='siddon_like',
            lineshape_model='gaussian',
            lineshape_params={'fwhm': fwhm_bins}
        )
        A, _ = art._initialize_system_matrix()

        # Center pixel (1,1) for 0 deg projects to p=0. Central bin is num_bins//2 = 5.
        # The ray for this bin (k_bin_idx = 5) should intersect pixel (1,1).
        # Intersection length for 0 deg through a 1x1 pixel is 1.0.
        # So, the column for pixel (1,1) should be a Gaussian centered at bin 5, with peak value 1.0*1.0 = 1.0.

        projection_for_center_pixel = A[:, center_pixel_flat_idx].toarray().flatten()
        expected_center_bin = num_bins // 2

        self.assertEqual(np.argmax(projection_for_center_pixel), expected_center_bin,
                         "Siddon+Lineshape: Peak of lineshape not at projected center bin.")

        # w_geom for 0-degree projection through center of pixel (1,1) should be 1.0 (pixel_width)
        # Ray origin for bin 5 (p_val=0): (image_center_x, image_center_y). Dir (0,1).
        # Pixel (1,1) is from x=1,x=2, y=1,y=2. Image center (1.5,1.5)
        # Intersection length should be 1.0.
        expected_w_geom = 1.0
        self.assertAlmostEqual(projection_for_center_pixel[expected_center_bin], 1.0 * expected_w_geom, places=3, # Peak of Gaussian is 1.0
                               "Siddon+Lineshape: Peak value of lineshape incorrect.")

        lineshape_x_axis = np.arange(num_bins) - expected_center_bin
        self._check_fwhm_for_A(lineshape_x_axis, projection_for_center_pixel / expected_w_geom, fwhm_bins, tolerance=0.25)


if __name__ == '__main__':
    unittest.main()
