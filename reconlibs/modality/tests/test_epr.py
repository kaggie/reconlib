import unittest
import abc
import numpy as np

# Adjust import paths based on the actual structure and how tests are run.
# If tests are run from the project root, these imports should work.
from reconlibs.modality.epr.base import EPRImaging
from reconlibs.modality.epr.continuous_wave import ContinuousWaveEPR
from reconlibs.modality.epr.pulse import PulseEPR
from reconlibs.modality.epr.reconstruction import (
    radial_recon_2d,
    radial_recon_3d,
    deconvolve_cw_spectrum,
    apply_kspace_corrections,
    preprocess_cw_epr_data,
    ARTReconstructor
)

# Dummy class to test EPRImaging
class DummyEPR(EPRImaging):
    def get_physics_model(self):
        return super().get_physics_model() # Should raise NotImplementedError

    def reconstruct(self, *args, **kwargs):
        return super().reconstruct(*args, **kwargs) # Should raise NotImplementedError

class TestEPR(unittest.TestCase):
    """Tests for EPR modality classes and functions."""

    def test_epr_imaging_abstract_methods(self):
        """Test that EPRImaging abstract methods raise NotImplementedError."""
        dummy_epr = DummyEPR(metadata={}, data={})
        with self.assertRaises(NotImplementedError):
            dummy_epr.get_physics_model()
        with self.assertRaises(NotImplementedError):
            dummy_epr.reconstruct()

    def test_continuous_wave_epr_instantiation(self):
        """Test instantiation of ContinuousWaveEPR."""
        metadata = {"experiment_id": "cw_test_001"}
        # data = {"raw_spectra": [1, 2, 3]} # Old way
        data_np = np.array([[1.0, 2.0, 3.0]]) # New: data is a 2D numpy array
        sweep_params = {"center_field_mT": 350, "sweep_width_mT": 10}
        cw_epr = ContinuousWaveEPR(metadata=metadata, data=data_np, sweep_parameters=sweep_params)
        self.assertIsInstance(cw_epr, ContinuousWaveEPR)
        self.assertEqual(cw_epr.metadata["experiment_id"], "cw_test_001")
        self.assertEqual(cw_epr.sweep_parameters["center_field_mT"], 350)
        self.assertTrue(np.array_equal(cw_epr.data, data_np))

    def test_continuous_wave_epr_get_physics_model(self):
        """Test the get_physics_model method of ContinuousWaveEPR."""
        # data = {} # Old way
        data_np = np.array([[0.0]]) # Minimal valid 2D numpy array
        cw_epr = ContinuousWaveEPR(metadata={}, data=data_np, sweep_parameters={})
        model = cw_epr.get_physics_model()
        self.assertIsInstance(model, dict)
        self.assertEqual(model["technique"], "Continuous Wave EPR")
        self.assertIn("key_parameters", model)
        self.assertIn("common_artifacts", model)
        self.assertIsInstance(model["key_parameters"], list)

    def test_continuous_wave_epr_reconstruct(self):
        """Test the reconstruct method of ContinuousWaveEPR using ART."""
        # Simple 1x1 case
        projection_data = np.array([[10.0]]) # One projection, one bin
        metadata = {'experiment_id': 'cw_art_test1'}
        sweep_params = {'center_field_mT': 350, 'sweep_width_mT': 10}
        gradient_angles_deg = [0.0]

        cw_epr = ContinuousWaveEPR(
            metadata=metadata,
            data=projection_data,
            sweep_parameters=sweep_params
        )

        reconstructed_image = cw_epr.reconstruct(
            gradient_angles=gradient_angles_deg,
            grid_size=(1, 1),
            num_iterations=1,
            relaxation_param=1.0,
            regularization_weight=0.0, # Disable for direct comparison
            non_negativity=False      # Disable for direct comparison
        )
        expected_image = np.array([[10.0]])
        np.testing.assert_array_almost_equal(reconstructed_image, expected_image, decimal=5)

    def test_cw_epr_reconstruct_with_art_and_preprocessing(self):
        """Test CW EPR reconstruct with ART and simple preprocessing."""
        # 1x2 image, 2 projections (0 and 90 deg)
        # Raw data that needs normalization
        raw_projection_data = np.array([[2.0, 4.0], [15.0, 15.0]])
        gradient_angles_deg = [0.0, 90.0] # For a 1x2 image, these project to different things
                                          # 0 deg: proj0 maps to pixel0, proj1 maps to pixel1
                                          # 90 deg: proj0 maps to (pix0+pix1)/2 , proj1 maps to (pix0+pix1)/2 (simplified)
                                          # This test is more about flow than perfect recon values with this data

        metadata = {'experiment_id': 'cw_art_test2'}
        sweep_params = {'center_field_mT': 350, 'sweep_width_mT': 10}

        cw_epr = ContinuousWaveEPR(
            metadata=metadata,
            data=raw_projection_data,
            sweep_parameters=sweep_params
        )

        reconstructed_image = cw_epr.reconstruct(
            gradient_angles=gradient_angles_deg,
            grid_size=(1, 2), # 1 row, 2 columns
            num_iterations=2, # More iterations
            relaxation_param=0.5,
            regularization_weight=0.01,
            non_negativity=True,
            art_preprocessing_params={'normalize_per_projection': True}
        )
        self.assertEqual(reconstructed_image.shape, (1,2))
        # The exact values are hard to predict without running the ART for this specific case,
        # but we can check if it ran and produced an image of the correct shape
        # and that values are non-negative.
        self.assertTrue(np.all(reconstructed_image >= 0))
        # Further checks could involve a known phantom and its expected reconstruction.


    def test_pulse_epr_instantiation(self):
        """Test instantiation of PulseEPR."""
        metadata = {"experiment_id": "pulse_test_001"}
        data = {"raw_echo": [0.1, 0.2, 0.15]}
        pulse_seq_details = {"sequence_name": "HahnEcho", "tau_ns": 100}
        p_epr = PulseEPR(metadata=metadata, data=data, pulse_sequence_details=pulse_seq_details)
        self.assertIsInstance(p_epr, PulseEPR)
        self.assertEqual(p_epr.metadata["experiment_id"], "pulse_test_001")
        self.assertEqual(p_epr.pulse_sequence_details["tau_ns"], 100)

    def test_pulse_epr_get_physics_model(self):
        """Test the get_physics_model method of PulseEPR."""
        p_epr = PulseEPR(metadata={}, data={}, pulse_sequence_details={})
        model = p_epr.get_physics_model()
        self.assertIsInstance(model, dict)
        self.assertEqual(model["technique"], "Pulse EPR")
        self.assertIn("common_sequences", model)
        self.assertIn("critical_factors_and_challenges", model)
        self.assertIsInstance(model["common_sequences"], list)

    def test_pulse_epr_reconstruct(self):
        """Test the reconstruct method of PulseEPR."""
        p_epr = PulseEPR(metadata={}, data={}, pulse_sequence_details={})
        self.assertEqual(p_epr.reconstruct(), "Reconstruction for Pulse EPR using gridding_fft is not yet implemented.")
        self.assertEqual(p_epr.reconstruct(algorithm="FBP"), "Reconstruction for Pulse EPR using FBP is not yet implemented.")

    def test_radial_recon_2d(self):
        """Test the radial_recon_2d function."""
        dummy_projections = [[1,2,3], [4,5,6]]
        dummy_angles = [0, 90]
        result = radial_recon_2d(data=dummy_projections, angles=dummy_angles)
        self.assertEqual(result, "2D Radial Reconstruction Placeholder")

    def test_radial_recon_3d(self):
        """Test the radial_recon_3d function."""
        dummy_projections = [[1,2,3], [4,5,6]]
        dummy_angles_phi = [0, 90]
        dummy_angles_theta = [0, 0]
        result = radial_recon_3d(data=dummy_projections, angles_phi=dummy_angles_phi, angles_theta=dummy_angles_theta)
        self.assertEqual(result, "3D Radial Reconstruction Placeholder")

    def test_deconvolve_cw_spectrum(self):
        """Test the deconvolve_cw_spectrum function."""
        dummy_spectrum = [1, 2, 1]
        mod_amp = 0.1
        result = deconvolve_cw_spectrum(spectrum=dummy_spectrum, modulation_amplitude_mT=mod_amp)
        self.assertEqual(result, "Deconvolved CW spectrum placeholder")

    def test_apply_kspace_corrections(self):
        """Test the apply_kspace_corrections function."""
        dummy_kspace = [[1+1j, 2+2j], [3+3j, 4+4j]]
        corr_params = {"dead_time_points": 2}
        result = apply_kspace_corrections(raw_kspace_data=dummy_kspace, correction_parameters=corr_params)
        self.assertEqual(result, "Corrected k-space data placeholder")

    def test_preprocess_cw_epr_data(self):
        """Tests for preprocess_cw_epr_data function."""
        data1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        processed1 = preprocess_cw_epr_data(data1)
        np.testing.assert_array_equal(processed1, data1)
        self.assertIsNot(processed1, data1)

        data2 = np.array([[1.0, 2.0, 4.0], [10.0, 15.0, 5.0]])
        expected2 = np.array([[0.25, 0.5, 1.0], [10.0/15.0, 1.0, 5.0/15.0]])
        processed2 = preprocess_cw_epr_data(data2, params={'normalize_per_projection': True})
        np.testing.assert_array_almost_equal(processed2, expected2)

        data3 = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        expected3 = np.array([[0.0, 0.0, 0.0], [1.0/3.0, 2.0/3.0, 1.0]])
        processed3 = preprocess_cw_epr_data(data3, params={'normalize_per_projection': True})
        np.testing.assert_array_almost_equal(processed3, expected3)

        data4 = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
        expected4 = np.array([[-2.0, -1.0, 0.0], [1.0/3.0, 2.0/3.0, 1.0]])
        processed4 = preprocess_cw_epr_data(data4, params={'normalize_per_projection': True})
        np.testing.assert_array_almost_equal(processed4, expected4)

        data5 = np.array([[-4.0, -2.0, -1.0], [1.0, 2.0, 3.0]])
        expected5 = np.array([[4.0, 2.0, 1.0], [1.0/3.0, 2.0/3.0, 1.0]])
        processed5 = preprocess_cw_epr_data(data5, params={'normalize_per_projection': True})
        np.testing.assert_array_almost_equal(processed5, expected5)

        with self.assertRaises(TypeError):
            preprocess_cw_epr_data([[1,2],[3,4]])
        with self.assertRaises(ValueError):
            preprocess_cw_epr_data(np.array([1,2,3]))

class TestARTReconstructor(unittest.TestCase):
    """Tests for the ARTReconstructor class."""

    def setUp(self):
        """Set up common parameters for ART tests."""
        self.projection_data = np.random.rand(10, 100) # 10 angles, 100 bins
        self.gradient_angles = np.arange(0, 180, 18) # 10 angles
        self.grid_size = (32, 32) # 32x32 image
        self.num_iterations = 10
        self.relaxation_param = 0.1
        self.regularization_weight = 0.01
        self.non_negativity = True

    def test_art_init_valid(self):
        """Test ARTReconstructor instantiation with valid parameters."""
        art = ARTReconstructor(
            self.projection_data, self.gradient_angles, self.grid_size,
            self.num_iterations, self.relaxation_param,
            self.regularization_weight, self.non_negativity
        )
        self.assertIsInstance(art, ARTReconstructor)
        self.assertEqual(art.num_angles, 10)
        self.assertEqual(art.num_bins_per_projection, 100)
        self.assertEqual(art.num_pixels_x, 32)
        self.assertEqual(art.num_pixels_y, 32)
        self.assertEqual(art.num_pixels, 32*32)
        self.assertEqual(art.total_projection_bins, 10*100)
        self.assertEqual(art.image_estimate.shape, (32*32,))

    def test_art_init_invalid_projection_data(self):
        """Test ARTReconstructor with invalid projection_data types and shapes."""
        with self.assertRaisesRegex(TypeError, "projection_data must be a NumPy array"):
            ARTReconstructor([[1,2],[3,4]], self.gradient_angles, self.grid_size, self.num_iterations, self.relaxation_param)
        with self.assertRaisesRegex(ValueError, "projection_data must be a 2D array"):
            ARTReconstructor(np.random.rand(10), self.gradient_angles, self.grid_size, self.num_iterations, self.relaxation_param)

    def test_art_init_invalid_gradient_angles(self):
        """Test ARTReconstructor with invalid gradient_angles."""
        with self.assertRaisesRegex(TypeError, "gradient_angles must be a list or NumPy array"):
            ARTReconstructor(self.projection_data, "not_a_list", self.grid_size, self.num_iterations, self.relaxation_param)
        with self.assertRaisesRegex(ValueError, "Length of gradient_angles must match"):
            ARTReconstructor(self.projection_data, [0, 90], self.grid_size, self.num_iterations, self.relaxation_param) # Only 2 angles

    def test_art_init_invalid_grid_size(self):
        """Test ARTReconstructor with invalid grid_size."""
        with self.assertRaisesRegex(ValueError, "grid_size must be a tuple of two positive integers"):
            ARTReconstructor(self.projection_data, self.gradient_angles, (32, 0), self.num_iterations, self.relaxation_param)
        with self.assertRaisesRegex(ValueError, "grid_size must be a tuple of two positive integers"):
            ARTReconstructor(self.projection_data, self.gradient_angles, (32, 32, 32), self.num_iterations, self.relaxation_param)
        with self.assertRaisesRegex(ValueError, "grid_size must be a tuple of two positive integers"):
            ARTReconstructor(self.projection_data, self.gradient_angles, ("32", "32"), self.num_iterations, self.relaxation_param)

    def test_art_init_invalid_num_iterations(self):
        """Test ARTReconstructor with invalid num_iterations."""
        with self.assertRaisesRegex(ValueError, "num_iterations must be a positive integer"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, 0, self.relaxation_param)
        with self.assertRaisesRegex(ValueError, "num_iterations must be a positive integer"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, -5, self.relaxation_param)
        with self.assertRaisesRegex(ValueError, "num_iterations must be a positive integer"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, 10.5, self.relaxation_param)

    def test_art_init_invalid_relaxation_param(self):
        """Test ARTReconstructor with invalid relaxation_param."""
        with self.assertRaisesRegex(TypeError, "relaxation_param must be a float"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, self.num_iterations, "0.1")
        # Test for the warning (cannot directly test print, but check if it runs)
        try:
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, self.num_iterations, 2.5)
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, self.num_iterations, 0.0)
        except Exception as e:
            self.fail(f"ARTReconstructor raised an unexpected exception for relaxation_param warning: {e}")


    def test_art_init_invalid_regularization_weight(self):
        """Test ARTReconstructor with invalid regularization_weight."""
        with self.assertRaisesRegex(TypeError, "regularization_weight must be a float"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, self.num_iterations, self.relaxation_param, "0.01")
        with self.assertRaisesRegex(ValueError, "regularization_weight must be non-negative"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, self.num_iterations, self.relaxation_param, -0.1)

    def test_art_init_invalid_non_negativity(self):
        """Test ARTReconstructor with invalid non_negativity."""
        with self.assertRaisesRegex(TypeError, "non_negativity must be a boolean value"):
            ARTReconstructor(self.projection_data, self.gradient_angles, self.grid_size, self.num_iterations, self.relaxation_param, self.regularization_weight, "True")

    def test_art_initialize_image_method(self):
        """Test the _initialize_image method of ARTReconstructor."""
        art = ARTReconstructor(
            self.projection_data, self.gradient_angles, self.grid_size,
            self.num_iterations, self.relaxation_param
        )
        initialized_image = art._initialize_image()
        self.assertIsInstance(initialized_image, np.ndarray)
        self.assertEqual(initialized_image.shape, self.grid_size) # grid_size is (rows, cols) or (y, x)
        self.assertEqual(initialized_image.dtype, np.float64)
        self.assertTrue(np.all(initialized_image == 0))

    def test_art_initialize_system_matrix_simple_1x2_0deg(self):
        """Test _initialize_system_matrix with a 1x2 grid and 0-degree projection."""
        proj_data = np.array([[1.0, 1.0]]) # 1 angle, 2 bins
        angles = [0.0]
        grid_size = (1, 2) # 1 row, 2 columns
        num_iter = 1
        relax_param = 0.1

        art = ARTReconstructor(proj_data, angles, grid_size, num_iter, relax_param)
        A, row_norms_sq = art._initialize_system_matrix()

        # Expected A: (num_angles*num_bins_per_projection, num_pixels_y*num_pixels_x)
        # (1*2, 1*2) = (2, 2)
        # Pixels: (0,0) flat_idx=0; (0,1) flat_idx=1
        # Image center: (1/2, 2/2) = (0.5, 1.0)
        # Pixel centers:
        # P0 (0,0): center (0.5,0.5). Relative to img center: (0.0, -0.5)
        # P1 (0,1): center (0.5,1.5). Relative to img center: (0.0, 0.5)

        # Theta = 0 rad (cos=1, sin=0)
        # p = center_x_rel * 1 + center_y_rel * 0 = center_x_rel
        # P0: p = -0.5
        # P1: p = 0.5

        # Projection axis for 1x2 grid: max_img_dim = 2. p_min=-1, p_max=1. Length=2.
        # num_bins_per_projection = 2. bin_width = 2/2 = 1.
        # p_shifted = p - p_min = p + 1
        # bin_idx = floor(p_shifted / bin_width) = floor(p+1)

        # P0: p=-0.5. p_shifted=0.5. bin_idx_float=0.5. bin_idx=0.
        # P1: p=0.5. p_shifted=1.5. bin_idx_float=1.5. bin_idx=1.

        # Global row index = angle_idx * num_bins_per_projection + bin_idx
        # angle_idx = 0. num_bins_per_projection = 2.
        # P0 -> bin 0: global_row_idx = 0*2 + 0 = 0. A[0,0] = 1
        # P1 -> bin 1: global_row_idx = 0*2 + 1 = 1. A[1,1] = 1

        expected_A = np.array([
            [1.0, 0.0], # bin 0: pixel 0
            [0.0, 1.0]  # bin 1: pixel 1
        ])
        expected_row_norms_sq = np.array([1.0, 1.0])

        np.testing.assert_array_almost_equal(A, expected_A, decimal=5)
        np.testing.assert_array_almost_equal(row_norms_sq, expected_row_norms_sq, decimal=5)

    def test_art_initialize_system_matrix_simple_1x2_90deg(self):
        """Test _initialize_system_matrix with a 1x2 grid and 90-degree projection."""
        proj_data = np.array([[1.0, 1.0]]) # 1 angle, 2 bins
        angles = [90.0]
        grid_size = (1, 2) # 1 row, 2 columns (height=1, width=2)
        num_iter = 1
        relax_param = 0.1

        art = ARTReconstructor(proj_data, angles, grid_size, num_iter, relax_param)
        A, row_norms_sq = art._initialize_system_matrix()

        # Image center: (0.5, 1.0)
        # Pixel centers:
        # P0 (0,0): center (0.5,0.5). Relative to img center: (0.0, -0.5) -> (y_rel, x_rel)
        # P1 (0,1): center (0.5,1.5). Relative to img center: (0.0, 0.5)

        # Theta = 90 deg (cos=0, sin=1)
        # p = center_x_rel * 0 + center_y_rel * 1 = center_y_rel
        # P0: p = 0.0
        # P1: p = 0.0

        # Projection axis for 1x2 grid: max_img_dim = 2. p_min=-1, p_max=1. Length=2.
        # num_bins_per_projection = 2. bin_width = 2/2 = 1.
        # p_shifted = p - p_min = p + 1
        # bin_idx = floor(p_shifted / bin_width) = floor(p+1)

        # P0: p=0.0. p_shifted=1.0. bin_idx_float=1.0. bin_idx=1.
        # P1: p=0.0. p_shifted=1.0. bin_idx_float=1.0. bin_idx=1.

        # Global row index = angle_idx * num_bins_per_projection + bin_idx
        # angle_idx = 0. num_bins_per_projection = 2.
        # P0 -> bin 1: global_row_idx = 0*2 + 1 = 1. A[1,0] = 1
        # P1 -> bin 1: global_row_idx = 0*2 + 1 = 1. A[1,1] = 1

        expected_A = np.array([
            [0.0, 0.0], # bin 0
            [1.0, 1.0]  # bin 1: pixel 0 and pixel 1
        ])
        expected_row_norms_sq = np.array([0.0, 2.0])

        np.testing.assert_array_almost_equal(A, expected_A, decimal=5)
        np.testing.assert_array_almost_equal(row_norms_sq, expected_row_norms_sq, decimal=5)

    def test_art_reconstruct_simple_1x1(self):
        """Test ART reconstruct method with a 1x1 grid."""
        proj_data = np.array([[10.0]]) # 1 angle, 1 bin
        angles = [0.0]
        grid_size = (1, 1)
        num_iter = 1
        relax_param = 1.0

        art = ARTReconstructor(
            projection_data=proj_data,
            gradient_angles=angles,
            grid_size=grid_size,
            num_iterations=num_iter,
            relaxation_param=relax_param,
            regularization_weight=0.0, # No regularization for this simple test
            non_negativity=False       # No non-negativity for this simple test
        )

        # Manually override _initialize_system_matrix for this simple case if needed,
        # or ensure the main one works.
        # For 1x1 grid, 1 angle, 1 bin:
        # num_pixels_x=1, num_pixels_y=1. Image center (0.5,0.5)
        # Pixel (0,0) center (0.5,0.5). Relative (0,0).
        # Angle 0: p = 0 * cos(0) + 0 * sin(0) = 0.
        # max_img_dim = 1. p_min = -0.5, p_max = 0.5. Proj axis length = 1.
        # num_bins_per_projection = 1. bin_width = 1.
        # p_shifted = p - p_min = 0 - (-0.5) = 0.5.
        # bin_idx_float = 0.5 / 1 = 0.5. bin_idx = floor(0.5) = 0.
        # A should be [[1.0]], A_row_norms_sq should be [1.0]

        reconstructed_image = art.reconstruct()

        expected_image = np.array([[10.0]])
        np.testing.assert_array_almost_equal(reconstructed_image, expected_image, decimal=5)

    def test_art_reconstruct_simple_1x2_0deg_1iter(self):
        """Test ART reconstruct with 1x2 grid, 0 deg angle, 1 iteration."""
        proj_data = np.array([[1.0, 2.0]]) # Pixel 0 projects to bin 0, Pixel 1 to bin 1
        angles = [0.0]
        grid_size = (1, 2)
        num_iter = 1
        relax_param = 1.0

        art = ARTReconstructor(
            proj_data, angles, grid_size, num_iter, relax_param,
            non_negativity=False # Disable constraints for simplicity
        )
        reconstructed_image = art.reconstruct()

        # Initial image_flat = [0,0]
        # A = [[1,0],[0,1]], norms_sq = [1,1]
        # Iteration 1:
        # Row 0 (angle 0, bin 0): A_row=[1,0], norm_sq=1. measured=1. predicted=0. residual=1.
        #   image_flat += 1.0 * (1/1) * [1,0] = [1,0]
        # Row 1 (angle 0, bin 1): A_row=[0,1], norm_sq=1. measured=2. predicted=np.dot([0,1],[1,0])=0. residual=2.
        #   image_flat += 1.0 * (2/1) * [0,1] = [1,0] + [0,2] = [1,2]
        expected_image = np.array([[1.0, 2.0]])
        np.testing.assert_array_almost_equal(reconstructed_image, expected_image, decimal=5)

    def test_art_apply_tikhonov_regularization(self):
        """Test the _apply_tikhonov_regularization method."""
        art = ARTReconstructor(
            self.projection_data, self.gradient_angles, self.grid_size,
            self.num_iterations, self.relaxation_param,
            regularization_weight=0.1, # Specific weight for this test
            non_negativity=False
        )
        image_flat = np.array([1.0, 2.0, -1.0, 0.0], dtype=np.float64)
        regularized_image = art._apply_tikhonov_regularization(image_flat.copy()) # Pass a copy
        expected_image = image_flat / (1.0 + 0.1)
        np.testing.assert_array_almost_equal(regularized_image, expected_image)

        # Test with zero regularization weight (should not change image)
        art.regularization_weight = 0.0
        regularized_image_zero_weight = art._apply_tikhonov_regularization(image_flat.copy())
        np.testing.assert_array_almost_equal(regularized_image_zero_weight, image_flat)


    def test_art_enforce_non_negativity(self):
        """Test the _enforce_non_negativity method."""
        art = ARTReconstructor(
            self.projection_data, self.gradient_angles, self.grid_size,
            self.num_iterations, self.relaxation_param,
            non_negativity=True # Ensure non_negativity is active if method relies on it
        )
        image_flat = np.array([1.0, -2.0, 0.0, -0.5, 3.0], dtype=np.float64)
        constrained_image = art._enforce_non_negativity(image_flat.copy()) # Pass a copy
        expected_image = np.array([1.0, 0.0, 0.0, 0.0, 3.0])
        np.testing.assert_array_almost_equal(constrained_image, expected_image)

    def test_art_reconstruct_simple_1x1_with_constraints(self):
        """Test ART reconstruct method with a 1x1 grid, with regularization and non-negativity."""
        proj_data = np.array([[-10.0]]) # Intentionally negative to test non-negativity
        angles = [0.0]
        grid_size = (1, 1)
        num_iter = 1
        relax_param = 1.0
        reg_weight = 0.1

        art = ARTReconstructor(
            projection_data=proj_data,
            gradient_angles=angles,
            grid_size=grid_size,
            num_iterations=num_iter,
            relaxation_param=relax_param,
            regularization_weight=reg_weight,
            non_negativity=True
        )

        reconstructed_image = art.reconstruct()

        # Step-by-step:
        # Initial image_flat = [0.0]
        # A = [[1.0]], A_row_norms_sq = [1.0]
        # Iteration 1:
        #   measured_val = -10.0, predicted_val = 0.0, residual = -10.0
        #   image_flat_updated = 0.0 + 1.0 * (-10.0 / 1.0) * 1.0 = -10.0
        #   Regularization: image_flat_reg = -10.0 / (1 + 0.1) = -10.0 / 1.1 = -9.0909...
        #   Non-negativity: image_flat_nonneg = 0.0
        expected_image = np.array([[0.0]])
        np.testing.assert_array_almost_equal(reconstructed_image, expected_image, decimal=5)


if __name__ == '__main__':
    unittest.main()
