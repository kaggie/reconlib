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

from reconlib.utils import calculate_density_compensation
from reconlib.voronoi_utils import EPSILON # For small value comparisons

class TestCalculateDensityCompensation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape_2d = (64, 64)
        self.image_shape_3d = (32, 32, 32)
        # Use float32 for k-trajectories as it's common in recon pipelines
        self.dtype = torch.float32 
        self.test_tol = EPSILON * 100 # Tolerance for DCF value checks, can be adjusted

    # --- General DCF Tests ---
    def test_dcf_invalid_method(self):
        points = torch.rand(10, 2, device=self.device, dtype=self.dtype)
        with self.assertRaisesRegex(NotImplementedError, "Density compensation method 'unknown_method' is not implemented."):
            calculate_density_compensation(points, self.image_shape_2d, method='unknown_method', device=self.device)

    # --- Voronoi DCF Tests ---
    def test_voronoi_dcf_2d_simple(self):
        # Simple square of points + center point
        points_2d = torch.tensor([
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], # Square
            [0.0, 0.0] # Center
        ], device=self.device, dtype=self.dtype)
        
        weights = calculate_density_compensation(points_2d, self.image_shape_2d, method='voronoi', device=self.device)
        
        self.assertEqual(weights.shape, (points_2d.shape[0],))
        self.assertTrue(torch.all(weights >= 0))
        # Center point (index 4) should have a relatively larger Voronoi cell area, thus smaller weight
        # Corner points (0-3) should have smaller cells, thus larger weights
        # This is heuristic: 1/area means smaller area -> larger weight.
        # The center point's cell is bounded by the 4 outer points. Outer points have cells extending outwards.
        # So center point should have a smaller area, thus larger weight than unbounded outer points.
        # If we assume this test runs without bounds, outer points might get very small weights (large areas).
        # Let's check that weights are not extremely small (e.g. > EPSILON)
        self.assertTrue(torch.all(weights > EPSILON))


    def test_voronoi_dcf_2d_with_bounds(self):
        points_2d = torch.tensor([
            [-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25], # Inner square
            [0.0, 0.0] 
        ], device=self.device, dtype=self.dtype)
        
        # Define bounds that are larger than the point extent
        bounds = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], device=self.device, dtype=self.dtype)
        
        weights = calculate_density_compensation(points_2d, self.image_shape_2d, method='voronoi', bounds=bounds, device=self.device)
        
        self.assertEqual(weights.shape, (points_2d.shape[0],))
        self.assertTrue(torch.all(weights > 0)) # With bounds, all finite cells should have positive area.
        
        # The center point (index 4) should have the smallest Voronoi cell area, thus the largest weight.
        # The four corner points (indices 0-3) should have similar, smaller weights than the center.
        center_weight = weights[4]
        corner_weights_mean = torch.mean(weights[:4])
        self.assertTrue(center_weight > corner_weights_mean)


    def test_voronoi_dcf_2d_degenerate_collinear(self):
        # Collinear points. compute_voronoi_density_weights handles this by returning uniform weights (1/N)
        # or small weights if Voronoi fails.
        points_2d = torch.tensor([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]
        ], device=self.device, dtype=self.dtype)
        
        num_points = points_2d.shape[0]
        weights = calculate_density_compensation(points_2d, self.image_shape_2d, method='voronoi', device=self.device)
        
        self.assertEqual(weights.shape, (num_points,))
        # Check if it falls back to uniform weights (1/N for N<=dim, or EPSILON if Voronoi fails for N>dim)
        # For 4 points in 2D, Voronoi might run. If it fails due to collinearity, expect EPSILON for each.
        # If it runs but cells are problematic, behavior depends on ConvexHull of degenerate cells.
        # The current compute_voronoi_density_weights has try-except for Voronoi returning EPSILON.
        # And if n_points <= space_dim it returns 1.0/n_points. Here 4 > 2.
        # Let's assume it might fail robustly or produce very small weights.
        # For collinear, SciPy Voronoi often fails or gives weird results.
        # Our wrapper should catch this and return EPSILON per point.
        expected_fallback_weights = torch.full((num_points,), EPSILON, dtype=self.dtype, device=self.device)
        # This assertion might be too strict if SciPy/Qhull manages to produce some result,
        # however, the current error handling in compute_voronoi_density_weights for Voronoi failure is to return EPSILON.
        torch.testing.assert_close(weights, expected_fallback_weights, rtol=0, atol=self.test_tol)


    def test_voronoi_dcf_3d_simple(self):
        # Simple cube of points + center point
        points_3d = torch.tensor([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], # Bottom face
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],   # Top face
            [0.0, 0.0, 0.0] # Center point
        ], device=self.device, dtype=self.dtype)
        
        weights = calculate_density_compensation(points_3d, self.image_shape_3d, method='voronoi', device=self.device)
        
        self.assertEqual(weights.shape, (points_3d.shape[0],))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights > EPSILON)) # Expect non-trivial weights for unbounded cells


    def test_voronoi_dcf_3d_with_bounds(self):
        points_3d = torch.tensor([
            [-0.25, -0.25, -0.25], [0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
            [-0.25, -0.25, 0.25], [0.25, -0.25, 0.25], [0.25, 0.25, 0.25], [-0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0]
        ], device=self.device, dtype=self.dtype)
        
        bounds = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device=self.device, dtype=self.dtype)
        
        weights = calculate_density_compensation(points_3d, self.image_shape_3d, method='voronoi', bounds=bounds, device=self.device)
        
        self.assertEqual(weights.shape, (points_3d.shape[0],))
        self.assertTrue(torch.all(weights > 0))
        
        # Center point (index 8) should have the smallest Voronoi cell volume, thus the largest weight.
        center_weight = weights[8]
        corner_weights_mean = torch.mean(weights[:8])
        self.assertTrue(center_weight > corner_weights_mean)


    def test_voronoi_dcf_empty_input(self):
        points_empty = torch.empty((0, 2), device=self.device, dtype=self.dtype)
        weights = calculate_density_compensation(points_empty, self.image_shape_2d, method='voronoi', device=self.device)
        self.assertEqual(weights.shape, (0,))

    def test_voronoi_dcf_less_than_required_points_2d(self):
        # Test with 2 points in 2D (n_points <= space_dim)
        points_2d = torch.tensor([[0.0,0.0], [0.1,0.1]], device=self.device, dtype=self.dtype)
        num_points = points_2d.shape[0]
        weights = calculate_density_compensation(points_2d, self.image_shape_2d, method='voronoi', device=self.device)
        # Expected: uniform weights 1.0 / num_points
        expected_weights = torch.full((num_points,), 1.0/num_points, dtype=self.dtype, device=self.device)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=self.test_tol)

    def test_voronoi_dcf_less_than_required_points_3d(self):
        # Test with 3 points in 3D (n_points <= space_dim)
        points_3d = torch.tensor([[0.0,0.0,0.0], [0.1,0.1,0.1], [0.2,0.0,0.0]], device=self.device, dtype=self.dtype)
        num_points = points_3d.shape[0]
        weights = calculate_density_compensation(points_3d, self.image_shape_3d, method='voronoi', device=self.device)
        # Expected: uniform weights 1.0 / num_points
        expected_weights = torch.full((num_points,), 1.0/num_points, dtype=self.dtype, device=self.device)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=self.test_tol)


if __name__ == '__main__':
    unittest.main()
