import unittest
import numpy as np
import torch # Ensure torch is imported
import sys
from unittest.mock import patch
import io

# No longer conditional, as functions now expect torch.Tensor
# try:
#     import torch
#     HAS_TORCH = True
# except ImportError:
#     HAS_TORCH = False

from reconlib.voronoi_utils import compute_polygon_area, compute_convex_hull_volume, normalize_weights, EPSILON
# QhullError is not directly raised by the PyTorch wrappers anymore, handled internally by the ConvexHull class.

class TestComputePolygonArea(unittest.TestCase):
    def test_square_area(self):
        vertices_np = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float64)
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertAlmostEqual(compute_polygon_area(vertices_torch), 1.0, delta=EPSILON*10)

    def test_triangle_area(self):
        vertices_np = np.array([[0,0], [1,0], [0.5,1]], dtype=np.float64)
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertAlmostEqual(compute_polygon_area(vertices_torch), 0.5, delta=EPSILON*10)

    def test_collinear_points(self):
        # The PyTorch ConvexHull class (using SciPy fallback) should handle collinear points gracefully
        # and return an area of 0.
        vertices_np = np.array([[0,0], [1,1], [2,2]], dtype=np.float64) # Collinear
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertAlmostEqual(compute_polygon_area(vertices_torch), 0.0, delta=EPSILON*10)

    def test_area_less_than_epsilon(self):
        vertices_np = np.array([[0,0], [EPSILON/10, 0], [EPSILON/20, EPSILON/10]], dtype=np.float64)
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertEqual(compute_polygon_area(vertices_torch), 0.0)

    def test_invalid_input_not_enough_points(self):
        vertices_torch = torch.tensor([[0,0], [1,1]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "At least 3 points are required"):
            compute_polygon_area(vertices_torch)

    def test_invalid_input_wrong_dimensions(self):
        vertices_torch = torch.tensor([[0,0,0], [1,1,0], [0,1,0]], dtype=torch.float64) # 3D points for 2D area
        with self.assertRaisesRegex(ValueError, "Input points must be a 2D tensor with shape"):
            compute_polygon_area(vertices_torch)
        
        vertices_torch_1d = torch.tensor([0,0,1,1,0,1], dtype=torch.float64) # 1D tensor
        with self.assertRaisesRegex(ValueError, "Input points must be a 2D tensor with shape"):
            compute_polygon_area(vertices_torch_1d)


    def test_invalid_input_not_torch_tensor(self):
        vertices_list = [[0,0], [1,0], [1,1], [0,1]]
        with self.assertRaisesRegex(ValueError, "Input points must be a PyTorch tensor."):
            compute_polygon_area(vertices_list)
        
        vertices_np = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "Input points must be a PyTorch tensor."):
            compute_polygon_area(vertices_np) # Pass numpy array directly


class TestComputeConvexHullVolume(unittest.TestCase):
    def test_cube_volume(self):
        vertices_np = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ], dtype=np.float64)
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertAlmostEqual(compute_convex_hull_volume(vertices_torch), 1.0, delta=EPSILON*10)

    def test_tetrahedron_volume(self):
        vertices_np = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertAlmostEqual(compute_convex_hull_volume(vertices_torch), 1.0/6.0, delta=EPSILON*10)

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_degenerate_input_coplanar(self, mock_stderr):
        # PyTorch ConvexHull class falls back to SciPy which handles QhullError and prints a warning.
        vertices_np = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=np.float64) # Coplanar
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertAlmostEqual(compute_convex_hull_volume(vertices_torch), 0.0, delta=EPSILON*10)
        self.assertIn("QhullError in 3D", mock_stderr.getvalue()) # Check for SciPy's QhullError message via wrapper

    def test_volume_less_than_epsilon(self):
        vertices_np = np.array([
            [0,0,0], [EPSILON/10,0,0], [0,EPSILON/10,0], [0,0,EPSILON/10]
        ], dtype=np.float64)
        vertices_torch = torch.tensor(vertices_np, dtype=torch.float64)
        self.assertEqual(compute_convex_hull_volume(vertices_torch), 0.0)

    def test_invalid_input_not_enough_points(self):
        vertices_torch = torch.tensor([[0,0,0], [1,1,0], [0,1,0]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "At least 4 points are required"):
            compute_convex_hull_volume(vertices_torch)

    def test_invalid_input_wrong_dimensions(self):
        vertices_torch_2d = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=torch.float64) # 2D points for 3D volume
        with self.assertRaisesRegex(ValueError, "Input points must be a 3D tensor with shape"):
            compute_convex_hull_volume(vertices_torch_2d)

        vertices_torch_1d = torch.tensor([0,0,0,1,0,0,0,1,0,0,0,1], dtype=torch.float64) # 1D tensor
        with self.assertRaisesRegex(ValueError, "Input points must be a 3D tensor with shape"):
            compute_convex_hull_volume(vertices_torch_1d)

    def test_invalid_input_not_torch_tensor(self):
        vertices_list = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        with self.assertRaisesRegex(ValueError, "Input points must be a PyTorch tensor."):
            compute_convex_hull_volume(vertices_list)

        vertices_np = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "Input points must be a PyTorch tensor."):
            compute_convex_hull_volume(vertices_np)


class TestNormalizeWeights(unittest.TestCase):
    def test_simple_normalization(self):
        weights_np = np.array([1, 2, 3], dtype=np.float64)
        weights_torch = torch.tensor(weights_np, dtype=torch.float64)
        
        normalized_torch = normalize_weights(weights_torch)
        expected_torch = torch.tensor([1/6.0, 2/6.0, 3/6.0], dtype=torch.float64)
        
        torch.testing.assert_close(normalized_torch, expected_torch, rtol=0, atol=EPSILON*10)
        self.assertAlmostEqual(torch.sum(normalized_torch).item(), 1.0, delta=EPSILON*10)

    def test_normalization_with_negatives_clamped(self):
        weights_np = np.array([1, -1, 2], dtype=np.float64) # sum = 2
        weights_torch = torch.tensor(weights_np, dtype=torch.float64)
        
        normalized_torch = normalize_weights(weights_torch)
        # Expected: [1/2, -1/2, 2/2] -> [0.5, -0.5, 1.0]
        # After clamping (torch.clamp(min=0.0)): [0.5, 0, 1.0]
        expected_torch = torch.tensor([0.5, 0, 1.0], dtype=torch.float64)
        
        torch.testing.assert_close(normalized_torch, expected_torch, rtol=0, atol=EPSILON*10)
        # Sum after clamping is 1.5, not 1.0. This is expected per current function logic.
        self.assertAlmostEqual(torch.sum(normalized_torch).item(), 1.5, delta=EPSILON*10)

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_sum_is_zero(self, mock_stderr):
        weights_torch = torch.tensor([1, -1, 0], dtype=torch.float64)
        normalized_torch = normalize_weights(weights_torch)
        torch.testing.assert_close(normalized_torch, weights_torch, rtol=0, atol=EPSILON*10) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_sum_is_near_zero(self, mock_stderr):
        weights_torch = torch.tensor([EPSILON/10, -EPSILON/10, EPSILON/20], dtype=torch.float64)
        normalized_torch = normalize_weights(weights_torch)
        torch.testing.assert_close(normalized_torch, weights_torch, rtol=0, atol=EPSILON*10) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_all_zeros(self, mock_stderr):
        weights_torch = torch.tensor([0, 0, 0], dtype=torch.float64)
        normalized_torch = normalize_weights(weights_torch)
        torch.testing.assert_close(normalized_torch, weights_torch, rtol=0, atol=EPSILON*10) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_empty_array(self, mock_stderr):
        weights_torch = torch.tensor([], dtype=torch.float64)
        normalized_torch = normalize_weights(weights_torch)
        torch.testing.assert_close(normalized_torch, weights_torch, rtol=0, atol=EPSILON*10) # Expect original weights
        # The PyTorch sum of an empty tensor is 0, so it should trigger the warning.
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())
        
    def test_invalid_input_not_torch_tensor(self):
        weights_list = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Input weights must be a PyTorch tensor."):
            normalize_weights(weights_list)

        weights_np = np.array([1,2,3])
        with self.assertRaisesRegex(ValueError, "Input weights must be a PyTorch tensor."):
            normalize_weights(weights_np)


if __name__ == '__main__':
    unittest.main()
