import unittest
import numpy as np
import sys
from unittest.mock import patch
import io

# Attempt to import PyTorch and use it for tensor creation where feasible,
# but ensure tests fall back to NumPy if PyTorch is not available or not practical.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from reconlib.voronoi_utils import compute_polygon_area, compute_convex_hull_volume, normalize_weights, EPSILON

class TestComputePolygonArea(unittest.TestCase):
    def test_square_area(self):
        vertices = np.array([[0,0], [1,0], [1,1], [0,1]])
        self.assertAlmostEqual(compute_polygon_area(vertices), 1.0, delta=EPSILON*10)
        if HAS_TORCH:
            vertices_torch = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=torch.float64)
            self.assertAlmostEqual(compute_polygon_area(vertices_torch.numpy()), 1.0, delta=EPSILON*10)

    def test_triangle_area(self):
        vertices = np.array([[0,0], [1,0], [0.5,1]])
        self.assertAlmostEqual(compute_polygon_area(vertices), 0.5, delta=EPSILON*10)

    def test_collinear_points(self):
        vertices = np.array([[0,0], [1,1], [2,2]]) # Collinear
        self.assertAlmostEqual(compute_polygon_area(vertices), 0.0, delta=EPSILON*10)

    def test_area_less_than_epsilon(self):
        vertices = np.array([[0,0], [EPSILON/10, 0], [EPSILON/20, EPSILON/10]])
        self.assertEqual(compute_polygon_area(vertices), 0.0)

    def test_invalid_input_not_enough_points(self):
        vertices = np.array([[0,0], [1,1]])
        with self.assertRaises(ValueError):
            compute_polygon_area(vertices)

    def test_invalid_input_wrong_dimensions(self):
        vertices = np.array([[0,0,0], [1,1,0], [0,1,0]])
        with self.assertRaises(ValueError):
            compute_polygon_area(vertices)

    def test_invalid_input_not_numpy_array(self):
        vertices = [[0,0], [1,0], [1,1], [0,1]]
        with self.assertRaises(ValueError):
            compute_polygon_area(vertices)


class TestComputeConvexHullVolume(unittest.TestCase):
    def test_cube_volume(self):
        vertices = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ])
        self.assertAlmostEqual(compute_convex_hull_volume(vertices), 1.0, delta=EPSILON*10)
        if HAS_TORCH:
            vertices_torch = torch.tensor(vertices, dtype=torch.float64)
            self.assertAlmostEqual(compute_convex_hull_volume(vertices_torch.numpy()), 1.0, delta=EPSILON*10)


    def test_tetrahedron_volume(self):
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.assertAlmostEqual(compute_convex_hull_volume(vertices), 1.0/6.0, delta=EPSILON*10)

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_degenerate_input_coplanar(self, mock_stderr):
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]) # Coplanar
        self.assertAlmostEqual(compute_convex_hull_volume(vertices), 0.0, delta=EPSILON*10)
        self.assertIn("QhullError encountered", mock_stderr.getvalue())

    def test_volume_less_than_epsilon(self):
        vertices = np.array([
            [0,0,0], [EPSILON/10,0,0], [0,EPSILON/10,0], [0,0,EPSILON/10]
        ])
        # The volume will be (EPSILON/10)^3 / 6, which is much smaller than EPSILON
        self.assertEqual(compute_convex_hull_volume(vertices), 0.0)


    def test_invalid_input_not_enough_points(self):
        vertices = np.array([[0,0,0], [1,1,0], [0,1,0]])
        with self.assertRaises(ValueError):
            compute_convex_hull_volume(vertices)

    def test_invalid_input_wrong_dimensions(self):
        vertices = np.array([[0,0], [1,0], [1,1], [0,1]])
        with self.assertRaises(ValueError):
            compute_convex_hull_volume(vertices)

    def test_invalid_input_not_numpy_array(self):
        vertices = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        with self.assertRaises(ValueError):
            compute_convex_hull_volume(vertices)


class TestNormalizeWeights(unittest.TestCase):
    def test_simple_normalization(self):
        weights = np.array([1, 2, 3], dtype=float)
        normalized = normalize_weights(weights)
        expected = np.array([1/6.0, 2/6.0, 3/6.0])
        np.testing.assert_array_almost_equal(normalized, expected, decimal=7)
        self.assertAlmostEqual(np.sum(normalized), 1.0, delta=EPSILON*10)
        if HAS_TORCH:
            weights_torch = torch.tensor([1,2,3], dtype=torch.float64)
            normalized_torch = normalize_weights(weights_torch.numpy())
            np.testing.assert_array_almost_equal(normalized_torch, expected, decimal=7)
            self.assertAlmostEqual(np.sum(normalized_torch), 1.0, delta=EPSILON*10)


    def test_normalization_with_negatives_clamped(self):
        weights = np.array([1, -1, 2], dtype=float) # sum = 2
        normalized = normalize_weights(weights)
        # Expected: [1/2, -1/2, 2/2] -> [0.5, -0.5, 1.0]
        # After clamping: [0.5, 0, 1.0]
        expected = np.array([0.5, 0, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected, decimal=7)
        # Sum after clamping is 1.5, not 1.0. This is expected per current function logic.
        self.assertAlmostEqual(np.sum(normalized), 1.5, delta=EPSILON*10)


    @patch('sys.stderr', new_callable=io.StringIO)
    def test_sum_is_zero(self, mock_stderr):
        weights = np.array([1, -1, 0], dtype=float)
        normalized = normalize_weights(weights)
        np.testing.assert_array_equal(normalized, weights) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_sum_is_near_zero(self, mock_stderr):
        weights = np.array([EPSILON/10, -EPSILON/10, EPSILON/20], dtype=float)
        normalized = normalize_weights(weights)
        np.testing.assert_array_equal(normalized, weights) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_all_zeros(self, mock_stderr):
        weights = np.array([0, 0, 0], dtype=float)
        normalized = normalize_weights(weights)
        np.testing.assert_array_equal(normalized, weights) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())

    @patch('sys.stderr', new_callable=io.StringIO)
    def test_empty_array(self, mock_stderr):
        weights = np.array([], dtype=float)
        normalized = normalize_weights(weights)
        np.testing.assert_array_equal(normalized, weights) # Expect original weights
        self.assertIn("Sum of weights is zero or near-zero", mock_stderr.getvalue())
        
    def test_invalid_input_not_numpy_array(self):
        weights = [1, 2, 3]
        with self.assertRaises(ValueError):
            normalize_weights(weights)


if __name__ == '__main__':
    unittest.main()
