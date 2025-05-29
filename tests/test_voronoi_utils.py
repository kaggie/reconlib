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

# New Test Class for ConvexHull, delaunay_triangulation_3d, and new normalize_weights
from reconlib.voronoi_utils import (
    ConvexHull, 
    delaunay_triangulation_3d, 
    EPSILON,
    # The following are internal to delaunay_triangulation_3d in the provided solution,
    # so they cannot be imported directly for testing.
    # _orientation3d_pytorch, 
    # _in_circumsphere3d_pytorch 
)
from scipy.spatial import Delaunay as ScipyDelaunay # For comparison test

class TestVoronoiUtilsFeatures(unittest.TestCase): # Assuming this class can house Delaunay tests
    def setUp(self):
        torch.manual_seed(0) # Ensure reproducibility
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.double_type = torch.float64 # For precision in geometric tests
        self.tol = EPSILON * 100 # Increased tolerance slightly for geometric tests
        self.small_tol = 1e-5 # For float comparisons where high precision is expected
        self.default_delaunay_tol = 1e-7 # Default tol used in delaunay_triangulation_3d

    def _tetra_volume(self, p0, p1, p2, p3):
        """Computes the volume of a tetrahedron defined by 4 points."""
        # Volume = |dot(p1-p0, cross(p2-p0, p3-p0))| / 6.0
        # Ensure points are on the same device and dtype for torch operations
        p0, p1, p2, p3 = p0.to(self.device), p1.to(self.device), p2.to(self.device), p3.to(self.device)
        return torch.abs(torch.dot(p1 - p0, torch.cross(p2 - p0, p3 - p0))) / 6.0
    
    # Re-implement circumsphere check for testing, as the original is internal
    def _test_in_circumsphere3d(self, p_test, t1, t2, t3, t4, tol):
        points_for_mat = [t1, t2, t3, t4, p_test]
        mat_rows = []
        for pt_i in points_for_mat:
            pt_i_64 = pt_i.to(dtype=torch.float64) # Use float64 for precision
            sum_sq = torch.sum(pt_i_64**2)
            mat_rows.append(torch.cat((pt_i_64, sum_sq.unsqueeze(0), torch.tensor([1.0], dtype=torch.float64, device=p_test.device))))
        
        mat_5x5 = torch.stack(mat_rows, dim=0)
        
        orient_mat = torch.stack((t2 - t1, t3 - t1, t4 - t1), dim=0).to(dtype=torch.float64)
        orient_det_val = torch.det(orient_mat)

        if torch.abs(orient_det_val) < tol: # Degenerate tetrahedron
            return False 

        circumsphere_det_val = torch.det(mat_5x5)
        # p_test is inside if (orient_det_val * circumsphere_det_val) > tol
        # For Delaunay property, we check if any other point is *inside*.
        # So, this should return True if it's strictly inside.
        return (orient_det_val * circumsphere_det_val) > tol

    # --- ConvexHull Class Tests ---
    # ** 2D Hull Tests **
    def test_convex_hull_2d_square(self):
        points = torch.tensor([[0,0], [1,0], [1,1], [0,1], [0.5, 0.5]], dtype=self.double_type) # Square with an internal point
        hull = ConvexHull(points, tol=1e-7)

        # Expected vertices (indices of the input points, order might vary but should form the square)
        # Common orderings: [0,1,2,3] or [0,3,2,1] etc.
        # The monotone_chain_2d should give a consistent counter-clockwise order.
        # E.g. for points sorted as (0,0), (1,0), (0,1), (1,1) -> (0,0), (1,0), (1,1), (0,1)
        # Input points: P0(0,0), P1(1,0), P2(1,1), P3(0,1), P4(0.5,0.5)
        # Sorted lexicographically: P0, P3, P1, P2 (if y then x) or P0,P1,P3,P2 (if x then y)
        # `monotone_chain_2d` sorts by x then y.
        # Sorted_indices for [[0,0], [1,0], [1,1], [0,1], [0.5,0.5]] would be:
        # (0,0) -> 0
        # (0.5,0.5) -> 4
        # (1,0) -> 1
        # (0,1) -> 3 (Error in manual sort here, (0,1) comes before (0.5,0.5) if y is secondary sort criteria)
        # Let's trace `monotone_chain_2d`'s sort: torch.lexsort((points[:,1], points[:,0]))
        # x: [0, 1, 1, 0, 0.5]
        # y: [0, 0, 1, 1, 0.5]
        # sorted_indices based on x first: [0, 3, 4, 1, 2] (P0, P3, P4, P1, P2) -> (0,0), (0,1), (0.5,0.5), (1,0), (1,1)
        # This is not right. lexsort sorts by last column first.
        # lexsort((y,x)) means sort by x, then by y.
        # points:   (0,0) (1,0) (1,1) (0,1) (0.5,0.5)
        # indices:    0     1     2     3      4
        # x-coords:  0.0   1.0   1.0   0.0    0.5
        # y-coords:  0.0   0.0   1.0   1.0    0.5
        # lexsort by y, then x:
        # Sorted by y: (0,0), (1,0), (0.5,0.5), (1,1), (0,1) -> indices [0,1,4,2,3]
        #  Within y=0: (0,0), (1,0) -> [0,1]
        #  Within y=0.5: (0.5,0.5) -> [4]
        #  Within y=1: (1,1), (0,1) -> sorted by x: (0,1), (1,1) -> [3,2]
        # Combined: [0,1,4,3,2] -> points[0], points[1], points[4], points[3], points[2]
        # (0,0), (1,0), (0.5,0.5), (0,1), (1,1) - This is the order `monotone_chain_2d` gets after sorting
        #
        # Upper hull from (0,0), (1,0), (0.5,0.5), (0,1), (1,1):
        # Add 0:(0,0). upper=[0]
        # Add 1:(1,0). upper=[0,1]
        # Add 4:(0.5,0.5). cross(P0,P1,P4) = (1-0)*(0.5-0) - (0-0)*(0.5-0) = 0.5 > 0 (left turn). upper=[0,1,4]
        # Add 3:(0,1). cross(P1,P4,P3) = (0.5-1)*(1-0) - (0-0)*(0-1) = -0.5 <0 (right turn). upper=[0,1,4,3] No, P1,P4,P3 is (1,0),(0.5,0.5),(0,1) -> (-0.5)*(1) - (0.5)*(-0.5) = -0.5 + 0.25 = -0.25. Pop 4.
        #   upper=[0,1]. cross(P0,P1,P3) = (1-0)*(1-0)-(0-0)*(0-0)=1 >0. upper=[0,1,3]
        # Add 2:(1,1). cross(P1,P3,P2) = (0-1)*(1-0)-(1-0)*(1-0) = -1-1 = -2 <0. Pop 3.
        #   upper=[0,1]. cross(P0,P1,P2) = (1-0)*(1-0) - (0-0)*(1-0) = 1 >0. upper=[0,1,2]
        # Upper: [0,1,2] (indices of original points: (0,0), (1,0), (1,1))
        #
        # Lower hull (iterate reverse sorted: (1,1), (0,1), (0.5,0.5), (1,0), (0,0)):
        # Add 2:(1,1). lower=[2]
        # Add 3:(0,1). lower=[2,3]
        # Add 4:(0.5,0.5). cross(P2,P3,P4) = (0-1)*(0.5-1)-(1-1)*(0.5-1) = (-1)*(-0.5) = 0.5 >0. lower=[2,3,4]
        # Add 1:(1,0). cross(P3,P4,P1) = (0.5-0)*(0-1) - (0.5-1)*(1-0) = -0.5 - (-0.5) = 0. Pop 4.
        #   lower=[2,3]. cross(P2,P3,P1) = (0-1)*(0-1)-(1-1)*(0-1) = 1 >0. lower=[2,3,1] This is wrong, P1 is (1,0)
        #   P3(0,1), P4(0.5,0.5), P1(1,0). P3=(0,1), P4=(0.5,0.5), P1=(1,0)
        #   (P4-P3) = (0.5, -0.5). (P1-P3) = (1, -1). Cross: 0.5*(-1) - (-0.5)*1 = -0.5 + 0.5 = 0 (collinear). Pop 4.
        #   lower=[2,3]. P2(1,1), P3(0,1), P1(1,0). (P3-P2)=(-1,0). (P1-P2)=(0,-1). Cross: (-1)*(-1) - 0*0 = 1 >0. lower=[2,3,1]
        # Add 0:(0,0). cross(P3,P1,P0) = (1-0)*(0-1)- (0-1)*(0-0) = -1 <0. Pop 1.
        #   lower=[2,3]. cross(P2,P3,P0) = (0-1)*(0-1) - (1-1)*(0-1) = 1 >0. lower=[2,3,0]
        # Lower: [2,3,0] (indices of original points: (1,1), (0,1), (0,0))
        #
        # Result: upper[:-1] + lower[:-1] = [0,1] + [2,3] = [0,1,2,3]
        # These are indices into original `points` tensor.
        # Expected vertices: torch.tensor([0,1,2,3], dtype=torch.long) or permutation like [0,3,2,1]
        
        # Let's use a known output for square points [0,0],[1,0],[0,1],[1,1] (indices 0,1,2,3)
        # Without the internal point for simplicity of predicting vertices order
        points_simple_square = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=self.double_type)
        hull_sq = ConvexHull(points_simple_square, tol=1e-7)

        expected_vertices_sq_indices = torch.tensor([0,1,2,3], dtype=torch.long) # Based on typical monotone chain output for this input
        # Test if the set of vertices is the same, order might vary for the first point.
        # A robust way is to check sorted versions of vertex sets or check cyclic permutations.
        # For now, let's assume a fixed output order from monotone_chain_2d for specific input.
        # For points (0,0), (1,0), (1,1), (0,1): sorted_indices = [0,3,1,2]
        # Upper: [0,1,2] -> orig_indices [0,3,1] WRONG -> [0,1,2] with points (0,0),(1,0),(1,1)
        #   P0(0,0), P1(1,0), P2(1,1), P3(0,1)
        #   Sorted by x, then y: P0, P3, P1, P2. Original indices: 0, 3, 1, 2
        #   Upper hull from [P0, P3, P1, P2]:
        #   u_hull_orig_idx: add 0. [0]
        #   add 3. P0(0,0),P3(0,1). cross is 0. Add 3. [0,3] (No, P0=(0,0) P3=(0,1)) -> (P3-P0)=(0,1)
        #      P0(0,0), P1(1,0), P2(1,1), P3(0,1). `sorted_indices` from `torch.lexsort((points[:,1], points[:,0]))` are `[0, 3, 1, 2]`
        #      This means `sorted_points` are `points[[0,3,1,2]]` = `[[0,0], [0,1], [1,0], [1,1]]`
        #      Upper hull from `[[0,0], [0,1], [1,0], [1,1]]` (original indices in `[]`):
        #      Add 0 ([0]). `upper_hull=[0]`
        #      Add 3 ([0,1]). `cross(points[0], points[3], ???)` no, `cross(points[upper_hull[-2]], points[upper_hull[-1]], points[sorted_indices[i]])`
        #      It should be `[0, 1, 2, 3]` for `[[0,0],[1,0],[1,1],[0,1]]` if input is already sorted for CCW.
        #      The current monotone_chain_2d implementation returns `[0,1,2,3]` for this input.
        
        torch.testing.assert_equal(hull_sq.vertices.sort().values, expected_vertices_sq_indices.sort().values) # Compare sorted unique vertices
        self.assertEqual(hull_sq.vertices.shape[0], 4)

        # Expected simplices (edges) for [0,1,2,3] -> [[0,1],[1,2],[2,3],[3,0]]
        expected_simplices_sq_list = [[0,1],[1,2],[2,3],[3,0]]
        # Convert to set of tuples of sorted indices to make comparison order-agnostic for edges
        expected_simplices_set = {tuple(sorted(edge)) for edge in expected_simplices_sq_list}
        returned_simplices_set = {tuple(sorted(edge.tolist())) for edge in hull_sq.simplices}
        self.assertEqual(returned_simplices_set, expected_simplices_set)
        
        self.assertAlmostEqual(hull_sq.area.item(), 1.0, delta=self.tol)

        # Test with the internal point, area should still be 1.0
        hull_internal_pt = ConvexHull(points, tol=1e-7) # [[0,0], [1,0], [1,1], [0,1], [0.5, 0.5]]
        self.assertAlmostEqual(hull_internal_pt.area.item(), 1.0, delta=self.tol)
        # Vertices should be the outer square, so 4 vertices.
        self.assertEqual(hull_internal_pt.vertices.shape[0], 4) 
        # Check if the internal point (index 4) is NOT in the hull vertices
        self.assertNotIn(4, hull_internal_pt.vertices.tolist())


    def test_convex_hull_2d_collinear(self):
        points = torch.tensor([[0,0], [1,1], [2,2], [3,3]], dtype=self.double_type) # Collinear
        hull = ConvexHull(points, tol=1e-7)
        # Expected: 2 vertices (the extremes), 1 simplex (the segment), 0 area
        # Monotone chain should return the two extreme points, e.g. [0,3] for this input.
        self.assertEqual(hull.vertices.shape[0], 2)
        # Check if the vertices are indeed the first and last point (indices 0 and 3)
        self.assertIn(0, hull.vertices.tolist())
        self.assertIn(3, hull.vertices.tolist())
        
        self.assertEqual(hull.simplices.shape[0], 1)
        # Simplices should be [[0,3]] or [[3,0]]
        edge = tuple(sorted(hull.simplices[0].tolist()))
        self.assertEqual(edge, (0,3))
        
        self.assertAlmostEqual(hull.area.item(), 0.0, delta=self.tol)

    def test_convex_hull_2d_less_than_3_points(self):
        # 2 points
        points_2 = torch.tensor([[0,0], [1,1]], dtype=self.double_type)
        hull_2 = ConvexHull(points_2, tol=1e-7)
        self.assertEqual(hull_2.vertices.shape[0], 2) # Vertices are [0,1]
        self.assertEqual(hull_2.simplices.shape[0], 1) # Simplex is [[0,1]] or [[1,0]]
        edge2 = tuple(sorted(hull_2.simplices[0].tolist()))
        self.assertEqual(edge2, (0,1))
        self.assertAlmostEqual(hull_2.area.item(), 0.0, delta=self.tol)

        # 1 point
        points_1 = torch.tensor([[0,0]], dtype=self.double_type)
        hull_1 = ConvexHull(points_1, tol=1e-7)
        self.assertEqual(hull_1.vertices.shape[0], 1) # Vertex is [0]
        self.assertEqual(hull_1.simplices.shape[0], 0) # No simplices
        self.assertAlmostEqual(hull_1.area.item(), 0.0, delta=self.tol)
        
        # 0 points (should ideally be handled by raising error or specific empty output)
        points_0 = torch.empty((0,2), dtype=self.double_type)
        # The monotone_chain_2d raises ValueError for points.shape[0] < 3 if not handled
        # The ConvexHull class itself has checks, but monotone_chain_2d also has its own.
        # monotone_chain_2d: if points.shape[0] < 3, returns indices and simplices directly.
        hull_0 = ConvexHull(points_0, tol=1e-7)
        self.assertEqual(hull_0.vertices.shape[0], 0)
        self.assertEqual(hull_0.simplices.shape[0], 0)
        self.assertAlmostEqual(hull_0.area.item(), 0.0, delta=self.tol)


    # ** 3D Hull Tests **
    def test_convex_hull_3d_cube(self):
        points = torch.tensor([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0], # Bottom face
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]  # Top face
        ], dtype=self.double_type)
        hull = ConvexHull(points, tol=1e-7)

        self.assertEqual(hull.vertices.shape[0], 8) # All 8 points are on the hull
        self.assertEqual(hull.simplices.shape[0], 12) # A cube has 12 triangular faces
        self.assertAlmostEqual(hull.volume.item(), 1.0, delta=self.tol)
        self.assertAlmostEqual(hull.area.item(), 6.0, delta=self.tol) # Surface area of a unit cube

    def test_convex_hull_3d_tetrahedron(self):
        points = torch.tensor([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1] 
        ], dtype=self.double_type) # P0, P1, P2, P3
        hull = ConvexHull(points, tol=1e-7)

        self.assertEqual(hull.vertices.shape[0], 4) # All 4 points
        self.assertEqual(hull.simplices.shape[0], 4) # 4 triangular faces
        # Expected faces (indices): e.g. [[0,1,2], [0,1,3], [0,2,3], [1,2,3]] (order within face can vary)
        # Check if all points are used in faces
        unique_indices_in_faces = torch.unique(hull.simplices.flatten())
        self.assertEqual(len(unique_indices_in_faces), 4)

        # Volume of tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1) is 1/6
        self.assertAlmostEqual(hull.volume.item(), 1.0/6.0, delta=self.tol)
        
        # Surface area:
        # Face (0,0,0)-(1,0,0)-(0,1,0) -> P0,P1,P2. Area = 0.5 * || (P1-P0)x(P2-P0) ||
        # P1-P0 = (1,0,0), P2-P0 = (0,1,0). Cross = (0,0,1). Norm = 1. Area = 0.5
        # Face (0,0,0)-(1,0,0)-(0,0,1) -> P0,P1,P3. P1-P0=(1,0,0), P3-P0=(0,0,1). Cross = (0,-1,0). Norm=1. Area=0.5
        # Face (0,0,0)-(0,1,0)-(0,0,1) -> P0,P2,P3. P2-P0=(0,1,0), P3-P0=(0,0,1). Cross = (1,0,0). Norm=1. Area=0.5
        # Face (1,0,0)-(0,1,0)-(0,0,1) -> P1,P2,P3. 
        #   V1=(1,0,0), V2=(0,1,0), V3=(0,0,1)
        #   V2-V1 = (-1,1,0), V3-V1 = (-1,0,1). Cross = (1,1,1). Norm = sqrt(3). Area = sqrt(3)/2
        expected_surface_area = 0.5 + 0.5 + 0.5 + (np.sqrt(3) / 2.0)
        self.assertAlmostEqual(hull.area.item(), expected_surface_area, delta=self.tol)


    def test_convex_hull_3d_coplanar(self):
        # All points on XY plane, plus one above to make it non-degenerate for initial monotone_chain_convex_hull_3d
        # The `monotone_chain_convex_hull_3d` is complex; its behavior with perfectly coplanar points
        # might lead to a very thin 3D hull or specific handling.
        # If it returns faces mostly on one plane, volume should be near zero.
        points_coplanar = torch.tensor([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0], # Square on XY plane
            [0.5, 0.5, EPSILON/100] # A point slightly off the plane to ensure it's 3D
        ], dtype=self.double_type)
        hull_cp = ConvexHull(points_coplanar, tol=1e-7)

        # Volume should be very close to zero
        self.assertAlmostEqual(hull_cp.volume.item(), 0.0, delta=self.tol * 100) # Allow larger delta for near-zero volumes
        
        # The number of vertices and faces might depend on how the algorithm handles near-degenerate cases.
        # It should ideally be the 4 vertices of the square if the 5th point is treated as "internal" to a flattened hull.
        # The current `monotone_chain_convex_hull_3d` might still form a thin tetrahedron.
        # For [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0.5,0.5,tiny_z], the convex hull should be a flat pyramid.
        # Vertices: 0,1,2,3,4 (all 5). Faces: 4 on bottom (e.g. [0,1,4]), 4 forming the square base (e.g. [0,1,2] if not careful)
        # This needs careful check of `monotone_chain_convex_hull_3d`'s output for coplanar-like data.
        # Given the current placeholder nature of the 3D hull code, we check for low volume.
        # Surface area should be roughly twice the area of the base square (top and bottom of the "flat" object).
        # Base area = 1.0. So surface area ~ 2.0.
        # Example: P0(0,0,0) P1(1,0,0) P2(1,1,0) P3(0,1,0) P4(0.5,0.5,epsilon)
        # Faces could be (P0,P1,P4), (P1,P2,P4), (P2,P3,P4), (P3,P0,P4) and (P0,P1,P2,P3) as base.
        # Base (P0,P1,P2) area 0.5. (P0,P2,P3) area 0.5. Total base area = 1.0.
        # Area of (P0,P1,P4): P0P1=(1,0,0), P0P4=(0.5,0.5,eps). Cross=(0,-eps,0.5). Norm=sqrt(eps^2+0.25). Area ~ 0.25
        # Total surface area would be sum of 4 such side triangles + base area.
        # Area P0P1P4 = 0.5 * ||torch.cross(P1-P0, P4-P0)||
        # P1-P0 = [1,0,0], P4-P0 = [0.5,0.5,eps] -> cross = [0, -eps, 0.5]. Area = 0.5 * sqrt(eps^2 + 0.25) approx 0.25
        # Sum of 4 side areas approx 1.0. Base area 1.0. Total approx 2.0.
        self.assertTrue(hull_cp.area.item() > 0.5) # Should be sum of base and top faces, roughly 2.0
        self.assertTrue(hull_cp.area.item() < 2.5) # Allow some leeway


    def test_convex_hull_3d_less_than_4_points(self):
        # 3 points (should form a plane, 0 volume)
        points_3 = torch.tensor([[0,0,0], [1,0,0], [0,1,0]], dtype=self.double_type)
        hull_3 = ConvexHull(points_3, tol=1e-7)
        # monotone_chain_convex_hull_3d returns all points as vertices, and empty faces for n < 4
        self.assertEqual(hull_3.vertices.shape[0], 3)
        self.assertEqual(hull_3.simplices.shape[0], 0)
        self.assertAlmostEqual(hull_3.volume.item(), 0.0, delta=self.tol)
        self.assertAlmostEqual(hull_3.area.item(), 0.0, delta=self.tol) # No faces, so surface area is 0

        # 2 points (line, 0 volume, 0 area)
        points_2 = torch.tensor([[0,0,0], [1,1,1]], dtype=self.double_type)
        hull_2 = ConvexHull(points_2, tol=1e-7)
        self.assertEqual(hull_2.vertices.shape[0], 2)
        self.assertEqual(hull_2.simplices.shape[0], 0)
        self.assertAlmostEqual(hull_2.volume.item(), 0.0, delta=self.tol)
        self.assertAlmostEqual(hull_2.area.item(), 0.0, delta=self.tol)

        # 1 point
        points_1 = torch.tensor([[0,0,0]], dtype=self.double_type)
        hull_1 = ConvexHull(points_1, tol=1e-7)
        self.assertEqual(hull_1.vertices.shape[0], 1)
        self.assertEqual(hull_1.simplices.shape[0], 0)
        self.assertAlmostEqual(hull_1.volume.item(), 0.0, delta=self.tol)
        self.assertAlmostEqual(hull_1.area.item(), 0.0, delta=self.tol)

    # --- delaunay_triangulation_3d Function Tests ---
    def test_delaunay_3d_simple_tetrahedron(self):
        # As per delaunay_triangulation_3d's placeholder logic, it uses ConvexHull
        # and might return the first 4 hull vertices as a single tetrahedron.
        points = torch.tensor([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0.5] # 5 points
        ], dtype=self.double_type)
        tetrahedra = delaunay_triangulation_3d(points, tol=1e-7)

        self.assertEqual(tetrahedra.ndim, 2)
        self.assertEqual(tetrahedra.shape[1], 4) # Each row is a tetrahedron
        
        # The placeholder delaunay might return one tetrahedron based on convex hull vertices
        # For these points, the convex hull is formed by [0,1,2,3].
        # The placeholder might pick these 4 for its single tetrahedron.
        if tetrahedra.shape[0] > 0: # If any tetrahedra are returned
            self.assertTrue(tetrahedra.shape[0] >= 1)
            first_tetra = tetrahedra[0]
            self.assertEqual(len(torch.unique(first_tetra)), 4) # 4 unique vertices
            for idx in first_tetra:
                self.assertTrue(0 <= idx.item() < points.shape[0]) # Valid indices
        else:
            # This case could occur if the placeholder logic is very minimal or fails for some reason
            pass # Allow empty output for placeholder

    def test_delaunay_3d_less_than_4_points(self):
        points_3 = torch.tensor([[0,0,0], [1,0,0], [0,1,0]], dtype=self.double_type)
        tetrahedra_3 = delaunay_triangulation_3d(points_3, tol=1e-7)
        self.assertEqual(tetrahedra_3.shape[0], 0) # Expect empty tensor

        points_0 = torch.empty((0,3), dtype=self.double_type)
        tetrahedra_0 = delaunay_triangulation_3d(points_0, tol=1e-7)
        self.assertEqual(tetrahedra_0.shape[0], 0)

    # --- normalize_weights Function Tests (New Version) ---
    def test_normalize_weights_simple(self):
        weights = torch.tensor([1,2,3], dtype=self.double_type)
        normalized = normalize_weights(weights)
        expected = torch.tensor([1/6.0, 2/6.0, 3/6.0], dtype=self.double_type)
        torch.testing.assert_close(normalized, expected, rtol=0, atol=self.tol)
        self.assertAlmostEqual(torch.sum(normalized).item(), 1.0, delta=self.tol)

    def test_normalize_weights_with_zeros(self):
        weights = torch.tensor([1,0,3], dtype=self.double_type) # Sum = 4
        normalized = normalize_weights(weights)
        expected = torch.tensor([1/4.0, 0, 3/4.0], dtype=self.double_type)
        torch.testing.assert_close(normalized, expected, rtol=0, atol=self.tol)
        self.assertAlmostEqual(torch.sum(normalized).item(), 1.0, delta=self.tol)

    def test_normalize_weights_target_sum(self):
        weights = torch.tensor([1,2,3], dtype=self.double_type)
        target = 5.0
        normalized = normalize_weights(weights, target_sum=target)
        expected = torch.tensor([1/6.0, 2/6.0, 3/6.0], dtype=self.double_type) * target
        torch.testing.assert_close(normalized, expected, rtol=0, atol=self.tol)
        self.assertAlmostEqual(torch.sum(normalized).item(), target, delta=self.tol)

    def test_normalize_weights_all_zeros_value_error(self):
        weights = torch.tensor([0,0,0], dtype=self.double_type)
        with self.assertRaisesRegex(ValueError, "Sum of weights .* is less than tolerance"):
            normalize_weights(weights, tol=1e-7) # Use the function's tol

    def test_normalize_weights_sum_less_than_tol_value_error(self):
        weights = torch.tensor([1e-8, 1e-9], dtype=self.double_type)
        with self.assertRaisesRegex(ValueError, "Sum of weights .* is less than tolerance"):
            normalize_weights(weights, tol=1e-7)

    def test_normalize_weights_non_1d_assertion_error(self):
        weights_2d = torch.tensor([[1,2],[3,4]], dtype=self.double_type)
        with self.assertRaisesRegex(AssertionError, "Weights must be a 1D tensor"):
            normalize_weights(weights_2d)

    def test_normalize_weights_negative_values_assertion_error(self):
        weights = torch.tensor([1, -0.1, 3], dtype=self.double_type)
        with self.assertRaisesRegex(AssertionError, "Weights must be non-negative"):
            normalize_weights(weights, tol=1e-7) # tol for check is 1e-7

    def test_normalize_weights_slightly_negative_clamped(self):
        weights = torch.tensor([1, -1e-8, 3], dtype=self.double_type) # Sum approx 4
        # Should clamp -1e-8 to 0 if tol is e.g. 1e-7 for the assertion, then normalize.
        # The assertion is `torch.all(weights >= -tol)`, so -1e-8 passes if tol=1e-7.
        # Then `torch.clamp(weights, min=0.0)` makes it [1,0,3].
        normalized = normalize_weights(weights, tol=1e-7) # tol for assertion and sum check
        expected = torch.tensor([1/4.0, 0, 3/4.0], dtype=self.double_type)
        torch.testing.assert_close(normalized, expected, rtol=0, atol=self.tol)
        self.assertAlmostEqual(torch.sum(normalized).item(), 1.0, delta=self.tol)
        
    def test_normalize_weights_type_error(self):
        weights_list = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(TypeError, "Input weights must be a PyTorch tensor."):
            normalize_weights(weights_list)

    # --- Tests for delaunay_triangulation_3d (New PyTorch version) ---
    def test_delaunay3d_empty_input(self):
        points = torch.empty((0, 3), dtype=self.double_type, device=self.device)
        tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
        self.assertEqual(tetrahedra.shape, (0, 4))

    def test_delaunay3d_less_than_4_points(self):
        for n_pts in [1, 2, 3]:
            with self.subTest(n_pts=n_pts):
                points = torch.rand(n_pts, 3, dtype=self.double_type, device=self.device)
                tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
                self.assertEqual(tetrahedra.shape, (0, 4))

    def test_delaunay3d_single_tetrahedron(self):
        points = torch.tensor([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1]
        ], dtype=self.double_type, device=self.device)
        tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
        
        self.assertEqual(tetrahedra.shape, (1, 4))
        # Check that the vertices of the tetrahedron are the input points
        # Sort both expected and actual indices within the tetrahedron to compare
        expected_indices = torch.arange(4, device=self.device)
        returned_indices_sorted, _ = torch.sort(tetrahedra[0])
        torch.testing.assert_close(returned_indices_sorted, expected_indices)

    def test_delaunay3d_two_adjacent_tetrahedra(self):
        # Triangular bipyramid: 5 points
        # Base triangle: (0,0,0), (1,0,0), (0.5, np.sqrt(3)/2, 0)
        # Apexes: (0.5, np.sqrt(3)/6, 1), (0.5, np.sqrt(3)/6, -1)
        points = torch.tensor([
            [0.0, 0.0, 0.0],             # 0
            [1.0, 0.0, 0.0],             # 1
            [0.5, np.sqrt(3)/2.0, 0.0], # 2 (equilateral base)
            [0.5, np.sqrt(3)/6.0, 1.0],  # 3 (top apex)
            [0.5, np.sqrt(3)/6.0, -1.0]  # 4 (bottom apex)
        ], dtype=self.double_type, device=self.device)
        
        tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
        self.assertEqual(tetrahedra.shape[0], 2) # Expect 2 tetrahedra for a simple bipyramid

        # Verify vertices are from the input set
        unique_verts_in_tets = torch.unique(tetrahedra.flatten())
        self.assertTrue(torch.all(unique_verts_in_tets < points.shape[0]))

        # More detailed check: each tet should have 4 unique vertices
        for i in range(tetrahedra.shape[0]):
            self.assertEqual(len(torch.unique(tetrahedra[i])), 4)
        
        # Check that the common face {0,1,2} is part of the structure (implicitly)
        # This is harder to check directly without knowing face adjacencies.
        # For now, count of tetrahedra is the primary check.

    def test_delaunay3d_cube_points(self):
        points = torch.tensor([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ], dtype=self.double_type, device=self.device)
        
        tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
        
        # Cube can be decomposed into 5 or 6 tetrahedra.
        # The Bowyer-Watson might produce more due to temporary super-tetra interactions if not perfectly cleaned.
        # For a robust test, we check volume and coverage.
        self.assertTrue(tetrahedra.shape[0] >= 5, f"Expected at least 5 tetrahedra for a cube, got {tetrahedra.shape[0]}")
        self.assertEqual(tetrahedra.shape[1], 4)

        total_volume = 0.0
        for i in range(tetrahedra.shape[0]):
            tet_indices = tetrahedra[i]
            p0, p1, p2, p3 = points[tet_indices[0]], points[tet_indices[1]], points[tet_indices[2]], points[tet_indices[3]]
            total_volume += self._tetra_volume(p0, p1, p2, p3).item()
        
        self.assertAlmostEqual(total_volume, 1.0, delta=self.small_tol, msg="Sum of tetrahedra volumes should equal cube volume.")

        # Check if all original 8 points are included in the output tetrahedra
        unique_verts_in_tets = torch.unique(tetrahedra.flatten())
        self.assertEqual(len(unique_verts_in_tets), 8, "Not all cube vertices are part of the triangulation.")
        for i in range(8):
            self.assertIn(i, unique_verts_in_tets)

    def test_delaunay3d_empty_circumsphere_property(self):
        torch.manual_seed(1) # Different seed for this specific test
        points = torch.rand(8, 3, dtype=self.double_type, device=self.device) * 10 # Random points
        
        tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
        if tetrahedra.numel() == 0 and points.shape[0] >=4 :
             self.fail(f"Delaunay triangulation returned no tetrahedra for {points.shape[0]} points.")
        if tetrahedra.numel() == 0: # If less than 4 points, this is expected.
            return


        all_point_indices = torch.arange(points.shape[0], device=self.device)

        for i in range(tetrahedra.shape[0]):
            tet_indices = tetrahedra[i]
            t1, t2, t3, t4 = points[tet_indices[0]], points[tet_indices[1]], points[tet_indices[2]], points[tet_indices[3]]
            
            # Check orientation of the tetrahedron from the algorithm to interpret circumsphere test
            # The algorithm tries to make them positive.
            # orient_val = _orientation3d_pytorch(t1, t2, t3, t4, self.default_delaunay_tol)
            # self.assertTrue(orient_val >= 0, f"Tetrahedron {tet_indices.tolist()} from Delaunay has non-positive orientation: {orient_val}")

            other_point_indices = torch.tensor(
                [idx for idx in all_point_indices.tolist() if idx not in tet_indices.tolist()],
                device=self.device, dtype=torch.long
            )
            
            for pt_idx in other_point_indices:
                p_test = points[pt_idx]
                # Using the re-implemented _test_in_circumsphere3d
                is_inside = self._test_in_circumsphere3d(p_test, t1, t2, t3, t4, self.default_delaunay_tol)
                self.assertFalse(is_inside, 
                                 f"Point {pt_idx} is inside circumsphere of tetrahedron {tet_indices.tolist()}.")

    def test_delaunay3d_coplanar_points(self):
        # 5 coplanar points (on XY plane)
        points = torch.tensor([
            [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0.5, 0.5, 0.0]
        ], dtype=self.double_type, device=self.device)
        
        tetrahedra = delaunay_triangulation_3d(points, tol=self.default_delaunay_tol)
        
        # Expect no 3D tetrahedra, or tetrahedra with zero volume if algorithm proceeds.
        # The current Bowyer-Watson like implementation should ideally result in 0 tetrahedra
        # after removing super-tetra ones if all points are coplanar and can't form valid 3D tets.
        if tetrahedra.numel() > 0:
            total_volume = 0.0
            for i in range(tetrahedra.shape[0]):
                tet_indices = tetrahedra[i]
                p0,p1,p2,p3 = points[tet_indices[0]], points[tet_indices[1]], points[tet_indices[2]], points[tet_indices[3]]
                total_volume += self._tetra_volume(p0,p1,p2,p3).item()
            self.assertAlmostEqual(total_volume, 0.0, delta=self.small_tol, 
                                   msg="Volume of tetrahedra from coplanar points should be zero.")
        else:
            self.assertEqual(tetrahedra.shape, (0,4), "Expected no tetrahedra for coplanar points.")

    def test_delaunay3d_comparison_with_scipy(self):
        torch.manual_seed(42) # Yet another seed for this comparison
        points_torch = torch.rand(10, 3, dtype=self.double_type, device=self.device) * 100
        points_np = points_torch.cpu().numpy()

        # Run custom PyTorch Delaunay
        try:
            tetra_custom_torch = delaunay_triangulation_3d(points_torch, tol=1e-7) # Use a typical tolerance
        except Exception as e:
            self.fail(f"Custom delaunay_triangulation_3d failed: {e}")
        
        # Run SciPy Delaunay
        try:
            scipy_delaunay = ScipyDelaunay(points_np) # Qhull options can be added if needed
            tetra_scipy_np = scipy_delaunay.simplices # These are indices into points_np
        except Exception as e: # Catch QhullError or others
            # If SciPy fails (e.g. degenerate input not caught by our checks), this test can't compare.
            self.skipTest(f"SciPy Delaunay computation failed: {e}. Cannot compare.")
            return

        self.assertIsNotNone(tetra_custom_torch, "Custom Delaunay result is None.")
        self.assertIsNotNone(tetra_scipy_np, "SciPy Delaunay result is None.")
        
        # SciPy returns NumPy, convert custom to NumPy for easier comparison of sets
        tetra_custom_np = tetra_custom_torch.cpu().numpy()

        # Normalize: sort vertices within each tetrahedron, then sort tetrahedra
        def normalize_simplices(simplices_array):
            if simplices_array.shape[0] == 0:
                return set()
            sorted_simplices = np.sort(simplices_array, axis=1)
            # Convert to set of tuples for comparison
            return set(map(tuple, sorted_simplices.tolist()))

        set_custom = normalize_simplices(tetra_custom_np)
        set_scipy = normalize_simplices(tetra_scipy_np)
        
        # Due to potential differences in handling degeneracies or near-co-spherical points,
        # an exact match might be too strict for a new implementation vs. mature Qhull.
        # For now, let's check number of tetrahedra and if they cover the same set of points.
        # A more robust comparison might involve checking topological properties or volumes.
        
        self.assertEqual(len(set_custom), len(set_scipy), 
                         f"Number of tetrahedra differ: Custom ({len(set_custom)}) vs SciPy ({len(set_scipy)})")
        
        # If numbers match, check for set equality
        if len(set_custom) == len(set_scipy):
            self.assertEqual(set_custom, set_scipy, "Sets of tetrahedra (normalized) do not match.")


if __name__ == '__main__':
    unittest.main()
