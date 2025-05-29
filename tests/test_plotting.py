import unittest
import numpy as np
import matplotlib.pyplot as plt

try:
    from reconlib.plotting import (
        plot_voronoi_diagram_2d,
        plot_density_weights_2d,
        plot_voronoi_diagram_3d_slice,
        plot_density_weights_3d_slice,
        # Newly added functions
        plot_3d_hull,
        plot_3d_voronoi_with_hull,
        plot_3d_delaunay,
        plot_voronoi_kspace 
    )
    from reconlib.voronoi_utils import ConvexHull, delaunay_triangulation_3d, EPSILON
    import torch # Ensure torch is imported for test data generation
    # Assuming these might be the other existing functions based on previous context
    # from reconlib.plotting import plot_phase_image, plot_unwrapped_phase_map, plot_b0_field_map
    HAS_RECONLIB_PLOTTING = True
    HAS_VORONOI_UTILS_FOR_PLOTTING = True
except ImportError:
    HAS_RECONLIB_PLOTTING = False
    HAS_VORONOI_UTILS_FOR_PLOTTING = False # For skipping new tests if utils are missing

# Mocking plt.show() for non-interactive testing - not strictly needed with Agg backend,
# but good if any function internally calls it. The instructions include this import.
from unittest.mock import patch


class TestPlottingBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure matplotlib backend is non-interactive for tests
        try:
            plt.switch_backend('Agg')
        except Exception as e:
            print(f"Could not switch matplotlib backend to Agg: {e}")
            # Depending on strictness, you might want to skip tests if backend switch fails
            # For now, we'll proceed, assuming it might already be non-interactive or tests might still pass.


    def tearDown(self):
        plt.close('all') # Close all figures after each test


@unittest.skipIf(not HAS_RECONLIB_PLOTTING, "reconlib.plotting module not found or functions missing.")
class TestPlotVoronoi2D(TestPlottingBase):
    def test_plot_voronoi_diagram_2d_runs(self):
        points = np.random.rand(5, 2)
        ax = plot_voronoi_diagram_2d(points)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_voronoi_diagram_2d_with_ax(self):
        points = np.random.rand(5, 2)
        fig, expected_ax = plt.subplots()
        returned_ax = plot_voronoi_diagram_2d(points, ax=expected_ax)
        self.assertIs(returned_ax, expected_ax)

    def test_plot_voronoi_diagram_2d_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_2d(np.random.rand(5, 3)) # Wrong dimension
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_2d(np.random.rand(2, 2)) # Not enough points
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_2d("not_an_array")


@unittest.skipIf(not HAS_RECONLIB_PLOTTING, "reconlib.plotting module not found or functions missing.")
class TestPlotDensity2D(TestPlottingBase):
    def test_plot_density_weights_2d_runs(self):
        weights = np.random.rand(10, 10)
        ax = plot_density_weights_2d(weights)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_density_weights_2d_with_ax(self):
        weights = np.random.rand(10, 10)
        fig, expected_ax = plt.subplots()
        returned_ax = plot_density_weights_2d(weights, ax=expected_ax)
        self.assertIs(returned_ax, expected_ax)

    def test_plot_density_weights_2d_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_density_weights_2d(np.random.rand(10)) # 1D array
        with self.assertRaises(ValueError):
            plot_density_weights_2d(np.random.rand(10,10,10)) # 3D array
        with self.assertRaises(ValueError):
            plot_density_weights_2d("not_an_array")


@unittest.skipIf(not HAS_RECONLIB_PLOTTING, "reconlib.plotting module not found or functions missing.")
class TestPlotVoronoi3DSlice(TestPlottingBase):
    def test_plot_voronoi_diagram_3d_slice_runs(self):
        points = np.random.rand(10, 3)
        for axis in [0, 1, 2]:
            with self.subTest(slice_axis=axis):
                ax = plot_voronoi_diagram_3d_slice(points, slice_axis=axis, slice_coord=0.5)
                self.assertIsInstance(ax, plt.Axes)

    def test_plot_voronoi_diagram_3d_slice_with_ax(self):
        points = np.random.rand(10, 3)
        fig, expected_ax = plt.subplots()
        returned_ax = plot_voronoi_diagram_3d_slice(points, ax=expected_ax)
        self.assertIs(returned_ax, expected_ax)

    def test_plot_voronoi_diagram_3d_slice_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_3d_slice(np.random.rand(10, 2)) # Wrong dimension
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_3d_slice(np.random.rand(3, 3)) # Not enough points
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_3d_slice(np.random.rand(10, 3), slice_axis=3) # Invalid slice_axis
        with self.assertRaises(ValueError):
            plot_voronoi_diagram_3d_slice("not_an_array")


@unittest.skipIf(not HAS_RECONLIB_PLOTTING, "reconlib.plotting module not found or functions missing.")
class TestPlotDensity3DSlice(TestPlottingBase):
    def test_plot_density_weights_3d_slice_runs(self):
        weights_volume = np.random.rand(5, 5, 5)
        for axis in [0, 1, 2]:
            with self.subTest(slice_axis=axis):
                ax = plot_density_weights_3d_slice(weights_volume, slice_axis=axis, slice_index=2)
                self.assertIsInstance(ax, plt.Axes)

    def test_plot_density_weights_3d_slice_with_ax(self):
        weights_volume = np.random.rand(5, 5, 5)
        fig, expected_ax = plt.subplots()
        returned_ax = plot_density_weights_3d_slice(weights_volume, ax=expected_ax)
        self.assertIs(returned_ax, expected_ax)

    def test_plot_density_weights_3d_slice_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_density_weights_3d_slice(np.random.rand(5, 5)) # 2D array
        with self.assertRaises(ValueError):
            plot_density_weights_3d_slice(np.random.rand(5,5,5), slice_axis=3) # Invalid slice_axis
        with self.assertRaises(ValueError): # Changed from IndexError based on typical implementation
            plot_density_weights_3d_slice(np.random.rand(5,5,5), slice_index=10)
        with self.assertRaises(ValueError):
            plot_density_weights_3d_slice("not_an_array")


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(not HAS_RECONLIB_PLOTTING or not HAS_VORONOI_UTILS_FOR_PLOTTING, 
                 "reconlib.plotting or reconlib.voronoi_utils functions for new tests missing.")
class TestNewPlottingFunctions(TestPlottingBase):
    def setUp(self):
        super().setUp() # Call parent setUp if it has any
        torch.manual_seed(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype_real = torch.float32
        self.dtype_complex = torch.complex64


    def test_plot_3d_hull_execution(self):
        points = torch.rand(10, 3, device=self.device, dtype=self.dtype_real)
        # Compute ConvexHull using reconlib.voronoi_utils.ConvexHull
        # This hull computation is now PyTorch-native
        hull = ConvexHull(points, tol=EPSILON) # Use EPSILON from voronoi_utils for tol

        # plot_3d_hull expects simplices (faces) and points.
        # The 'vertices' argument in plot_3d_hull is for API consistency but not strictly used if simplices index into original points.
        # hull.vertices are the indices of points on the hull.
        # hull.simplices are the faces of the hull, indexing into the original 'points' tensor.
        if hull.simplices is not None and hull.simplices.numel() > 0:
            plot_3d_hull(points, hull.vertices, hull.simplices, show_points=True, show_hull=True)
            plt.close('all') # Explicitly close after plot
        else:
            self.skipTest("Skipping plot_3d_hull test as ConvexHull computation resulted in no simplices (e.g., degenerate input).")


    def test_plot_3d_voronoi_with_hull_execution(self):
        points = torch.rand(10, 3, device=self.device, dtype=self.dtype_real) 
        # Compute overall convex hull
        overall_hull = ConvexHull(points, tol=EPSILON)
        
        if overall_hull.simplices is not None and overall_hull.simplices.numel() > 0:
            plot_3d_voronoi_with_hull(points, overall_hull.simplices, 
                                      show_points=True, show_voronoi=True, show_hull=True)
            plt.close('all')
        else:
            self.skipTest("Skipping plot_3d_voronoi_with_hull test as overall ConvexHull resulted in no simplices.")


    def test_plot_3d_delaunay_execution(self):
        points = torch.rand(10, 3, device=self.device, dtype=self.dtype_real) # Need at least 4 for tetrahedra
        
        # Generate tetrahedra using delaunay_triangulation_3d (placeholder in voronoi_utils)
        # The placeholder delaunay_triangulation_3d might return minimal/empty tetrahedra depending on its internal logic.
        tetrahedra = delaunay_triangulation_3d(points, tol=EPSILON)
        
        # Compute ConvexHull for the points (optional for the plot function, but good for testing that path)
        convex_hull_obj = ConvexHull(points, tol=EPSILON)

        if tetrahedra.numel() > 0 : # Only plot if some tetrahedra were generated
            plot_3d_delaunay(points, tetrahedra, convex_hull_obj, 
                             show_points=True, show_tetrahedra=True, show_hull=True)
            plt.close('all')
        else:
            # If delaunay_triangulation_3d returns empty, we can still test plotting without tetrahedra
            plot_3d_delaunay(points, tetrahedra, convex_hull_obj, 
                             show_points=True, show_tetrahedra=False, show_hull=True)
            plt.close('all')
            # self.skipTest("Skipping plot_3d_delaunay test as delaunay_triangulation_3d returned no tetrahedra.")


    def test_plot_voronoi_kspace_execution(self):
        kspace_points = torch.rand(20, 2, device=self.device, dtype=self.dtype_real) * 2 - 1 # Scale to [-1, 1]
        weights = torch.rand(20, device=self.device, dtype=self.dtype_real)
        bounds_min = kspace_points.min(dim=0).values - 0.1
        bounds_max = kspace_points.max(dim=0).values + 0.1
        bounds = torch.stack([bounds_min, bounds_max])

        # Test with all arguments
        plot_voronoi_kspace(kspace_points, weights=weights, bounds=bounds, title='Test K-space Voronoi (Full)')
        plt.close('all')

        # Test with no weights and no bounds
        plot_voronoi_kspace(kspace_points, weights=None, bounds=None, title='Test K-space Voronoi (Minimal)')
        plt.close('all')

        # Test with only weights
        plot_voronoi_kspace(kspace_points, weights=weights, bounds=None, title='Test K-space Voronoi (Weights Only)')
        plt.close('all')

        # Test with only bounds
        plot_voronoi_kspace(kspace_points, weights=None, bounds=bounds, title='Test K-space Voronoi (Bounds Only)')
        plt.close('all')
