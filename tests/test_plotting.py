import unittest
import numpy as np
import matplotlib.pyplot as plt

try:
    from reconlib.plotting import (
        plot_voronoi_diagram_2d,
        plot_density_weights_2d,
        plot_voronoi_diagram_3d_slice,
        plot_density_weights_3d_slice
    )
    # Assuming these might be the other existing functions based on previous context
    # from reconlib.plotting import plot_phase_image, plot_unwrapped_phase_map, plot_b0_field_map
    HAS_RECONLIB_PLOTTING = True
except ImportError:
    HAS_RECONLIB_PLOTTING = False

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
