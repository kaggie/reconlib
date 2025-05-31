import torch
import numpy as np
import unittest
from typing import List, Dict, Any, Optional, Tuple
import traceback

from reconlib.modalities.pet.voronoi_reconstructor import VoronoiPETReconstructor2D

try:
    from reconlib.voronoi.geometry_core import EPSILON as GEOMETRY_EPSILON
except ImportError:
    if 'GEOMETRY_EPSILON' not in globals(): GEOMETRY_EPSILON = 1e-7


class TestVoronoiPETReconstructor2D(unittest.TestCase):
    def setUp(self):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.default_epsilon = GEOMETRY_EPSILON
        self.verbose = False # Add verbose flag for test-specific prints

    def test_init(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, verbose=False)
        self.assertIsNotNone(recon)
        self.assertEqual(recon.num_iterations, 10)
        self.assertEqual(recon.positivity_constraint, True)

    def test_gen_points_valid(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=self.default_epsilon, verbose=False)
        points = torch.tensor([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]], dtype=torch.float32, device=self.device)
        is_invalid, status = recon._validate_generator_points_2d(points)
        self.assertFalse(is_invalid, f"Valid points flagged as invalid: {status}")
        self.assertIn("success", status.lower())

    def test_gen_points_insufficient(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, verbose=False)
        points = torch.tensor([[0.0,0.0],[1.0,1.0]], dtype=torch.float32, device=self.device)
        is_invalid, status = recon._validate_generator_points_2d(points)
        self.assertTrue(is_invalid)
        self.assertIn("insufficient", status.lower())

    def test_gen_points_duplicates(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=self.default_epsilon, verbose=False)
        points = torch.tensor([[0.0,0.0],[1.0,1.0],[0.0,0.00000001],[2.0,2.0]], dtype=torch.float32, device=self.device)
        is_invalid, status = recon._validate_generator_points_2d(points)
        self.assertTrue(is_invalid)
        self.assertIn("duplicate", status.lower())

    def test_gen_points_collinear(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=self.default_epsilon, verbose=False)
        points = torch.tensor([[0.0,0.0],[1.0,1.0],[2.0,2.0],[3.0,3.0]], dtype=torch.float32, device=self.device)
        is_invalid, status = recon._validate_generator_points_2d(points)
        self.assertTrue(is_invalid)
        self.assertIn("degenerate", status.lower())

    def test_compute_voronoi_diagram_valid_points(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, verbose=False)
        points = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]], dtype=torch.float32, device=self.device)
        cells_verts, unique_verts, status = recon._compute_voronoi_diagram_2d(points)
        self.assertIsNotNone(cells_verts, f"Voronoi computation failed for valid points: {status}")
        if cells_verts is not None:
             self.assertEqual(len(cells_verts), points.shape[0])
        self.assertIn("success", status.lower())

    def test_validate_voronoi_cells_valid(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, verbose=False)
        valid_cell = [torch.tensor([0.,0.], device=self.device), torch.tensor([1.,0.], device=self.device), torch.tensor([0.5,0.5], device=self.device)]
        is_invalid, status = recon._validate_voronoi_cells_2d([valid_cell])
        self.assertFalse(is_invalid, f"Valid cell flagged as invalid: {status}")
        self.assertIn("success", status.lower())

    def test_validate_voronoi_cells_degenerate_cell(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=1e-7, verbose=True)
        degenerate_cell_line = [torch.tensor([0.,0.], device=self.device), torch.tensor([1.,1.], device=self.device), torch.tensor([2.,2.], device=self.device)]
        is_invalid, status = recon._validate_voronoi_cells_2d([degenerate_cell_line])
        self.assertTrue(is_invalid, "Degenerate cell (line) not flagged.")
        self.assertTrue("area" in status.lower() or "degenerate" in status.lower() or "vertices form a line or point" in status.lower())

        degenerate_cell_point = [torch.tensor([0.,0.], device=self.device), torch.tensor([0.,1e-8], device=self.device), torch.tensor([1e-8,0.], device=self.device)]
        is_invalid_pt, status_pt = recon._validate_voronoi_cells_2d([degenerate_cell_point])
        self.assertTrue(is_invalid_pt, "Degenerate cell (point-like) not flagged.")
        self.assertTrue("area" in status_pt.lower() or "degenerate" in status_pt.lower() or "vertices form a line or point" in status_pt.lower())

    def test_line_segment_intersection(self):
        p1 = torch.tensor([0.,0.], device=self.device); p2 = torch.tensor([2.,2.], device=self.device)
        p3 = torch.tensor([0.,2.], device=self.device); p4 = torch.tensor([2.,0.], device=self.device)
        intersect = VoronoiPETReconstructor2D._line_segment_intersection_2d(p1,p2,p3,p4, self.default_epsilon)
        self.assertIsNotNone(intersect)
        if intersect is not None:
            self.assertTrue(torch.allclose(intersect, torch.tensor([1.,1.], device=self.device), atol=1e-6))

        p5 = torch.tensor([3.,3.], device=self.device)
        no_intersect = VoronoiPETReconstructor2D._line_segment_intersection_2d(p1,p2,p3,p5, self.default_epsilon)
        self.assertIsNone(no_intersect)

        p6 = torch.tensor([1.,-1.], device=self.device); p7 = torch.tensor([3.,1.], device=self.device)
        parallel_intersect = VoronoiPETReconstructor2D._line_segment_intersection_2d(p1,p2,p6,p7, self.default_epsilon)
        self.assertIsNone(parallel_intersect)

    def test_is_point_inside_polygon(self):
        poly = torch.tensor([[0,0],[1,0],[1,1],[0,1]], dtype=torch.float32, device=self.device)
        pt_inside = torch.tensor([0.5,0.5], device=self.device)
        pt_outside = torch.tensor([2.0,0.5], device=self.device)
        pt_on_edge = torch.tensor([0.5,0.0], device=self.device)
        pt_vertex = torch.tensor([1.0,1.0], device=self.device)

        self.assertTrue(VoronoiPETReconstructor2D._is_point_inside_polygon_2d(pt_inside, poly, self.default_epsilon))
        self.assertFalse(VoronoiPETReconstructor2D._is_point_inside_polygon_2d(pt_outside, poly, self.default_epsilon))
        self.assertTrue(VoronoiPETReconstructor2D._is_point_inside_polygon_2d(pt_on_edge, poly, self.default_epsilon))
        self.assertTrue(VoronoiPETReconstructor2D._is_point_inside_polygon_2d(pt_vertex, poly, self.default_epsilon))

    def test_lor_cell_intersection_simple_hit(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=1e-6, verbose=False)
        lor_p1 = torch.tensor([-1.0, 0.5], device=self.device, dtype=torch.float32)
        lor_p2 = torch.tensor([ 2.0, 0.5], device=self.device, dtype=torch.float32)
        cell_verts_list = [
            torch.tensor([0.0,0.0], device=self.device, dtype=torch.float32), torch.tensor([1.0,0.0], device=self.device, dtype=torch.float32),
            torch.tensor([1.0,1.0], device=self.device, dtype=torch.float32), torch.tensor([0.0,1.0], device=self.device, dtype=torch.float32)
        ]
        expected_length = 1.0
        length = recon._compute_lor_cell_intersection_2d(lor_p1, lor_p2, cell_verts_list)
        self.assertAlmostEqual(length, expected_length, delta=1e-5)

    def test_lor_cell_intersection_miss(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=1e-6, verbose=False)
        lor_p1 = torch.tensor([-1.0, 2.5], device=self.device, dtype=torch.float32)
        lor_p2 = torch.tensor([ 2.0, 2.5], device=self.device, dtype=torch.float32)
        cell_verts_list = [
            torch.tensor([0.0,0.0], device=self.device, dtype=torch.float32), torch.tensor([1.0,0.0], device=self.device, dtype=torch.float32),
            torch.tensor([1.0,1.0], device=self.device, dtype=torch.float32), torch.tensor([0.0,1.0], device=self.device, dtype=torch.float32)
        ]
        expected_length = 0.0
        length = recon._compute_lor_cell_intersection_2d(lor_p1, lor_p2, cell_verts_list)
        self.assertAlmostEqual(length, expected_length, delta=1e-5)

    def test_lor_cell_intersection_lor_endpoint_inside(self):
        recon = VoronoiPETReconstructor2D(device=self.device_str, epsilon=1e-6, verbose=False)
        lor_p1 = torch.tensor([0.5, 0.5], device=self.device, dtype=torch.float32)
        lor_p2 = torch.tensor([2.0, 0.5], device=self.device, dtype=torch.float32)
        cell_verts_list = [
            torch.tensor([0.0,0.0], device=self.device, dtype=torch.float32), torch.tensor([1.0,0.0], device=self.device, dtype=torch.float32),
            torch.tensor([1.0,1.0], device=self.device, dtype=torch.float32), torch.tensor([0.0,1.0], device=self.device, dtype=torch.float32)
        ]
        expected_length = 0.5
        length = recon._compute_lor_cell_intersection_2d(lor_p1, lor_p2, cell_verts_list)
        self.assertAlmostEqual(length, expected_length, delta=1e-5)

    def test_reconstruct_mlem_runs_simple_phantom(self):
        recon = VoronoiPETReconstructor2D(num_iterations=3, device=self.device_str, verbose=True, epsilon=1e-6)

        gen_points = torch.tensor([
            [10.0, 10.0], [20.0, 10.0], [15.0, 20.0]
        ], dtype=torch.float32, device=self.device)
        num_cells = gen_points.shape[0]

        num_angles_test = 8
        num_radial_bins_test = 10
        fov_width_test = 40.0

        angles_np = np.linspace(0, np.pi, num_angles_test, endpoint=False)
        angles_t = torch.tensor(angles_np, device=self.device, dtype=torch.float32)

        radial_step = fov_width_test / num_radial_bins_test
        radial_offsets_t = torch.arange(-fov_width_test/2 + radial_step/2, fov_width_test/2, radial_step, device=self.device)

        test_lor_descriptor = {
            'angles_rad': angles_t, 'radial_offsets': radial_offsets_t, 'fov_width': fov_width_test
        }
        num_total_lors = num_angles_test * num_radial_bins_test

        # Mock internal methods to bypass Voronoi generation issues for this MLEM test
        original_validate_gen = recon._validate_generator_points_2d
        recon._validate_generator_points_2d = lambda x: (False, "Mocked generator validation: OK")

        mock_cells_verts_data = [ # Simple cells for 3 generator points
            [[5,5],[15,5],[15,15],[5,15]],
            [[15,5],[25,5],[25,15],[15,15]],
            [[5,15],[25,15],[25,25],[5,25]]
        ]
        mock_cells_verts = [[torch.tensor(v, dtype=torch.float32, device=self.device) for v in cell] for cell in mock_cells_verts_data]
        dummy_unique_verts = torch.tensor([[0,0]], device=self.device)
        original_compute_voronoi = recon._compute_voronoi_diagram_2d
        recon._compute_voronoi_diagram_2d = lambda x: (mock_cells_verts, dummy_unique_verts, "Mocked Voronoi diagram")

        original_validate_cells = recon._validate_voronoi_cells_2d
        recon._validate_voronoi_cells_2d = lambda x: (False, "Mocked cell validation: OK")

        dummy_system_matrix = torch.rand(num_total_lors, num_cells, device=self.device, dtype=torch.float32) * 0.1
        if num_total_lors > 0 and num_cells > 0:
            # Ensure a specific LOR sees a specific cell to test MLEM update direction
            # e.g. LOR 0 strongly sees cell 0
            dummy_system_matrix[0, 0] = 1.0
            if num_cells > 1 and num_total_lors > 1:
                 dummy_system_matrix[1, min(1, num_cells-1)] = 1.0 # LOR 1 sees cell 1

        original_compute_sm = recon._compute_system_matrix_2d
        recon._compute_system_matrix_2d = lambda ld, cvl: dummy_system_matrix

        sinogram_2d_flat = torch.ones(num_total_lors, device=self.device, dtype=torch.float32) * 0.5
        if num_total_lors > 0 and num_cells > 0:
            sinogram_2d_flat[0] = 10.0 # LOR 0 has high counts (should boost cell 0 activity)
        sinogram_2d_reshaped = sinogram_2d_flat.reshape(num_angles_test, num_radial_bins_test)

        initial_activity = torch.ones(num_cells, device=self.device, dtype=torch.float32)

        result = recon.reconstruct(sinogram_2d_reshaped, gen_points, test_lor_descriptor, initial_estimate=initial_activity)

        recon._validate_generator_points_2d = original_validate_gen
        recon._compute_voronoi_diagram_2d = original_compute_voronoi
        recon._validate_voronoi_cells_2d = original_validate_cells
        recon._compute_system_matrix_2d = original_compute_sm

        if "completed" not in result['status'].lower():
            print("Reconstructor Error Log for MLEM test:")
            for log_item in result.get('error_log', []): print(f"  - {log_item}")
        self.assertIn("completed", result['status'].lower(), f"Reconstruction did not complete successfully: {result['status']}")
        self.assertFalse(result['degenerate_input'])
        self.assertEqual(result['activity'].shape[0], num_cells)
        self.assertTrue(torch.all(result['activity'] >= 0.0).item(), "Activity estimates should be non-negative.")

        if self.verbose:
            print(f"Final activity estimates for MLEM test (mocked SM): {result['activity'].cpu().numpy()}")
            if num_cells > 0 and num_total_lors > 0:
                 self.assertTrue(result['activity'][0] > (result['activity'][1] if num_cells > 1 else -1.0), "Cell 0 activity not highest as expected from mock SM.")

if __name__ == '__main__':
    unittest.main()
