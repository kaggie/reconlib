import unittest
import torch
import numpy as np
from reconlib.modalities.spect.voronoi_reconstructor import VoronoiSPECTReconstructor2D

class TestVoronoiSPECTReconstructor2D(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.default_reconstructor_params = {
            'num_iterations': 3,
            'num_subsets': 2,
            'verbose': False,
            'device': self.device,
            'positivity_constraint': True
        }
        self.reconstructor = VoronoiSPECTReconstructor2D(**self.default_reconstructor_params)

        # Common LOR descriptor for SPECT projection geometry
        self.num_angles = 10
        self.num_radial_bins = 10
        self.fov_width = 100.0
        angles = torch.linspace(0, torch.pi, self.num_angles, endpoint=False, device=self.device)
        radial_offsets = torch.linspace(-self.fov_width / 2, self.fov_width / 2, self.num_radial_bins, device=self.device)
        self.lor_descriptor = {
            'angles_rad': angles,
            'radial_offsets': radial_offsets,
            'fov_width': self.fov_width
        }

    def test_input_validation_degenerate_points(self):
        # Test with too few points
        points_too_few = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device)
        result_too_few = self.reconstructor.reconstruct(
            torch.rand((self.num_angles, self.num_radial_bins), device=self.device),
            points_too_few,
            self.lor_descriptor
        )
        self.assertTrue(result_too_few["degenerate_input"], "Failed for too few points")
        self.assertIn("Insufficient generator points", result_too_few["status"])

        # Test with duplicate points
        points_duplicate = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]], device=self.device)
        result_duplicate = self.reconstructor.reconstruct(
            torch.rand((self.num_angles, self.num_radial_bins), device=self.device),
            points_duplicate,
            self.lor_descriptor
        )
        self.assertTrue(result_duplicate["degenerate_input"], "Failed for duplicate points")
        self.assertIn("Duplicate or near-duplicate", result_duplicate["status"])

        # Test with collinear points
        points_collinear = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=self.device)
        result_collinear = self.reconstructor.reconstruct(
            torch.rand((self.num_angles, self.num_radial_bins), device=self.device),
            points_collinear,
            self.lor_descriptor
        )
        self.assertTrue(result_collinear["degenerate_input"] or "degenerate" in result_collinear["status"].lower() or "failed" in result_collinear["status"].lower(),
                        f"Failed for collinear points. Status: {result_collinear['status']}")

    def test_basic_osem_reconstruction(self):
        # Define generator points for Voronoi cells
        # Cell 0: "Hot" region
        # Others: Background
        generator_points = torch.tensor([
            [self.fov_width * 0.5, self.fov_width * 0.5],  # Center for the "hot" cell
            [self.fov_width * 0.25, self.fov_width * 0.25],
            [self.fov_width * 0.75, self.fov_width * 0.25],
            [self.fov_width * 0.25, self.fov_width * 0.75],
            [self.fov_width * 0.75, self.fov_width * 0.75],
        ], device=self.device)
        num_cells = generator_points.shape[0]

        # True activities for each cell
        hot_cell_activity = 10.0
        background_activity = 1.0 # OSEM works better with non-zero background
        true_cell_activities = torch.full((num_cells,), background_activity, device=self.device, dtype=torch.float32)
        # Assume cell 0 (corresponding to generator_points[0]) is the hot cell.
        true_cell_activities[0] = hot_cell_activity

        # Create a temporary reconstructor instance for system matrix calculation utilities
        sim_reconstructor_params = {**self.default_reconstructor_params, 'num_iterations': 1, 'num_subsets':1}
        sim_reconstructor = VoronoiSPECTReconstructor2D(**sim_reconstructor_params)

        # Validate points and compute Voronoi diagram for system matrix
        is_gen_invalid, gen_status = sim_reconstructor._validate_generator_points_2d(generator_points)
        self.assertFalse(is_gen_invalid, f"Generator points validation failed for simulation: {gen_status}")

        cells_verts, _, vor_status = sim_reconstructor._compute_voronoi_diagram_2d(generator_points)
        self.assertIsNotNone(cells_verts, f"Voronoi diagram computation failed for simulation: {vor_status}")

        is_cell_invalid, cell_status = sim_reconstructor._validate_voronoi_cells_2d(cells_verts)
        self.assertFalse(is_cell_invalid, f"Voronoi cell validation failed for simulation: {cell_status}")

        # Manually compute system matrix
        # NOTE: This uses a simplified geometric system matrix. For realistic SPECT,
        # this should incorporate attenuation and collimator-detector response.
        system_matrix = sim_reconstructor._compute_system_matrix_2d(self.lor_descriptor, cells_verts)
        self.assertIsNotNone(system_matrix, "System matrix computation failed for simulation.")

        # Generate sinogram using the reconstructor's own forward projection (no noise for simplicity)
        sinogram_flat = sim_reconstructor._forward_project_2d(true_cell_activities, system_matrix)
        sinogram_2d = sinogram_flat.reshape(self.num_angles, self.num_radial_bins)

        # Initialize the reconstructor for the actual test
        test_reconstructor_params = {**self.default_reconstructor_params} # Use setUp defaults
        test_reconstructor = VoronoiSPECTReconstructor2D(**test_reconstructor_params)

        # Perform reconstruction
        result = test_reconstructor.reconstruct(sinogram_2d, generator_points, self.lor_descriptor)

        self.assertIn("completed", result["status"].lower(), f"Reconstruction did not complete successfully: {result['status']}")
        self.assertFalse(result["degenerate_input"], "Reconstruction flagged input as degenerate unexpectedly.")

        reconstructed_activity = result["activity"]
        self.assertEqual(reconstructed_activity.shape[0], num_cells, "Output activity shape mismatch.")

        if test_reconstructor.positivity_constraint:
            self.assertTrue(torch.all(reconstructed_activity >= 0), "Positivity constraint violated.")

        # Basic checks on reconstructed values
        # Expect the cell corresponding to the "hot" region (assumed cell 0) to have higher activity.
        # This is a soft check due to few iterations/subsets and simplified physics.
        # print(f"Reconstructed activities: {reconstructed_activity}")
        # print(f"True cell activities: {true_cell_activities}")

        hot_cell_reco_val = reconstructed_activity[0]
        other_cells_mean_reco_val = reconstructed_activity[1:].mean() if num_cells > 1 else torch.tensor(0.0, device=self.device)

        self.assertTrue(hot_cell_reco_val > other_cells_mean_reco_val,
                        f"Hot cell activity {hot_cell_reco_val.item()} not significantly greater than others mean {other_cells_mean_reco_val.item()}")

        # Check that the hot cell's activity is closer to hot_cell_activity than background_activity
        # and vice-versa for a background cell (e.g. cell 1 if num_cells > 1)
        self.assertTrue(abs(hot_cell_reco_val - hot_cell_activity) < abs(hot_cell_reco_val - background_activity),
                        "Hot cell reconstructed value seems closer to background than true hot activity.")
        if num_cells > 1:
            background_cell_reco_val = reconstructed_activity[1]
            self.assertTrue(abs(background_cell_reco_val - background_activity) < abs(background_cell_reco_val - hot_cell_activity),
                            "Background cell reconstructed value seems closer to hot activity than true background.")

if __name__ == '__main__':
    unittest.main()
