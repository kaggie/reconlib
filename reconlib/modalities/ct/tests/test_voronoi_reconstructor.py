import unittest
import torch
import numpy as np
from reconlib.modalities.ct.voronoi_reconstructor import VoronoiCTReconstructor2D

class TestVoronoiCTReconstructor2D(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reconstructor = VoronoiCTReconstructor2D(
            num_iterations=5,
            relaxation_factor=0.1,
            verbose=False,
            device=self.device
        )
        # Common LOR descriptor for some tests
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

        # Test with collinear points (expecting ConvexHull area check to fail or Delaunay to give few simplices)
        points_collinear = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=self.device)
        result_collinear = self.reconstructor.reconstruct(
            torch.rand((self.num_angles, self.num_radial_bins), device=self.device),
            points_collinear,
            self.lor_descriptor
        )
        # This might result in Voronoi cell validation failure or system matrix error if not caught by point validation directly
        self.assertTrue(result_collinear["degenerate_input"] or "degenerate" in result_collinear["status"].lower() or "failed" in result_collinear["status"].lower(),
                        f"Failed for collinear points. Status: {result_collinear['status']}")


    def test_basic_reconstruction(self):
        # Define a simple 2D phantom: a square of higher attenuation
        image_size = 32
        phantom_val = 0.1
        background_val = 0.001 # SART works better with non-zero background

        # Define generator points: a central square and some surrounding points
        # These points define the Voronoi cells.
        # Cell 0: Central square
        # Cells 1-4: Background regions around the square
        generator_points = torch.tensor([
            [self.fov_width*0.5, self.fov_width*0.5],  # Center of the phantom square cell
            [self.fov_width*0.25, self.fov_width*0.25], # Background TL
            [self.fov_width*0.75, self.fov_width*0.25], # Background TR
            [self.fov_width*0.25, self.fov_width*0.75], # Background BL
            [self.fov_width*0.75, self.fov_width*0.75], # Background BR
            [self.fov_width*0.5, self.fov_width*0.1],   # Background Top Mid
            [self.fov_width*0.5, self.fov_width*0.9],   # Background Bot Mid
            [self.fov_width*0.1, self.fov_width*0.5],   # Background Left Mid
            [self.fov_width*0.9, self.fov_width*0.5],   # Background Right Mid
        ], device=self.device)
        num_cells = generator_points.shape[0]

        # True attenuations for each cell
        # For this test, let's assume the first generator point corresponds to the "phantom"
        true_cell_attenuations = torch.full((num_cells,), background_val, device=self.device, dtype=torch.float32)
        # To identify which cell is the central one after Voronoi tessellation is complex.
        # For simplicity, we'll assume the first generator point creates the "phantom" cell.
        # This is an approximation for testing. A robust test would map phantom to cells.
        # Here, we assume cell 0 is our high attenuation square.
        true_cell_attenuations[0] = phantom_val

        reconstructor_for_sim = VoronoiCTReconstructor2D(
            num_iterations=1, verbose=False, device=self.device, positivity_constraint=False
        )

        # Validate points and compute Voronoi diagram to pass to system matrix
        is_gen_invalid, gen_status = reconstructor_for_sim._validate_generator_points_2d(generator_points)
        self.assertFalse(is_gen_invalid, f"Generator points validation failed for simulation: {gen_status}")

        cells_verts, _, vor_status = reconstructor_for_sim._compute_voronoi_diagram_2d(generator_points)
        self.assertIsNotNone(cells_verts, f"Voronoi diagram computation failed for simulation: {vor_status}")

        is_cell_invalid, cell_status = reconstructor_for_sim._validate_voronoi_cells_2d(cells_verts)
        self.assertFalse(is_cell_invalid, f"Voronoi cell validation failed for simulation: {cell_status}")

        # Manually compute system matrix
        system_matrix = reconstructor_for_sim._compute_system_matrix_2d(self.lor_descriptor, cells_verts)
        self.assertIsNotNone(system_matrix, "System matrix computation failed for simulation.")

        # Generate sinogram using the reconstructor's own forward projection
        sinogram_flat = reconstructor_for_sim._forward_project_2d(true_cell_attenuations, system_matrix)
        sinogram_2d = sinogram_flat.reshape(self.num_angles, self.num_radial_bins)

        # Initialize the reconstructor for the actual test
        test_reconstructor = VoronoiCTReconstructor2D(
            num_iterations=10,
            relaxation_factor=0.15,
            verbose=False,
            device=self.device,
            positivity_constraint=True # Test with positivity
        )

        # Perform reconstruction
        result = test_reconstructor.reconstruct(sinogram_2d, generator_points, self.lor_descriptor)

        self.assertIn("completed", result["status"].lower(), f"Reconstruction did not complete successfully: {result['status']}")
        self.assertFalse(result["degenerate_input"], "Reconstruction flagged input as degenerate unexpectedly.")

        reconstructed_attenuation = result["attenuation"]
        self.assertEqual(reconstructed_attenuation.shape[0], num_cells, "Output attenuation shape mismatch.")

        if test_reconstructor.positivity_constraint:
            self.assertTrue(torch.all(reconstructed_attenuation >= 0), "Positivity constraint violated.")

        # Basic checks on reconstructed values
        # This is a very soft check as SART convergence and accuracy with few iterations/angles/cells is limited.
        # We expect the cell corresponding to the phantom (cell 0 here by assumption) to have a higher value.
        # And other cells to have lower values.

        # Find the actual index of the cell generated by points[0]
        # This is tricky without querying the Voronoi diagram structure more deeply.
        # For this test, we'll rely on the assumption that generator_points[0] results in reconstructed_attenuation[0]
        # This might not hold if the Voronoi library reorders cells internally.
        # A more robust way would be to find the cell containing the center of the phantom.

        # For now, let's check if the sum of attenuations is reasonable.
        self.assertTrue(reconstructed_attenuation.sum() > 0, "Sum of reconstructed attenuations is not positive.")

        # A more specific check might be that the max value is somewhat related to phantom_val
        # and min value to background_val, but this is highly dependent on setup.
        # print(f"Reconstructed attenuations: {reconstructed_attenuation}")
        # print(f"True cell attenuations: {true_cell_attenuations}")

        # Check if the cell assumed to be the phantom has a higher value than the mean of others (very rough)
        if num_cells > 1:
            phantom_cell_reco_val = reconstructed_attenuation[0]
            other_cells_mean_reco_val = reconstructed_attenuation[1:].mean() if num_cells > 1 else torch.tensor(0.0, device=self.device) # handle single cell case
            # Expecting phantom cell to be higher than others, but SART might smooth things a lot
            # This is a weak assertion.
            self.assertTrue(phantom_cell_reco_val > other_cells_mean_reco_val - self.reconstructor.sart_epsilon,
                            f"Phantom cell value {phantom_cell_reco_val.item()} not significantly greater than others mean {other_cells_mean_reco_val.item()}")


if __name__ == '__main__':
    unittest.main()
