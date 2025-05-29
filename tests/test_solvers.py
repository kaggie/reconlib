import unittest
import torch
import numpy as np # For phantom creation if needed simply

from reconlib.solvers import iterative_reconstruction
from reconlib.nufft import NUFFT2D # Using NUFFT2D for testing
from reconlib.utils import calculate_density_compensation # For generating Voronoi weights

class TestIterativeReconstruction(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.image_shape_2d = (16, 16) # Small for faster tests
        self.num_k_points = 100
        self.dtype_real = torch.float32
        self.dtype_complex = torch.complex64

        # Create simple 2D k-space trajectory (radial-like for simplicity)
        angles = torch.linspace(0, np.pi, self.num_k_points // 2, device=self.device, dtype=self.dtype_real)
        radii = torch.linspace(0, 0.5, self.num_k_points // (self.num_k_points // 2), device=self.device, dtype=self.dtype_real).unsqueeze(0)
        kx = radii * torch.cos(angles.unsqueeze(-1))
        ky = radii * torch.sin(angles.unsqueeze(-1))
        # Stack to create a simple radial pattern, then flatten
        self.sampling_points = torch.stack((kx.flatten(), ky.flatten()), dim=1)
        # Ensure num_k_points is updated if flattening changes it (though it shouldn't here)
        self.num_k_points = self.sampling_points.shape[0] 


        # NUFFT parameters for NUFFT2D
        self.nufft_kwargs = {
            'oversamp_factor': (2.0, 2.0),
            'kb_J': (4, 4),
            # kb_alpha calculated based on J usually, e.g. 2.34 * J
            # For J=4, alpha is often around 2.34*4 = 9.36. Let's use a common value or derive it.
            # Simplified from MIRT: alpha = oversamp_factor * pi * sqrt( (J/oversamp_factor)^2 * (os-0.5)^2 - 0.8 )
            # Or simply use a value like 2.34 * J
            'kb_alpha': (2.34 * 4, 2.34 * 4), 
            'Ld': (2**10, 2**10), # Table length for interpolation
            'kb_m': (0.0, 0.0) # Standard Kaiser-Bessel (not generalized)
        }

        # Create a simple phantom image (e.g., a small square in the center)
        self.phantom = torch.zeros(self.image_shape_2d, device=self.device, dtype=self.dtype_complex)
        center_x, center_y = self.image_shape_2d[0] // 2, self.image_shape_2d[1] // 2
        square_size = self.image_shape_2d[0] // 4
        self.phantom[
            center_x - square_size // 2 : center_x + square_size // 2,
            center_y - square_size // 2 : center_y + square_size // 2
        ] = 1.0 + 0.0j # Make it complex

        # Instantiate NUFFT2D operator
        # Note: iterative_reconstruction will instantiate its own NUFFT operator.
        # This instance is just for generating test kspace_data.
        # No density_comp_weights passed here for generating "raw" kspace_data
        nufft_op_for_data_gen = NUFFT2D(
            image_shape=self.image_shape_2d,
            k_trajectory=self.sampling_points,
            device=self.device,
            **self.nufft_kwargs 
        )
        self.kspace_data = nufft_op_for_data_gen.forward(self.phantom)
        
        # Ensure kspace_data is 1D as expected by iterative_reconstruction
        if self.kspace_data.ndim > 1:
            self.kspace_data = self.kspace_data.flatten() 
            if self.kspace_data.shape[0] != self.num_k_points: # Should match if sampling_points was (N,D)
                 # Adjust num_k_points if k_trajectory was complex and forward changed output points
                 # This should not happen with current NUFFT forward for (N,D) k_traj
                 pass


    def test_iterative_recon_2d_basic(self):
        reconstructed_image = iterative_reconstruction(
            kspace_data=self.kspace_data,
            sampling_points=self.sampling_points,
            image_shape=self.image_shape_2d,
            nufft_operator_class=NUFFT2D, # Pass the class itself
            nufft_kwargs=self.nufft_kwargs,
            use_voronoi=False,
            max_iters=5 # Few iterations for speed
        )
        self.assertEqual(reconstructed_image.shape, self.image_shape_2d)
        self.assertEqual(reconstructed_image.dtype, self.dtype_complex)
        self.assertTrue(torch.sum(torch.abs(reconstructed_image)) > 1e-6, "Reconstruction is all zeros.")
        # Check for NaNs
        self.assertFalse(torch.isnan(reconstructed_image).any(), "Reconstruction contains NaNs.")

    def test_iterative_recon_2d_voronoi(self):
        # Define bounds for Voronoi calculation (e.g., covering k-space extents)
        bounds_min = self.sampling_points.min(dim=0).values - 0.1 # Small margin
        bounds_max = self.sampling_points.max(dim=0).values + 0.1
        bounds = torch.stack([bounds_min, bounds_max]).to(self.device, dtype=self.dtype_real)

        # Generate Voronoi weights
        # calculate_density_compensation expects k_trajectory on CPU for SciPy if not handled internally
        # However, iterative_reconstruction passes sampling_points directly.
        # compute_voronoi_density_weights (used by calculate_density_compensation) handles CPU conversion.
        voronoi_weights = calculate_density_compensation(
            k_trajectory=self.sampling_points, # Already on self.device
            image_shape=self.image_shape_2d, # Not directly used by voronoi method but part of API
            method='voronoi',
            device=self.device, # Passed to calculate_density_compensation
            bounds=bounds
        )
        self.assertEqual(voronoi_weights.shape[0], self.num_k_points)

        reconstructed_image = iterative_reconstruction(
            kspace_data=self.kspace_data,
            sampling_points=self.sampling_points,
            image_shape=self.image_shape_2d,
            nufft_operator_class=NUFFT2D,
            nufft_kwargs=self.nufft_kwargs,
            use_voronoi=True,
            voronoi_weights=voronoi_weights,
            max_iters=5
        )
        self.assertEqual(reconstructed_image.shape, self.image_shape_2d)
        self.assertEqual(reconstructed_image.dtype, self.dtype_complex)
        self.assertTrue(torch.sum(torch.abs(reconstructed_image)) > 1e-6, "Voronoi reconstruction is all zeros.")
        self.assertFalse(torch.isnan(reconstructed_image).any(), "Voronoi reconstruction contains NaNs.")

    def test_iterative_recon_voronoi_errors(self):
        with self.assertRaisesRegex(ValueError, "If use_voronoi is True, voronoi_weights must be provided."):
            iterative_reconstruction(
                kspace_data=self.kspace_data,
                sampling_points=self.sampling_points,
                image_shape=self.image_shape_2d,
                nufft_operator_class=NUFFT2D,
                nufft_kwargs=self.nufft_kwargs,
                use_voronoi=True,
                voronoi_weights=None # Error case
            )

        wrong_shape_weights = torch.ones(self.num_k_points + 10, device=self.device, dtype=self.dtype_real)
        with self.assertRaisesRegex(ValueError, "Shape mismatch: voronoi_weights .* and kspace_data .* must have the same length."):
            iterative_reconstruction(
                kspace_data=self.kspace_data,
                sampling_points=self.sampling_points,
                image_shape=self.image_shape_2d,
                nufft_operator_class=NUFFT2D,
                nufft_kwargs=self.nufft_kwargs,
                use_voronoi=True,
                voronoi_weights=wrong_shape_weights # Error case
            )

    def test_iterative_recon_convergence_params(self):
        # Test with max_iters=1
        recon_iters_1 = iterative_reconstruction(
            kspace_data=self.kspace_data,
            sampling_points=self.sampling_points,
            image_shape=self.image_shape_2d,
            nufft_operator_class=NUFFT2D,
            nufft_kwargs=self.nufft_kwargs,
            max_iters=1
        )
        self.assertEqual(recon_iters_1.shape, self.image_shape_2d)
        self.assertFalse(torch.isnan(recon_iters_1).any())

        # Test with a very loose tolerance for early stopping (if possible to observe)
        # This primarily checks that the parameter is used and doesn't crash.
        # Actual early stopping depends on the data and gradient behavior.
        recon_loose_tol = iterative_reconstruction(
            kspace_data=self.kspace_data,
            sampling_points=self.sampling_points,
            image_shape=self.image_shape_2d,
            nufft_operator_class=NUFFT2D,
            nufft_kwargs=self.nufft_kwargs,
            max_iters=10, # Allow a few iterations
            tol=1e5 # Very loose tolerance
        )
        self.assertEqual(recon_loose_tol.shape, self.image_shape_2d)
        self.assertFalse(torch.isnan(recon_loose_tol).any())

if __name__ == '__main__':
    unittest.main()
