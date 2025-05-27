import unittest
import torch
import numpy as np
from reconlib.operators import RadioInterferometryOperator # Assuming Operator base class is not directly tested here

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestRadioInterferometryOperator(unittest.TestCase):
    def setUp(self):
        self.image_shape = (16, 16) # Ny, Nx
        self.Ny, self.Nx = self.image_shape
        # uv_coordinates are zero-centered, integer indices for fftshifted grid
        self.uv_coords_np = np.array([
            [0, 0],    # DC component
            [1, 0],    # Horizontal frequency
            [0, 1],    # Vertical frequency
            [-2, -2],
            [self.Nx//2 - 1, self.Ny//2 - 1] # Max frequency
        ])
        # Ensure uv_coordinates is on DEVICE before passing to operator
        self.uv_coordinates_tensor = torch.from_numpy(self.uv_coords_np).long().to(DEVICE)
        self.num_visibilities = self.uv_coordinates_tensor.shape[0]
        
        self.operator = RadioInterferometryOperator(
            uv_coordinates=self.uv_coordinates_tensor, # Use tensor here
            image_shape=self.image_shape,
            device=DEVICE # Pass DEVICE explicitly
        )

    def test_instantiation(self):
        self.assertIsInstance(self.operator, RadioInterferometryOperator)
        # Compare with the tensor version that was passed to constructor
        self.assertTrue(torch.equal(self.operator.uv_coordinates, self.uv_coordinates_tensor))
        self.assertEqual(self.operator.image_shape, self.image_shape)
        
        # Test invalid uv_coordinates shapes
        with self.assertRaises(ValueError): # Wrong uv dim
            RadioInterferometryOperator(torch.randn(5, 3, device=DEVICE).long(), self.image_shape, device=DEVICE)
        with self.assertRaises(ValueError): # Wrong uv ndim
            RadioInterferometryOperator(torch.randn(5, device=DEVICE).long(), self.image_shape, device=DEVICE)
        
        # Test uv_coordinates out of range
        invalid_uv_large = torch.tensor([[self.Nx // 2 + 10, 0]], dtype=torch.long, device=DEVICE)
        with self.assertRaises(ValueError):
            RadioInterferometryOperator(invalid_uv_large, self.image_shape, device=DEVICE)
            
        invalid_uv_small = torch.tensor([[-self.Nx // 2 - 10, 0]], dtype=torch.long, device=DEVICE)
        with self.assertRaises(ValueError):
            RadioInterferometryOperator(invalid_uv_small, self.image_shape, device=DEVICE)


    def test_op_forward_point_source_at_center(self):
        # Point source at (0,0) in image space for FFT (top-left corner for non-shifted).
        # The operator handles the shift internally. The input image is standard.
        sky_image_delta_at_origin = torch.zeros(self.image_shape, dtype=torch.complex64, device=DEVICE)
        sky_image_delta_at_origin[0, 0] = 1.0 # Standard image space delta

        visibilities = self.operator.op(sky_image_delta_at_origin)
        self.assertEqual(visibilities.shape, (self.num_visibilities,))
        self.assertTrue(visibilities.is_complex())
        
        # FFT of delta[0,0] (image space) is constant in Fourier domain.
        # With 'ortho' norm, FFT(delta_image)[u,v] = 1.0 / sqrt(Ny*Nx) for all u,v.
        expected_value = 1.0 / np.sqrt(self.Ny * self.Nx)
        expected_vis_tensor = torch.full((self.num_visibilities,), expected_value, dtype=torch.complex64, device=DEVICE)
        torch.testing.assert_close(visibilities, expected_vis_tensor)

    def test_op_adj_dc_visibilities(self):
        vis_values = torch.zeros(self.num_visibilities, dtype=torch.complex64, device=DEVICE)
        dc_uv_coord_index = -1
        # Find the index of the DC component in the original uv_coords_np
        for i, coord_np in enumerate(self.uv_coords_np):
            if coord_np[0] == 0 and coord_np[1] == 0:
                dc_uv_coord_index = i
                break
        self.assertNotEqual(dc_uv_coord_index, -1, "DC coordinate (0,0) not found in test uv_coordinates.")
        
        # For IFFT (norm='ortho') of a grid with only DC value V_dc, the image is V_dc / sqrt(Ny*Nx).
        # So, to get an image of constant 1.0, V_dc should be sqrt(Ny*Nx).
        vis_values[dc_uv_coord_index] = 1.0 * np.sqrt(self.Ny * self.Nx) 

        dirty_image = self.operator.op_adj(vis_values)
        self.assertEqual(dirty_image.shape, self.image_shape)
        self.assertTrue(dirty_image.is_complex())
        
        expected_image = torch.ones(self.image_shape, dtype=torch.complex64, device=DEVICE)
        torch.testing.assert_close(dirty_image, expected_image)

    def test_adjoint_property(self):
        torch.manual_seed(0) # For reproducibility
        sky_image_x = torch.randn(self.image_shape, dtype=torch.complex64, device=DEVICE)
        visibilities_y = torch.randn(self.num_visibilities, dtype=torch.complex64, device=DEVICE)

        op_x = self.operator.op(sky_image_x)
        op_adj_y = self.operator.op_adj(visibilities_y)

        lhs = torch.vdot(op_x, visibilities_y)
        # op_adj_y is (Ny, Nx), sky_image_x is (Ny, Nx). Flatten for vdot.
        rhs = torch.vdot(sky_image_x.flatten(), op_adj_y.flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)

if __name__ == '__main__':
    unittest.main()

from reconlib.operators import FieldCorrectedNUFFTOperator, NUFFTOperator # Added imports

class TestFieldCorrectedNUFFTOperator(unittest.TestCase):
    def setUp(self):
        self.device = DEVICE # Use global DEVICE
        self.image_shape_2d = (16, 16) # Ny, Nx
        self.num_k_points_2d = 64

        # Simple linear k-space trajectory (more for testing logic than realism)
        kx = torch.linspace(-0.5, 0.5, self.num_k_points_2d, device=self.device)
        ky = torch.zeros(self.num_k_points_2d, device=self.device)
        # For NUFFT2D, k_trajectory shape should be (num_k_points, 2)
        # The order usually expected is (kx, ky) or (ku, kv)
        # If reconlib.nufft.NUFFT2D expects (ky, kx) for (Ny, Nx) image, then this is fine.
        # Let's assume it expects (ku, kv) where u maps to x-dim and v to y-dim.
        # So, stack([kx, ky], dim=-1) would be more standard if kx is for image_shape[1] (width).
        # The example in FieldCorrectedNUFFTOperator shows k_trajectory is (num_k_points, ndims)
        # For a (Ny, Nx) image, ndims=2. Coordinates are (u,v)
        # u corresponds to x (width, image_shape[1]), v to y (height, image_shape[0])
        # NUFFT2D.py likely maps k_trajectory[:,0] to first spatial dim (y), k_trajectory[:,1] to second (x)
        # if its image_shape is (Ny,Nx). So k_trajectory should be (v,u).
        # If kx is for the x-dim (width) and ky for y-dim (height):
        self.k_trajectory_2d = torch.stack([ky, kx], dim=-1) # (v,u) for (Ny,Nx) image if NUFFT maps k_traj dim 0 to img_dim 0

        # Time vector: simple linear ramp
        self.time_per_kpoint_2d = torch.linspace(0, 0.01, self.num_k_points_2d, device=self.device) # 10 ms readout

        # B0 map: simple linear gradient in x (image columns) from -20Hz to 20Hz
        self.b0_map_2d_hz = torch.zeros(self.image_shape_2d, device=self.device)
        for c in range(self.image_shape_2d[1]): # Iterate through columns (x-dimension)
            self.b0_map_2d_hz[:, c] = (c / (self.image_shape_2d[1] -1) - 0.5) * 40.0 # -20Hz to 20Hz

        self.num_segments = 4 # Fewer segments for faster testing

        # Basic NUFFT parameters (can be simple for unit testing the logic)
        self.oversamp_factor = (1.25, 1.25) # Small oversampling for speed
        self.kb_J = (3, 3) # Smaller kernel
        self.kb_alpha = tuple(2.34 * j for j in self.kb_J)
        # Ld will be calculated internally by FieldCorrectedNUFFTOperator's __init__

        self.fc_nufft_op_2d = FieldCorrectedNUFFTOperator(
            k_trajectory=self.k_trajectory_2d,
            image_shape=self.image_shape_2d,
            b0_map=self.b0_map_2d_hz,
            time_per_kspace_point=self.time_per_kpoint_2d,
            num_segments=self.num_segments,
            oversamp_factor=self.oversamp_factor,
            kb_J=self.kb_J,
            kb_alpha=self.kb_alpha,
            device=self.device.type # pass 'cpu' or 'cuda' string
        )
        
        # Create a simple test image (e.g., point source)
        self.test_image_2d = torch.zeros(self.image_shape_2d, dtype=torch.complex64, device=self.device)
        # Place point source at image center
        self.test_image_2d[self.image_shape_2d[0]//2, self.image_shape_2d[1]//2] = 1.0 + 0.5j


    def test_instantiation(self):
        self.assertIsInstance(self.fc_nufft_op_2d, FieldCorrectedNUFFTOperator)
        # The number of segments might be adjusted internally if num_segments > num_k_points
        # So, we check against the operator's actual num_segments
        self.assertEqual(self.fc_nufft_op_2d.num_segments, min(self.num_segments, self.num_k_points_2d))
        self.assertEqual(len(self.fc_nufft_op_2d._segment_indices_list), self.fc_nufft_op_2d.num_segments)
        self.assertEqual(len(self.fc_nufft_op_2d.segment_avg_times_list), self.fc_nufft_op_2d.num_segments)

    def test_op_forward_basic_run(self):
        # Test if op runs and produces output of correct shape
        k_space_out = self.fc_nufft_op_2d.op(self.test_image_2d)
        self.assertEqual(k_space_out.shape, (self.num_k_points_2d,))
        self.assertTrue(k_space_out.is_complex())
        self.assertEqual(k_space_out.device, self.device)

    def test_op_adj_basic_run(self):
        # Test if op_adj runs and produces output of correct shape
        test_k_space_data = torch.randn(self.num_k_points_2d, dtype=torch.complex64, device=self.device)
        image_out = self.fc_nufft_op_2d.op_adj(test_k_space_data)
        self.assertEqual(image_out.shape, self.image_shape_2d)
        self.assertTrue(image_out.is_complex())
        self.assertEqual(image_out.device, self.device)

    def test_adjoint_property_field_corrected(self):
        # Test adjoint property: <A(x), y> = <x, A_adj(y)>
        # This is a more rigorous test of correctness.
        torch.manual_seed(1) # For reproducibility
        img_x = torch.randn(self.image_shape_2d, dtype=torch.complex64, device=self.device)
        kspace_y = torch.randn(self.num_k_points_2d, dtype=torch.complex64, device=self.device)

        op_x = self.fc_nufft_op_2d.op(img_x)
        op_adj_y = self.fc_nufft_op_2d.op_adj(kspace_y)

        lhs = torch.vdot(op_x, kspace_y) # inner product in k-space
        rhs = torch.vdot(img_x.flatten(), op_adj_y.flatten()) # inner product in image space
        
        # Tolerance might need to be adjusted depending on NUFFT precision and segmentation effects
        torch.testing.assert_close(lhs, rhs, rtol=1e-3, atol=1e-4)

    def test_op_no_b0_effect_if_b0_zero(self):
        # If B0 map is zero, FieldCorrectedNUFFTOperator should behave like standard NUFFT
        b0_map_zeros = torch.zeros_like(self.b0_map_2d_hz)
        fc_nufft_op_no_b0 = FieldCorrectedNUFFTOperator(
            k_trajectory=self.k_trajectory_2d,
            image_shape=self.image_shape_2d,
            b0_map=b0_map_zeros, # Zero B0 map
            time_per_kspace_point=self.time_per_kpoint_2d,
            num_segments=1, # Can even be 1 segment if b0 is zero for true comparison
            oversamp_factor=self.oversamp_factor,
            kb_J=self.kb_J,
            kb_alpha=self.kb_alpha,
            device=self.device.type
        )
        k_space_fc_no_b0 = fc_nufft_op_no_b0.op(self.test_image_2d)

        # Compare with a standard NUFFTOperator (assuming one exists and is imported)
        # NUFFTOperator is already imported
        std_nufft_op = NUFFTOperator(
            k_trajectory=self.k_trajectory_2d,
            image_shape=self.image_shape_2d,
            oversamp_factor=self.oversamp_factor,
            kb_J=self.kb_J,
            kb_alpha=self.kb_alpha,
            device=self.device.type
        )
        k_space_std = std_nufft_op.op(self.test_image_2d)
        torch.testing.assert_close(k_space_fc_no_b0, k_space_std, rtol=1e-4, atol=1e-5)

# This ensures that if the script is run directly, unittest.main() is called.
# It should be at the very end of the file.
if __name__ == '__main__':
    unittest.main()
