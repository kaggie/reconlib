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
