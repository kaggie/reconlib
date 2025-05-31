import unittest
import torch
import numpy as np
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.modalities.astronomical.operators import AstronomicalInterferometryOperator
    from reconlib.modalities.astronomical.reconstructors import tv_reconstruction_astro
    # tv_reconstruction_astro uses UltrasoundTVCustomRegularizer
    from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
    RECONLIB_ASTRO_AVAILABLE = True
except ImportError as e:
    print(f"Could not import Astronomical (and potentially ultrasound regularizer) modules for testing: {e}")
    RECONLIB_ASTRO_AVAILABLE = False


@unittest.skipIf(not RECONLIB_ASTRO_AVAILABLE, "reconlib.modalities.astronomical module or its dependencies not available")
class TestAstronomicalInterferometryOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Ny, self.Nx = 32, 32  # Image height, width for tests
        self.image_shape_astro = (self.Ny, self.Nx)

        self.num_visibilities = 100
        # Simulate (u,v) coordinates as floats, operator handles rounding for indexing
        u_coords = torch.randint(-self.Nx // 2, self.Nx // 2, (self.num_visibilities,), device=self.device).float()
        v_coords = torch.randint(-self.Ny // 2, self.Ny // 2, (self.num_visibilities,), device=self.device).float()
        self.uv_coordinates = torch.stack((u_coords, v_coords), dim=1)

        self.operator_params = {
            'image_shape': self.image_shape_astro,
            'uv_coordinates': self.uv_coordinates,
            'device': self.device
        }
        self.astro_op = AstronomicalInterferometryOperator(**self.operator_params)

        self.phantom_sky_map = torch.randn(self.image_shape_astro, dtype=torch.complex64, device=self.device)
        self.dummy_visibilities = torch.randn(self.num_visibilities, dtype=torch.complex64, device=self.device)

    def test_operator_instantiation(self):
        self.assertIsInstance(self.astro_op, AstronomicalInterferometryOperator)
        print("TestAstronomicalInterferometryOperator: Instantiation OK.")

    def test_forward_op_shape_dtype(self):
        visibilities = self.astro_op.op(self.phantom_sky_map)
        self.assertEqual(visibilities.shape, (self.num_visibilities,))
        self.assertTrue(visibilities.is_complex())
        self.assertEqual(visibilities.device, self.device)
        print("TestAstronomicalInterferometryOperator: Forward op shape and dtype OK.")

    def test_adjoint_op_shape_dtype(self):
        dirty_image = self.astro_op.op_adj(self.dummy_visibilities)
        self.assertEqual(dirty_image.shape, self.image_shape_astro)
        self.assertTrue(dirty_image.is_complex())
        self.assertEqual(dirty_image.device, self.device)
        print("TestAstronomicalInterferometryOperator: Adjoint op shape and dtype OK.")

    def test_dot_product(self):
        x_dp = torch.randn_like(self.phantom_sky_map)
        y_dp_rand = torch.randn_like(self.dummy_visibilities)

        Ax = self.astro_op.op(x_dp)
        Aty = self.astro_op.op_adj(y_dp_rand)

        lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

        print(f"Astronomical Dot Product Test - LHS: {lhs.item():.6f}, RHS: {rhs.item():.6f}")
        # FFT/IFFT with 'ortho' and accumulate=True in gridding should hold well.
        self.assertAlmostEqual(lhs.real.item(), rhs.real.item(), delta=1e-4 * (abs(lhs.real.item()) + abs(rhs.real.item()) + 1e-9))
        self.assertAlmostEqual(lhs.imag.item(), rhs.imag.item(), delta=1e-4 * (abs(lhs.imag.item()) + abs(rhs.imag.item()) + 1e-9))
        print("TestAstronomicalInterferometryOperator: Dot product test PASSED.")


@unittest.skipIf(not RECONLIB_ASTRO_AVAILABLE, "reconlib.modalities.astronomical module or its dependencies not available")
class TestAstronomicalReconstructors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Ny, self.Nx = 32, 32
        self.image_shape_astro = (self.Ny, self.Nx)
        self.num_visibilities = 100

        # Using a Mock Operator for reconstructor test for speed and isolation
        class MockAstroOp:
            def __init__(self, image_shape, num_vis, device):
                self.image_shape = image_shape
                self.num_vis = num_vis
                self.device = device
            def op(self, x): return torch.randn(self.num_vis, dtype=torch.complex64, device=self.device)
            def op_adj(self, y): return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device)

        self.astro_operator_inst = MockAstroOp(
            image_shape=self.image_shape_astro,
            num_vis=self.num_visibilities,
            device=self.device
        )

        self.y_visibilities = torch.randn(self.num_visibilities, dtype=torch.complex64, device=self.device)

    def test_tv_reconstruction_astro_execution(self):
        # Test basic execution and output shape
        recon_sky_map = tv_reconstruction_astro(
            y_visibilities=self.y_visibilities,
            astro_operator=self.astro_operator_inst,
            lambda_tv=0.001,
            iterations=2,    # Minimal iterations for speed
            step_size=0.01,
            tv_prox_iterations=1,
            tv_prox_step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_sky_map.shape, self.image_shape_astro)
        self.assertTrue(recon_sky_map.is_complex())
        self.assertEqual(recon_sky_map.device, self.device)
        print("TestAstronomicalReconstructors: tv_reconstruction_astro execution OK.")

if __name__ == '__main__':
    if RECONLIB_ASTRO_AVAILABLE:
        unittest.main()
    else:
        print("Skipping Astronomical tests as module or its dependencies are not available.")
