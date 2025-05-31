import unittest
import torch
import numpy as np # For np.isclose in main block, if used directly
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.modalities.sar.operators import SARForwardOperator
    from reconlib.modalities.sar.reconstructors import tv_reconstruction_sar
    # tv_reconstruction_sar uses UltrasoundTVCustomRegularizer
    from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
    RECONLIB_SAR_AVAILABLE = True
except ImportError as e:
    print(f"Could not import SAR (and potentially ultrasound regularizer) modules for testing: {e}")
    RECONLIB_SAR_AVAILABLE = False


@unittest.skipIf(not RECONLIB_SAR_AVAILABLE, "reconlib.modalities.sar module or its dependencies not available")
class TestSARForwardOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Ny, self.Nx = 32, 32  # Image height, width for tests
        self.image_shape_sar = (self.Ny, self.Nx)

        self.num_visibilities = 100
        # Simulate (u,v) coordinates as integers for simple FFT indexing in tests
        u_coords = torch.randint(-self.Nx // 2, self.Nx // 2, (self.num_visibilities,), device=self.device).float()
        v_coords = torch.randint(-self.Ny // 2, self.Ny // 2, (self.num_visibilities,), device=self.device).float()
        self.uv_coordinates = torch.stack((u_coords, v_coords), dim=1)

        self.operator_params = {
            'image_shape': self.image_shape_sar,
            'uv_coordinates': self.uv_coordinates,
            'device': self.device
        }
        self.sar_op = SARForwardOperator(**self.operator_params)

        self.phantom = torch.randn(self.image_shape_sar, dtype=torch.complex64, device=self.device)
        self.dummy_visibilities = torch.randn(self.num_visibilities, dtype=torch.complex64, device=self.device)

    def test_operator_instantiation(self):
        self.assertIsInstance(self.sar_op, SARForwardOperator)
        print("TestSARForwardOperator: Instantiation OK.")

    def test_forward_op_shape_dtype(self):
        visibilities = self.sar_op.op(self.phantom)
        self.assertEqual(visibilities.shape, (self.num_visibilities,))
        self.assertTrue(visibilities.is_complex())
        self.assertEqual(visibilities.device, self.device)
        print("TestSARForwardOperator: Forward op shape and dtype OK.")

    def test_adjoint_op_shape_dtype(self):
        dirty_image = self.sar_op.op_adj(self.dummy_visibilities)
        self.assertEqual(dirty_image.shape, self.image_shape_sar)
        self.assertTrue(dirty_image.is_complex())
        self.assertEqual(dirty_image.device, self.device)
        print("TestSARForwardOperator: Adjoint op shape and dtype OK.")

    def test_dot_product(self):
        x_dp = torch.randn_like(self.phantom)
        y_dp_rand = torch.randn_like(self.dummy_visibilities)

        Ax = self.sar_op.op(x_dp)
        Aty = self.sar_op.op_adj(y_dp_rand)

        lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

        print(f"SAR Dot Product Test - LHS: {lhs.item():.6f}, RHS: {rhs.item():.6f}")
        # Tolerance might need to be a bit looser due to nearest neighbor gridding/sampling
        # compared to a perfect NUFFT adjoint pair.
        self.assertAlmostEqual(lhs.real.item(), rhs.real.item(), delta=1e-3 * (abs(lhs.real.item()) + abs(rhs.real.item()) + 1e-9))
        self.assertAlmostEqual(lhs.imag.item(), rhs.imag.item(), delta=1e-3 * (abs(lhs.imag.item()) + abs(rhs.imag.item()) + 1e-9))
        print("TestSARForwardOperator: Dot product test PASSED (within tolerance).")


@unittest.skipIf(not RECONLIB_SAR_AVAILABLE, "reconlib.modalities.sar module or its dependencies not available")
class TestSARReconstructors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Ny, self.Nx = 32, 32
        self.image_shape_sar = (self.Ny, self.Nx)
        self.num_visibilities = 100

        u_coords = torch.randint(-self.Nx // 2, self.Nx // 2, (self.num_visibilities,), device=self.device).float()
        v_coords = torch.randint(-self.Ny // 2, self.Ny // 2, (self.num_visibilities,), device=self.device).float()
        self.uv_coordinates = torch.stack((u_coords, v_coords), dim=1)

        self.sar_operator_inst = SARForwardOperator(
            image_shape=self.image_shape_sar,
            uv_coordinates=self.uv_coordinates,
            device=self.device
        )

        self.y_sar_data = torch.randn(self.num_visibilities, dtype=torch.complex64, device=self.device)

    def test_tv_reconstruction_sar_execution(self):
        recon_image = tv_reconstruction_sar(
            y_sar_data=self.y_sar_data,
            sar_operator=self.sar_operator_inst,
            lambda_tv=0.001, # Small lambda for quick test
            iterations=2,    # Minimal iterations
            step_size=0.01,
            tv_prox_iterations=1,
            tv_prox_step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_image.shape, self.image_shape_sar)
        self.assertTrue(recon_image.is_complex())
        self.assertEqual(recon_image.device, self.device)
        print("TestSARReconstructors: tv_reconstruction_sar execution OK.")

if __name__ == '__main__':
    if RECONLIB_SAR_AVAILABLE:
        unittest.main()
    else:
        print("Skipping SAR tests as module or its dependencies are not available.")
