import unittest
import torch
import numpy as np
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.modalities.em.operators import EMForwardOperator
    from reconlib.modalities.em.reconstructors import tv_reconstruction_em
    # tv_reconstruction_em uses UltrasoundTVCustomRegularizer
    from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
    # EM operators might use utils for rotation
    from reconlib.modalities.em.utils import rotate_volume_z_axis, project_volume
    RECONLIB_EM_AVAILABLE = True
except ImportError as e:
    print(f"Could not import EM (and potentially ultrasound regularizer or EM utils) modules for testing: {e}")
    RECONLIB_EM_AVAILABLE = False


@unittest.skipIf(not RECONLIB_EM_AVAILABLE, "reconlib.modalities.em module or its dependencies not available")
class TestEMForwardOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Keep dimensions small for testing speed, especially for 3D
        self.D, self.H, self.W = 8, 16, 16
        self.volume_shape_em = (self.D, self.H, self.W)

        self.num_angles = 5 # Fewer angles for faster tests
        self.angles_rad = torch.tensor(np.linspace(0, np.pi, self.num_angles, endpoint=False),
                                       dtype=torch.float32, device=self.device)
        self.projection_axis = 0 # Project along Depth

        # Expected shape of a single projection if projecting along D
        self.single_proj_shape = (self.H, self.W)

        self.operator_params = {
            'volume_shape': self.volume_shape_em,
            'angles_rad': self.angles_rad,
            'projection_axis': self.projection_axis,
            'device': self.device
        }
        self.em_op = EMForwardOperator(**self.operator_params)

        self.phantom_volume = torch.randn(self.volume_shape_em, dtype=torch.complex64, device=self.device)

        self.dummy_projections = torch.randn(
            (self.num_angles,) + self.single_proj_shape,
            dtype=torch.complex64, device=self.device
        )

    def test_operator_instantiation(self):
        self.assertIsInstance(self.em_op, EMForwardOperator)
        print("TestEMForwardOperator: Instantiation OK.")

    def test_forward_op_shape_dtype(self):
        projections = self.em_op.op(self.phantom_volume)
        expected_shape = (self.num_angles,) + self.single_proj_shape
        self.assertEqual(projections.shape, expected_shape)
        self.assertTrue(projections.is_complex())
        self.assertEqual(projections.device, self.device)
        print("TestEMForwardOperator: Forward op shape and dtype OK.")

    def test_adjoint_op_shape_dtype(self):
        recon_volume = self.em_op.op_adj(self.dummy_projections)
        self.assertEqual(recon_volume.shape, self.volume_shape_em)
        self.assertTrue(recon_volume.is_complex())
        self.assertEqual(recon_volume.device, self.device)
        print("TestEMForwardOperator: Adjoint op shape and dtype OK.")

    def test_dot_product(self):
        x_dp = torch.randn_like(self.phantom_volume)
        y_dp_rand = torch.randn_like(self.dummy_projections)

        Ax = self.em_op.op(x_dp)
        Aty = self.em_op.op_adj(y_dp_rand)

        lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

        print(f"EM Dot Product Test - LHS: {lhs.item():.4f}, RHS: {rhs.item():.4f}")
        # Interpolation in rotation (grid_sample) can lead to larger differences
        # than pure FFT based operators.
        self.assertAlmostEqual(lhs.real.item(), rhs.real.item(), delta=5e-2 * (abs(lhs.real.item()) + abs(rhs.real.item()) + 1e-9))
        self.assertAlmostEqual(lhs.imag.item(), rhs.imag.item(), delta=5e-2 * (abs(lhs.imag.item()) + abs(rhs.imag.item()) + 1e-9))
        print("TestEMForwardOperator: Dot product test PASSED (within tolerance for rotation/interpolation).")


@unittest.skipIf(not RECONLIB_EM_AVAILABLE, "reconlib.modalities.em module or its dependencies not available")
class TestEMReconstructors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.D, self.H, self.W = 8, 16, 16
        self.volume_shape_em = (self.D, self.H, self.W)
        self.num_angles = 5
        self.angles_rad = torch.tensor(np.linspace(0, np.pi, self.num_angles, endpoint=False),
                                       dtype=torch.float32, device=self.device)
        self.projection_axis = 0
        self.single_proj_shape = (self.H, self.W)

        self.em_operator_inst = EMForwardOperator(
            volume_shape=self.volume_shape_em,
            angles_rad=self.angles_rad,
            projection_axis=self.projection_axis,
            device=self.device
        )

        self.y_projections = torch.randn(
            (self.num_angles,) + self.single_proj_shape,
            dtype=torch.complex64, device=self.device
        )

    def test_tv_reconstruction_em_execution(self):
        # Test basic execution and output shape
        recon_volume = tv_reconstruction_em(
            y_projections=self.y_projections,
            em_operator=self.em_operator_inst,
            lambda_tv=0.001,
            iterations=1,    # Minimal iterations for 3D test speed
            step_size=0.01,
            tv_prox_iterations=1, # Minimal inner iterations
            tv_prox_step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_volume.shape, self.volume_shape_em)
        self.assertTrue(recon_volume.is_complex())
        self.assertEqual(recon_volume.device, self.device)
        print("TestEMReconstructors: tv_reconstruction_em execution OK.")

if __name__ == '__main__':
    if RECONLIB_EM_AVAILABLE:
        unittest.main()
    else:
        print("Skipping EM tests as module or its dependencies are not available.")
