import unittest
import torch
import numpy as np
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.modalities.xray_phase_contrast.operators import XRayPhaseContrastOperator
    from reconlib.modalities.xray_phase_contrast.reconstructors import tv_reconstruction_xrpc
    # tv_reconstruction_xrpc uses UltrasoundTVCustomRegularizer
    from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
    RECONLIB_XPCI_AVAILABLE = True
except ImportError as e:
    print(f"Could not import XPCI (and potentially ultrasound regularizer) modules for testing: {e}")
    RECONLIB_XPCI_AVAILABLE = False


@unittest.skipIf(not RECONLIB_XPCI_AVAILABLE, "reconlib.modalities.xray_phase_contrast module or its dependencies not available")
class TestXRayPhaseContrastOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.H, self.W = 32, 32
        self.image_shape_xrpc = (self.H, self.W)

        self.k_wave_number = 2 * np.pi / 1e-10 # Example k
        self.pixel_size = 1e-5 # Example pixel size

        self.operator_params = {
            'image_shape': self.image_shape_xrpc,
            'k_wave_number': self.k_wave_number,
            'pixel_size_m': self.pixel_size,
            'device': self.device
        }
        self.xrpc_op = XRayPhaseContrastOperator(**self.operator_params)

        # Phase phantom - should be float, gradients are real
        self.phantom_phase = torch.randn(self.image_shape_xrpc, dtype=torch.float32, device=self.device)

        # Dummy differential phase data for adjoint testing - should match output of op
        self.dummy_diff_phase_data = torch.randn(
            self.image_shape_xrpc, dtype=torch.float32, device=self.device
        )

    def test_operator_instantiation(self):
        self.assertIsInstance(self.xrpc_op, XRayPhaseContrastOperator)
        print("TestXRayPhaseContrastOperator: Instantiation OK.")

    def test_forward_op_shape_dtype(self):
        diff_phase_simulated = self.xrpc_op.op(self.phantom_phase)
        self.assertEqual(diff_phase_simulated.shape, self.image_shape_xrpc)
        # Output can be float if input phase is float and k is float
        self.assertTrue(diff_phase_simulated.dtype == torch.float32 or diff_phase_simulated.is_complex())
        self.assertEqual(diff_phase_simulated.device, self.device)
        print("TestXRayPhaseContrastOperator: Forward op shape and dtype OK.")

    def test_adjoint_op_shape_dtype(self):
        recon_phase_image = self.xrpc_op.op_adj(self.dummy_diff_phase_data)
        self.assertEqual(recon_phase_image.shape, self.image_shape_xrpc)
        self.assertTrue(recon_phase_image.dtype == torch.float32 or recon_phase_image.is_complex())
        self.assertEqual(recon_phase_image.device, self.device)
        print("TestXRayPhaseContrastOperator: Adjoint op shape and dtype OK.")

    def test_dot_product(self):
        x_dp = torch.randn_like(self.phantom_phase) # Real phase image
        y_dp_rand = torch.randn_like(self.dummy_diff_phase_data) # Real diff phase data

        Ax = self.xrpc_op.op(x_dp)
        Aty = self.xrpc_op.op_adj(y_dp_rand)

        # Ensure complex for vdot if inputs are real, as vdot expects complex or real pairs
        Ax_c = Ax.to(torch.complex64) if not Ax.is_complex() else Ax
        y_dp_rand_c = y_dp_rand.to(torch.complex64) if not y_dp_rand.is_complex() else y_dp_rand
        x_dp_c = x_dp.to(torch.complex64) if not x_dp.is_complex() else x_dp
        Aty_c = Aty.to(torch.complex64) if not Aty.is_complex() else Aty

        lhs = torch.vdot(Ax_c.flatten(), y_dp_rand_c.flatten())
        rhs = torch.vdot(x_dp_c.flatten(), Aty_c.flatten())

        print(f"XPCI Dot Product Test - LHS: {lhs.item():.4e}, RHS: {rhs.item():.4e}")
        # Finite difference operators should satisfy this well.
        self.assertAlmostEqual(lhs.real.item(), rhs.real.item(), delta=1e-4 * (abs(lhs.real.item()) + abs(rhs.real.item()) + 1e-9))
        # Imaginary part should be close to zero if inputs are real and ops are real
        if not x_dp.is_complex() and not y_dp_rand.is_complex():
             self.assertAlmostEqual(lhs.imag.item(), 0.0, delta=1e-5)
             self.assertAlmostEqual(rhs.imag.item(), 0.0, delta=1e-5)
        else: # If inputs can be complex, compare imag parts
             self.assertAlmostEqual(lhs.imag.item(), rhs.imag.item(), delta=1e-4 * (abs(lhs.imag.item()) + abs(rhs.imag.item()) + 1e-9))
        print("TestXRayPhaseContrastOperator: Dot product test PASSED.")


@unittest.skipIf(not RECONLIB_XPCI_AVAILABLE, "reconlib.modalities.xray_phase_contrast module or its dependencies not available")
class TestXPCReconstructors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.H, self.W = 32,32
        self.image_shape_xrpc = (self.H, self.W)

        self.k_wave_number = 2 * np.pi / 1e-10
        self.pixel_size = 1e-5

        self.xrpc_operator_inst = XRayPhaseContrastOperator(
            image_shape=self.image_shape_xrpc,
            k_wave_number=self.k_wave_number,
            pixel_size_m=self.pixel_size,
            device=self.device
        )

        self.y_diff_phase_data = torch.randn(
            self.image_shape_xrpc, dtype=torch.float32, device=self.device
        )

    def test_tv_reconstruction_xrpc_execution(self):
        # Test basic execution and output shape
        recon_phase_image = tv_reconstruction_xrpc(
            y_differential_phase_data=self.y_diff_phase_data,
            xrpc_operator=self.xrpc_operator_inst,
            lambda_tv=0.001,
            iterations=2,    # Minimal iterations for speed
            step_size=0.01,
            tv_prox_iterations=1,
            tv_prox_step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_phase_image.shape, self.image_shape_xrpc)
        # Output should be float if input measurement and initial phase are float
        self.assertTrue(recon_phase_image.dtype == torch.float32 or recon_phase_image.is_complex())
        self.assertEqual(recon_phase_image.device, self.device)
        print("TestXPCReconstructors: tv_reconstruction_xrpc execution OK.")

if __name__ == '__main__':
    if RECONLIB_XPCI_AVAILABLE:
        unittest.main()
    else:
        print("Skipping XPCI tests as module or its dependencies are not available.")
