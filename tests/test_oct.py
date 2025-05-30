import unittest
import torch
import numpy as np # For np.isclose in main block, if used directly
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.modalities.oct.operators import OCTForwardOperator
    from reconlib.modalities.oct.reconstructors import tv_reconstruction_oct
    # For tv_reconstruction_oct, it internally uses UltrasoundTVCustomRegularizer
    # So, that path also needs to be valid for this test to import fully.
    from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
    RECONLIB_OCT_AVAILABLE = True
except ImportError as e:
    print(f"Could not import OCT (and potentially ultrasound regularizer) modules for testing: {e}")
    RECONLIB_OCT_AVAILABLE = False


@unittest.skipIf(not RECONLIB_OCT_AVAILABLE, "reconlib.modalities.oct module or its dependencies not available")
class TestOCTForwardOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ascans = 16 # Smaller for faster tests
        self.depth_pixels = 32
        self.image_shape_oct = (self.num_ascans, self.depth_pixels)

        self.operator_params = {
            'image_shape': self.image_shape_oct,
            'lambda_w': 850e-9, # 850 nm
            'z_max_m': 0.002,    # 2 mm
            'n_refractive_index': 1.35,
            'device': self.device
        }
        self.oct_op = OCTForwardOperator(**self.operator_params)

        # Create a simple phantom (reflectivity profile)
        self.phantom = torch.randn(self.image_shape_oct, dtype=torch.complex64, device=self.device)

        # Create dummy k-space data for adjoint testing
        self.dummy_k_space_data = torch.randn(
            self.image_shape_oct, dtype=torch.complex64, device=self.device
        )

    def test_operator_instantiation(self):
        self.assertIsInstance(self.oct_op, OCTForwardOperator)
        print("TestOCTForwardOperator: Instantiation OK.")

    def test_forward_op_shape_dtype(self):
        k_space_simulated = self.oct_op.op(self.phantom)
        self.assertEqual(k_space_simulated.shape, self.image_shape_oct)
        self.assertTrue(k_space_simulated.is_complex())
        self.assertEqual(k_space_simulated.device, self.device)
        print("TestOCTForwardOperator: Forward op shape and dtype OK.")

    def test_adjoint_op_shape_dtype(self):
        recon_image = self.oct_op.op_adj(self.dummy_k_space_data)
        self.assertEqual(recon_image.shape, self.image_shape_oct)
        self.assertTrue(recon_image.is_complex())
        self.assertEqual(recon_image.device, self.device)
        print("TestOCTForwardOperator: Adjoint op shape and dtype OK.")

    def test_adjoint_consistency_fft_ifft(self):
        # For an operator that is just FFT (ortho), A^H A x should be x
        A_H_A_x = self.oct_op.op_adj(self.oct_op.op(self.phantom))
        self.assertTrue(torch.allclose(A_H_A_x, self.phantom, atol=1e-5),
                        "A^H A x is not close to x for FFT-based operator.")
        print("TestOCTForwardOperator: Adjoint consistency (A^H A x = x) OK.")

    def test_dot_product(self):
        x_dp = torch.randn_like(self.phantom)
        y_dp_rand = torch.randn_like(self.dummy_k_space_data)

        Ax = self.oct_op.op(x_dp)
        Aty = self.oct_op.op_adj(y_dp_rand)

        lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

        print(f"OCT Dot Product Test - LHS: {lhs.item():.6f}, RHS: {rhs.item():.6f}")
        # FFT/IFFT with 'ortho' norm should satisfy adjoint property very well
        self.assertAlmostEqual(lhs.real.item(), rhs.real.item(), delta=1e-5 * (abs(lhs.real.item()) + abs(rhs.real.item()) + 1e-9))
        self.assertAlmostEqual(lhs.imag.item(), rhs.imag.item(), delta=1e-5 * (abs(lhs.imag.item()) + abs(rhs.imag.item()) + 1e-9))
        print("TestOCTForwardOperator: Dot product test PASSED.")


@unittest.skipIf(not RECONLIB_OCT_AVAILABLE, "reconlib.modalities.oct module or its dependencies not available")
class TestOCTReconstructors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ascans = 16
        self.depth_pixels = 32
        self.image_shape_oct = (self.num_ascans, self.depth_pixels)

        # Use a real OCTForwardOperator instance
        self.oct_operator_inst = OCTForwardOperator(
            image_shape=self.image_shape_oct,
            lambda_w=850e-9,
            z_max_m=0.002,
            device=self.device
        )

        self.y_oct_data = torch.randn(
            self.image_shape_oct, dtype=torch.complex64, device=self.device
        )

    def test_tv_reconstruction_oct_execution(self):
        # Test basic execution and output shape
        recon_image = tv_reconstruction_oct(
            y_oct_data=self.y_oct_data,
            oct_operator=self.oct_operator_inst,
            lambda_tv=0.01,
            iterations=2, # Minimal iterations for speed
            step_size=0.01,
            tv_prox_iterations=1,
            tv_prox_step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_image.shape, self.image_shape_oct)
        self.assertTrue(recon_image.is_complex()) # Output of prox grad will match input type generally
        self.assertEqual(recon_image.device, self.device)
        print("TestOCTReconstructors: tv_reconstruction_oct execution OK.")

if __name__ == '__main__':
    if RECONLIB_OCT_AVAILABLE:
        unittest.main()
    else:
        print("Skipping OCT tests as module or its dependencies (e.g., ultrasound regularizer) are not available.")
