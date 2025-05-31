import unittest
import torch
import numpy as np

# Attempt to import from reconlib.modalities.sim
# This assumes that reconlib is in the PYTHONPATH or installed
try:
    from reconlib.modalities.sim.operators import SIMOperator
    from reconlib.modalities.sim.utils import generate_sim_patterns # For creating patterns
    from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf # For detection PSF
except ImportError:
    # Fallback for local testing if reconlib is not installed (e.g. running script directly from tests folder)
    # This requires adjusting sys.path or ensuring the script is run in an environment where reconlib is discoverable.
    # For a robust package structure, the first import should work when tests are run by a test runner.
    print("Local import fallback for SIMOperator test - ensure PYTHONPATH is set if reconlib is not installed.")
    import sys
    from pathlib import Path
    # Assuming tests are in reconlib/modalities/tests/
    # Go up three levels to reach the root of reconlib project if reconlib is the main folder
    # Or adjust based on actual project structure if reconlib is a subfolder of a larger project
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.sim.operators import SIMOperator
    from reconlib.modalities.sim.utils import generate_sim_patterns
    from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf


class TestSIMOperator(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hr_image_shape = (32, 32) # Smaller for faster tests
        self.num_angles = 2
        self.num_phases = 2
        self.num_patterns = self.num_angles * self.num_phases

        self.sim_patterns = generate_sim_patterns(
            self.hr_image_shape,
            num_angles=self.num_angles,
            num_phases=self.num_phases,
            device=self.device
        )

        self.psf_detection = generate_gaussian_psf(
            shape=(5,5),
            sigma=1.0,
            device=self.device
        )

        self.sim_operator = SIMOperator(
            hr_image_shape=self.hr_image_shape,
            num_patterns=self.num_patterns,
            psf_detection=self.psf_detection,
            patterns=self.sim_patterns,
            device=self.device
        )

        self.hr_image_test_data = torch.randn(self.hr_image_shape, device=self.device, dtype=torch.float32)
        self.raw_images_test_data = torch.randn(
            (self.num_patterns, *self.hr_image_shape),
            device=self.device,
            dtype=torch.float32
        )

    def test_operator_instantiation(self):
        self.assertIsInstance(self.sim_operator, SIMOperator)
        print("SIMOperator instantiated successfully for tests.")

    def test_forward_op_shape_dtype(self):
        raw_sim_images = self.sim_operator.op(self.hr_image_test_data)
        self.assertEqual(raw_sim_images.shape, (self.num_patterns, *self.hr_image_shape))
        self.assertEqual(raw_sim_images.dtype, self.hr_image_test_data.dtype)
        print(f"SIMOperator forward op output shape and dtype correct: {raw_sim_images.shape}, {raw_sim_images.dtype}")

    def test_adjoint_op_shape_dtype(self):
        hr_estimate = self.sim_operator.op_adj(self.raw_images_test_data)
        self.assertEqual(hr_estimate.shape, self.hr_image_shape)
        self.assertEqual(hr_estimate.dtype, self.raw_images_test_data.dtype) # Adjoint output matches input data type
        print(f"SIMOperator adjoint op output shape and dtype correct: {hr_estimate.shape}, {hr_estimate.dtype}")

    def test_dot_product(self):
        # Ensure data is float32 for dot product precision with this placeholder
        hr_image_dp = torch.randn(self.hr_image_shape, device=self.device, dtype=torch.float32)
        raw_images_dp = torch.randn(
            (self.num_patterns, *self.hr_image_shape),
            device=self.device,
            dtype=torch.float32
        )

        Ax = self.sim_operator.op(hr_image_dp)
        Aty = self.sim_operator.op_adj(raw_images_dp)

        # LHS: <Ax, y>
        lhs = torch.dot(Ax.flatten(), raw_images_dp.flatten())
        # RHS: <x, A^H y>
        rhs = torch.dot(hr_image_dp.flatten(), Aty.flatten())

        print(f"SIMOperator Dot Product Test: LHS = {lhs.item():.6f}, RHS = {rhs.item():.6f}")
        self.assertAlmostEqual(lhs.item(), rhs.item(), delta=1e-3, msg="Dot product test failed for SIMOperator.")
        print("SIMOperator dot product test passed.")

if __name__ == '__main__':
    # This allows running the tests directly using `python test_sim_operator.py`
    # It's also common to use a test runner like `python -m unittest discover reconlib/modalities/tests`
    print("Running SIMOperator tests directly...")
    unittest.main()
