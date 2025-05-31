import unittest
import torch
import numpy as np

try:
    from reconlib.modalities.sim.operators import SIMOperator
    from reconlib.modalities.sim.utils import generate_sim_patterns
    from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf
except ImportError:
    print("Local import fallback for SIMOperator test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.sim.operators import SIMOperator
    from reconlib.modalities.sim.utils import generate_sim_patterns
    from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf

class TestSIMOperator(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hr_image_shape = (32, 32)
        self.num_angles = 2
        self.num_phases = 2
        self.num_patterns = self.num_angles * self.num_phases
        self.pattern_k_max_rel = 0.7

        self.psf_detection = generate_gaussian_psf(
            shape=(5,5),
            sigma=1.0,
            device=self.device
        ).to(torch.float32) # Ensure PSF is float32

        # For testing with externally provided patterns
        self.external_patterns = generate_sim_patterns(
            self.hr_image_shape,
            num_angles=self.num_angles,
            num_phases=self.num_phases,
            k_vector_max_rel=self.pattern_k_max_rel,
            device=self.device
        ).to(torch.float32)

        # Operator that will generate patterns internally
        self.sim_operator_internal_pats = SIMOperator(
            hr_image_shape=self.hr_image_shape,
            psf_detection=self.psf_detection,
            num_angles=self.num_angles, # Provide these for internal generation
            num_phases=self.num_phases, # Provide these
            pattern_k_max_rel=self.pattern_k_max_rel, # Provide this
            patterns=None, # Force internal generation
            device=self.device
        )

        # Operator that uses externally provided patterns
        self.sim_operator_external_pats = SIMOperator(
            hr_image_shape=self.hr_image_shape,
            psf_detection=self.psf_detection,
            patterns=self.external_patterns, # Provide patterns
            device=self.device
        )

        self.hr_image_test_data = torch.randn(self.hr_image_shape, device=self.device, dtype=torch.float32)
        self.raw_images_test_data = torch.randn(
            (self.num_patterns, *self.hr_image_shape),
            device=self.device,
            dtype=torch.float32
        )

    def test_operator_instantiation_internal_patterns(self):
        op = self.sim_operator_internal_pats
        self.assertIsInstance(op, SIMOperator)
        self.assertEqual(op.patterns.shape, (self.num_patterns, *self.hr_image_shape))
        print("SIMOperator (internal patterns) instantiated successfully.")

    def test_operator_instantiation_external_patterns(self):
        op = self.sim_operator_external_pats
        self.assertIsInstance(op, SIMOperator)
        self.assertTrue(torch.equal(op.patterns, self.external_patterns))
        print("SIMOperator (external patterns) instantiated successfully.")

    def _run_op_tests(self, operator_instance, op_label):
        raw_sim_images = operator_instance.op(self.hr_image_test_data)
        self.assertEqual(raw_sim_images.shape, (operator_instance.num_patterns, *self.hr_image_shape))
        self.assertEqual(raw_sim_images.dtype, self.hr_image_test_data.dtype)
        print(f"SIMOperator forward op ({op_label}) output shape and dtype correct.")

        hr_estimate = operator_instance.op_adj(self.raw_images_test_data)
        self.assertEqual(hr_estimate.shape, self.hr_image_shape)
        # Adjoint output dtype should match input stack dtype after internal casting consistency
        self.assertEqual(hr_estimate.dtype, self.raw_images_test_data.dtype)
        print(f"SIMOperator adjoint op ({op_label}) output shape and dtype correct.")

    def test_ops_internal_patterns(self):
        self._run_op_tests(self.sim_operator_internal_pats, "internal patterns")

    def test_ops_external_patterns(self):
        self._run_op_tests(self.sim_operator_external_pats, "external patterns")

    def _run_dot_product_test(self, operator_instance, test_label):
        hr_image_dp = torch.randn(self.hr_image_shape, device=self.device, dtype=torch.float32)
        # Ensure raw_images_dp matches the num_patterns of the specific operator instance
        raw_images_dp = torch.randn(
            (operator_instance.num_patterns, *self.hr_image_shape),
            device=self.device,
            dtype=torch.float32
        )

        Ax = operator_instance.op(hr_image_dp)
        Aty = operator_instance.op_adj(raw_images_dp)

        lhs = torch.dot(Ax.flatten(), raw_images_dp.flatten())
        rhs = torch.dot(hr_image_dp.flatten(), Aty.flatten())

        print(f"SIMOperator Dot Product Test ({test_label}): LHS = {lhs.item():.6f}, RHS = {rhs.item():.6f}")
        # Increased delta slightly due to multiple ops and potential float32 acc.
        self.assertAlmostEqual(lhs.item(), rhs.item(), delta=1e-2, msg=f"Dot product test failed for SIMOperator ({test_label}).")
        print(f"SIMOperator dot product test passed ({test_label}).")

    def test_dot_product_internal_patterns(self):
        self._run_dot_product_test(self.sim_operator_internal_pats, "Internal Patterns")

    def test_dot_product_external_patterns(self):
        self._run_dot_product_test(self.sim_operator_external_pats, "External Patterns")

if __name__ == '__main__':
    print("Running SIMOperator tests directly...")
    unittest.main()
