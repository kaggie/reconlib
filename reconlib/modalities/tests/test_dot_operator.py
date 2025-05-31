import unittest
import torch
import numpy as np

# Attempt to import from reconlib.modalities.dot
try:
    from reconlib.modalities.dot.operators import DOTOperator
except ImportError:
    print("Local import fallback for DOTOperator test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.dot.operators import DOTOperator

class TestDOTOperator(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (16, 16) # Smaller for faster tests
        self.num_pixels = self.image_shape[0] * self.image_shape[1]
        self.num_measurements = 24 # Example number of DOT measurements

        # Test case 1: Operator generates its own random J
        self.dot_operator_random_J = DOTOperator(
            image_shape=self.image_shape,
            num_measurements=self.num_measurements,
            device=self.device
        )

        # Test case 2: Provide a sensitivity matrix J
        self.custom_J = torch.randn(self.num_measurements, self.num_pixels,
                                    device=self.device, dtype=torch.float32)
        self.dot_operator_custom_J = DOTOperator(
            image_shape=self.image_shape,
            num_measurements=self.num_measurements,
            sensitivity_matrix_J=self.custom_J,
            device=self.device
        )

        self.delta_mu_test_data = torch.randn(self.image_shape, device=self.device, dtype=torch.float32)
        self.delta_y_test_data = torch.randn(self.num_measurements, device=self.device, dtype=torch.float32)

    def test_operator_instantiation(self):
        self.assertIsInstance(self.dot_operator_random_J, DOTOperator)
        self.assertEqual(self.dot_operator_random_J.J.shape, (self.num_measurements, self.num_pixels))

        self.assertIsInstance(self.dot_operator_custom_J, DOTOperator)
        self.assertTrue(torch.equal(self.dot_operator_custom_J.J, self.custom_J))
        print("DOTOperator instantiated successfully (random J and custom J).")

    def test_forward_op_shape_dtype(self):
        # Test with random J operator
        delta_y = self.dot_operator_random_J.op(self.delta_mu_test_data)
        self.assertEqual(delta_y.shape, (self.num_measurements,))
        self.assertEqual(delta_y.dtype, self.delta_mu_test_data.dtype)

        # Test with custom J operator
        delta_y_custom = self.dot_operator_custom_J.op(self.delta_mu_test_data)
        self.assertEqual(delta_y_custom.shape, (self.num_measurements,))
        self.assertEqual(delta_y_custom.dtype, self.delta_mu_test_data.dtype)
        print(f"DOTOperator forward op output shape and dtype correct.")

    def test_adjoint_op_shape_dtype(self):
        # Test with random J operator
        delta_mu_adj = self.dot_operator_random_J.op_adj(self.delta_y_test_data)
        self.assertEqual(delta_mu_adj.shape, self.image_shape)
        self.assertEqual(delta_mu_adj.dtype, self.delta_y_test_data.dtype)

        # Test with custom J operator
        delta_mu_adj_custom = self.dot_operator_custom_J.op_adj(self.delta_y_test_data)
        self.assertEqual(delta_mu_adj_custom.shape, self.image_shape)
        self.assertEqual(delta_mu_adj_custom.dtype, self.delta_y_test_data.dtype)
        print(f"DOTOperator adjoint op output shape and dtype correct.")

    def _run_dot_product_test(self, operator_instance, test_label):
        # Ensure J is float32 for consistent testing if it was randomly init with other types
        if operator_instance.J.dtype != torch.float32:
            operator_instance.J = operator_instance.J.to(torch.float32)

        delta_mu_dp = torch.randn(self.image_shape, device=self.device, dtype=torch.float32)
        delta_y_dp = torch.randn(self.num_measurements, device=self.device, dtype=torch.float32)

        Ax = operator_instance.op(delta_mu_dp)
        Aty = operator_instance.op_adj(delta_y_dp)

        lhs = torch.dot(Ax.flatten(), delta_y_dp.flatten())
        rhs = torch.dot(delta_mu_dp.flatten(), Aty.flatten())

        print(f"DOTOperator Dot Product Test ({test_label}): LHS = {lhs.item():.6f}, RHS = {rhs.item():.6f}")
        self.assertAlmostEqual(lhs.item(), rhs.item(), delta=1e-3, msg=f"Dot product test failed for DOTOperator ({test_label}).")
        print(f"DOTOperator dot product test passed ({test_label}).")

    def test_dot_product_random_J(self):
        self._run_dot_product_test(self.dot_operator_random_J, "Random J")

    def test_dot_product_custom_J(self):
        self._run_dot_product_test(self.dot_operator_custom_J, "Custom J")

if __name__ == '__main__':
    print("Running DOTOperator tests directly...")
    unittest.main()
