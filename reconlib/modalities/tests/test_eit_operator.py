import unittest
import torch
import numpy as np

# Attempt to import from reconlib.modalities.eit
try:
    from reconlib.modalities.eit.operators import EITOperator
except ImportError:
    print("Local import fallback for EITOperator test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.eit.operators import EITOperator

class TestEITOperator(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (16, 16) # Smaller for faster tests
        self.num_pixels = self.image_shape[0] * self.image_shape[1]
        self.num_measurements = 32 # Example number of EIT measurements

        # Test case 1: Operator generates its own random J
        self.eit_operator_random_J = EITOperator(
            image_shape=self.image_shape,
            num_measurements=self.num_measurements,
            device=self.device
        )

        # Test case 2: Provide a sensitivity matrix J
        self.custom_J = torch.randn(self.num_measurements, self.num_pixels,
                                    device=self.device, dtype=torch.float32)
        self.eit_operator_custom_J = EITOperator(
            image_shape=self.image_shape,
            num_measurements=self.num_measurements, # num_measurements derived from J if J is given
            sensitivity_matrix_J=self.custom_J,
            device=self.device
        )

        self.delta_sigma_test_data = torch.randn(self.image_shape, device=self.device, dtype=torch.float32)
        self.delta_v_test_data = torch.randn(self.num_measurements, device=self.device, dtype=torch.float32)

    def test_operator_instantiation(self):
        self.assertIsInstance(self.eit_operator_random_J, EITOperator)
        self.assertEqual(self.eit_operator_random_J.J.shape, (self.num_measurements, self.num_pixels))

        self.assertIsInstance(self.eit_operator_custom_J, EITOperator)
        self.assertTrue(torch.equal(self.eit_operator_custom_J.J, self.custom_J))
        print("EITOperator instantiated successfully (random J and custom J).")

    def test_forward_op_shape_dtype(self):
        # Test with random J operator
        delta_v = self.eit_operator_random_J.op(self.delta_sigma_test_data)
        self.assertEqual(delta_v.shape, (self.num_measurements,))
        self.assertEqual(delta_v.dtype, self.delta_sigma_test_data.dtype) # Assuming J matches or input is cast

        # Test with custom J operator
        delta_v_custom = self.eit_operator_custom_J.op(self.delta_sigma_test_data)
        self.assertEqual(delta_v_custom.shape, (self.num_measurements,))
        self.assertEqual(delta_v_custom.dtype, self.delta_sigma_test_data.dtype)
        print(f"EITOperator forward op output shape and dtype correct.")

    def test_adjoint_op_shape_dtype(self):
        # Test with random J operator
        delta_sigma_adj = self.eit_operator_random_J.op_adj(self.delta_v_test_data)
        self.assertEqual(delta_sigma_adj.shape, self.image_shape)
        self.assertEqual(delta_sigma_adj.dtype, self.delta_v_test_data.dtype)

        # Test with custom J operator
        delta_sigma_adj_custom = self.eit_operator_custom_J.op_adj(self.delta_v_test_data)
        self.assertEqual(delta_sigma_adj_custom.shape, self.image_shape)
        self.assertEqual(delta_sigma_adj_custom.dtype, self.delta_v_test_data.dtype)
        print(f"EITOperator adjoint op output shape and dtype correct.")

    def _run_dot_product_test(self, operator_instance, test_label):
        delta_sigma_dp = torch.randn(self.image_shape, device=self.device, dtype=torch.float32)
        delta_v_dp = torch.randn(self.num_measurements, device=self.device, dtype=torch.float32)

        # Ensure operator's J matrix is float32 for this test if it was not already
        if operator_instance.J.dtype != torch.float32:
            operator_instance.J = operator_instance.J.to(torch.float32)


        Ax = operator_instance.op(delta_sigma_dp)
        Aty = operator_instance.op_adj(delta_v_dp)

        lhs = torch.dot(Ax.flatten(), delta_v_dp.flatten())
        rhs = torch.dot(delta_sigma_dp.flatten(), Aty.flatten())

        print(f"EITOperator Dot Product Test ({test_label}): LHS = {lhs.item():.6f}, RHS = {rhs.item():.6f}")
        self.assertAlmostEqual(lhs.item(), rhs.item(), delta=1e-3, msg=f"Dot product test failed for EITOperator ({test_label}).")
        print(f"EITOperator dot product test passed ({test_label}).")

    def test_dot_product_random_J(self):
        self._run_dot_product_test(self.eit_operator_random_J, "Random J")

    def test_dot_product_custom_J(self):
        self._run_dot_product_test(self.eit_operator_custom_J, "Custom J")


if __name__ == '__main__':
    print("Running EITOperator tests directly...")
    unittest.main()
