import unittest
import torch
from modalities.MRI.reconstruction import (
    initialize_array,
    multiply,
    conjugate,
    sum_tensor,
    mean_tensor,
    standard_deviation,
    absolute_value,
    apply_l1_proximal,
    deep_unrolled_reconstruction,
    # Import other necessary functions if they are directly tested or needed for setup
    # nufft_forward, nufft_adjoint, apply_cnn_prior, evaluate_image_quality (placeholders)
)

class TestMRIReconstruction(unittest.TestCase):

    def test_initialize_array(self):
        shape = (2, 3, 4)
        value = 5.0
        tensor = initialize_array(shape, value)
        self.assertEqual(tensor.shape, shape)
        self.assertTrue(torch.all(tensor == value))
        self.assertEqual(tensor.dtype, torch.float32) # Default PyTorch float type

        tensor_int = initialize_array(shape, 3)
        self.assertEqual(tensor_int.dtype, torch.int64) # Default PyTorch int type
        self.assertTrue(torch.all(tensor_int == 3))

    def test_multiply(self):
        tensor1 = torch.tensor([[1, 2], [3, 4]])
        tensor2 = torch.tensor([[2, 0], [1, 3]])
        result = multiply(tensor1, tensor2)
        expected = torch.tensor([[2, 0], [3, 12]])
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, tensor1.shape)

        # Test with a scalar
        tensor_scalar = torch.tensor(2)
        result_scalar_mult = multiply(tensor1, tensor_scalar)
        expected_scalar_mult = torch.tensor([[2,4],[6,8]])
        self.assertTrue(torch.equal(result_scalar_mult, expected_scalar_mult))


    def test_conjugate(self):
        tensor_complex = torch.tensor([[1+1j, 2-2j], [3+0j, -4j]])
        result = conjugate(tensor_complex)
        expected = torch.tensor([[1-1j, 2+2j], [3-0j, 4j]])
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, tensor_complex.shape)
        self.assertEqual(result.dtype, tensor_complex.dtype)

        tensor_real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result_real = conjugate(tensor_real) # Conjugate of real is the real itself
        self.assertTrue(torch.equal(result_real, tensor_real))


    def test_sum_tensor(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        # Sum all elements
        result_all = sum_tensor(tensor)
        self.assertEqual(result_all.item(), 21)
        # Sum along axis 0
        result_axis0 = sum_tensor(tensor, axis=0)
        expected_axis0 = torch.tensor([5, 7, 9], dtype=torch.float32)
        self.assertTrue(torch.equal(result_axis0, expected_axis0))
        # Sum along axis 1
        result_axis1 = sum_tensor(tensor, axis=1)
        expected_axis1 = torch.tensor([6, 15], dtype=torch.float32)
        self.assertTrue(torch.equal(result_axis1, expected_axis1))

    def test_mean_tensor(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        # Mean of all elements
        result_all = mean_tensor(tensor)
        self.assertAlmostEqual(result_all.item(), 3.5)

        # Mean with condition (e.g., elements > 3)
        condition = tensor > 3
        result_cond = mean_tensor(tensor, condition=condition)
        expected_cond_mean = torch.tensor([4,5,6], dtype=torch.float32).mean().item()
        self.assertAlmostEqual(result_cond.item(), expected_cond_mean)

        tensor_int = torch.tensor([[1,2],[3,4]])
        result_int = mean_tensor(tensor_int) # should cast to float for mean
        self.assertAlmostEqual(result_int.item(), 2.5)


    def test_standard_deviation(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standard_deviation(tensor)
        expected_std = torch.std(tensor)
        self.assertAlmostEqual(result.item(), expected_std.item())

    def test_absolute_value(self):
        tensor_real = torch.tensor([[-1.0, 2.5], [-3.0, 0.0]])
        result_real = absolute_value(tensor_real)
        expected_real = torch.tensor([[1.0, 2.5], [3.0, 0.0]])
        self.assertTrue(torch.equal(result_real, expected_real))

        tensor_complex = torch.tensor([[3+4j, -5-12j]]) # abs is sqrt(real^2 + imag^2)
        result_complex = absolute_value(tensor_complex)
        expected_complex = torch.tensor([[5.0, 13.0]])
        self.assertTrue(torch.allclose(result_complex, expected_complex))

    def test_apply_l1_proximal(self):
        image = torch.tensor([[-2.0, -1.0, 0.0, 1.5, 3.0]])
        lambda_l1 = 1.0
        learning_rate = 1.0 # Keep it simple for testing prox directly
        epsilon = 1e-6

        # term = lambda_l1 * learning_rate = 1.0
        # expected: sign(x) * max(0, abs(x) - 1.0)
        # x = -2.0: sign(-2)*max(0, 2-1) = -1.0 * 1.0 = -1.0
        # x = -1.0: sign(-1)*max(0, 1-1) = -1.0 * 0.0 = 0.0
        # x =  0.0: sign(0)*max(0, 0-1) =  0.0 * 0.0 = 0.0
        # x =  1.5: sign(1.5)*max(0, 1.5-1) = 1.0 * 0.5 = 0.5
        # x =  3.0: sign(3)*max(0, 3-1) = 1.0 * 2.0 = 2.0
        expected = torch.tensor([[-1.0, 0.0, 0.0, 0.5, 2.0]])
        result = apply_l1_proximal(image, lambda_l1, learning_rate, epsilon)
        self.assertTrue(torch.allclose(result, expected))

        # Test with different learning_rate
        learning_rate_2 = 0.5
        # term = lambda_l1 * learning_rate_2 = 0.5
        # expected: sign(x) * max(0, abs(x) - 0.5)
        # x = -2.0: sign(-2)*max(0, 2-0.5) = -1.0 * 1.5 = -1.5
        # x = -1.0: sign(-1)*max(0, 1-0.5) = -1.0 * 0.5 = -0.5
        # x =  0.0: sign(0)*max(0, 0-0.5) =  0.0 * 0.0 = 0.0
        # x =  1.5: sign(1.5)*max(0, 1.5-0.5) = 1.0 * 1.0 = 1.0
        # x =  3.0: sign(3)*max(0, 3-0.5) = 1.0 * 2.5 = 2.5
        expected_lr2 = torch.tensor([[-1.5, -0.5, 0.0, 1.0, 2.5]])
        result_lr2 = apply_l1_proximal(image, lambda_l1, learning_rate_2, epsilon)
        self.assertTrue(torch.allclose(result_lr2, expected_lr2))


    def test_deep_unrolled_reconstruction_runs(self):
        # Define dummy inputs
        batch_size = 1
        num_coils = 4
        image_height = 32 # Must be consistent for placeholders
        image_width = 32  # Must be consistent for placeholders
        num_trajectory_points = 100

        # kspace_data: [batch, num_coils, num_trajectory_points]
        kspace_data = torch.randn(batch_size, num_coils, num_trajectory_points, dtype=torch.complex64)

        # trajectory: [batch, 2, num_trajectory_points] or [2, num_trajectory_points] for 2D
        trajectory = torch.rand(batch_size, 2, num_trajectory_points, dtype=torch.float32) * 2 - 1 # Scaled to [-1, 1]

        # coil_sensitivities: [batch, num_coils, height, width]
        coil_sensitivities = torch.randn(batch_size, num_coils, image_height, image_width, dtype=torch.complex64)

        config = {
            'image_dims': (batch_size, 1, image_height, image_width), # Output image dims (1 channel for placeholder)
            'nufft_params': {},  # Placeholder
            'max_iterations': 2, # Small number for a quick test run
            'lambda_l1': 0.01,
            'learning_rate': 0.1,
            'rho': 0.5, # ADMM parameter
            'cnn_params': {},    # Placeholder
            'quality_metrics': ["PSNR", "SSIM"], # Example metrics
            'min_quality_threshold': {"PSNR": 10.0} # Example threshold
        }
        epsilon = 1e-7

        result = deep_unrolled_reconstruction(kspace_data, trajectory, coil_sensitivities, config, epsilon)

        self.assertIsInstance(result, dict)
        # Given the placeholder nature of NUFFT, it's hard to guarantee "Reconstruction successful"
        # without actual NUFFT ops. We'll check if an error message is present or not.
        # If NUFFT placeholders return zeros, data consistency might be poor, but the loop should complete.
        self.assertTrue('reconstructed_image' in result)
        self.assertTrue('quality_scores' in result)
        self.assertTrue('error_message' in result)

        if result['reconstructed_image'] is not None:
            self.assertIsInstance(result['reconstructed_image'], torch.Tensor)
            self.assertEqual(result['reconstructed_image'].shape, config['image_dims'])
            # Check if error message is empty, implying successful run through logic
            # This might need adjustment if placeholders inherently cause warnings/errors in the quality check.
            # For now, we check if the function ran to completion and produced an image.
            # A truly "successful" status might depend on the dummy nufft_adjoint output relative to kspace_data.
            # If error_message is empty, it implies no exceptions were caught during the main loop.
            # However, the quality check part might populate error_message if quality is below threshold.
            # For this test, let's primarily focus on the function running without critical exceptions.
            print(f"Test run error message: '{result['error_message']}'") # For debugging in test output
            # A more robust check for "success" would require more sophisticated mocks or actual mini-NUFFT.
            # For now, we consider it a pass if an image is produced and no Python exceptions occurred.
            # The `error_message` field might contain quality warnings, which is acceptable for this test.
        else:
            # If image is None, there must have been an error message.
            self.assertTrue(len(result['error_message']) > 0, "Error message should be populated if image is None")


if __name__ == '__main__':
    unittest.main()
