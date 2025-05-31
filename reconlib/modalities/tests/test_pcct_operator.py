import unittest
import torch
import numpy as np

try:
    from reconlib.modalities.pcct.operators import PCCTProjectorOperator, simple_radon_transform, simple_back_projection
    from reconlib.operators import Operator
except ImportError:
    print("Local import fallback for PCCTProjectorOperator test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.pcct.operators import PCCTProjectorOperator, simple_radon_transform, simple_back_projection
    from reconlib.operators import Operator

class SimpleRadonLinearOperator(Operator): # Helper from previous test
    def __init__(self, image_shape, num_angles, num_detector_pixels, device):
        super().__init__(); self.image_shape=image_shape; self.num_angles=num_angles
        self.num_detector_pixels=num_detector_pixels; self.device=device
    def op(self, image): return simple_radon_transform(image, self.num_angles, self.num_detector_pixels, self.device)
    def op_adj(self, sinogram): return simple_back_projection(sinogram, self.image_shape, self.device)

class TestPCCTProjectorOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (16, 16)
        self.num_angles = 10
        self.num_detector_pixels = 20
        self.energy_bins_keV = [(20., 50.), (50., 80.)]
        self.num_bins = len(self.energy_bins_keV)
        self.source_photons_per_bin = torch.tensor([20000., 22000.], device=self.device, dtype=torch.float32) # Higher I0 for better noise statistics
        self.energy_scaling_factors = torch.tensor([1.0, 0.85], device=self.device, dtype=torch.float32)

        self.mu_map_test = torch.rand(self.image_shape, device=self.device, dtype=torch.float32) * 0.02 # Low attenuation for testing counts

        self.pcct_op_no_noise = PCCTProjectorOperator(
            image_shape=self.image_shape, num_angles=self.num_angles, num_detector_pixels=self.num_detector_pixels,
            energy_bins_keV=self.energy_bins_keV, source_photons_per_bin=self.source_photons_per_bin,
            energy_scaling_factors=self.energy_scaling_factors, add_poisson_noise=False, device=self.device
        )
        self.pcct_op_with_noise = PCCTProjectorOperator(
            image_shape=self.image_shape, num_angles=self.num_angles, num_detector_pixels=self.num_detector_pixels,
            energy_bins_keV=self.energy_bins_keV, source_photons_per_bin=self.source_photons_per_bin,
            energy_scaling_factors=self.energy_scaling_factors, add_poisson_noise=True, device=self.device
        )
        self.counts_stack_test_data = torch.rand( # For adjoint input
            (self.num_bins, self.num_angles, self.num_detector_pixels), device=self.device, dtype=torch.float32
        ) * 10000


    def test_operator_instantiation(self):
        self.assertIsInstance(self.pcct_op_no_noise, PCCTProjectorOperator)
        self.assertIsInstance(self.pcct_op_with_noise, PCCTProjectorOperator)
        self.assertFalse(self.pcct_op_no_noise.add_poisson_noise)
        self.assertTrue(self.pcct_op_with_noise.add_poisson_noise)
        print("PCCTProjectorOperator (with and without noise) instantiated.")

    def test_forward_op_shape_dtype(self):
        # No noise
        counts_no_noise = self.pcct_op_no_noise.op(self.mu_map_test)
        expected_shape = (self.num_bins, self.num_angles, self.num_detector_pixels)
        self.assertEqual(counts_no_noise.shape, expected_shape)
        self.assertEqual(counts_no_noise.dtype, torch.float32)

        # With noise
        counts_with_noise = self.pcct_op_with_noise.op(self.mu_map_test)
        self.assertEqual(counts_with_noise.shape, expected_shape)
        self.assertEqual(counts_with_noise.dtype, torch.float32) # Poisson output is float if input rate is float

        # Check if noise actually made a difference (statistically)
        # For high counts, Poisson(lambda) is very unlikely to be exactly lambda
        # unless lambda is integer and very low.
        self.assertFalse(torch.allclose(counts_no_noise, counts_with_noise, atol=1e-1),
                         "Noisy counts should differ from noise-free counts.")
        print(f"PCCT forward op (with/without noise) shape/dtype correct, noise effect observed.")

    def test_adjoint_op_shape_dtype(self):
        mu_adj_no_noise_op = self.pcct_op_no_noise.op_adj(self.counts_stack_test_data)
        self.assertEqual(mu_adj_no_noise_op.shape, self.image_shape)
        self.assertEqual(mu_adj_no_noise_op.dtype, torch.float32)

        mu_adj_with_noise_op = self.pcct_op_with_noise.op_adj(self.counts_stack_test_data)
        self.assertEqual(mu_adj_with_noise_op.shape, self.image_shape)
        self.assertEqual(mu_adj_with_noise_op.dtype, torch.float32)
        print(f"PCCT adjoint op output shape/dtype correct.")

    def test_radon_linear_part_dot_product(self):
        print("\nTesting Radon/Back-projection part of PCCT operator for adjointness...")
        radon_op_test = SimpleRadonLinearOperator(
            self.image_shape, self.num_angles, self.num_detector_pixels, self.device
        )
        x_radon_dp = torch.randn(self.image_shape, device=self.device, dtype=torch.float32)
        y_radon_dp = torch.randn(
            (self.num_angles, self.num_detector_pixels), device=self.device, dtype=torch.float32
        )
        Ax_radon = radon_op_test.op(x_radon_dp)
        Aty_radon = radon_op_test.op_adj(y_radon_dp)
        lhs_radon = torch.dot(Ax_radon.flatten(), y_radon_dp.flatten())
        rhs_radon = torch.dot(x_radon_dp.flatten(), Aty_radon.flatten())
        print(f"  Radon part Dot Test: LHS={lhs_radon.item():.4f}, RHS={rhs_radon.item():.4f}")
        self.assertTrue(np.isclose(lhs_radon.item(), rhs_radon.item(), rtol=0.2),
                        f"Dot product test for Radon part failed: LHS={lhs_radon.item()}, RHS={rhs_radon.item()}")
        print("  Radon part dot product test passed (with loose tolerance).")

if __name__ == '__main__':
    unittest.main()
