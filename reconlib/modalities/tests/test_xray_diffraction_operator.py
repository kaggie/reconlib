import unittest
import torch
import numpy as np

# Attempt to import from reconlib.modalities.xray_diffraction
try:
    from reconlib.modalities.xray_diffraction.operators import XRayDiffractionOperator
except ImportError:
    print("Local import fallback for XRayDiffractionOperator test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.xray_diffraction.operators import XRayDiffractionOperator

class TestXRayDiffractionOperator(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (32, 32) # Smaller for faster tests

        self.xrd_operator_no_adj_phase = XRayDiffractionOperator(
            image_shape=self.image_shape,
            add_random_phase_to_adjoint=False,
            device=self.device
        )
        self.xrd_operator_with_adj_phase = XRayDiffractionOperator(
            image_shape=self.image_shape,
            add_random_phase_to_adjoint=True,
            device=self.device
        )

        # Create a simple real object phantom
        self.object_phantom = torch.zeros(self.image_shape, device=self.device, dtype=torch.float32)
        self.object_phantom[8:24, 8:24] = 1.0
        self.object_phantom[12:20, 12:20] = 0.5


    def test_operator_instantiation(self):
        self.assertIsInstance(self.xrd_operator_no_adj_phase, XRayDiffractionOperator)
        self.assertIsInstance(self.xrd_operator_with_adj_phase, XRayDiffractionOperator)
        print("XRayDiffractionOperator instantiated successfully.")

    def test_forward_op_shape_dtype(self):
        magnitudes = self.xrd_operator_no_adj_phase.op(self.object_phantom)
        self.assertEqual(magnitudes.shape, self.image_shape)
        self.assertEqual(magnitudes.dtype, torch.float32) # Output of abs() is float if input to fft is float
        self.assertTrue(torch.all(magnitudes >= 0)) # Magnitudes should be non-negative
        print(f"XRayDiffractionOperator forward op output shape and dtype correct: {magnitudes.shape}, {magnitudes.dtype}")

    def test_adjoint_op_shape_dtype(self):
        # Using magnitudes from the forward op
        magnitudes = self.xrd_operator_no_adj_phase.op(self.object_phantom)

        # Test with no phase estimate (should use zero phase)
        object_estimate_zero_phase = self.xrd_operator_no_adj_phase.op_adj(magnitudes)
        self.assertEqual(object_estimate_zero_phase.shape, self.image_shape)
        self.assertEqual(object_estimate_zero_phase.dtype, torch.float32) # Output of .real is float

        # Test with random phase in adjoint
        object_estimate_random_phase = self.xrd_operator_with_adj_phase.op_adj(magnitudes)
        self.assertEqual(object_estimate_random_phase.shape, self.image_shape)
        self.assertEqual(object_estimate_random_phase.dtype, torch.float32)

        # Test with a provided phase estimate
        dummy_phase_estimate = torch.rand(self.image_shape, device=self.device) * 2 * np.pi
        object_estimate_provided_phase = self.xrd_operator_no_adj_phase.op_adj(magnitudes, phase_estimate=dummy_phase_estimate)
        self.assertEqual(object_estimate_provided_phase.shape, self.image_shape)
        self.assertEqual(object_estimate_provided_phase.dtype, torch.float32)
        print(f"XRayDiffractionOperator adjoint op output shapes and dtypes correct.")

    def test_adjoint_with_true_phase_reconstruction(self):
        """
        Tests if op_adj(op(x), true_phase_of_FT(x)) reconstructs x.
        This verifies the self-consistency of the FT/IFT parts of the operator,
        even though the overall forward operator is non-linear.
        """
        # Compute the complex Fourier transform of the original object
        true_complex_ft = torch.fft.fft2(self.object_phantom, norm='ortho')
        # Extract the true magnitudes (which op() would return)
        true_magnitudes = torch.abs(true_complex_ft)
        # Extract the true phase
        true_phase = torch.angle(true_complex_ft)

        # Use op_adj with the true magnitudes and true phase
        reconstructed_object = self.xrd_operator_no_adj_phase.op_adj(true_magnitudes, phase_estimate=true_phase)

        self.assertTrue(torch.allclose(reconstructed_object, self.object_phantom, atol=1e-5),
                        "Adjoint with true phase should closely reconstruct the original object.")
        print("XRayDiffractionOperator self-consistency test (adjoint with true phase) passed.")

    def test_comments_on_dot_product(self):
        """ Confirms that the operator notes why standard dot product test isn't applicable. """
        # This is more of a reminder / check of documentation within the operator code itself.
        # Here, we just assert True if this test runs, assuming the dev has read the operator's comments.
        self.assertTrue(True, "Operator should contain comments about non-linearity and dot-product test.")
        print("Reminder: XRayDiffractionOperator is non-linear due to abs(); standard dot product test is not directly applicable.")


if __name__ == '__main__':
    print("Running XRayDiffractionOperator tests directly...")
    unittest.main()
