import unittest
import torch

try:
    from reconlib.modalities.sim.operators import SIMOperator
    from reconlib.modalities.sim.reconstructors import fourier_domain_sim_reconstruction
    from reconlib.modalities.sim.utils import generate_sim_patterns
    from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf
except ImportError:
    print("Local import fallback for SIMReconstructor test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.sim.operators import SIMOperator
    from reconlib.modalities.sim.reconstructors import fourier_domain_sim_reconstruction
    from reconlib.modalities.sim.utils import generate_sim_patterns
    from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf

class TestSIMReconstructor(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hr_image_shape = (32, 32) # Small for testing
        self.num_angles = 2
        self.num_phases = 2
        self.num_patterns = self.num_angles * self.num_phases

        self.psf_detection = generate_gaussian_psf(
            shape=(5,5), sigma=1.0, device=self.device
        ).to(torch.float32)

        # Reconstructor needs a SIMOperator instance
        self.sim_operator = SIMOperator(
            hr_image_shape=self.hr_image_shape,
            psf_detection=self.psf_detection,
            num_angles=self.num_angles,
            num_phases=self.num_phases,
            device=self.device
        )

        # Dummy raw SIM images stack
        self.raw_sim_images_stack_test = torch.rand(
            (self.num_patterns, *self.hr_image_shape),
            device=self.device,
            dtype=torch.float32
        )

    def test_reconstructor_runs_and_shape(self):
        print("\nTesting fourier_domain_sim_reconstruction execution and output shape...")
        reconstructed_hr = fourier_domain_sim_reconstruction(
            raw_sim_images_stack=self.raw_sim_images_stack_test,
            sim_operator=self.sim_operator,
            wiener_reg=0.1
        )
        self.assertEqual(reconstructed_hr.shape, self.hr_image_shape)
        self.assertEqual(reconstructed_hr.dtype, torch.float32) # Should output real image
        self.assertTrue(torch.all(reconstructed_hr >= 0)) # Check non-negativity constraint
        print(f"fourier_domain_sim_reconstruction output shape {reconstructed_hr.shape} and dtype {reconstructed_hr.dtype} correct.")

    def test_reconstructor_with_different_otf_params(self):
        print("\nTesting fourier_domain_sim_reconstruction with different OTF params...")
        reconstructed_hr = fourier_domain_sim_reconstruction(
            raw_sim_images_stack=self.raw_sim_images_stack_test,
            sim_operator=self.sim_operator,
            otf_cutoff_rel=0.8, # Different cutoff
            wiener_reg=0.01     # Different Wiener reg
        )
        self.assertEqual(reconstructed_hr.shape, self.hr_image_shape)
        print("fourier_domain_sim_reconstruction ran with different OTF params.")


if __name__ == '__main__':
    print("Running SIMReconstructor tests directly...")
    unittest.main()
