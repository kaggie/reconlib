import unittest
import torch
import numpy as np

try:
    from reconlib.modalities.pcct.operators import PCCTProjectorOperator # For parameters
    from reconlib.modalities.pcct.reconstructors import tv_reconstruction_pcct_mu_ref, iterative_reconstruction_pcct_bin
    from reconlib.modalities.pcct.utils import generate_pcct_phantom_material_maps, combine_material_maps_to_mu_ref, get_pcct_energy_scaling_factors
    # For LinearRadonOperator used by iterative_reconstruction_pcct_bin's test in reconstructors.py
    from reconlib.modalities.pcct.reconstructors import LinearRadonOperator
except ImportError:
    print("Local import fallback for PCCTReconstructor test - ensure PYTHONPATH.")
    import sys
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(base_path))
    from reconlib.modalities.pcct.operators import PCCTProjectorOperator
    from reconlib.modalities.pcct.reconstructors import tv_reconstruction_pcct_mu_ref, iterative_reconstruction_pcct_bin, LinearRadonOperator
    from reconlib.modalities.pcct.utils import generate_pcct_phantom_material_maps, combine_material_maps_to_mu_ref, get_pcct_energy_scaling_factors


class TestPCCTReconstructors(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (16, 16) # Small for testing
        self.num_angles = 10
        self.num_detector_pixels = 20
        self.energy_bins_keV = [(20., 50.), (50., 80.)]
        self.num_bins = len(self.energy_bins_keV)
        self.source_photons_per_bin = torch.tensor([10000., 12000.], device=self.device, dtype=torch.float32)
        self.energy_scaling_factors = get_pcct_energy_scaling_factors(
            self.energy_bins_keV, device=self.device
        ) # Use the utility

        # Operator for tv_reconstruction_pcct_mu_ref
        self.pcct_operator_global = PCCTProjectorOperator(
            image_shape=self.image_shape,
            num_angles=self.num_angles,
            num_detector_pixels=self.num_detector_pixels,
            energy_bins_keV=self.energy_bins_keV,
            source_photons_per_bin=self.source_photons_per_bin,
            energy_scaling_factors=self.energy_scaling_factors,
            add_poisson_noise=False, # For reconstructor test, start with clean data usually
            device=self.device
        )

        # Dummy noisy counts stack for global recon
        self.noisy_counts_stack_global = torch.rand(
            (self.num_bins, self.num_angles, self.num_detector_pixels),
            device=self.device, dtype=torch.float32
        ) * self.source_photons_per_bin.view(self.num_bins, 1, 1) * 0.5 + 100

        # Dummy noisy counts for single bin recon
        self.noisy_counts_single_bin = torch.rand(
            (self.num_angles, self.num_detector_pixels),
            device=self.device, dtype=torch.float32
        ) * self.source_photons_per_bin[0] * 0.5 + 100


    def test_tv_reconstruction_pcct_mu_ref_runs(self):
        print("\nTesting tv_reconstruction_pcct_mu_ref execution and output shape...")
        reconstructed_mu_ref = tv_reconstruction_pcct_mu_ref(
            y_photon_counts_stack=self.noisy_counts_stack_global,
            pcct_operator=self.pcct_operator_global,
            lambda_tv=1e-5,
            iterations=3, # Quick test
            step_size=1e-7, # Very small for stability with this placeholder
            verbose=False
        )
        self.assertEqual(reconstructed_mu_ref.shape, self.image_shape)
        self.assertEqual(reconstructed_mu_ref.dtype, torch.float32)
        self.assertTrue(torch.all(reconstructed_mu_ref >= 0)) # Non-negativity
        print(f"tv_reconstruction_pcct_mu_ref output shape {reconstructed_mu_ref.shape} and dtype {reconstructed_mu_ref.dtype} correct.")


    def test_iterative_reconstruction_pcct_bin_runs(self):
        print("\nTesting iterative_reconstruction_pcct_bin execution and output shape...")
        selected_bin_idx = 0
        I0_single_bin = self.source_photons_per_bin[selected_bin_idx]

        reconstructed_mu_eff_bin = iterative_reconstruction_pcct_bin(
            noisy_counts_sinogram_bin=self.noisy_counts_single_bin,
            source_photons_bin=I0_single_bin,
            image_shape=self.image_shape,
            num_angles=self.num_angles,
            num_detector_pixels=self.num_detector_pixels,
            lambda_tv=0.001,
            pgd_iterations=3, # Quick test
            device=self.device,
            verbose=False
        )
        self.assertEqual(reconstructed_mu_eff_bin.shape, self.image_shape)
        self.assertEqual(reconstructed_mu_eff_bin.dtype, torch.float32)
        print(f"iterative_reconstruction_pcct_bin output shape {reconstructed_mu_eff_bin.shape} and dtype {reconstructed_mu_eff_bin.dtype} correct.")

if __name__ == '__main__':
    print("Running PCCTReconstructor tests directly...")
    unittest.main()
