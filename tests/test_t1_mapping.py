import unittest
import torch
import math
import sys
import os

# Add project root to sys.path to allow importing from reconlib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from reconlib.modalities.MRI.T1_mapping import spgr_signal, fit_t1_vfa

class TestVFAT1Mapping(unittest.TestCase):
    def setUp(self):
        """Set up test parameters and synthetic data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running VFA T1 mapping tests on device: {self.device}")

        self.TR_ms = 15.0
        self.flip_angles_deg = torch.tensor([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], device=self.device)

        # Define ground truth parameters for a single pixel or small image
        self.true_T1_ms = torch.tensor([[1200.0, 800.0], [600.0, 1500.0]], device=self.device, dtype=torch.float32) # 2x2 T1 map
        self.true_M0 = torch.tensor([[2000.0, 1800.0], [2200.0, 1900.0]], device=self.device, dtype=torch.float32)   # 2x2 M0 map
        self.spatial_dims = self.true_T1_ms.shape

        # Generate synthetic signals
        _T1 = self.true_T1_ms.unsqueeze(0)
        _M0 = self.true_M0.unsqueeze(0)
        _FA_rad = (self.flip_angles_deg * math.pi / 180.0).view(-1, 1, 1)

        self.signals_clean = spgr_signal(_T1, _M0, _FA_rad, self.TR_ms)

        self.noise_level_percent = 5.0
        noise_std_dev = (self.noise_level_percent / 100.0) * torch.max(torch.abs(self.signals_clean)) # Use abs for complex signals if any
        # Ensure noise is scaled by a positive std dev, handle case where max signal is 0
        if noise_std_dev == 0: noise_std_dev = 1e-3
        self.signals_noisy = self.signals_clean + torch.randn_like(self.signals_clean) * noise_std_dev

        self.b1_map_ideal = torch.ones(self.spatial_dims, device=self.device, dtype=torch.float32)
        self.b1_map_varied = torch.tensor([[0.9, 1.1], [1.05, 0.95]], device=self.device, dtype=torch.float32)

        self.t1_tolerance_percent = 20.0 # Increased tolerance
        self.m0_tolerance_percent = 25.0 # Increased tolerance

    def test_spgr_signal_basic(self):
        """Test spgr_signal with scalar inputs and known values."""
        t1 = torch.tensor(1000.0, device=self.device)
        m0 = torch.tensor(1.0, device=self.device)
        fa_deg = torch.tensor(10.0, device=self.device)
        fa_rad = fa_deg * math.pi / 180.0
        tr = 15.0

        expected_signal = 0.08659
        calculated_signal = spgr_signal(t1, m0, fa_rad, tr)
        self.assertAlmostEqual(calculated_signal.item(), expected_signal, places=4)

    def test_spgr_signal_tensor_input(self):
        """Test spgr_signal with tensor inputs (like those in setUp)."""
        self.assertEqual(self.signals_clean.shape, (self.flip_angles_deg.shape[0], *self.spatial_dims))
        self.assertFalse(torch.isnan(self.signals_clean).any())
        self.assertFalse(torch.isinf(self.signals_clean).any())

    def _run_fit_and_check(self, signals_to_fit, true_t1, true_m0, b1_map_to_use, test_name_suffix="", optimizer_type='adam', lr=5e-2, iterations=200):
        T1_fit, M0_fit = fit_t1_vfa(
            signals_to_fit,
            self.flip_angles_deg,
            self.TR_ms,
            b1_map=b1_map_to_use,
            initial_T1_ms_guess=1000.0,
            initial_M0_guess=-1,
            num_iterations=iterations,
            learning_rate=lr,
            optimizer_type=optimizer_type,
            device=self.device,
            verbose=False
        )
        self.assertEqual(T1_fit.shape, self.spatial_dims)
        self.assertEqual(M0_fit.shape, self.spatial_dims)

        t1_rel_diff = torch.abs(T1_fit - true_t1) / true_t1 * 100
        print(f"\nMax T1 relative difference (%) {test_name_suffix}: {torch.max(t1_rel_diff).item():.2f}%")
        print(f"T1 values {test_name_suffix}: Fit:\n{T1_fit.cpu().numpy()}\nTrue:\n{true_t1.cpu().numpy()}")
        self.assertTrue(torch.all(t1_rel_diff < self.t1_tolerance_percent),
                        f"T1 fit out of tolerance {test_name_suffix}. Max diff: {torch.max(t1_rel_diff).item()}%")

        m0_rel_diff = torch.abs(M0_fit - true_m0) / true_m0 * 100
        print(f"Max M0 relative difference (%) {test_name_suffix}: {torch.max(m0_rel_diff).item():.2f}%")
        print(f"M0 values {test_name_suffix}: Fit:\n{M0_fit.cpu().numpy()}\nTrue:\n{true_m0.cpu().numpy()}")
        self.assertTrue(torch.all(m0_rel_diff < self.m0_tolerance_percent),
                        f"M0 fit out of tolerance {test_name_suffix}. Max diff: {torch.max(m0_rel_diff).item()}%")

        self.assertFalse(torch.isnan(T1_fit).any(), f"NaN in T1_fit {test_name_suffix}")
        self.assertFalse(torch.isinf(T1_fit).any(), f"Inf in T1_fit {test_name_suffix}")
        self.assertFalse(torch.isnan(M0_fit).any(), f"NaN in M0_fit {test_name_suffix}")
        self.assertFalse(torch.isinf(M0_fit).any(), f"Inf in M0_fit {test_name_suffix}")

    def test_fit_t1_vfa_clean_no_b1_map(self):
        """Test fit_t1_vfa with clean signals and no B1 map."""
        self._run_fit_and_check(self.signals_clean, self.true_T1_ms, self.true_M0, None, "clean_no_b1")

    def test_fit_t1_vfa_noisy_no_b1_map(self):
        """Test fit_t1_vfa with noisy signals and no B1 map."""
        self._run_fit_and_check(self.signals_noisy, self.true_T1_ms, self.true_M0, None, "noisy_no_b1", iterations=250, lr=0.1) # More iterations for noisy

    def test_fit_t1_vfa_clean_with_ideal_b1_map(self):
        """Test fit_t1_vfa with clean signals and an ideal (all ones) B1 map."""
        self._run_fit_and_check(self.signals_clean, self.true_T1_ms, self.true_M0, self.b1_map_ideal, "clean_ideal_b1")

    def test_fit_t1_vfa_noisy_with_varied_b1_map_corrected(self):
        """Test fit_t1_vfa with noisy signals (generated with varied B1) and correcting with the varied B1 map."""
        _T1 = self.true_T1_ms.unsqueeze(0)
        _M0 = self.true_M0.unsqueeze(0)
        _FA_rad_eff = (self.flip_angles_deg * math.pi / 180.0).view(-1, 1, 1) * self.b1_map_varied.unsqueeze(0)

        signals_clean_varied_b1 = spgr_signal(_T1, _M0, _FA_rad_eff, self.TR_ms)
        noise_std_dev = (self.noise_level_percent / 100.0) * torch.max(torch.abs(signals_clean_varied_b1))
        if noise_std_dev == 0: noise_std_dev = 1e-3
        signals_noisy_varied_b1 = signals_clean_varied_b1 + torch.randn_like(signals_clean_varied_b1) * noise_std_dev

        self._run_fit_and_check(signals_noisy_varied_b1, self.true_T1_ms, self.true_M0, self.b1_map_varied, "noisy_varied_b1_corrected", iterations=250, lr=0.1)

    def test_fit_t1_vfa_lbfgs_optimizer_clean(self):
        """Test fit_t1_vfa with LBFGS optimizer (on clean signals)."""
        # LBFGS needs fewer iterations typically, but each can be slower. lr is more like max step.
        self._run_fit_and_check(self.signals_clean, self.true_T1_ms, self.true_M0, None,
                                "clean_lbfgs", optimizer_type='lbfgs', lr=0.8, iterations=30)

if __name__ == '__main__':
    unittest.main()
