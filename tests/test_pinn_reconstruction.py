import unittest
import torch
import sys
import os

# Add project root to sys.path to allow importing from modalities and reconlib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modalities.MRI.physics_loss import calculate_bloch_residual, calculate_girf_gradient_error
from modalities.MRI.pinn_reconstructor import SimpleCNN, PINNReconstructor
from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator # Assuming this can be imported

class TestPhysicsLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_calculate_bloch_residual(self):
        # image_estimate: (batch_size, num_maps, Z, Y, X)
        dummy_image_estimate = torch.rand(1, 1, 8, 8, 8, device=self.device)
        dummy_scan_params = {"TE": 0.05, "TR": 2.0}
        loss = calculate_bloch_residual(dummy_image_estimate, dummy_scan_params)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0) # Placeholder returns 0

    def test_calculate_girf_gradient_error(self):
        # kspace_trajectory_ideal, kspace_trajectory_actual: (num_points, dims)
        dummy_traj_ideal = torch.rand(100, 3, device=self.device)
        dummy_traj_actual = torch.rand(100, 3, device=self.device)
        loss = calculate_girf_gradient_error(dummy_traj_ideal, dummy_traj_actual)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0) # Placeholder returns 0

        with self.assertRaises(ValueError): # Test shape mismatch
            dummy_traj_wrong_shape = torch.rand(101, 3, device=self.device)
            calculate_girf_gradient_error(dummy_traj_ideal, dummy_traj_wrong_shape)


class TestSimpleCNN(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_instantiation_and_forward_2d(self):
        cnn_2d = SimpleCNN(n_channels_in=1, n_channels_out=2, n_spatial_dims=2).to(self.device)
        # Input: (batch_size, n_channels_in, Y, X)
        dummy_input_2d = torch.rand(1, 1, 16, 16, device=self.device)
        output_2d = cnn_2d(dummy_input_2d)
        self.assertEqual(output_2d.shape, (1, 2, 16, 16))

    def test_instantiation_and_forward_3d(self):
        cnn_3d = SimpleCNN(n_channels_in=1, n_channels_out=1, n_spatial_dims=3).to(self.device)
        # Input: (batch_size, n_channels_in, Z, Y, X)
        dummy_input_3d = torch.rand(1, 1, 8, 8, 8, device=self.device)
        output_3d = cnn_3d(dummy_input_3d)
        self.assertEqual(output_3d.shape, (1, 1, 8, 8, 8))


class MockSCNUFFT:
    """Mock Single-Coil NUFFT operator for testing PINNReconstructor."""
    def __init__(self, image_shape, k_trajectory_shape_k_points, device='cpu'):
        self.image_shape = image_shape # Spatial dimensions, e.g., (H, W) or (D, H, W)
        self.k_trajectory_shape_k_points = k_trajectory_shape_k_points # Number of k-space points
        self.device = device
        # Mock k_trajectory attribute, actual values not critical for these tests
        num_dims = len(image_shape)
        self.k_trajectory = torch.zeros(k_trajectory_shape_k_points, num_dims, device=device)


    def op(self, image_data: torch.Tensor) -> torch.Tensor:
        # Input: single coil image (e.g., H, W or D,H,W)
        # Output: single coil k-space (num_k_points,)
        if image_data.shape != self.image_shape:
            raise ValueError(f"MockSCNUFFT.op input shape {image_data.shape} mismatch, expected {self.image_shape}")
        # Return zeros but dependent on input for autograd testing
        return torch.sum(image_data) * 0.0 + torch.zeros(self.k_trajectory_shape_k_points, dtype=torch.complex64, device=self.device)

    def op_adj(self, kspace_data: torch.Tensor) -> torch.Tensor:
        # Input: single coil k-space (num_k_points,)
        # Output: single coil image (e.g., H, W or D,H,W)
        if kspace_data.shape != (self.k_trajectory_shape_k_points,):
             raise ValueError(f"MockSCNUFFT.op_adj input shape {kspace_data.shape} mismatch, expected ({self.k_trajectory_shape_k_points},)")
        # Return zeros but dependent on input for autograd testing
        return torch.sum(kspace_data) * 0.0 + torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)


class TestPINNReconstructor(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_shape_2d = (16, 16) # Y, X
        self.num_coils = 2
        self.num_k_points = 100
        self.spatial_dims = 2

        # Mock NUFFT setup
        mock_sc_nufft = MockSCNUFFT(
            image_shape=self.image_shape_2d,
            k_trajectory_shape_k_points=self.num_k_points,
            device=self.device
        )
        self.mock_mc_nufft_op = MultiCoilNUFFTOperator(mock_sc_nufft)

        # CNN model (2D for these tests)
        # PINNReconstructor.reconstruct feeds RSS to CNN if n_channels_in=1
        # PINNReconstructor.loss_function expects CNN output to be multi-coil if data fidelity uses mc_nufft_op
        self.cnn_model = SimpleCNN(
            n_channels_in=1, # Matching RSS input logic in PINNReconstructor.reconstruct
            n_channels_out=self.num_coils, # Matching data fidelity needs in PINNReconstructor.loss_function
            n_spatial_dims=self.spatial_dims
        ).to(self.device)

        # Config for PINNReconstructor
        self.config = {
            "learning_rate": 1e-4,
            "loss_weights": {"data_fidelity": 1.0, "bloch": 0.1, "girf": 0.1},
            "device": self.device
        }

    def test_initialization(self):
        reconstructor = PINNReconstructor(
            nufft_op=self.mock_mc_nufft_op,
            cnn_model=self.cnn_model,
            config=self.config
        )
        self.assertIsInstance(reconstructor, PINNReconstructor)
        self.assertEqual(reconstructor.device, self.device)

    def test_loss_function(self):
        reconstructor = PINNReconstructor(
            nufft_op=self.mock_mc_nufft_op,
            cnn_model=self.cnn_model,
            config=self.config
        )

        # Dummy data for loss function
        # predicted_image_mc: (num_coils, *image_shape) - this is output of CNN
        predicted_image_mc = torch.rand(self.num_coils, *self.image_shape_2d, dtype=torch.complex64, device=self.device, requires_grad=True)
        # true_kspace_data_mc: (num_coils, num_k_points)
        true_kspace_data_mc = torch.rand(self.num_coils, self.num_k_points, dtype=torch.complex64, device=self.device)
        # trajectories: (num_k_points, dims)
        trajectory_ideal = torch.rand(self.num_k_points, self.spatial_dims, device=self.device)
        trajectory_actual = torch.rand(self.num_k_points, self.spatial_dims, device=self.device)
        scan_parameters = {"TE": 0.03, "TR": 1.0, "flip_angle": 15, "T1_assumed":1.0, "T2_assumed":0.1}

        total_loss, loss_components = reconstructor.loss_function(
            predicted_image_mc,
            true_kspace_data_mc,
            trajectory_ideal,
            trajectory_actual,
            scan_parameters
        )

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.ndim, 0)
        self.assertIsInstance(loss_components, dict)
        self.assertIn("data_fidelity", loss_components)
        self.assertIn("bloch", loss_components)
        self.assertIn("girf", loss_components)
        self.assertIn("total", loss_components)

        # Check if gradients can be computed (mock ops return sum * 0.0 + zeros)
        # If predicted_image_mc requires grad, total_loss should too.
        self.assertTrue(total_loss.requires_grad)


    def test_reconstruct_runs(self):
        reconstructor = PINNReconstructor(
            nufft_op=self.mock_mc_nufft_op,
            cnn_model=self.cnn_model,
            config=self.config
        )

        initial_kspace_data_mc = torch.rand(self.num_coils, self.num_k_points, dtype=torch.complex64, device=self.device)
        trajectory_ideal = torch.rand(self.num_k_points, self.spatial_dims, device=self.device)
        trajectory_actual = torch.rand(self.num_k_points, self.spatial_dims, device=self.device)
        scan_parameters = {"TE": 0.03, "TR": 1.0, "flip_angle": 15, "T1_assumed":1.0, "T2_assumed":0.1}
        num_epochs = 2

        reconstructed_image = reconstructor.reconstruct(
            initial_kspace_data_mc,
            trajectory_ideal,
            trajectory_actual,
            scan_parameters,
            num_epochs
        )
        self.assertIsInstance(reconstructed_image, torch.Tensor)
        # Expected output shape from CNN: (num_coils, *image_shape_2d)
        self.assertEqual(reconstructed_image.shape, (self.num_coils, *self.image_shape_2d))
        self.assertEqual(reconstructed_image.device.type, self.device.type)


if __name__ == '__main__':
    unittest.main()
