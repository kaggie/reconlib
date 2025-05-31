import unittest
import torch
import sys
import os

# Add project root to sys.path to allow importing from modalities and reconlib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from reconlib.modalities.MRI.physics_loss import PhysicsLossTerm, BlochResidualLoss, GIRFErrorLoss, B0OffResonanceLoss # Updated imports
from reconlib.modalities.MRI.pinn_reconstructor import SimpleCNN, PINNReconstructor
from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator # This should be fine as is

class TestPhysicsLossClasses(unittest.TestCase): # Renamed class
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_physics_loss_term_abc(self):
        with self.assertRaises(TypeError):
            # Cannot instantiate ABC with abstract methods
            PhysicsLossTerm(name="TestAbstract")
        with self.assertRaises(ValueError): # Test name validation
            BlochResidualLoss(name="")


    def test_bloch_residual_loss(self):
        bloch_term = BlochResidualLoss(weight=0.5, name="TestBloch")
        self.assertEqual(bloch_term.name, "TestBloch")
        self.assertEqual(bloch_term.weight, 0.5)
        dummy_image_estimate = torch.rand(1, 1, 8, 8, 8, device=self.device, requires_grad=True)
        dummy_scan_params = {"TE": 0.05, "TR": 2.0}
        loss = bloch_term.compute_loss(predicted_image=dummy_image_estimate, scan_parameters=dummy_scan_params)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0)
        if dummy_image_estimate.requires_grad: # Check grad propagation
            loss.backward()
            self.assertIsNotNone(dummy_image_estimate.grad)


    def test_girf_error_loss(self):
        girf_term = GIRFErrorLoss(weight=0.1)
        self.assertEqual(girf_term.name, "GIRFError") # Default name
        self.assertEqual(girf_term.weight, 0.1)
        dummy_traj_ideal = torch.rand(100, 3, device=self.device, requires_grad=True)
        dummy_traj_actual = torch.rand(100, 3, device=self.device)
        dummy_predicted_image = torch.rand(1,1,8,8,8, device=self.device)
        loss = girf_term.compute_loss(
            trajectory_ideal=dummy_traj_ideal,
            trajectory_actual=dummy_traj_actual,
            predicted_image=dummy_predicted_image
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0)
        if dummy_traj_ideal.requires_grad: # Check grad propagation
            loss.backward()
            self.assertIsNotNone(dummy_traj_ideal.grad)


        with self.assertRaises(ValueError):
            dummy_traj_wrong_shape = torch.rand(101, 3, device=self.device)
            girf_term.compute_loss(
                trajectory_ideal=dummy_traj_ideal,
                trajectory_actual=dummy_traj_wrong_shape,
                predicted_image=dummy_predicted_image
            )

    def test_b0_off_resonance_loss(self):
        b0_map_shape_2d = (8, 8)
        b0_map_shape_3d = (8, 8, 8) # Z,Y,X for 3D example

        dummy_b0_map_2d = torch.rand(*b0_map_shape_2d, device=self.device)
        dummy_scan_params_epi = {'echo_spacing_ms': 0.5, 'phase_encoding_lines': b0_map_shape_2d[0]} # phase_encoding_lines = Ny

        b0_term = B0OffResonanceLoss(
            b0_map=dummy_b0_map_2d,
            scan_parameters_epi=dummy_scan_params_epi,
            weight=0.2,
            name="TestB0Loss"
        )
        self.assertEqual(b0_term.name, "TestB0Loss")
        self.assertEqual(b0_term.weight, 0.2)

        # Test compute_loss with matching 2D image (B, C, Y, X)
        dummy_predicted_image_2d = torch.rand(1, 1, *b0_map_shape_2d, device=self.device, requires_grad=True)
        loss = b0_term.compute_loss(predicted_image=dummy_predicted_image_2d)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0) # Placeholder returns 0
        if dummy_predicted_image_2d.requires_grad:
            loss.backward() # Test gradient flow
            self.assertIsNotNone(dummy_predicted_image_2d.grad)


        # Test compute_loss with matching 3D image (B, C, Z, Y, X)
        dummy_b0_map_3d = torch.rand(*b0_map_shape_3d, device=self.device)
        dummy_scan_params_epi_3d = {'echo_spacing_ms': 0.5, 'phase_encoding_lines': b0_map_shape_3d[1]} # phase_encoding_lines = Ny
        b0_term_3d = B0OffResonanceLoss(
            b0_map=dummy_b0_map_3d,
            scan_parameters_epi=dummy_scan_params_epi_3d
        )
        dummy_predicted_image_3d = torch.rand(1, 1, *b0_map_shape_3d, device=self.device)
        loss_3d = b0_term_3d.compute_loss(predicted_image=dummy_predicted_image_3d)
        self.assertEqual(loss_3d.item(), 0.0)


        # Test ValueError for mismatched spatial dimensions
        with self.assertRaises(ValueError):
            dummy_predicted_image_wrong_shape = torch.rand(1, 1, 7, 7, device=self.device) # Wrong spatial shape
            b0_term.compute_loss(predicted_image=dummy_predicted_image_wrong_shape)

        # Test ValueError for missing scan_parameters_epi keys
        with self.assertRaises(ValueError):
            B0OffResonanceLoss(b0_map=dummy_b0_map_2d, scan_parameters_epi={'phase_encoding_lines': 8}) # Missing echo_spacing_ms
        with self.assertRaises(ValueError):
            B0OffResonanceLoss(b0_map=dummy_b0_map_2d, scan_parameters_epi={'echo_spacing_ms': 0.5}) # Missing phase_encoding_lines


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
        # Config for PINNReconstructor
        self.config = {
            "learning_rate": 1e-4,
            "data_fidelity_weight": 1.0,
            "device": self.device
        }

        # Modular physics loss terms
        self.mock_bloch_term = BlochResidualLoss(weight=0.1, name="TestBloch")
        self.mock_girf_term = GIRFErrorLoss(weight=0.05, name="TestGIRF")

        self.dummy_b0_map = torch.rand(self.image_shape_2d, device=self.device) # Y,X
        self.dummy_scan_params_epi = {'echo_spacing_ms': 0.5, 'phase_encoding_lines': self.image_shape_2d[0]}
        self.mock_b0_term = B0OffResonanceLoss(
            b0_map=self.dummy_b0_map,
            scan_parameters_epi=self.dummy_scan_params_epi,
            weight=0.2,
            name="TestB0"
        )
        self.physics_terms_list = [self.mock_bloch_term, self.mock_girf_term, self.mock_b0_term]


    def test_initialization(self):
        reconstructor = PINNReconstructor(
            nufft_op=self.mock_mc_nufft_op,
            cnn_model=self.cnn_model,
            config=self.config,
            physics_terms=self.physics_terms_list
        )
        self.assertIsInstance(reconstructor, PINNReconstructor)
        self.assertEqual(reconstructor.device, self.device)
        self.assertEqual(len(reconstructor.physics_terms), 3) # Now includes B0 term

    def test_loss_function(self):
        reconstructor = PINNReconstructor(
            nufft_op=self.mock_mc_nufft_op,
            cnn_model=self.cnn_model,
            config=self.config,
            physics_terms=self.physics_terms_list
        )

        current_cnn_output = torch.rand(self.num_coils, *self.image_shape_2d, dtype=torch.complex64, device=self.device, requires_grad=True)
        true_kspace_data_mc = torch.rand(self.num_coils, self.num_k_points, dtype=torch.complex64, device=self.device)

        loss_fn_kwargs = {
            "trajectory_ideal": torch.rand(self.num_k_points, self.spatial_dims, device=self.device),
            "trajectory_actual": torch.rand(self.num_k_points, self.spatial_dims, device=self.device),
            "scan_parameters": {"TE": 0.03, "TR": 1.0, "flip_angle": 15, "T1_assumed":1.0, "T2_assumed":0.1},
            "b0_map": self.dummy_b0_map, # Added for B0 term
            "scan_parameters_epi": self.dummy_scan_params_epi # Added for B0 term
        }

        total_loss, loss_components = reconstructor.loss_function(
            current_cnn_output=current_cnn_output,
            true_kspace_data_mc=true_kspace_data_mc,
            **loss_fn_kwargs
        )

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.ndim, 0)
        self.assertIsInstance(loss_components, dict)
        self.assertIn("data_fidelity", loss_components)
        self.assertIn(self.mock_bloch_term.name, loss_components)
        self.assertIn(self.mock_girf_term.name, loss_components)
        self.assertIn(self.mock_b0_term.name, loss_components) # Check for B0 term
        self.assertIn("total", loss_components)

        self.assertTrue(total_loss.requires_grad)

        # Verify total loss calculation
        expected_total_loss = reconstructor.data_fidelity_weight * loss_components["data_fidelity"]
        for term in self.physics_terms_list:
            expected_total_loss += term.weight * loss_components[term.name]
        self.assertAlmostEqual(total_loss.item(), expected_total_loss.item(), places=5)


    def test_reconstruct_runs(self):
        reconstructor = PINNReconstructor(
            nufft_op=self.mock_mc_nufft_op,
            cnn_model=self.cnn_model,
            config=self.config,
            physics_terms=self.physics_terms_list
        )

        initial_kspace_data_mc = torch.rand(self.num_coils, self.num_k_points, dtype=torch.complex64, device=self.device)
        loss_fn_kwargs_for_reconstruct = {
            "trajectory_ideal": torch.rand(self.num_k_points, self.spatial_dims, device=self.device),
            "trajectory_actual": torch.rand(self.num_k_points, self.spatial_dims, device=self.device),
            "scan_parameters": {"TE": 0.03, "TR": 1.0, "flip_angle": 15, "T1_assumed":1.0, "T2_assumed":0.1},
            "b0_map": self.dummy_b0_map, # Added for B0 term
            "scan_parameters_epi": self.dummy_scan_params_epi # Added for B0 term
        }
        num_epochs = 1

        reconstructed_image = reconstructor.reconstruct(
            initial_kspace_data_mc=initial_kspace_data_mc,
            num_epochs=num_epochs,
            loss_fn_kwargs=loss_fn_kwargs_for_reconstruct
        )
        self.assertIsInstance(reconstructed_image, torch.Tensor)
        self.assertEqual(reconstructed_image.shape, (self.num_coils, *self.image_shape_2d))
        self.assertEqual(reconstructed_image.device.type, self.device.type)


if __name__ == '__main__':
    unittest.main()
