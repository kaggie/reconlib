import sys
import os
# Add project root to sys.path to ensure reconlib can be found
# Assuming the script is always run from /app or that /app is the identifiable project root.
project_root = "/app"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(f"DEBUG: Current sys.path: {sys.path}")
# print(f"DEBUG: Contents of /app: {os.listdir('/app')}")
# if 'reconlib' in os.listdir('/app'):
#     print(f"DEBUG: Contents of /app/reconlib: {os.listdir('/app/reconlib')}")

import torch
import torch.nn as nn
import torch.optim as optim
# Assuming nufft and physics_loss will be imported from the same directory or reconlib
try:
    from .physics_loss import calculate_bloch_residual, calculate_girf_gradient_error
except ImportError:
    # Fallback for direct execution or if reconlib structure isn't fully set up for this new module
    from physics_loss import calculate_bloch_residual, calculate_girf_gradient_error

# Attempt to import actual NUFFT operators
NUFFT3D = None
MultiCoilNUFFTOperator = None
try:
    # print("DEBUG: Attempting to import from reconlib.nufft and reconlib.nufft_multi_coil...")
    from reconlib.nufft import NUFFT3D
    from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator
    # print("DEBUG: Successfully imported from reconlib.")
except ImportError as e:
    print(f"FATAL: Failed to import from reconlib. Error: {e}")
    import traceback
    traceback.print_exc()
    # For this subtask, we assume these imports will succeed. If not, the test run will fail.
    # NUFFT3D and MultiCoilNUFFTOperator remain None


# Adapter class for NUFFT3D
class NUFFT3DAdapter:
    def __init__(self, nufft3d_instance: NUFFT3D):
        self.nufft3d_instance = nufft3d_instance
        self.device = nufft3d_instance.device
        # NUFFT3D image_shape is likely (Z,Y,X) or whatever spatial dims it's configured for
        self.image_shape = nufft3d_instance.image_shape
        self.k_trajectory = nufft3d_instance.k_trajectory

    def op(self, x: torch.Tensor) -> torch.Tensor:
        # reconlib.nufft.NUFFT3D.forward expects image of shape (Z,Y,X) or (1,Z,Y,X)
        # x input to this op (from MultiCoilNUFFTOperator) will be single coil (Z,Y,X)
        if x.ndim == 3: # (Z,Y,X)
            return self.nufft3d_instance.forward(x)
        elif x.ndim == 4 and x.shape[0] == 1: # (1,Z,Y,X)
            return self.nufft3d_instance.forward(x.squeeze(0)) # Pass (Z,Y,X)
        else:
            raise ValueError(f"NUFFT3DAdapter.op expects single coil image (Z,Y,X) or (1,Z,Y,X), got {x.shape}")

    def op_adj(self, y: torch.Tensor) -> torch.Tensor:
        # reconlib.nufft.NUFFT3D.adjoint expects k-space data of shape (num_k_points,)
        # y input to this op_adj (from MultiCoilNUFFTOperator) will be single coil k-space (K,)
        if y.ndim == 1: # (K)
            return self.nufft3d_instance.adjoint(y)
        else:
            raise ValueError(f"NUFFT3DAdapter.op_adj expects single coil k-space (K,), got {y.shape}")


# Basic CNN model (e.g., a few conv layers, U-Net would be more complex)
class SimpleCNN(nn.Module):
    def __init__(self, n_channels_in=1, n_channels_out=1, n_spatial_dims=3):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        conv_layer = nn.Conv3d if n_spatial_dims == 3 else nn.Conv2d

        self.layers = nn.Sequential(
            conv_layer(n_channels_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            conv_layer(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            conv_layer(32, n_channels_out, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, n_channels_in, Z, Y, X) for 3D
        # Output: (batch_size, n_channels_out, Z, Y, X) for 3D
        return self.layers(x)

class PINNReconstructor:
    def __init__(self,
                 nufft_op: MultiCoilNUFFTOperator,
                 cnn_model: nn.Module,
                 config: dict):
        """
        PINN Reconstructor.

        Args:
            nufft_op: An instance of MultiCoilNUFFTOperator (or a compatible wrapper).
            cnn_model: A PyTorch neural network model (e.g., U-Net).
            config: Dictionary with configuration parameters:
                - learning_rate (float)
                - loss_weights (dict): e.g., {"data_fidelity": 1.0, "bloch": 0.1, "girf": 0.1}
                - device (str or torch.device)
        """
        self.nufft_op = nufft_op
        self.cnn_model = cnn_model.to(config.get("device", "cpu"))
        self.config = config
        self.device = config.get("device", "cpu")

        # Ensure loss_weights are present
        self.loss_weights = config.get("loss_weights", {
            "data_fidelity": 1.0, "bloch": 0.0, "girf": 0.0
        })
        if "data_fidelity" not in self.loss_weights: # Ensure data fidelity is always there
            self.loss_weights["data_fidelity"] = 1.0


    def loss_function(self,
                      predicted_image_mc: torch.Tensor, # Multi-coil image (C, Z, Y, X)
                      true_kspace_data_mc: torch.Tensor, # Multi-coil k-space (C, K)
                      trajectory_ideal: torch.Tensor,
                      trajectory_actual: torch.Tensor,
                      scan_parameters: dict) -> tuple[torch.Tensor, dict]:
        """
        Calculates the composite loss for the PINN.

        Args:
            predicted_image_mc: The image predicted by the CNN (multi-coil).
                                Shape (num_coils, Z, Y, X) or (num_coils, Y, X)
            true_kspace_data_mc: The acquired k-space data (multi-coil).
                                 Shape (num_coils, num_k_points)
            trajectory_ideal: Ideal k-space trajectory.
            trajectory_actual: Actual k-space trajectory (from GIRF).
            scan_parameters: Dictionary of scan parameters for Bloch equations.

        Returns:
            total_loss: A scalar torch.Tensor.
            loss_components: A dictionary with individual loss values.
        """
        loss_components = {}

        # 1. Data Fidelity Loss (in k-space)
        # MultiCoilNUFFTOperator.op expects (C, Z, Y, X) and returns (C, K)
        predicted_kspace_mc = self.nufft_op.op(predicted_image_mc)

        # Basic L2 loss in k-space
        data_fidelity_loss = torch.mean(torch.abs(predicted_kspace_mc - true_kspace_data_mc)**2)
        loss_components["data_fidelity"] = data_fidelity_loss

        # For physics losses, it's assumed they operate on a single physical map (e.g. proton density, T1)
        # If the CNN outputs multi-coil images, they need to be combined or a specific channel selected.
        # If the CNN outputs a single map, that can be used directly.
        # This logic depends heavily on what cnn_model.n_channels_out represents.
        # Current SimpleCNN test setup: n_channels_out = num_coils.
        # So, predicted_image_mc is (num_coils, Z, Y, X).
        # For physics loss, we need a single map representation. Let's use RSS for now.
        # image_for_physics should be (1, Z, Y, X) or (1, num_maps, Z, Y, X) for calculate_bloch_residual

        if predicted_image_mc.shape[0] > 1 and predicted_image_mc.ndim == 4: # (C,Z,Y,X) with C > 1
             image_for_physics = torch.sqrt(torch.sum(torch.abs(predicted_image_mc)**2, dim=0, keepdim=True)) # (1,Z,Y,X)
        elif predicted_image_mc.ndim == 3: # (Z,Y,X) - add batch/channel dim
             image_for_physics = predicted_image_mc.unsqueeze(0).unsqueeze(0) # (1,1,Z,Y,X) - assuming single map
        elif predicted_image_mc.ndim == 4 and predicted_image_mc.shape[0] == 1: # (1,Z,Y,X)
             image_for_physics = predicted_image_mc # Assumed to be a single map or already processed
        else: # Fallback - this should be made more robust
             print(f"Warning: Unexpected shape for predicted_image_mc in physics loss: {predicted_image_mc.shape}. Using as is.")
             image_for_physics = predicted_image_mc


        # 2. Bloch Equation Residual
        if self.loss_weights.get("bloch", 0.0) > 0:
            # calculate_bloch_residual expects (batch_size, num_maps, Z, Y, X)
            # image_for_physics is currently (1,Z,Y,X) from RSS, so it's (1,1,Z,Y,X) effectively.
            bloch_loss = calculate_bloch_residual(image_for_physics.unsqueeze(0), scan_parameters) # Add batch dim
            loss_components["bloch"] = bloch_loss
        else:
            bloch_loss = torch.tensor(0.0, device=self.device)

        # 3. GIRF-Predicted Gradient Error
        if self.loss_weights.get("girf", 0.0) > 0:
            girf_loss = calculate_girf_gradient_error(trajectory_ideal, trajectory_actual)
            loss_components["girf"] = girf_loss
        else:
            girf_loss = torch.tensor(0.0, device=self.device)

        # Total Loss
        total_loss = (self.loss_weights["data_fidelity"] * data_fidelity_loss +
                      self.loss_weights.get("bloch", 0.0) * bloch_loss +
                      self.loss_weights.get("girf", 0.0) * girf_loss)

        loss_components["total"] = total_loss
        return total_loss, loss_components

    def reconstruct(self,
                    initial_kspace_data_mc: torch.Tensor, # (C, K)
                    trajectory_ideal: torch.Tensor,
                    trajectory_actual: torch.Tensor,
                    scan_parameters: dict,
                    num_epochs: int) -> torch.Tensor:
        """
        Performs the PINN reconstruction.

        Args:
            initial_kspace_data_mc: Acquired k-space data (multi-coil).
            trajectory_ideal: Ideal k-space trajectory.
            trajectory_actual: Actual k-space trajectory.
            scan_parameters: Scan parameters for Bloch equations.
            num_epochs: Number of training epochs.

        Returns:
            reconstructed_image: The final reconstructed image.
                                 Shape depends on cnn_model output, e.g. (C_out, Z, Y, X)
        """
        self.cnn_model.train()
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.config.get("learning_rate", 1e-3))

        # Initial image estimate: Adjoint NUFFT of initial k-space data
        initial_image_mc = self.nufft_op.op_adj(initial_kspace_data_mc) # (C, Z,Y,X)

        # Prepare input for CNN
        # SimpleCNN as defined: n_channels_in=1. So, combine coils from initial_image_mc.
        if self.cnn_model.n_channels_in == 1 and initial_image_mc.shape[0] > 1:
            cnn_input_image = torch.sqrt(torch.sum(torch.abs(initial_image_mc)**2, dim=0, keepdim=True)) # (1, Z,Y,X)
        elif initial_image_mc.shape[0] == self.cnn_model.n_channels_in:
            cnn_input_image = initial_image_mc # (C_in, Z,Y,X)
        else:
            print(f"Warning: Mismatch between initial image channels ({initial_image_mc.shape[0]}) "
                  f"and CNN input channels ({self.cnn_model.n_channels_in}). Attempting RSS.")
            cnn_input_image = torch.sqrt(torch.sum(torch.abs(initial_image_mc)**2, dim=0, keepdim=True))


        cnn_input_image_batched = cnn_input_image.unsqueeze(0).to(self.device) # (1, C_in, Z,Y,X)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            predicted_output_batched = self.cnn_model(cnn_input_image_batched) # (1, C_out, Z,Y,X)
            predicted_image_for_loss = predicted_output_batched[0] # (C_out, Z,Y,X)

            # Ensure predicted_image_for_loss matches nufft_op coil dimension expectations
            # If cnn_model.n_channels_out is not num_coils, this will be an issue for data_fidelity.
            # Current test setup: SimpleCNN n_channels_out = num_coils.

            total_loss, loss_comp = self.loss_function(
                predicted_image_mc=predicted_image_for_loss,
                true_kspace_data_mc=initial_kspace_data_mc.to(self.device), # Ensure target is on same device
                trajectory_ideal=trajectory_ideal.to(self.device),
                trajectory_actual=trajectory_actual.to(self.device),
                scan_parameters=scan_parameters
            )

            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == num_epochs -1 :
                loss_str = ", ".join([f"{k}: {v.item():.4e}" for k,v in loss_comp.items()])
                print(f"Epoch {epoch}/{num_epochs}, Losses: [{loss_str}]")

        self.cnn_model.eval()
        with torch.no_grad():
            final_prediction_batched = self.cnn_model(cnn_input_image_batched)

        reconstructed_image = final_prediction_batched[0] # (C_out, Z,Y,X)
        return reconstructed_image


if __name__ == '__main__':
    print("Testing PINNReconstructor structure with reconlib NUFFT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_coils = 2
    img_dims = 3
    image_shape_spatial = (16, 16, 16) # Z, Y, X (spatial dimensions for NUFFT)
    num_k_points = 100

    # k_trajectory should be FloatTensor for NUFFT3D
    dummy_k_trajectory = (torch.rand(num_k_points, img_dims, device=device) - 0.5).float()

    # Parameters for NUFFT3D from reconlib
    # Ensure these are appropriate for 3D
    nufft_params = {
        'oversamp_factor': (1.5, 1.5, 1.5), # Tuple of 3 floats for 3D
        'kb_J': (4, 4, 4),                 # Tuple of 3 ints for 3D
        'Ld': (32, 32, 32)                 # Tuple of 3 ints for 3D (can be smaller for speed if works)
    }

    # Instantiate actual NUFFT3D from reconlib
    try:
        actual_nufft3d_instance = NUFFT3D(
            image_shape=image_shape_spatial,
            k_trajectory=dummy_k_trajectory,
            device=device,
            **nufft_params
        )
        print("Successfully instantiated reconlib.nufft.NUFFT3D.")
    except Exception as e:
        print(f"FATAL: Could not instantiate reconlib.nufft.NUFFT3D. Error: {e}")
        print("Ensure reconlib is correctly installed and NUFFT3D parameters are valid.")
        # In a real scenario, might re-raise or exit. For testing, we'll let it fail if mc_nufft is None.
        actual_nufft3d_instance = None

    if actual_nufft3d_instance is None:
        raise RuntimeError("Failed to initialize NUFFT3D from reconlib. Cannot proceed with test.")

    # Wrap with adapter
    sc_nufft_adapted = NUFFT3DAdapter(actual_nufft3d_instance)
    print("NUFFT3DAdapter created.")

    # Pass the adapted instance to MultiCoilNUFFTOperator
    # MultiCoilNUFFTOperator expects the single-coil operator to have op/op_adj methods.
    mc_nufft = MultiCoilNUFFTOperator(sc_nufft_adapted)
    print("MultiCoilNUFFTOperator created with adapted reconlib.NUFFT3D.")

    # CNN model: Input 1 channel (RSS), Output `num_coils` channels
    # Image data for NUFFT3D should be ComplexFloat (complex64)
    # CNN typically works with FloatTensor. Conversion might be needed.
    # For this test, let SimpleCNN use FloatTensor. NUFFTAdapter/MultiCoil... handle complex.
    cnn = SimpleCNN(n_channels_in=1, n_channels_out=num_coils, n_spatial_dims=img_dims).to(device)

    pinn_config = {
        "learning_rate": 1e-3, # Faster LR for quick test
        "loss_weights": {"data_fidelity": 1.0, "bloch": 0.01, "girf": 0.01},
        "device": device
    }

    reconstructor = PINNReconstructor(nufft_op=mc_nufft, cnn_model=cnn, config=pinn_config)
    print("PINNReconstructor instantiated.")

    dummy_initial_kspace_mc = torch.randn(num_coils, num_k_points, dtype=torch.complex64, device=device)
    dummy_traj_ideal = (torch.rand(num_k_points, img_dims, device=device) - 0.5)
    dummy_traj_actual = dummy_traj_ideal + torch.randn_like(dummy_traj_ideal) * 0.01
    dummy_scan_params = {"TE": 0.05, "TR": 2.0, "flip_angle": 15, "T1_assumed":1.0, "T2_assumed":0.1} # Added T1/T2 for bloch placeholder

    print("Running reconstructor...")
    try:
        reconstructed_img = reconstructor.reconstruct(
            initial_kspace_data_mc=dummy_initial_kspace_mc, # Complex
            trajectory_ideal=dummy_traj_ideal.float(), # Ensure float
            trajectory_actual=dummy_traj_actual.float(), # Ensure float
            scan_parameters=dummy_scan_params,
            num_epochs=3 # Reduced epochs for faster test with actual NUFFT
        )
        print(f"Reconstruction finished. Output image shape: {reconstructed_img.shape}")
        # CNN outputs (num_coils, Z, Y, X)
        expected_output_shape = (num_coils, *image_shape_spatial)
        assert reconstructed_img.shape == expected_output_shape
        assert reconstructed_img.device.type == device.type # Check device consistency
        print("PINNReconstructor structural test with reconlib NUFFT potentially passed.")
    except Exception as e:
        print(f"Error during PINNReconstructor test run with reconlib NUFFT: {e}")
        import traceback
        traceback.print_exc()
        print("PINNReconstructor structural test failed during run with reconlib NUFFT.")
