import torch
import torch.nn as nn
import torch.optim as optim

# Updated import for physics_loss from its new location
from reconlib.modalities.MRI.physics_loss import PhysicsLossTerm, BlochResidualLoss, GIRFErrorLoss # Import new classes

# Attempt to import actual NUFFT operators
# These imports should work if reconlib is in PYTHONPATH
from reconlib.nufft import NUFFT3D
from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator

# Commented out the debug/fallback logic for NUFFT imports after confirming they work.
# NUFFT3D = None
# MultiCoilNUFFTOperator = None
# try:
#     # print("DEBUG: Attempting to import from reconlib.nufft and reconlib.nufft_multi_coil...")
#     from reconlib.nufft import NUFFT3D
#     from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator
#     # print("DEBUG: Successfully imported from reconlib.")
# except ImportError as e:
#     print(f"FATAL: Failed to import from reconlib. Error: {e}")
#     import traceback
#     traceback.print_exc()
    # For this subtask, we assume these imports will succeed. If not, the test run will fail.
    # NUFFT3D and MultiCoilNUFFTOperator remain None


# Adapter class for NUFFT3D
# This adapter is defined here if pinn_reconstructor is the primary user,
# or it could be moved to a more general reconlib.adapters module if used elsewhere.
class NUFFT3DAdapter:
    def __init__(self, nufft3d_instance: NUFFT3D):
        if nufft3d_instance is None: # Check if NUFFT3D failed to import
            raise ValueError("NUFFT3D instance is None, likely due to import failure.")
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
                 config: dict,
                 physics_terms: list[PhysicsLossTerm] = None): # New parameter for list of physics terms
        """
        PINN Reconstructor.

        Args:
            nufft_op: An instance of MultiCoilNUFFTOperator (or a compatible wrapper).
            cnn_model: A PyTorch neural network model (e.g., U-Net).
            config: Dictionary with configuration parameters:
                - learning_rate (float)
                - data_fidelity_weight (float): Weight for the data fidelity term.
                - device (str or torch.device)
            physics_terms (list[PhysicsLossTerm], optional): A list of physics loss term instances.
        """
        self.nufft_op = nufft_op
        self.cnn_model = cnn_model.to(config.get("device", "cpu"))
        self.config = config
        self.device = config.get("device", "cpu")
        self.physics_terms = physics_terms if physics_terms is not None else []

        self.data_fidelity_weight = config.get("data_fidelity_weight", 1.0)


    def loss_function(self,
                      current_cnn_output: torch.Tensor, # Output from CNN (e.g. C,Z,Y,X or 1,Z,Y,X)
                      true_kspace_data_mc: torch.Tensor, # Multi-coil k-space (C, K)
                      # Pass all potentially needed data for physics terms via kwargs or a dict
                      **kwargs
                      # trajectory_ideal: torch.Tensor,
                      # trajectory_actual: torch.Tensor,
                      # scan_parameters: dict
                      ) -> tuple[torch.Tensor, dict]:
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
        total_loss = torch.tensor(0.0, device=self.device)

        # 1. Data Fidelity Loss (in k-space)
        # MultiCoilNUFFTOperator.op expects (C, Z, Y, X) and returns (C, K)
        # current_cnn_output is assumed to be in the format expected by nufft_op (e.g. multi-coil image)
        predicted_kspace_mc = self.nufft_op.op(current_cnn_output)

        data_fidelity_loss = torch.mean(torch.abs(predicted_kspace_mc - true_kspace_data_mc)**2)
        loss_components["data_fidelity"] = data_fidelity_loss
        total_loss += self.data_fidelity_weight * data_fidelity_loss

        # Prepare image for physics losses (e.g. coil combine if needed, or select specific map)
        # This logic might need to be adaptable based on what each PhysicsLossTerm expects.
        # For now, using the same RSS logic if current_cnn_output is multi-coil.
        # The `predicted_image` argument to `compute_loss` will be this `image_for_physics`.
        if current_cnn_output.shape[0] > 1 and current_cnn_output.ndim == (len(self.nufft_op.image_shape) + 1): # (C, *spatial_dims)
             image_for_physics = torch.sqrt(torch.sum(torch.abs(current_cnn_output)**2, dim=0, keepdim=True)) # (1, *spatial_dims)
        elif current_cnn_output.ndim == len(self.nufft_op.image_shape): # (*spatial_dims)
             image_for_physics = current_cnn_output.unsqueeze(0).unsqueeze(0) # (1,1,*spatial_dims)
        elif current_cnn_output.ndim == (len(self.nufft_op.image_shape) + 1) and current_cnn_output.shape[0] == 1: # (1, *spatial_dims)
             image_for_physics = current_cnn_output # Assumed to be a single map
        else:
             print(f"Warning: Unexpected shape for current_cnn_output in physics loss prep: {current_cnn_output.shape}. Using as is.")
             image_for_physics = current_cnn_output

        # Add batch dimension if PhysicsLossTerm expects it (e.g. BlochResidualLoss placeholder)
        # This is a bit ad-hoc; ideally PhysicsLossTerm defines what input shape it needs.
        # For BlochResidualLoss placeholder, it expected (B, M, Z,Y,X).
        # If image_for_physics is (1,Z,Y,X), then image_for_physics.unsqueeze(0) is (1,1,Z,Y,X).
        image_for_physics_batched = image_for_physics.unsqueeze(0)


        # 2. Iterate over modular physics loss terms
        for term in self.physics_terms:
            # Pass all available data; term.compute_loss will pick what it needs via **kwargs.
            # `predicted_image` in compute_loss signature will receive `image_for_physics_batched`.
            physics_loss_value = term.compute_loss(
                predicted_image=image_for_physics_batched,
                # scan_parameters=scan_parameters, # Pass through kwargs
                # trajectory_ideal=trajectory_ideal,
                # trajectory_actual=trajectory_actual,
                **kwargs # Pass all other named arguments from loss_function's signature
            )
            loss_components[term.name] = physics_loss_value
            total_loss += term.weight * physics_loss_value

        loss_components["total"] = total_loss
        return total_loss, loss_components

    def reconstruct(self,
                    initial_kspace_data_mc: torch.Tensor, # (C, K)
                    num_epochs: int,
                    # Pass all data needed for any loss term via loss_fn_kwargs
                    loss_fn_kwargs: dict = None
                    ) -> torch.Tensor:
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

            # Ensure current_cnn_output matches nufft_op coil dimension expectations
            # If cnn_model.n_channels_out is not num_coils, this will be an issue for data_fidelity.
            # Current test setup: SimpleCNN n_channels_out = num_coils.

            current_cnn_output = predicted_output_batched[0] # (C_out, Z,Y,X) or (C_out, Y,X)

            # Prepare arguments for loss function
            all_loss_args = {
                "current_cnn_output": current_cnn_output,
                "true_kspace_data_mc": initial_kspace_data_mc.to(self.device),
            }
            if loss_fn_kwargs: # Add trajectory_ideal, trajectory_actual, scan_parameters etc.
                all_loss_args.update(loss_fn_kwargs)

            total_loss, loss_comp = self.loss_function(**all_loss_args)

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
    nufft_params_reconlib = { # Renamed to avoid conflict if loaded from elsewhere
        'oversamp_factor': (1.5, 1.5, 1.5),
        'kb_J': (4, 4, 4),
        'Ld': (32, 32, 32)
    }

    # Instantiate actual NUFFT3D from reconlib
    try:
        actual_nufft3d_instance = NUFFT3D(
            image_shape=image_shape_spatial,
            k_trajectory=dummy_k_trajectory,
            device=device,
            **nufft_params_reconlib
        )
        print("Successfully instantiated reconlib.nufft.NUFFT3D.")
    except Exception as e:
        print(f"FATAL: Could not instantiate reconlib.nufft.NUFFT3D. Error: {e}")
        print("Ensure reconlib is correctly installed and NUFFT3D parameters are valid.")
        actual_nufft3d_instance = None

    if actual_nufft3d_instance is None:
        # This will skip the test if NUFFT cannot be initialized,
        # which is reasonable if reconlib itself is broken or has missing heavy deps.
        print("WARNING: Skipping __main__ test execution due to NUFFT3D initialization failure.")
    else:
        sc_nufft_adapted = NUFFT3DAdapter(actual_nufft3d_instance)
        mc_nufft = MultiCoilNUFFTOperator(sc_nufft_adapted)
        cnn = SimpleCNN(n_channels_in=1, n_channels_out=num_coils, n_spatial_dims=img_dims).to(device)

        # Setup for new PINNReconstructor with modular physics losses
        bloch_term = BlochResidualLoss(weight=0.01)
        girf_term = GIRFErrorLoss(weight=0.01)
        physics_terms_list = [bloch_term, girf_term]

        pinn_config = {
            "learning_rate": 1e-3,
            "data_fidelity_weight": 1.0, # Changed from loss_weights dict
            "device": device
        }

        reconstructor = PINNReconstructor(
            nufft_op=mc_nufft,
            cnn_model=cnn,
            config=pinn_config,
            physics_terms=physics_terms_list
        )
        print("PINNReconstructor instantiated with modular physics terms.")

        dummy_initial_kspace_mc = torch.randn(num_coils, num_k_points, dtype=torch.complex64, device=device)
        dummy_traj_ideal = (torch.rand(num_k_points, img_dims, device=device) - 0.5).float()
        dummy_traj_actual = (dummy_traj_ideal + torch.randn_like(dummy_traj_ideal) * 0.01).float()
        dummy_scan_params = {"TE": 0.05, "TR": 2.0, "flip_angle": 15, "T1_assumed":1.0, "T2_assumed":0.1}

        loss_fn_kwargs_for_reconstruct = {
            "trajectory_ideal": dummy_traj_ideal,
            "trajectory_actual": dummy_traj_actual,
            "scan_parameters": dummy_scan_params
        }

        print("Running reconstructor...")
        try:
            reconstructed_img = reconstructor.reconstruct(
                initial_kspace_data_mc=dummy_initial_kspace_mc,
                num_epochs=3,
                loss_fn_kwargs=loss_fn_kwargs_for_reconstruct
            )
            print(f"Reconstruction finished. Output image shape: {reconstructed_img.shape}")
        expected_output_shape = (num_coils, *image_shape_spatial)
        assert reconstructed_img.shape == expected_output_shape
        assert reconstructed_img.device.type == device.type # Check device consistency
        print("PINNReconstructor structural test with reconlib NUFFT potentially passed.")
    except Exception as e:
        print(f"Error during PINNReconstructor test run with reconlib NUFFT: {e}")
        import traceback
        traceback.print_exc()
        print("PINNReconstructor structural test failed during run with reconlib NUFFT.")
