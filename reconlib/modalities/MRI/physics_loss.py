import torch
import abc

class PhysicsLossTerm(abc.ABC):
    def __init__(self, weight: float = 1.0, name: str = "UnnamedPhysicsLoss"):
        self.weight = weight
        self.name = name
        if not isinstance(name, str) or not name:
            raise ValueError("PhysicsLossTerm name must be a non-empty string.")

    @abc.abstractmethod
    def compute_loss(self, predicted_image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the physics-specific loss.

        Args:
            predicted_image: The current estimate of the image (e.g., output from CNN).
                             Shape might vary based on CNN output (e.g. (C,Z,Y,X) or (1,Z,Y,X)).
            **kwargs: Other necessary data (e.g., trajectories, scan_params, b0_map).

        Returns:
            A scalar torch.Tensor representing the loss.
        """
        pass

class BlochResidualLoss(PhysicsLossTerm):
    def __init__(self, weight: float = 1.0, name: str = "BlochResidual"):
        super().__init__(weight=weight, name=name)

    def compute_loss(self, predicted_image: torch.Tensor, scan_parameters: dict = None, **kwargs) -> torch.Tensor:
        """
        Placeholder for calculating the Bloch equation residual.
        Args:
            predicted_image: The current estimate of the image.
                             For this placeholder, assumed to be shape (Batch, Maps, Z,Y,X) or similar
                             that calculate_bloch_residual's placeholder logic handled.
            scan_parameters: Dictionary containing MRI sequence parameters.
        """
        if scan_parameters is None:
            # scan_parameters = {} # Or raise error if essential
            print(f"Warning: {self.name}.compute_loss called without scan_parameters.")
            # Return a zero loss that still depends on predicted_image for grad purposes
            return torch.mean(torch.abs(predicted_image)) * 0.0


        # Current placeholder logic from old calculate_bloch_residual:
        # It expected `image_estimate` (now `predicted_image`).
        # The original took image_estimate of shape (B, M, Z,Y,X).
        # The `PINNReconstructor.loss_function` passes `image_for_physics.unsqueeze(0)`
        # which is (1, 1, Z,Y,X) if image_for_physics was (1,Z,Y,X) or (1,Z,Y,X) from RSS.
        # This matches the expected input format.
        print(f"WARNING: {self.name} ({self.__class__.__name__}) is a placeholder and returns 0 loss.")

        # Return a scalar tensor that depends on predicted_image to maintain computation graph, but is zero.
        return torch.mean(torch.abs(predicted_image)) * 0.0

class GIRFErrorLoss(PhysicsLossTerm):
    def __init__(self, weight: float = 1.0, name: str = "GIRFError"):
        super().__init__(weight=weight, name=name)

    def compute_loss(self, trajectory_ideal: torch.Tensor, trajectory_actual: torch.Tensor, predicted_image: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Placeholder for calculating the GIRF-predicted gradient error.
        Args:
            trajectory_ideal: The ideal k-space trajectory.
            trajectory_actual: The actual k-space trajectory.
            predicted_image: Included for interface consistency, not used by this placeholder.
        """
        if trajectory_ideal.shape != trajectory_actual.shape:
            raise ValueError(f"{self.name}: Ideal and actual k-space trajectories must have the same shape.")

        print(f"WARNING: {self.name} ({self.__class__.__name__}) is a placeholder and returns 0 loss.")

        # Return a scalar tensor that depends on inputs to maintain computation graph, but is zero.
        return torch.mean(torch.abs(trajectory_ideal - trajectory_actual)) * 0.0


class B0OffResonanceLoss(PhysicsLossTerm):
    def __init__(self,
                 b0_map: torch.Tensor,
                 scan_parameters_epi: dict,
                 weight: float = 1.0,
                 name: str = "B0OffResonanceLoss"):
        """
        Physics loss term for B0 off-resonance effects in EPI.

        Args:
            b0_map: Tensor representing the B0 field inhomogeneity map (e.g., in Hz).
                    Shape should be compatible with the image (e.g., (Z, Y, X) or (Y, X)).
            scan_parameters_epi: Dictionary containing EPI-specific parameters.
                                 Expected keys:
                                 - 'echo_spacing_ms' (float): Echo spacing in milliseconds.
                                 - 'phase_encoding_lines' (int): Number of phase encoding lines.
                                 - 'gamma_hz_t' (float, optional): Gyromagnetic ratio in Hz/T.
                                                              Defaults to 42.577478518e6.
            weight: Weight for this loss term.
            name: Name of this loss term.
        """
        super().__init__(weight=weight, name=name)
        self.b0_map = b0_map
        self.scan_parameters_epi = scan_parameters_epi
        self.gamma_hz_t = scan_parameters_epi.get('gamma_hz_t', 42.577478518e6) # Hz/T

        if 'echo_spacing_ms' not in scan_parameters_epi:
            raise ValueError("scan_parameters_epi must contain 'echo_spacing_ms'.")
        if 'phase_encoding_lines' not in scan_parameters_epi:
            raise ValueError("scan_parameters_epi must contain 'phase_encoding_lines'.")

    def compute_loss(self, predicted_image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the loss related to B0 off-resonance effects.
        This is a placeholder for a more sophisticated simulation.

        The core idea:
        1. The `predicted_image` from the CNN should be the 'corrected' image.
        2. If this `predicted_image` were the true underlying image, what would the
           acquired signal/image look like given the B0 map and EPI readout?
        3. This might involve simulating phase evolution line-by-line for an EPI sequence.
        4. The loss would then compare some property of this simulated off-resonant
           signal/image with the `predicted_image` (if it's supposed to be corrected)
           or with a signal derived from the network's input (if the network learns a forward model).

        For this placeholder:
        - We'll acknowledge the b0_map and predicted_image.
        - We'll simulate a very simplified phase error based on an effective TE and b0_map.
        - The loss will be a dummy zero value but ensure it's gradient-friendly.

        Args:
            predicted_image: The image output by the CNN.
                             Expected shape e.g., (batch, channels, Z, Y, X) or (batch, channels, Y, X).
                             The B0 map spatial dimensions should match the image's spatial dimensions.
            **kwargs: Other potential arguments (not used in this placeholder).

        Returns:
            A scalar torch.Tensor representing the loss.
        """

        # Ensure b0_map is on the same device as predicted_image
        b0_map_device = self.b0_map.to(predicted_image.device)

        # Basic validation for spatial dimension agreement
        # Assuming predicted_image is (B, C, Z, Y, X) or (B, C, Y, X)
        # and b0_map is (Z, Y, X) or (Y, X)
        image_spatial_shape = predicted_image.shape[2:]
        b0_spatial_shape = b0_map_device.shape
        if image_spatial_shape != b0_spatial_shape:
            raise ValueError(
                f"Spatial dimensions of predicted_image ({image_spatial_shape}) "
                f"must match B0 map ({b0_spatial_shape})."
            )

        # Placeholder: Calculate a simplified 'effective TE' for demonstration
        # This is NOT a real EPI TE, just for showing parameter usage.
        effective_te_s = (self.scan_parameters_epi['echo_spacing_ms'] / 1000.0) * \
                         (self.scan_parameters_epi['phase_encoding_lines'] / 2.0)

        # Placeholder: Calculate simplified phase error: exp(1j * gamma * B0 * TE_eff)
        # gamma_hz_t is in Hz/T. b0_map is in Hz. So gamma * B0 is not quite right.
        # If b0_map is in Hz, then phase = 2 * pi * b0_map_hz * time_s
        # If b0_map is in Tesla, then phase = gamma_hz_t * b0_map_tesla * time_s * 2 * pi (for radians)
        # Let's assume b0_map is directly in Hz representing the off-resonance frequency.

        # phase_error = 2 * torch.pi * b0_map_device * effective_te_s
        # For a loss, we might compare the phase of predicted_image (if complex)
        # with the expected phase_error.
        # e.g., loss = torch.mean(torch.abs(torch.angle(predicted_image) - phase_error_wrapped))

        # For now, just a gradient-friendly zero loss that uses the inputs
        # to avoid "unused parameter" errors if some inputs aren't used in a real calc yet.
        # Ensure predicted_image (which might have requires_grad=True) is part of the calculation.
        dummy_calc = torch.sum(torch.abs(predicted_image)) * torch.sum(torch.abs(b0_map_device)) * effective_te_s * 0.0

        print(f"WARNING: {self.name} ({self.__class__.__name__}) is a placeholder and returns 0 loss.")
        # Add dummy_calc to ensure it's part of graph if predicted_image requires grad
        return torch.tensor(0.0, device=predicted_image.device, dtype=torch.float32) + dummy_calc


if __name__ == '__main__':
    print("Testing PhysicsLossTerm classes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test BlochResidualLoss
    bloch_loss_term = BlochResidualLoss(weight=0.5)
    dummy_image_bloch = torch.rand(1, 1, 8, 8, 8, device=device, requires_grad=True) # B, C, Z, Y, X
    dummy_scan_params_bloch = {"TE": 0.05, "TR": 2.0}
    loss_b = bloch_loss_term.compute_loss(predicted_image=dummy_image_bloch, scan_parameters=dummy_scan_params_bloch)
    print(f"{bloch_loss_term.name} Loss: {loss_b.item()}, Weight: {bloch_loss_term.weight}")
    assert loss_b.ndim == 0
    assert loss_b.item() == 0.0
    # Test grad propagation
    if dummy_image_bloch.requires_grad:
        loss_b.backward() # Should not error
        assert dummy_image_bloch.grad is not None or torch.all(dummy_image_bloch.grad == 0) # Grad can be zero

    # Test GIRFErrorLoss
    girf_loss_term = GIRFErrorLoss(weight=0.1, name="MyGIRFLoss") # Custom name
    dummy_traj_ideal_girf = torch.rand(100, 3, device=device, requires_grad=True)
    dummy_traj_actual_girf = torch.rand(100, 3, device=device)
    dummy_image_girf = torch.rand(1,1,8,8,8, device=device) # Match expected B,C,Z,Y,X for image
    loss_g = girf_loss_term.compute_loss(
        trajectory_ideal=dummy_traj_ideal_girf,
        trajectory_actual=dummy_traj_actual_girf,
        predicted_image=dummy_image_girf # Predicted image not used in placeholder GIRF, but passed
    )
    print(f"{girf_loss_term.name} Loss: {loss_g.item()}, Weight: {girf_loss_term.weight}")
    assert loss_g.ndim == 0
    assert loss_g.item() == 0.0
    if dummy_traj_ideal_girf.requires_grad:
        loss_g.backward() # Should not error
        assert dummy_traj_ideal_girf.grad is not None or torch.all(dummy_traj_ideal_girf.grad == 0)


    # Test B0OffResonanceLoss
    b0_term = B0OffResonanceLoss(
        b0_map=torch.rand(8, 8, 8, device=device), # Z, Y, X
        scan_parameters_epi={'echo_spacing_ms': 0.5, 'phase_encoding_lines': 8},
        weight=0.2
    )
    dummy_image_b0 = torch.rand(1, 1, 8, 8, 8, device=device, requires_grad=True) # B, C, Z, Y, X
    loss_b0 = b0_term.compute_loss(predicted_image=dummy_image_b0)
    print(f"{b0_term.name} Loss: {loss_b0.item()}, Weight: {b0_term.weight}")
    assert loss_b0.ndim == 0
    assert loss_b0.item() == 0.0
    if dummy_image_b0.requires_grad:
        loss_b0.backward() # Should not error
        assert dummy_image_b0.grad is not None or torch.all(dummy_image_b0.grad == 0)


    print("PhysicsLossTerm tests completed.")
