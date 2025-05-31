import torch
from reconlib.operators import Operator
import numpy as np

class InfraredThermographyOperator(Operator):
    """
    Forward and Adjoint Operator for Infrared Thermography (IRT).

    Models the relationship between an internal heat generation pattern or
    subsurface thermal property variations (the 'image' to be reconstructed)
    and the resulting surface temperature distribution observed over time.

    This is a complex problem often involving solving the heat equation.
    - In 'active thermography', an external heat source is applied, and defects
      alter the heat flow, leading to surface temperature anomalies.
    - In 'passive thermography', the object's own heat emissions are imaged,
      often to find hot spots.

    This placeholder will assume a simplified scenario where the 'image'
    is a map of subsurface heat sources/sinks or thermal resistance anomalies,
    and the operator applies a blurring/diffusion kernel to simulate its
    effect on the surface temperature.
    """
    def __init__(self, image_shape: tuple[int, int], time_steps: int = 1,
                 diffusion_kernel: torch.Tensor | None = None, device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape  # (Ny, Nx) representing subsurface properties
        self.time_steps = time_steps      # Number of time points for surface temperature measurement
        self.device = torch.device(device)

        if diffusion_kernel is not None:
            self.diffusion_kernel = diffusion_kernel.to(self.device)
            # Kernel could be 2D (applied per time step) or 3D (image_shape -> surface_shape x time_steps)
        else:
            # Placeholder: a simple Gaussian blur kernel for each time step
            print("Warning: No diffusion_kernel provided. Using a placeholder Gaussian blur.")
            kernel_size = min(image_shape) // 8 | 1 # Odd kernel size
            sigma = kernel_size / 3
            ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel_2d = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            self.diffusion_kernel = (kernel_2d / kernel_2d.sum()).unsqueeze(0).unsqueeze(0) # For conv2d
            # Output shape will be (time_steps, Ny, Nx)
            self.surface_shape = image_shape # Assuming surface directly above subsurface image

        # The 'measurements' will be surface temperature maps over time
        self.measurement_shape = (self.time_steps, *self.surface_shape)

        print(f"InfraredThermographyOperator initialized for image shape {self.image_shape}, "
              f"{self.time_steps} time steps. Measurement shape: {self.measurement_shape}.")

    def op(self, subsurface_property_map: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Subsurface property map to surface temperature time series.

        Args:
            subsurface_property_map (torch.Tensor): The map of subsurface thermal properties
                                                    or heat sources/sinks. Shape: self.image_shape.

        Returns:
            torch.Tensor: Simulated surface temperature data over time.
                          Shape: (time_steps, surface_Ny, surface_Nx).
        """
        if subsurface_property_map.shape != self.image_shape:
            raise ValueError(f"Input map shape {subsurface_property_map.shape} must match {self.image_shape}.")
        if subsurface_property_map.device != self.device:
            subsurface_property_map = subsurface_property_map.to(self.device)

        # Placeholder: Apply the diffusion kernel for each time step
        # This is a gross oversimplification of heat transfer.
        # A real model would solve the heat equation numerically.

        # Input for conv2d: (batch, channels, H, W)
        # subsurface_property_map is (H, W), treat as 1 channel, 1 batch
        source_term_expanded = subsurface_property_map.unsqueeze(0).unsqueeze(0) # (1, 1, Ny, Nx)

        surface_temperature_sequence = torch.zeros(self.measurement_shape, device=self.device)

        # Simulate a very basic temporal effect (e.g. increasing blur or effect strength over time)
        for t in range(self.time_steps):
            # Simplistic: apply kernel, maybe scale by time factor
            # In reality, the kernel itself might evolve or the integration time matters.
            # For this placeholder, we just apply the same kernel. A better placeholder might
            # convolve multiple times to simulate more diffusion for later time steps.

            # Apply convolution
            # padding to keep size same: (kernel_size - 1) // 2
            padding_val = (self.diffusion_kernel.shape[-1] -1) // 2
            convolved_map = torch.nn.functional.conv2d(source_term_expanded, self.diffusion_kernel, padding=padding_val)

            # Simplistic temporal scaling: effect stronger or more spread out over time
            # This is arbitrary for a placeholder.
            time_scaling_factor = (t + 1.0) / self.time_steps
            surface_temperature_sequence[t, ...] = convolved_map.squeeze() * time_scaling_factor

        return surface_temperature_sequence

    def op_adj(self, surface_temperature_sequence: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Surface temperature time series to subsurface property map.
        This would be related to the adjoint of the heat equation solver, or
        correlation/matched filtering with the diffusion kernel.

        Args:
            surface_temperature_sequence (torch.Tensor): Surface temperature data over time.
                                                      Shape: (time_steps, surface_Ny, surface_Nx).

        Returns:
            torch.Tensor: Reconstructed subsurface property map. Shape: self.image_shape.
        """
        if surface_temperature_sequence.shape != self.measurement_shape:
            raise ValueError(f"Input data shape {surface_temperature_sequence.shape} must match {self.measurement_shape}.")
        if surface_temperature_sequence.device != self.device:
            surface_temperature_sequence = surface_temperature_sequence.to(self.device)

        # Placeholder: Adjoint of convolution is convolution with flipped kernel.
        # Sum contributions from all time steps.

        # Kernel for conv_transpose2d needs to be (in_channels, out_channels, H, W)
        # Our kernel is (1,1,H,W). For transpose, it's (out_channels, in_channels/groups, H,W)
        # If self.diffusion_kernel was (out_C, in_C, kH, kW)
        # Adjoint kernel for conv2d A*x is A^T * y.
        # If A is conv(kernel, .), A^T is conv(flipped_kernel, .)
        # Let's use conv2d with a flipped kernel.

        # Create flipped kernel for adjoint convolution
        # For a symmetric Gaussian kernel, flipping doesn't change it.
        # If kernel was (1,1,kH,kW), weight for conv2d is (out_ch, in_ch/groups, kH, kW)
        # Our input for adjoint is (batch=time_steps, channels=1, H, W)
        # Our output for adjoint is (1, 1, H, W)

        # The "adjoint" of our simplified forward op.
        reconstructed_map = torch.zeros(self.image_shape, device=self.device)
        padding_val = (self.diffusion_kernel.shape[-1] -1) // 2

        for t in range(self.time_steps):
            time_slice_expanded = surface_temperature_sequence[t, ...].unsqueeze(0).unsqueeze(0) # (1,1,H,W)

            # Simplistic temporal scaling factor from forward
            time_scaling_factor = (t + 1.0) / self.time_steps

            # Apply conv2d with the same kernel (since it's symmetric)
            # This is the adjoint of y = conv(k,x) * scale  => x_adj = conv(k_flipped, y/scale)
            # Assuming kernel is symmetric, k_flipped = k.
            # Note: for conv_transpose2d, kernel is (in_C, out_C, kH, kW).
            # For conv2d, kernel is (out_C, in_C, kH, kW).
            # We are using conv2d(kernel, data) as the adjoint of conv2d(kernel, image).
            # This is correct if the kernel is symmetric and real.

            # Apply scaling from forward op (adjoint means divide by it, or multiply if it was on x)
            # If forward: y = conv(k, x) * S  => adjoint: x_adj = conv(k_adj, y / S) if S is scalar.
            # Or if forward: y = conv(k, x*S) => adjoint: x_adj = conv(k_adj,y) * S_adj
            # Our forward was: y_t = conv(k, x) * S_t
            # So adjoint is sum_t conv(k_adj, y_t / S_t) or sum_t conv(k_adj, y_t) * S_t_adj
            # Let's assume the latter where S_t is part of the "source" effect.

            adj_conv_slice = torch.nn.functional.conv2d(time_slice_expanded, self.diffusion_kernel, padding=padding_val)
            reconstructed_map += adj_conv_slice.squeeze() * time_scaling_factor


        return reconstructed_map / self.time_steps # Average contribution

if __name__ == '__main__':
    print("Running basic InfraredThermographyOperator checks...")
    device_irt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape_irt = (32, 32) # Subsurface map
    time_steps_irt = 5       # Number of surface temperature frames

    try:
        irt_op_test = InfraredThermographyOperator(
            image_shape=img_shape_irt,
            time_steps=time_steps_irt,
            device=device_irt
        )
        print("InfraredThermographyOperator instantiated with default kernel.")

        # Create a simple phantom subsurface map (e.g., a heat source)
        phantom_subsurface = torch.zeros(img_shape_irt, device=device_irt)
        phantom_subsurface[img_shape_irt[0]//4:img_shape_irt[0]//4*3, img_shape_irt[1]//4:img_shape_irt[1]//4*3] = 1.0

        surface_temp_sim = irt_op_test.op(phantom_subsurface)
        print(f"Forward op output shape (surface temps): {surface_temp_sim.shape}")
        assert surface_temp_sim.shape == (time_steps_irt, *img_shape_irt)

        recon_subsurface_map = irt_op_test.op_adj(surface_temp_sim)
        print(f"Adjoint op output shape (reconstructed map): {recon_subsurface_map.shape}")
        assert recon_subsurface_map.shape == img_shape_irt

        # Basic dot product test (using real tensors for simplicity with this placeholder)
        x_dp_irt = torch.randn_like(phantom_subsurface)
        y_dp_rand_irt = torch.randn_like(surface_temp_sim)

        Ax_irt = irt_op_test.op(x_dp_irt)
        Aty_irt = irt_op_test.op_adj(y_dp_rand_irt)

        lhs_irt = torch.dot(Ax_irt.flatten(), y_dp_rand_irt.flatten())
        rhs_irt = torch.dot(x_dp_irt.flatten(), Aty_irt.flatten())

        print(f"IRT Dot product test: LHS={lhs_irt.item():.4f}, RHS={rhs_irt.item():.4f}")
        # This test should pass if op and op_adj are correctly implemented as A and A^H
        # (or A and A^T for real case) and kernel is symmetric.
        # The scaling factors in op and op_adj must be handled consistently.
        assert np.isclose(lhs_irt.item(), rhs_irt.item(), rtol=1e-3), "Dot product test failed for IRT operator."

        print("InfraredThermographyOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in InfraredThermographyOperator __main__ checks: {e}")
        import traceback
        traceback.print_exc()
