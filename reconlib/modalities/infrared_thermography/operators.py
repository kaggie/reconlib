import torch
from reconlib.operators import Operator
import numpy as np

class InfraredThermographyOperator(Operator):
    """
    Forward and Adjoint Operator for Infrared Thermography (IRT).

    Models the relationship between an initial subsurface heat distribution
    (the 'image' to be reconstructed) and the resulting surface temperature
    distribution observed over time. This simulates a simplified scenario of
    heat diffusion from an initial state.

    The forward operator applies a blurring (diffusion) kernel iteratively
    to simulate the heat spreading to the surface and its temporal evolution.
    The 'image' is a 2D map of initial heat distribution.
    The output is a 3D tensor (time, Ny, Nx) of surface temperatures.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx) representing initial subsurface heat
                 time_steps: int,
                 diffusion_kernel_sigma: float = 1.5, # Sigma for Gaussian blur kernel
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape  # (Ny, Nx)
        self.Ny, self.Nx = self.image_shape
        self.time_steps = time_steps
        self.device = torch.device(device)

        # Create a Gaussian blur kernel for diffusion
        kernel_size = int(6 * diffusion_kernel_sigma) | 1 # Kernel size based on sigma, ensure odd
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel_2d = torch.exp(-(xx**2 + yy**2) / (2. * diffusion_kernel_sigma**2))
        self.diffusion_kernel = (kernel_2d / kernel_2d.sum()).unsqueeze(0).unsqueeze(0) # For conv2d: (1,1,kH,kW)
        self.conv_padding = (kernel_size - 1) // 2

        self.measurement_shape = (self.time_steps, *self.image_shape) # Surface temp has same spatial shape

        print(f"InfraredThermographyOperator (Iterative Diffusion) initialized.")
        print(f"  Image (Initial Heat): {self.Ny}x{self.Nx}, Time Steps: {self.time_steps}")
        print(f"  Diffusion Kernel: Gaussian, sigma={diffusion_kernel_sigma}, size={kernel_size}x{kernel_size}")


    def op(self, initial_heat_map: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Initial heat map to surface temperature time series.
        Simulates heat diffusion by iteratively applying a blur kernel.

        Args:
            initial_heat_map (torch.Tensor): The map of initial heat distribution at t=0.
                                             Shape: self.image_shape (Ny, Nx).
        Returns:
            torch.Tensor: Simulated surface temperature data over time.
                          Shape: (time_steps, Ny, Nx).
        """
        if initial_heat_map.shape != self.image_shape:
            raise ValueError(f"Input map shape {initial_heat_map.shape} must match {self.image_shape}.")
        initial_heat_map = initial_heat_map.to(self.device)

        surface_temperature_sequence = torch.zeros(self.measurement_shape, device=self.device)

        # Current state starts as the initial heat map (perhaps representing the heat just reaching the surface)
        current_state = initial_heat_map.unsqueeze(0).unsqueeze(0) # (1, 1, Ny, Nx) for conv2d

        for t in range(self.time_steps):
            # Apply diffusion (convolution with kernel)
            current_state = torch.nn.functional.conv2d(
                current_state,
                self.diffusion_kernel,
                padding=self.conv_padding
            )
            surface_temperature_sequence[t, ...] = current_state.squeeze()

        return surface_temperature_sequence

    def op_adj(self, surface_temperature_sequence: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Surface temperature time series to initial heat map.
        This reverses the iterative blurring process.
        If Y_t = K * Y_{t-1}, then adj(Y_N) = K_adj * ... * K_adj * Y_N (for the last term effect)
        More accurately, if y_t = K^t x_0, then x_0_adj = sum_t (K^t)_adj y_t (this is complex)
        Let's simplify: the adjoint of Y_t = K * X_{t-1} (where X_{t-1} is the state at t-1)
        is X_{t-1}_adj = K_adj * Y_t.
        We want to reconstruct X_0.
        The forward process is: x_0 -> Kx_0 (=s_0) -> K(Kx_0) (=s_1) -> ...
        y_t = s_t.
        Adjoint: sum_t (K^t)^H y_t
        For iterative blur: Y_0 = K X_in, Y_1 = K Y_0, Y_2 = K Y_1 ...
        Adjoint is like running diffusion backwards with the flipped kernel, accumulating results.
        """
        if surface_temperature_sequence.shape != self.measurement_shape:
            raise ValueError(f"Input data shape {surface_temperature_sequence.shape} must match {self.measurement_shape}.")
        surface_temperature_sequence = surface_temperature_sequence.to(self.device)

        # Flipped kernel for adjoint convolution (Gaussian is symmetric, so flip does nothing)
        # However, for correctness with torch.nn.functional.conv2d, it expects (out_C, in_C/groups, kH, kW)
        # Our diffusion_kernel is (1,1,kH,kW). For adjoint, this structure is fine.
        # If we were using conv_transpose, it would be (in_C, out_C/groups, kH, kW)
        adjoint_kernel = self.diffusion_kernel # Assuming symmetric kernel

        reconstructed_map = torch.zeros(self.image_shape, device=self.device)

        # Iterate backwards in time for the adjoint of an iterative process
        current_adj_state = torch.zeros_like(surface_temperature_sequence[0,...].unsqueeze(0).unsqueeze(0)) # (1,1,H,W)

        for t in range(self.time_steps - 1, -1, -1):
            # Add the contribution from this time step's measurement
            current_adj_state = current_adj_state + surface_temperature_sequence[t, ...].unsqueeze(0).unsqueeze(0)
            # Apply adjoint diffusion (convolve with the same kernel as it's symmetric)
            current_adj_state = torch.nn.functional.conv2d(
                current_adj_state,
                adjoint_kernel, # or self.diffusion_kernel
                padding=self.conv_padding
            )
        reconstructed_map = current_adj_state.squeeze()

        return reconstructed_map


if __name__ == '__main__':
    print("\nRunning basic InfraredThermographyOperator (Iterative Diffusion) checks...")
    device_irt_op = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_s_irt = (32, 32) # Subsurface map
    time_s_irt = 5       # Number of surface temperature frames
    sigma_irt = 1.0

    try:
        irt_op = InfraredThermographyOperator(
            image_shape=img_s_irt,
            time_steps=time_s_irt,
            diffusion_kernel_sigma=sigma_irt,
            device=device_irt_op
        )
        print("InfraredThermographyOperator (Iterative Diffusion) instantiated.")

        phantom_initial_heat = torch.zeros(img_s_irt, device=device_irt_op)
        phantom_initial_heat[img_s_irt[0]//4:img_s_irt[0]*3//4, img_s_irt[1]//4:img_s_irt[1]*3//4] = 10.0 # Hot square
        phantom_initial_heat[img_s_irt[0]//3:img_s_irt[0]*2//3, img_s_irt[1]//3:img_s_irt[1]*2//3] = 20.0 # Hotter inner square


        surface_temp_sim = irt_op.op(phantom_initial_heat)
        print(f"Forward op output shape (surface temps): {surface_temp_sim.shape}")
        assert surface_temp_sim.shape == (time_s_irt, *img_s_irt)

        recon_initial_heat = irt_op.op_adj(surface_temp_sim)
        print(f"Adjoint op output shape (reconstructed initial heat): {recon_initial_heat.shape}")
        assert recon_initial_heat.shape == img_s_irt

        # Basic dot product test
        x_dp_irt = torch.randn_like(phantom_initial_heat)
        y_dp_rand_irt = torch.randn_like(surface_temp_sim)

        Ax_irt = irt_op.op(x_dp_irt)
        Aty_irt = irt_op.op_adj(y_dp_rand_irt)

        lhs_irt = torch.dot(Ax_irt.flatten(), y_dp_rand_irt.flatten())
        rhs_irt = torch.dot(x_dp_irt.flatten(), Aty_irt.flatten())

        print(f"IRT Iterative Diffusion Dot product test: LHS={lhs_irt.item():.6f}, RHS={rhs_irt.item():.6f}")
        assert np.isclose(lhs_irt.item(), rhs_irt.item(), rtol=1e-3), "Dot product test failed."

        print("InfraredThermographyOperator (Iterative Diffusion) __main__ checks completed.")

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(2, time_s_irt+1, figsize=((time_s_irt+1)*3, 6))
        # axes[0,0].imshow(phantom_initial_heat.cpu().numpy()); axes[0,0].set_title("Initial Heat")
        # axes[1,0].imshow(recon_initial_heat.cpu().numpy()); axes[1,0].set_title("Adjoint Recon")
        # for t in range(time_s_irt):
        #     axes[0,t+1].imshow(surface_temp_sim[t,...].cpu().numpy()); axes[0,t+1].set_title(f"Surf Temp t={t}")
        #     axes[1,t+1].axis('off')
        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"Error in InfraredThermographyOperator (Iterative Diffusion) __main__ checks: {e}")
        import traceback
        traceback.print_exc()
