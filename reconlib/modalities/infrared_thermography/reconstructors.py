import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # General TV regularizer
# from .operators import InfraredThermographyOperator # For type hinting

def tv_reconstruction_irt(
    y_surface_temperature_sequence: torch.Tensor,
    irt_operator: 'InfraredThermographyOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    # Parameters for TV regularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    is_3d_tv: bool = False, # Subsurface map is typically 2D or 3D (for volumetric defects)
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for Infrared Thermography (IRT)
    data, aiming to recover a subsurface property map from surface temperature observations.

    Args:
        y_surface_temperature_sequence (torch.Tensor): Acquired surface temperature data over time.
                                                       Shape: (time_steps, surface_Ny, surface_Nx).
        irt_operator (InfraredThermographyOperator): Configured IRT operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        is_3d_tv (bool, optional): Whether to use 3D TV for the subsurface map.
                                   irt_operator.image_shape determines if map is 2D/3D.
        initial_estimate_fn (callable, optional): Function to compute initial subsurface map.
                                                  If None, ProxGradReconstructor defaults (e.g. adjoint).
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed subsurface property map.
                      Shape: irt_operator.image_shape.
    """
    device = y_surface_temperature_sequence.device

    # Determine if the image to reconstruct (subsurface_property_map) is 3D
    # based on the operator's image_shape attribute.
    actual_is_3d_tv = len(irt_operator.image_shape) == 3

    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=actual_is_3d_tv, # Use actual dimension from operator
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    forward_op_fn_wrapper = lambda image_estimate, smaps: irt_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: irt_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"IRT TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_irt: Using adjoint of y_surface_temperature_sequence as initial estimate.")
        x_init_arg = irt_operator.op_adj(y_surface_temperature_sequence)

    reconstructed_subsurface_map = pg_reconstructor.reconstruct(
        kspace_data=y_surface_temperature_sequence, # Measurements
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=irt_operator.image_shape if x_init_arg is None else None
    )

    return reconstructed_subsurface_map

if __name__ == '__main__':
    print("Running basic execution checks for Infrared Thermography reconstructors...")
    import numpy as np
    try:
        from reconlib.modalities.infrared_thermography.operators import InfraredThermographyOperator
    except ImportError:
        print("Attempting local import for InfraredThermographyOperator for __main__ block.")
        from operators import InfraredThermographyOperator

    device_irt_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test with 2D subsurface map
    img_shape_test_irt_2d = (24, 24)
    time_steps_test_irt = 3

    try:
        irt_op_instance_2d = InfraredThermographyOperator(
            image_shape=img_shape_test_irt_2d,
            time_steps=time_steps_test_irt,
            device=device_irt_recon
        )
        print("Using actual InfraredThermographyOperator (2D) for test.")

        # Dummy surface temperature data: (time_steps, Ny, Nx)
        dummy_surface_data_2d = torch.randn(
            time_steps_test_irt, *img_shape_test_irt_2d,
            dtype=torch.float32, device=device_irt_recon
        )

        recon_map_irt_2d = tv_reconstruction_irt(
            y_surface_temperature_sequence=dummy_surface_data_2d,
            irt_operator=irt_op_instance_2d,
            lambda_tv=0.01,
            iterations=3,
            step_size=0.01,
            tv_prox_iterations=2,
            # is_3d_tv is determined by operator.image_shape in the function
            verbose=True
        )
        print(f"IRT TV reconstruction (2D) output shape: {recon_map_irt_2d.shape}, dtype: {recon_map_irt_2d.dtype}")
        assert recon_map_irt_2d.shape == img_shape_test_irt_2d
        print("tv_reconstruction_irt (2D) basic execution check PASSED.")

        # Test with 3D subsurface map (e.g. volumetric defect estimation)
        img_shape_test_irt_3d = (16, 16, 16) # Nz, Ny, Nx for subsurface
        # For 3D subsurface, surface might still be 2D. Operator needs to handle this.
        # The current placeholder operator assumes surface_shape = image_shape (if image_shape is 2D)
        # or surface_shape = image_shape[1:] (if image_shape is 3D for subsurface, surface is Ny,Nx on top).
        # Let's refine the operator slightly for this test or make assumptions.
        # For simplicity, the current placeholder op's diffusion is 2D. A 3D problem is more complex.
        # We will test with a 3D image_shape for the reconstructor, assuming the op handles it.
        # The current op's diffusion_kernel is 2D, so this test is more about the reconstructor's is_3d_tv flag.

        # To make this test meaningful with current simple operator, we assume the operator's
        # image_shape is 3D, and its op/op_adj somehow still work (even if physically crude).
        # The key is that tv_reconstruction_irt correctly sets is_3d for the regularizer.

        # Mock a 3D capable operator for testing the reconstructor's 3D TV aspect
        class MockIRT3DOperator:
            def __init__(self, image_shape, time_steps, device):
                self.image_shape = image_shape # (Nz, Ny, Nx)
                self.time_steps = time_steps
                self.surface_shape = image_shape[1:] # Assume surface is (Ny, Nx)
                self.measurement_shape = (time_steps, *self.surface_shape)
                self.device = device
                # Dummy kernel for testing purposes, not physically accurate for 3D->2D projection
                kernel_size = min(self.surface_shape) // 4 | 1
                sigma = kernel_size / 3
                ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
                xx, yy = torch.meshgrid(ax, ax, indexing='ij')
                kernel_2d = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
                self.diffusion_kernel = (kernel_2d / kernel_2d.sum()).unsqueeze(0).unsqueeze(0)


            def op(self, subsurface_map_3d): # (Nz, Ny, Nx)
                # Very crude: take a slice or projection for surface effect
                projected_map_2d = torch.sum(subsurface_map_3d, dim=0) # Sum over Z
                source_expanded = projected_map_2d.unsqueeze(0).unsqueeze(0)
                padding_val = (self.diffusion_kernel.shape[-1] -1) // 2
                surface_temp_seq = torch.zeros(self.measurement_shape, device=self.device)
                for t in range(self.time_steps):
                    convolved = torch.nn.functional.conv2d(source_expanded, self.diffusion_kernel, padding=padding_val)
                    surface_temp_seq[t,...] = convolved.squeeze() * ((t+1)/self.time_steps)
                return surface_temp_seq

            def op_adj(self, surface_temp_seq): # (time_steps, Ny, Nx)
                recon_2d_sum = torch.zeros(self.surface_shape, device=self.device)
                padding_val = (self.diffusion_kernel.shape[-1] -1) // 2
                for t in range(self.time_steps):
                    slice_expanded = surface_temp_seq[t,...].unsqueeze(0).unsqueeze(0)
                    convolved = torch.nn.functional.conv2d(slice_expanded, self.diffusion_kernel, padding=padding_val)
                    recon_2d_sum += convolved.squeeze() * ((t+1)/self.time_steps)
                recon_2d_avg = recon_2d_sum / self.time_steps
                # Crude backprojection to 3D
                return recon_2d_avg.unsqueeze(0).repeat(self.image_shape[0], 1, 1)


        irt_op_instance_3d = MockIRT3DOperator(
            image_shape=img_shape_test_irt_3d, # (Nz, Ny, Nx)
            time_steps=time_steps_test_irt,
            device=device_irt_recon
        )
        print("Using MockIRT3DOperator for 3D TV test.")

        dummy_surface_data_3d_input = torch.randn(
            time_steps_test_irt, *img_shape_test_irt_3d[1:], # Surface data is (time, Ny, Nx)
            dtype=torch.float32, device=device_irt_recon
        )

        recon_map_irt_3d = tv_reconstruction_irt(
            y_surface_temperature_sequence=dummy_surface_data_3d_input,
            irt_operator=irt_op_instance_3d, # This op has image_shape as 3D
            lambda_tv=0.01,
            iterations=3,
            step_size=0.01,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"IRT TV reconstruction (3D) output shape: {recon_map_irt_3d.shape}, dtype: {recon_map_irt_3d.dtype}")
        assert recon_map_irt_3d.shape == img_shape_test_irt_3d # Should be (Nz, Ny, Nx)
        print("tv_reconstruction_irt (3D) basic execution check PASSED.")


    except Exception as e:
        print(f"Error during tv_reconstruction_irt check: {e}")
        import traceback
        traceback.print_exc()
