import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
# from .operators import AstronomicalInterferometryOperator # For type hinting

def tv_reconstruction_astro(
    y_visibilities: torch.Tensor,
    astro_operator: 'AstronomicalInterferometryOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    # Parameters for UltrasoundTVCustomRegularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for Astronomical Imaging
    data (visibilities) using the Proximal Gradient algorithm.

    Args:
        y_visibilities (torch.Tensor): Acquired visibility data.
                                       Shape (num_visibilities,).
        astro_operator (AstronomicalInterferometryOperator): Configured astronomical operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for custom TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial sky map.
                                                  If None, ProxGradReconstructor defaults.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed sky brightness map.
                      Shape (Ny, Nx).
    """
    device = y_visibilities.device

    # Instantiate the custom TV regularizer (sky maps are 2D)
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    forward_op_fn_wrapper = lambda image_estimate, smaps: astro_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: astro_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"Astro TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    image_shape_for_zero_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_astro: Using adjoint of y_visibilities (dirty image) as initial estimate.")
        x_init_arg = astro_operator.op_adj(y_visibilities)
        # Or for zero init: image_shape_for_zero_init_arg = astro_operator.image_shape

    reconstructed_sky_map = pg_reconstructor.reconstruct(
        kspace_data=y_visibilities, # These are the 'measurements'
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=astro_operator.image_shape if x_init_arg is None else None
    )

    return reconstructed_sky_map

if __name__ == '__main__':
    print("Running basic execution checks for Astronomical reconstructors...")
    device = torch.device('cpu')

    # Mock AstronomicalInterferometryOperator for standalone testing
    class MockAstroOperator:
        def __init__(self, image_shape, num_vis, device):
            self.image_shape = image_shape # (Ny, Nx)
            self.num_vis = num_vis
            self.device = device
        def op(self, x_sky_map): # x is (Ny, Nx)
            return torch.randn(self.num_vis, dtype=torch.complex64, device=self.device)
        def op_adj(self, y_visibilities): # y is (num_vis,)
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device)

    img_shape_test_astro = (32, 32) # Ny, Nx
    num_vis_test_astro = 50

    mock_astro_op = MockAstroOperator(
        image_shape=img_shape_test_astro,
        num_vis=num_vis_test_astro,
        device=device
    )

    dummy_visibilities_astro = torch.randn(num_vis_test_astro, dtype=torch.complex64, device=device)

    try:
        recon_map_astro = tv_reconstruction_astro(
            y_visibilities=dummy_visibilities_astro,
            astro_operator=mock_astro_op,
            lambda_tv=0.005,
            iterations=3,
            step_size=0.02,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"Astronomical TV reconstruction output shape: {recon_map_astro.shape}")
        assert recon_map_astro.shape == img_shape_test_astro
        print("tv_reconstruction_astro basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_astro check: {e}")
