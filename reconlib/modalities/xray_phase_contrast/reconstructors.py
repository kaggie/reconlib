import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
# from .operators import XRayPhaseContrastOperator # For type hinting

def tv_reconstruction_xrpc(
    y_differential_phase_data: torch.Tensor,
    xrpc_operator: 'XRayPhaseContrastOperator',
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
    Performs Total Variation (TV) regularized phase reconstruction for X-ray
    Phase-Contrast Imaging (XPCI) data using the Proximal Gradient algorithm.

    Args:
        y_differential_phase_data (torch.Tensor): Acquired differential phase contrast data.
                                                 Shape (H, W).
        xrpc_operator (XRayPhaseContrastOperator): Configured XPCI forward/adjoint operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for custom TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial phase image.
                                                  If None, ProxGradReconstructor defaults.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed phase image.
                      Shape (H, W).
    """
    device = y_differential_phase_data.device

    # Instantiate the custom TV regularizer (XPCI images are 2D)
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    forward_op_fn_wrapper = lambda image_estimate, smaps: xrpc_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: xrpc_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"XPCI TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    image_shape_for_zero_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_xrpc: Using adjoint of y_differential_phase_data as initial estimate.")
        x_init_arg = xrpc_operator.op_adj(y_differential_phase_data)
        # Or for zero init: image_shape_for_zero_init_arg = xrpc_operator.image_shape

    reconstructed_phase_image = pg_reconstructor.reconstruct(
        kspace_data=y_differential_phase_data, # This is the 'measurement'
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=xrpc_operator.image_shape if x_init_arg is None else None
    )

    return reconstructed_phase_image

if __name__ == '__main__':
    print("Running basic execution checks for XPCI reconstructors...")
    device = torch.device('cpu')

    # Mock XRayPhaseContrastOperator for standalone testing
    class MockXRPCOperator:
        def __init__(self, image_shape, device):
            self.image_shape = image_shape # (H,W)
            self.device = device
        def op(self, x_phase_image): # x_phase_image is (H,W)
            # Return dummy differential phase data: (H,W)
            return torch.randn(self.image_shape, dtype=x_phase_image.dtype, device=self.device)
        def op_adj(self, y_diff_phase_data): # y_diff_phase_data is (H,W)
            # Return dummy phase image: (H,W)
            return torch.randn(self.image_shape, dtype=y_diff_phase_data.dtype, device=self.device)

    img_shape_test_xrpc = (32, 32) # H, W

    mock_xrpc_op = MockXRPCOperator(
        image_shape=img_shape_test_xrpc,
        device=device
    )

    dummy_xrpc_measurement = torch.randn(img_shape_test_xrpc, dtype=torch.float32, device=device)

    try:
        recon_img_xrpc = tv_reconstruction_xrpc(
            y_differential_phase_data=dummy_xrpc_measurement,
            xrpc_operator=mock_xrpc_op,
            lambda_tv=0.01,
            iterations=3,
            step_size=0.05,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"XPCI TV reconstruction output shape: {recon_img_xrpc.shape}")
        assert recon_img_xrpc.shape == img_shape_test_xrpc
        print("tv_reconstruction_xrpc basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_xrpc check: {e}")
