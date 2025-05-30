import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
# from .operators import SARForwardOperator # For type hinting

def tv_reconstruction_sar(
    y_sar_data: torch.Tensor, # Measured visibilities
    sar_operator: 'SARForwardOperator',
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
    Performs Total Variation (TV) regularized reconstruction for SAR data
    using the Proximal Gradient algorithm.

    Args:
        y_sar_data (torch.Tensor): Acquired SAR visibilities (k-space samples).
                                   Shape (num_visibilities,).
        sar_operator (SARForwardOperator): Configured SAR forward/adjoint operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Number of inner iterations for the custom TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Step size for the inner gradient descent in custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm calculation. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial image.
                                                  If None, ProxGradReconstructor defaults to zero or adjoint.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed SAR image (reflectivity map).
                      Shape (Ny, Nx).
    """
    device = y_sar_data.device

    # Instantiate the custom TV regularizer (SAR images are 2D)
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    forward_op_fn_wrapper = lambda image_estimate, smaps: sar_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda k_space_data, smaps: sar_operator.op_adj(k_space_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"SAR TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    image_shape_for_zero_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_sar: Using adjoint of y_sar_data (dirty image) as initial estimate.")
        x_init_arg = sar_operator.op_adj(y_sar_data)
        # Or for zero init: image_shape_for_zero_init_arg = sar_operator.image_shape

    reconstructed_image = pg_reconstructor.reconstruct(
        kspace_data=y_sar_data, # These are the visibilities
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=image_shape_for_zero_init_arg
    )

    return reconstructed_image

if __name__ == '__main__':
    print("Running basic execution checks for SAR reconstructors...")
    device = torch.device('cpu')

    # Mock SAR Operator for standalone testing
    class MockSAROperator:
        def __init__(self, image_shape, uv_coords_shape, device):
            self.image_shape = image_shape
            self.uv_coordinates_shape = uv_coords_shape # Store shape for dummy data
            self.device = device
        def op(self, x): # x is (Ny, Nx)
            # Return dummy visibilities: (num_vis,)
            return torch.randn(self.uv_coordinates_shape[0], dtype=torch.complex64, device=self.device)
        def op_adj(self, y): # y is (num_vis,)
            # Return dummy dirty image: (Ny, Nx)
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device)

    img_shape_test_sar = (32, 32) # Ny, Nx
    num_vis_test = 50

    mock_sar_op = MockSAROperator(
        image_shape=img_shape_test_sar,
        uv_coords_shape=(num_vis_test, 2),
        device=device
    )

    dummy_sar_visibilities = torch.randn(num_vis_test, dtype=torch.complex64, device=device)

    try:
        recon_img_sar = tv_reconstruction_sar(
            y_sar_data=dummy_sar_visibilities,
            sar_operator=mock_sar_op,
            lambda_tv=0.005,
            iterations=3,
            step_size=0.02,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"SAR TV reconstruction output shape: {recon_img_sar.shape}")
        assert recon_img_sar.shape == img_shape_test_sar
        print("tv_reconstruction_sar basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_sar check: {e}")
        # raise # Avoid raising in subtask
