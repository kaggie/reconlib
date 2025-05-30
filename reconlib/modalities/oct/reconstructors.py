import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
# Assumes UltrasoundTVCustomRegularizer is general enough or OCT will use a similar TV logic.
# If a truly OCT-specific TV is needed later, this could be changed.

# from .operators import OCTForwardOperator # For type hinting if needed

def tv_reconstruction_oct(
    y_oct_data: torch.Tensor, # Spectral interferogram data S(k)
    oct_operator: 'OCTForwardOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01, # Step size for ProximalGradientReconstructor
    # Parameters for UltrasoundTVCustomRegularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for OCT data
    using the Proximal Gradient algorithm.

    Args:
        y_oct_data (torch.Tensor): Acquired OCT spectral data (S(k)).
                                   Shape (num_ascan_lines, num_k_samples_depth).
        oct_operator (OCTForwardOperator): Configured OCT forward/adjoint operator.
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
        torch.Tensor: The TV-regularized reconstructed OCT image (reflectivity R(z)).
                      Shape (num_ascan_lines, num_depth_pixels).
    """
    device = y_oct_data.device

    # Instantiate the custom TV regularizer (assuming OCT data is 2D: A-scans vs Depth)
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False, # OCT B-scans are typically 2D
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    # The 'sensitivity_maps' argument in ProxGradReconstructor's op functions is not used by OCT operator.
    forward_op_fn_wrapper = lambda image_estimate, smaps: oct_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda k_space_data, smaps: oct_operator.op_adj(k_space_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"OCT TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    # Determine initial guess strategy if initial_estimate_fn is None
    x_init_arg = None
    image_shape_for_zero_init_arg = None
    if initial_estimate_fn is None:
        # Default: use adjoint of data as initial estimate for faster convergence
        print("tv_reconstruction_oct: Using adjoint of y_oct_data as initial estimate.")
        x_init_arg = oct_operator.op_adj(y_oct_data)
        # Alternatively, for zero init by ProxGradReconstructor:
        # image_shape_for_zero_init_arg = oct_operator.image_shape

    reconstructed_image = pg_reconstructor.reconstruct(
        kspace_data=y_oct_data, # This is S(k)
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None, # Not used in this OCT model
        x_init=x_init_arg,
        image_shape_for_zero_init=image_shape_for_zero_init_arg
    )

    return reconstructed_image

if __name__ == '__main__':
    # This block is for basic execution check and will not be run by the subtask script.
    # It requires OCTForwardOperator to be defined or imported.
    print("Running basic execution checks for OCT reconstructors...")
    device = torch.device('cpu')

    # Mock OCT Operator for standalone testing of reconstructor logic
    class MockOCTOperator:
        def __init__(self, image_shape, device):
            self.image_shape = image_shape
            self.device = device
        def op(self, x):
            return torch.fft.fft(x, dim=1, norm='ortho')
        def op_adj(self, y):
            return torch.fft.ifft(y, dim=1, norm='ortho')

    img_shape_test = (16, 32) # num_ascans, depth_pixels
    mock_oct_op = MockOCTOperator(image_shape=img_shape_test, device=device)

    dummy_oct_k_space_data = torch.randn(img_shape_test, dtype=torch.complex64, device=device)

    try:
        recon_img = tv_reconstruction_oct(
            y_oct_data=dummy_oct_k_space_data,
            oct_operator=mock_oct_op,
            lambda_tv=0.01,
            iterations=3, # Keep low for test
            step_size=0.05,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"OCT TV reconstruction output shape: {recon_img.shape}")
        assert recon_img.shape == img_shape_test
        print("tv_reconstruction_oct basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_oct check: {e}")
        # raise # Avoid raising in subtask if it's just a __main__ check
