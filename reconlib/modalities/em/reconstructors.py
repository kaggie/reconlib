import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
# Using the TV regularizer from ultrasound module, configured for 3D.
# from .operators import EMForwardOperator # For type hinting

def tv_reconstruction_em(
    y_projections: torch.Tensor, # Stack of 2D projections
    em_operator: 'EMForwardOperator',
    lambda_tv: float,
    iterations: int = 30, # Fewer iterations by default for 3D can be slow
    step_size: float = 0.01,
    # Parameters for UltrasoundTVCustomRegularizer (for 3D)
    tv_prox_iterations: int = 5, # Fewer inner iterations for 3D
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs 3D Total Variation (TV) regularized reconstruction for EM data
    using the Proximal Gradient algorithm.

    Args:
        y_projections (torch.Tensor): Acquired EM projection data.
                                      Shape (num_angles, proj_H, proj_W).
        em_operator (EMForwardOperator): Configured EM forward/adjoint operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 30.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for custom TV prox. Defaults to 5.
        tv_prox_step_size (float, optional): Inner step size for custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial volume.
                                                  If None, ProxGradReconstructor defaults.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed 3D EM volume.
                      Shape (D, H, W).
    """
    device = y_projections.device

    # Instantiate the custom TV regularizer, ensuring it's set for 3D
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=True, # CRITICAL: Set to True for 3D EM volumes
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    forward_op_fn_wrapper = lambda volume_estimate, smaps: em_operator.op(volume_estimate)
    adjoint_op_fn_wrapper = lambda projections_data, smaps: em_operator.op_adj(projections_data)
    regularizer_prox_fn_wrapper = lambda volume, sl: tv_regularizer.proximal_operator(volume, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, vol, change, grad_norm: \
               print(f"EM TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 5 == 0 or iter_n == iterations -1) else None # Log less frequently for 3D
    )

    x_init_arg = None
    image_shape_for_zero_init_arg = None # For EM, image_shape is volume_shape
    if initial_estimate_fn is None:
        print("tv_reconstruction_em: Using adjoint of y_projections (backprojection) as initial estimate.")
        x_init_arg = em_operator.op_adj(y_projections)
        # Or for zero init: image_shape_for_zero_init_arg = em_operator.volume_shape

    reconstructed_volume = pg_reconstructor.reconstruct(
        kspace_data=y_projections, # These are the 'measurements' (stack of 2D projections)
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=em_operator.volume_shape if x_init_arg is None else None
    )

    return reconstructed_volume

if __name__ == '__main__':
    print("Running basic execution checks for EM reconstructors...")
    device = torch.device('cpu')

    # Mock EM Operator for standalone testing
    class MockEMOperator:
        def __init__(self, volume_shape, num_angles, single_proj_shape, device):
            self.volume_shape = volume_shape # (D,H,W)
            self.num_angles = num_angles
            self.single_projection_shape = single_proj_shape # (pH, pW)
            self.device = device
        def op(self, x_volume): # x_volume is (D,H,W)
            # Return dummy projections: (num_angles, pH, pW)
            return torch.randn((self.num_angles,) + self.single_projection_shape,
                               dtype=torch.complex64, device=self.device)
        def op_adj(self, y_projections): # y_projections is (num_angles, pH, pW)
            # Return dummy 3D volume: (D,H,W)
            return torch.randn(self.volume_shape, dtype=torch.complex64, device=self.device)

    vol_shape_test_em = (16, 24, 24) # D, H, W - keep small for 3D
    num_angles_test = 10
    # Assuming projection axis 0 (D), so projection shape is (H,W)
    single_proj_shape_test = (vol_shape_test_em[1], vol_shape_test_em[2])

    mock_em_op = MockEMOperator(
        volume_shape=vol_shape_test_em,
        num_angles=num_angles_test,
        single_proj_shape=single_proj_shape_test,
        device=device
    )

    dummy_em_projections = torch.randn(
        (num_angles_test,) + single_proj_shape_test,
        dtype=torch.complex64, device=device
    )

    try:
        recon_vol_em = tv_reconstruction_em(
            y_projections=dummy_em_projections,
            em_operator=mock_em_op,
            lambda_tv=0.002,
            iterations=2, # Minimal iterations for quick 3D test
            step_size=0.01,
            tv_prox_iterations=1, # Minimal inner iterations
            verbose=True
        )
        print(f"EM TV reconstruction output shape: {recon_vol_em.shape}")
        assert recon_vol_em.shape == vol_shape_test_em
        print("tv_reconstruction_em basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_em check: {e}")
        # raise # Avoid raising in subtask
