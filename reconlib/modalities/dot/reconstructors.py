import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # Generic TV
# from .operators import DOTOperator

def tv_reconstruction_dot(
    y_delta_measurements: torch.Tensor,
    dot_operator: 'DOTOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    tv_prox_iterations: int = 10,
    initial_delta_mu_estimate: torch.Tensor | None = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Placeholder TV-regularized reconstruction for linearized DOT.
    Aims to reconstruct a map of optical property changes (delta_mu_a or delta_mu_s').
    """
    device = y_delta_measurements.device

    is_3d_map = len(dot_operator.image_shape) == 3
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_map,
        device=device
    )

    forward_op_fn_wrapper = lambda image_estimate, smaps: dot_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: dot_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=None,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"DOT TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = initial_delta_mu_estimate
    if x_init_arg is None:
        print("tv_reconstruction_dot: Using adjoint of measurements as initial delta_mu estimate.")
        x_init_arg = dot_operator.op_adj(y_delta_measurements)
    else:
        x_init_arg = x_init_arg.to(device)

    reconstructed_delta_mu = pg_reconstructor.reconstruct(
        kspace_data=y_delta_measurements, # Measurements (delta_y)
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=dot_operator.image_shape if x_init_arg is None and initial_delta_mu_estimate is None else None
    )
    return reconstructed_delta_mu

if __name__ == '__main__':
    print("Running basic DOT reconstructor checks...")
    dev_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_recon = (16,16)
    n_m_recon = 24

    try:
        from .operators import DOTOperator # Relative import
        dot_op_inst = DOTOperator(image_shape=img_s_recon, num_measurements=n_m_recon, device=dev_recon)

        dummy_delta_y = torch.randn(n_m_recon, device=dev_recon)

        recon_dm = tv_reconstruction_dot(
            y_delta_measurements=dummy_delta_y,
            dot_operator=dot_op_inst,
            lambda_tv=0.0001, # DOT often needs very small lambda with random J
            iterations=3,
            verbose=True
        )
        print(f"DOT reconstruction output shape: {recon_dm.shape}")
        assert recon_dm.shape == img_s_recon
        print("tv_reconstruction_dot basic check PASSED.")
    except Exception as e:
        print(f"Error in tv_reconstruction_dot check: {e}")
        import traceback; traceback.print_exc()
