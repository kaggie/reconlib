import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # Generic TV
# from .operators import EITOperator

def tv_reconstruction_eit(
    y_delta_v_measurements: torch.Tensor,
    eit_operator: 'EITOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    tv_prox_iterations: int = 10,
    initial_delta_sigma_estimate: torch.Tensor | None = None, # For delta imaging
    verbose: bool = False
) -> torch.Tensor:
    """
    Placeholder TV-regularized reconstruction for linearized EIT.
    Aims to reconstruct a map of conductivity changes (delta_sigma).
    """
    device = y_delta_v_measurements.device

    # Conductivity map is typically 2D or 3D. For this placeholder, assume 2D from operator.
    is_3d_map = len(eit_operator.image_shape) == 3
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_map,
        device=device
    )

    forward_op_fn_wrapper = lambda image_estimate, smaps: eit_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: eit_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=None, # Will use x_init_arg directly
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"EIT TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = initial_delta_sigma_estimate
    if x_init_arg is None:
        print("tv_reconstruction_eit: Using adjoint of measurements as initial delta_sigma estimate.")
        x_init_arg = eit_operator.op_adj(y_delta_v_measurements)
    else:
        x_init_arg = x_init_arg.to(device)

    reconstructed_delta_sigma = pg_reconstructor.reconstruct(
        kspace_data=y_delta_v_measurements, # These are the 'measurements' (delta_v)
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=eit_operator.image_shape if x_init_arg is None and initial_delta_sigma_estimate is None else None
    )
    return reconstructed_delta_sigma

if __name__ == '__main__':
    print("Running basic EIT reconstructor checks...")
    dev_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_recon = (16,16) # Small for test
    n_m_recon = 32

    try:
        from .operators import EITOperator # Relative import
        eit_op_inst = EITOperator(image_shape=img_s_recon, num_measurements=n_m_recon, device=dev_recon)

        dummy_delta_v = torch.randn(n_m_recon, device=dev_recon)

        recon_ds = tv_reconstruction_eit(
            y_delta_v_measurements=dummy_delta_v,
            eit_operator=eit_op_inst,
            lambda_tv=0.001,
            iterations=3,
            verbose=True
        )
        print(f"EIT reconstruction output shape: {recon_ds.shape}")
        assert recon_ds.shape == img_s_recon
        print("tv_reconstruction_eit basic check PASSED.")
    except Exception as e:
        print(f"Error in tv_reconstruction_eit check: {e}")
        import traceback; traceback.print_exc()
