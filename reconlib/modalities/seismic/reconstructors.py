import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
# from .operators import SeismicForwardOperator # For type hinting

def tv_reconstruction_seismic(
    y_seismic_traces: torch.Tensor,
    seismic_operator: 'SeismicForwardOperator',
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
    Performs Total Variation (TV) regularized reconstruction for Seismic Imaging data
    using the Proximal Gradient algorithm.

    Args:
        y_seismic_traces (torch.Tensor): Acquired seismic traces.
                                         Shape (num_receivers, num_time_samples).
        seismic_operator (SeismicForwardOperator): Configured seismic forward/adjoint operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for custom TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial reflectivity map.
                                                  If None, ProxGradReconstructor defaults.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed subsurface reflectivity map.
                      Shape (Nz, Nx).
    """
    device = y_seismic_traces.device

    # Instantiate the custom TV regularizer (Seismic reflectivity map is 2D)
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    forward_op_fn_wrapper = lambda image_estimate, smaps: seismic_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: seismic_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"Seismic TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    image_shape_for_zero_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_seismic: Using adjoint of y_seismic_traces (migrated image) as initial estimate.")
        x_init_arg = seismic_operator.op_adj(y_seismic_traces)
        # Or for zero init: image_shape_for_zero_init_arg = seismic_operator.reflectivity_map_shape

    reconstructed_map = pg_reconstructor.reconstruct(
        kspace_data=y_seismic_traces, # These are the 'measurements' (seismic traces)
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=seismic_operator.reflectivity_map_shape if x_init_arg is None else None
    )

    return reconstructed_map

if __name__ == '__main__':
    print("Running basic execution checks for Seismic reconstructors...")
    device = torch.device('cpu')

    # Mock SeismicForwardOperator for standalone testing
    class MockSeismicOperator:
        def __init__(self, reflectivity_map_shape, traces_shape, device):
            self.reflectivity_map_shape = reflectivity_map_shape # (Nz, Nx)
            self.traces_shape = traces_shape # (num_receivers, num_time_samples)
            self.device = device
        def op(self, x_reflectivity_map): # x is (Nz, Nx)
            # Return dummy traces
            return torch.randn(self.traces_shape, dtype=torch.float32, device=self.device)
        def op_adj(self, y_seismic_traces): # y is (num_receivers, num_time_samples)
            # Return dummy reflectivity map
            return torch.randn(self.reflectivity_map_shape, dtype=torch.float32, device=self.device)

    map_shape_test_seismic = (32, 48) # Nz, Nx
    num_receivers_test = 10
    num_time_samples_test = 200
    traces_shape_test = (num_receivers_test, num_time_samples_test)

    mock_seismic_op = MockSeismicOperator(
        reflectivity_map_shape=map_shape_test_seismic,
        traces_shape=traces_shape_test,
        device=device
    )

    dummy_seismic_traces = torch.randn(traces_shape_test, dtype=torch.float32, device=device)

    try:
        recon_map_seismic = tv_reconstruction_seismic(
            y_seismic_traces=dummy_seismic_traces,
            seismic_operator=mock_seismic_op,
            lambda_tv=0.01,
            iterations=3,
            step_size=0.05,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"Seismic TV reconstruction output shape: {recon_map_seismic.shape}")
        assert recon_map_seismic.shape == map_shape_test_seismic
        print("tv_reconstruction_seismic basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_seismic check: {e}")
