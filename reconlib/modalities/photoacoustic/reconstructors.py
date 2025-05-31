import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # Can reuse or adapt
# from .operators import PhotoacousticOperator # For type hinting

def tv_reconstruction_pat(
    y_sensor_data: torch.Tensor,
    pat_operator: 'PhotoacousticOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    # Parameters for TV regularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    is_3d_tv: bool = False, # PAT can be 2D or 3D
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for Photoacoustic Tomography
    data using the Proximal Gradient algorithm.

    Args:
        y_sensor_data (torch.Tensor): Acquired sensor data (time series).
                                      Shape (num_sensors, num_time_samples).
        pat_operator (PhotoacousticOperator): Configured photoacoustic operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for custom TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        is_3d_tv (bool, optional): Whether to use 3D TV (if image is 3D). Defaults to False.
        initial_estimate_fn (callable, optional): Function to compute initial pressure map.
                                                  If None, ProxGradReconstructor defaults (e.g. adjoint).
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed initial pressure map.
                      Shape: pat_operator.image_shape.
    """
    device = y_sensor_data.device

    # Instantiate the TV regularizer
    # Assuming UltrasoundTVCustomRegularizer can be adapted or a new one created if needed
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_tv, # Important for PAT which can be 2D or 3D
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    # Wrappers for ProximalGradientReconstructor
    # Note: sensitivity_maps are usually None for PAT unless specific methods are used
    forward_op_fn_wrapper = lambda image_estimate, smaps: pat_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: pat_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"PAT TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_pat: Using adjoint of y_sensor_data as initial estimate.")
        # The adjoint of the operator applied to the data is a common initial estimate
        x_init_arg = pat_operator.op_adj(y_sensor_data)

    reconstructed_pressure_map = pg_reconstructor.reconstruct(
        kspace_data=y_sensor_data, # These are the 'measurements'
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None, # Typically None for PAT
        x_init=x_init_arg,
        image_shape_for_zero_init=pat_operator.image_shape if x_init_arg is None else None
    )

    return reconstructed_pressure_map

if __name__ == '__main__':
    print("Running basic execution checks for Photoacoustic reconstructors...")
    # Need to import PhotoacousticOperator for standalone testing, adjust path if necessary
    # This might require adding reconlib/modalities to sys.path or using relative imports carefully
    try:
        from reconlib.modalities.photoacoustic.operators import PhotoacousticOperator
    except ImportError:
        # Fallback for direct execution if path issues, assumes operators.py is in the same folder
        # This is a common pattern for __main__ blocks in submodules.
        print("Attempting local import for PhotoacousticOperator for __main__ block.")
        from operators import PhotoacousticOperator


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mock PhotoacousticOperator for standalone testing (if not already available from operators.py)
    # Using the one from operators.py if import works, otherwise define a minimal mock here.

    img_shape_test_pat = (32, 32) # Ny, Nx
    num_sensors_test_pat = 16
    num_time_samples_test_pat = 50

    # Minimal mock if full operator is problematic for quick test
    class MockPATOperator:
        def __init__(self, image_shape, num_sensors, num_time_samples, device):
            self.image_shape = image_shape
            self.num_sensors = num_sensors
            self.num_time_samples = num_time_samples
            self.device = device
        def op(self, x_pressure_map): # x is (Ny, Nx)
            # Naive sum based signal
            signal = torch.sum(x_pressure_map) * 0.01
            return torch.full((self.num_sensors, self.num_time_samples), signal.item(), dtype=torch.float32, device=self.device)
        def op_adj(self, y_sensor_data): # y is (num_sensors, num_time_samples)
            # Naive sum based backprojection
            val = torch.sum(y_sensor_data) * 0.001
            return torch.full(self.image_shape, val.item(), dtype=torch.float32, device=self.device)

    # Use actual operator if available and working, otherwise mock
    try:
        # Sensor positions for the actual operator
        angles_test = torch.linspace(0, 2 * torch.pi, num_sensors_test_pat, device=device)
        radius_test = 25
        sensor_pos_test_pat = torch.stack([radius_test * torch.cos(angles_test), radius_test * torch.sin(angles_test)], dim=1)

        pat_op_instance = PhotoacousticOperator(
            image_shape=img_shape_test_pat,
            sensor_positions=sensor_pos_test_pat,
            sound_speed=1500.0,
            device=device
        )
        print("Using actual PhotoacousticOperator for test.")
    except Exception as e:
        print(f"Could not instantiate actual PhotoacousticOperator for test ({e}), using MockPATOperator.")
        pat_op_instance = MockPATOperator(
            image_shape=img_shape_test_pat,
            num_sensors=num_sensors_test_pat,
            num_time_samples=num_time_samples_test_pat,
            device=device
        )

    # Dummy sensor data: (num_sensors, num_time_samples)
    dummy_sensor_data_pat = torch.randn(num_sensors_test_pat, num_time_samples_test_pat, dtype=torch.float32, device=device)

    try:
        recon_map_pat = tv_reconstruction_pat(
            y_sensor_data=dummy_sensor_data_pat,
            pat_operator=pat_op_instance,
            lambda_tv=0.005,
            iterations=3, # Keep low for a quick test
            step_size=0.02,
            tv_prox_iterations=2,
            is_3d_tv=False, # For 2D test case
            verbose=True
        )
        print(f"Photoacoustic TV reconstruction output shape: {recon_map_pat.shape}")
        assert recon_map_pat.shape == img_shape_test_pat,             f"Output shape {recon_map_pat.shape} does not match expected {img_shape_test_pat}"
        print("tv_reconstruction_pat basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_pat check: {e}")
        import traceback
        traceback.print_exc()
