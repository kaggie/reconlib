import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # Can be used for 3D TV
# from .operators import HyperspectralImagingOperator # For type hinting

def tv_reconstruction_hsi(
    y_sensor_measurements: torch.Tensor,
    hsi_operator: 'HyperspectralImagingOperator',
    lambda_tv: float,
    iterations: int = 100, # HSI often needs more iterations
    step_size: float = 0.005, # May need tuning
    # Parameters for TV regularizer (will be 3D for HSI cube)
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for Hyperspectral Imaging (HSI)
    data, aiming to recover the full hyperspectral cube from (potentially compressed)
    sensor measurements. Uses 3D TV regularization.

    Args:
        y_sensor_measurements (torch.Tensor): Acquired sensor measurement data.
                                              Shape: (num_measurements,).
        hsi_operator (HyperspectralImagingOperator): Configured HSI operator.
                                                     Its image_shape is (Ny, Nx, N_bands).
        lambda_tv (float): Regularization strength for 3D TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 100.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.005.
        tv_prox_iterations (int, optional): Inner iterations for TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial HSI cube.
                                                  If None, ProxGradReconstructor defaults (e.g. adjoint).
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed hyperspectral cube.
                      Shape: hsi_operator.image_shape (Ny, Nx, N_bands).
    """
    device = y_sensor_measurements.device

    # HSI data cube is 3D (Ny, Nx, N_bands), so use 3D TV.
    # The UltrasoundTVCustomRegularizer supports is_3d.
    tv_regularizer_3d = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=True, # Crucial for HSI cube
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size,
        device=device
    )
    print(f"Using 3D TV regularization for HSI cube of shape {hsi_operator.image_shape}.")

    forward_op_fn_wrapper = lambda image_estimate, smaps: hsi_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: hsi_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer_3d.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"HSI TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_hsi: Using adjoint of y_sensor_measurements as initial estimate.")
        x_init_arg = hsi_operator.op_adj(y_sensor_measurements)

    reconstructed_hsi_cube = pg_reconstructor.reconstruct(
        kspace_data=y_sensor_measurements, # Measurements
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=hsi_operator.image_shape if x_init_arg is None else None
    )

    return reconstructed_hsi_cube

if __name__ == '__main__':
    print("Running basic execution checks for Hyperspectral Imaging reconstructors...")
    import numpy as np
    try:
        from reconlib.modalities.hyperspectral.operators import HyperspectralImagingOperator
    except ImportError:
        print("Attempting local import for HyperspectralImagingOperator for __main__ block.")
        from operators import HyperspectralImagingOperator

    device_hsi_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test with a small HSI cube
    img_shape_test_hsi = (16, 16, 8) # Ny, Nx, N_bands
    num_elements_test_hsi = np.prod(img_shape_test_hsi)
    num_measurements_test_hsi = num_elements_test_hsi // 2 # Compressed sensing scenario

    sensing_matrix_test_hsi = torch.randn(
        num_measurements_test_hsi, num_elements_test_hsi,
        dtype=torch.float32, device=device_hsi_recon
    )

    try:
        hsi_op_instance = HyperspectralImagingOperator(
            image_shape=img_shape_test_hsi,
            sensing_matrix=sensing_matrix_test_hsi,
            device=device_hsi_recon
        )
        print("Using actual HyperspectralImagingOperator for test.")

        dummy_measurements_hsi = torch.randn(
            num_measurements_test_hsi, dtype=torch.float32, device=device_hsi_recon
        )

        recon_cube_hsi = tv_reconstruction_hsi(
            y_sensor_measurements=dummy_measurements_hsi,
            hsi_operator=hsi_op_instance,
            lambda_tv=0.005, # Lambda might need careful tuning for HSI
            iterations=5,    # Low for quick test
            step_size=0.001, # Step size also critical
            tv_prox_iterations=3,
            verbose=True
        )
        print(f"HSI TV reconstruction output shape: {recon_cube_hsi.shape}, dtype: {recon_cube_hsi.dtype}")
        assert recon_cube_hsi.shape == img_shape_test_hsi
        print("tv_reconstruction_hsi basic execution check PASSED.")

    except Exception as e:
        print(f"Error during tv_reconstruction_hsi check: {e}")
        import traceback
        traceback.print_exc()
