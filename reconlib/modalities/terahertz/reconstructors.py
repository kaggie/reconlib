import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # Reusable for general image/volume TV
# from .operators import TerahertzOperator # For type hinting

def tv_reconstruction_thz(
    y_thz_data: torch.Tensor,
    thz_operator: 'TerahertzOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    # Parameters for TV regularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    is_3d_tv: bool = False, # THz imaging can be 2D or 3D
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for Terahertz (THz)
    imaging data using the Proximal Gradient algorithm.

    Args:
        y_thz_data (torch.Tensor): Acquired THz measurement data.
                                   Shape depends on thz_operator (e.g., (num_measurements,)).
        thz_operator (TerahertzOperator): Configured Terahertz operator.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        is_3d_tv (bool, optional): Whether to use 3D TV (if image is 3D). Defaults to False.
        initial_estimate_fn (callable, optional): Function to compute initial image.
                                                  If None, ProxGradReconstructor defaults (e.g. adjoint).
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed THz image/volume.
                      Shape: thz_operator.image_shape.
    """
    device = y_thz_data.device

    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_tv,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size
    )

    forward_op_fn_wrapper = lambda image_estimate, smaps: thz_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: thz_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"THz TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_thz: Using adjoint of y_thz_data as initial estimate.")
        x_init_arg = thz_operator.op_adj(y_thz_data)
        # The new TerahertzOperator (Fourier sampling) op_adj returns a real image.
        # If x_init_arg were complex for some reason, TV regularizer expects real.
        if x_init_arg.is_complex():
            print("Warning: Initial estimate for THz reconstruction is complex. Taking real part for TV.")
            x_init_arg = x_init_arg.real

    reconstructed_thz_image = pg_reconstructor.reconstruct(
        kspace_data=y_thz_data, # Or generic 'measurement_data'
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None, # Typically None for THz unless specific setup
        x_init=x_init_arg,
        image_shape_for_zero_init=thz_operator.image_shape if x_init_arg is None else None
    )

    # The new THz operator (Fourier sampling) assumes real image input to op()
    # and its op_adj() returns a real image.
    # The TV regularizer also expects a real image.
    # If ProxGradReconstructor somehow produces a complex image, take its real part.
    if reconstructed_thz_image.is_complex():
        print("Warning: Reconstructed THz image is complex. Taking real part.")
        reconstructed_thz_image = reconstructed_thz_image.real

    return reconstructed_thz_image

if __name__ == '__main__':
    print("Running basic execution checks for Terahertz reconstructors...")
    import numpy as np
    try:
        from reconlib.modalities.terahertz.operators import TerahertzOperator
    except ImportError:
        print("Attempting local import for TerahertzOperator for __main__ block.")
        from operators import TerahertzOperator

    device_thz_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_shape_test_thz = (24, 24)
    num_pixels_test_thz = np.prod(img_shape_test_thz)

    # For Fourier Sampling Operator
    num_measurements_test_thz = num_pixels_test_thz // 3
    kx_test = torch.randint(-img_shape_test_thz[1]//2, img_shape_test_thz[1]//2, (num_measurements_test_thz,), device=device_thz_recon).float()
    ky_test = torch.randint(-img_shape_test_thz[0]//2, img_shape_test_thz[0]//2, (num_measurements_test_thz,), device=device_thz_recon).float()
    k_locations_test = torch.stack([kx_test, ky_test], dim=1)

    try:
        thz_op_instance = TerahertzOperator(
            image_shape=img_shape_test_thz,
            k_space_locations=k_locations_test,
            device=device_thz_recon
        )
        print("Using actual TerahertzOperator (Fourier Sampling) for test.")

        # Dummy THz data (k-space measurements, should be complex)
        dummy_thz_data = torch.randn(num_measurements_test_thz, dtype=torch.complex64, device=device_thz_recon)

        recon_image_thz = tv_reconstruction_thz(
            y_thz_data=dummy_thz_data,
            thz_operator=thz_op_instance,
            lambda_tv=0.01,
            iterations=3,
            step_size=0.01,
            tv_prox_iterations=2,
            is_3d_tv=len(img_shape_test_thz) == 3, # False for this 2D test
            verbose=True
        )
        print(f"Terahertz TV reconstruction output shape: {recon_image_thz.shape}, dtype: {recon_image_thz.dtype}")
        assert recon_image_thz.shape == img_shape_test_thz
        assert not recon_image_thz.is_complex(), "Output should be real for the Fourier THz model."
        print("tv_reconstruction_thz basic execution check PASSED.")

    except Exception as e:
        print(f"Error during tv_reconstruction_thz check: {e}")
        import traceback
        traceback.print_exc()
