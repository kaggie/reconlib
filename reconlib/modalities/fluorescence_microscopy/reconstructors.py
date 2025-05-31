import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # General TV
# from .operators import FluorescenceMicroscopyOperator # For type hinting

def tv_deconvolution_fm(
    y_observed_image: torch.Tensor,
    fm_operator: 'FluorescenceMicroscopyOperator', # Contains PSF and image shape
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01, # May need careful tuning for deconvolution
    # Parameters for TV regularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized deconvolution for Fluorescence Microscopy
    data, aiming to recover the true fluorescence distribution from a blurred observation.

    Args:
        y_observed_image (torch.Tensor): The observed (blurred) microscope image.
                                         Shape: fm_operator.image_shape.
        fm_operator (FluorescenceMicroscopyOperator): Configured Fluorescence Microscopy operator,
                                                      which includes the PSF.
        lambda_tv (float): Regularization strength for TV (2D or 3D based on fm_operator).
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial fluorescence map.
                                                  If None, ProxGradReconstructor defaults (e.g. adjoint).
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized deconvolved fluorescence map.
                      Shape: fm_operator.image_shape.
    """
    device = y_observed_image.device
    is_3d_map = fm_operator.is_3d # Get from operator

    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_map,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size,
        device=device
    )
    print(f"Using {'3D' if is_3d_map else '2D'} TV regularization for Fluorescence Microscopy deconvolution.")

    forward_op_fn_wrapper = lambda image_estimate, smaps: fm_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: fm_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"FM TV Deconv Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    if initial_estimate_fn is None:
        print("tv_deconvolution_fm: Using observed image (y_observed_image) as initial estimate, "
              "or op_adj(y_observed_image) could also be used.")
        # For deconvolution, the blurred image itself is often a reasonable starting point.
        # Adjoint(blurred_image) is another common choice. Let's use observed for simplicity here.
        # x_init_arg = y_observed_image.clone()
        # Or using adjoint:
        x_init_arg = fm_operator.op_adj(y_observed_image)


    deconvolved_map = pg_reconstructor.reconstruct(
        kspace_data=y_observed_image, # 'kspace_data' is a generic name for measurements
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=fm_operator.image_shape if x_init_arg is None else None
    )

    return deconvolved_map

if __name__ == '__main__':
    print("Running basic execution checks for Fluorescence Microscopy reconstructors...")
    import numpy as np
    try:
        from reconlib.modalities.fluorescence_microscopy.operators import FluorescenceMicroscopyOperator, generate_gaussian_psf
    except ImportError:
        print("Attempting local import for FluorescenceMicroscopyOperator for __main__ block.")
        from operators import FluorescenceMicroscopyOperator, generate_gaussian_psf

    device_fm_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test 2D deconvolution
    img_shape_2d_fm_recon = (32, 32)
    psf_2d_recon = generate_gaussian_psf(shape=(5,5), sigma=1.0, device=device_fm_recon)

    try:
        fm_op_instance_2d = FluorescenceMicroscopyOperator(
            image_shape=img_shape_2d_fm_recon, psf=psf_2d_recon, device=device_fm_recon
        )
        print("Using actual FluorescenceMicroscopyOperator (2D) for deconvolution test.")

        # Create a dummy observed (blurred) image
        dummy_observed_2d = torch.randn(img_shape_2d_fm_recon, dtype=torch.float32, device=device_fm_recon)

        deconvolved_img_2d = tv_deconvolution_fm(
            y_observed_image=dummy_observed_2d,
            fm_operator=fm_op_instance_2d,
            lambda_tv=0.005,
            iterations=5,
            step_size=0.01,
            tv_prox_iterations=3,
            verbose=True
        )
        print(f"FM TV deconvolution (2D) output shape: {deconvolved_img_2d.shape}")
        assert deconvolved_img_2d.shape == img_shape_2d_fm_recon
        print("tv_deconvolution_fm (2D) basic execution check PASSED.")

    except Exception as e:
        print(f"Error during tv_deconvolution_fm (2D) check: {e}")
        import traceback; traceback.print_exc()

    # Test 3D deconvolution
    img_shape_3d_fm_recon = (16, 16, 8) # Nz, Ny, Nx
    psf_3d_recon = generate_gaussian_psf(shape=(3,3,3), sigma=0.8, device=device_fm_recon)
    try:
        fm_op_instance_3d = FluorescenceMicroscopyOperator(
            image_shape=img_shape_3d_fm_recon, psf=psf_3d_recon, device=device_fm_recon
        )
        print("Using actual FluorescenceMicroscopyOperator (3D) for deconvolution test.")

        dummy_observed_3d = torch.randn(img_shape_3d_fm_recon, dtype=torch.float32, device=device_fm_recon)

        deconvolved_img_3d = tv_deconvolution_fm(
            y_observed_image=dummy_observed_3d,
            fm_operator=fm_op_instance_3d,
            lambda_tv=0.005,
            iterations=5,
            step_size=0.01,
            tv_prox_iterations=3,
            verbose=True
        )
        print(f"FM TV deconvolution (3D) output shape: {deconvolved_img_3d.shape}")
        assert deconvolved_img_3d.shape == img_shape_3d_fm_recon
        print("tv_deconvolution_fm (3D) basic execution check PASSED.")

    except Exception as e:
        print(f"Error during tv_deconvolution_fm (3D) check: {e}")
        import traceback; traceback.print_exc()
