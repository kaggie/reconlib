import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
# For complex-valued images, TV can be defined in a few ways.
# One common way is to apply TV to magnitude and phase, or real and imag parts separately,
# or use a vectorial TV. The UltrasoundTVCustomRegularizer might need adaptation or
# a new regularizer for complex TV if its current form is only for real data.
# For this placeholder, we'll assume it can handle complex inputs,
# or we'll apply it to real and imaginary parts if needed.
from reconlib.proximal_operators import Regularizer # Base class
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer


class ComplexTotalVariationRegularizer(Regularizer):
    """
    Total Variation regularizer for complex-valued images.
    Applies TV to the real and imaginary parts separately and sums the results.
    lambda_reg is split equally between real and imaginary TV.
    """
    def __init__(self, lambda_reg: float, prox_iterations: int = 10, is_3d: bool = False,
                 epsilon_tv_grad: float = 1e-8, prox_step_size: float = 0.01,
                 device: str | torch.device = 'cpu'):
        super().__init__(lambda_reg)
        self.is_3d = is_3d
        self.device = torch.device(device)
        # Half of the lambda for real part, half for imaginary part
        lambda_half = lambda_reg / 2.0

        self.tv_real = UltrasoundTVCustomRegularizer(
            lambda_reg=lambda_half, prox_iterations=prox_iterations, is_3d=is_3d,
            epsilon_tv_grad=epsilon_tv_grad, prox_step_size=prox_step_size, device=device
        )
        self.tv_imag = UltrasoundTVCustomRegularizer(
            lambda_reg=lambda_half, prox_iterations=prox_iterations, is_3d=is_3d,
            epsilon_tv_grad=epsilon_tv_grad, prox_step_size=prox_step_size, device=device
        )
        print(f"ComplexTVRegularizer: lambda_total={lambda_reg}, lambda_part={lambda_half}, 3D={is_3d}")

    def proximal_operator(self, x_complex_image: torch.Tensor, step_size: float) -> torch.Tensor:
        if not x_complex_image.is_complex():
            # If image is real, just apply real TV with full lambda
            # This might happen if initial estimate is real.
            # However, MWI generally reconstructs complex permittivity.
            # For safety, we'll use the original lambda for the real part if input is real.
            # This scenario should ideally be handled by the reconstructor logic / initial estimate.
            # print("Warning: ComplexTVRegularizer received real input. Applying TV only to real part with full original lambda.")
            # temp_tv_real_full_lambda = UltrasoundTVCustomRegularizer(
            #     lambda_reg=self.lambda_reg, prox_iterations=self.tv_real.prox_iterations,
            #     is_3d=self.is_3d, epsilon_tv_grad=self.tv_real.epsilon_tv_grad,
            #     prox_step_size=self.tv_real.prox_step_size, device=self.device
            # )
            # return temp_tv_real_full_lambda.proximal_operator(x_complex_image, step_size)
            # Fallback: if reconstructor expects complex, but gets real, make it complex.
             x_complex_image = x_complex_image.to(torch.complex64)


        x_real = self.tv_real.proximal_operator(x_complex_image.real.clone(), step_size)
        x_imag = self.tv_imag.proximal_operator(x_complex_image.imag.clone(), step_size)
        return torch.complex(x_real, x_imag)

def tv_reconstruction_mwi(
    y_scattered_data: torch.Tensor,
    mwi_operator: 'MicrowaveImagingOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    # Parameters for Complex TV regularizer
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01,
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for Microwave Imaging (MWI)
    data, aiming to recover a complex dielectric contrast map.

    Args:
        y_scattered_data (torch.Tensor): Acquired microwave scattered field data (complex).
                                         Shape: (num_measurements,).
        mwi_operator (MicrowaveImagingOperator): Configured MWI operator.
        lambda_tv (float): Regularization strength for complex TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Inner iterations for TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Inner step size for TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial dielectric map.
                                                  If None, ProxGradReconstructor defaults (e.g. adjoint).
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed complex dielectric contrast map.
                      Shape: mwi_operator.image_shape.
    """
    device = y_scattered_data.device
    is_3d_map = len(mwi_operator.image_shape) == 3

    # Use the ComplexTotalVariationRegularizer for complex-valued images
    complex_tv_regularizer = ComplexTotalVariationRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_map,
        epsilon_tv_grad=tv_epsilon_grad,
        prox_step_size=tv_prox_step_size,
        device=device
    )

    forward_op_fn_wrapper = lambda image_estimate, smaps: mwi_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: mwi_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: complex_tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm:                print(f"MWI TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}")                if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_mwi: Using adjoint of y_scattered_data as initial estimate.")
        x_init_arg = mwi_operator.op_adj(y_scattered_data)

    # Ensure initial estimate is complex, as MWI reconstructs complex permittivity
    if x_init_arg is not None and not x_init_arg.is_complex():
        print("Warning: Initial estimate for MWI reconstruction is real. Converting to complex.")
        x_init_arg = x_init_arg.to(torch.complex64)
    elif mwi_operator.image_shape is not None and x_init_arg is None and initial_estimate_fn is None:
        # If default zero init is used by ProxGradReconstructor, ensure it's complex
        # This is implicitly handled if image_shape_for_zero_init leads to complex default.
        # ProxGradReconstructor's default init for complex data should be complex zeros.
        pass


    reconstructed_dielectric_map = pg_reconstructor.reconstruct(
        kspace_data=y_scattered_data, # Measurements
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        # image_shape_for_zero_init is used by ProxGrad if x_init is None
        # It should ideally create complex zeros if the problem is complex.
        image_shape_for_zero_init=mwi_operator.image_shape if x_init_arg is None else None,
        # Ensure reconstructor knows the output should be complex
        # This might need a flag in ProximalGradientReconstructor or rely on x_init's dtype
        # For now, we assume x_init (if provided) or the default zero init will set the stage.
        # The regularizer also expects complex input.
    )

    # Final output should be complex
    if not reconstructed_dielectric_map.is_complex():
        print("Warning: MWI reconstruction resulted in a real map. Converting to complex.")
        reconstructed_dielectric_map = reconstructed_dielectric_map.to(torch.complex64)


    return reconstructed_dielectric_map

if __name__ == '__main__':
    print("Running basic execution checks for Microwave Imaging reconstructors...")
    import numpy as np
    try:
        from reconlib.modalities.microwave.operators import MicrowaveImagingOperator
    except ImportError:
        print("Attempting local import for MicrowaveImagingOperator for __main__ block.")
        from operators import MicrowaveImagingOperator

    device_mwi_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape_test_mwi = (24, 24)
    num_pixels_test_mwi = np.prod(img_shape_test_mwi)
    num_measurements_test_mwi = num_pixels_test_mwi // 2

    system_matrix_test_mwi = torch.randn(
        num_measurements_test_mwi, num_pixels_test_mwi,
        dtype=torch.complex64, device=device_mwi_recon
    )

    try:
        mwi_op_instance = MicrowaveImagingOperator(
            image_shape=img_shape_test_mwi,
            system_matrix=system_matrix_test_mwi,
            device=device_mwi_recon
        )
        print("Using actual MicrowaveImagingOperator for test.")

        dummy_scatter_data_mwi = torch.randn(
            num_measurements_test_mwi, dtype=torch.complex64, device=device_mwi_recon
        )

        recon_map_mwi = tv_reconstruction_mwi(
            y_scattered_data=dummy_scatter_data_mwi,
            mwi_operator=mwi_op_instance,
            lambda_tv=0.01,
            iterations=3, # Low for quick test
            step_size=0.01,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"MWI TV reconstruction output shape: {recon_map_mwi.shape}, dtype: {recon_map_mwi.dtype}")
        assert recon_map_mwi.shape == img_shape_test_mwi
        assert recon_map_mwi.is_complex(), "MWI reconstruction output must be complex."
        print("tv_reconstruction_mwi basic execution check PASSED.")

    except Exception as e:
        print(f"Error during tv_reconstruction_mwi check: {e}")
        import traceback
        traceback.print_exc()
