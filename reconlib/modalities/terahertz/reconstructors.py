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
        # Ensure initial estimate matches expected dtype for the reconstructor if operator output complex things by default
        if x_init_arg.is_complex() and not thz_operator.system_matrix.is_complex(): # if system matrix was real, image should be real
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

    # If the system matrix was real, the reconstructed image should ideally be real.
    # The proximal gradient reconstructor might produce complex outputs if intermediate steps are complex.
    if not thz_operator.system_matrix.is_complex() and reconstructed_thz_image.is_complex():
        reconstructed_thz_image = reconstructed_thz_image.real


    return reconstructed_thz_image

if __name__ == '__main__':
    print("Running basic execution checks for Terahertz reconstructors...")
    import numpy as np
    try:
        # Adjust path as necessary if reconlib.modalities is not directly in PYTHONPATH
        from reconlib.modalities.terahertz.operators import TerahertzOperator
    except ImportError:
        print("Attempting local import for TerahertzOperator for __main__ block.")
        from operators import TerahertzOperator # Fallback for direct execution

    device_thz_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape_test_thz = (24, 24) # Smaller for quicker test
    num_pixels_test_thz = np.prod(img_shape_test_thz)
    num_measurements_test_thz = num_pixels_test_thz // 2

    # System matrix can be real or complex for testing
    is_complex_system_test = False # Or True
    dtype_system_test = torch.complex64 if is_complex_system_test else torch.float32

    try:
        # Try with a pre-defined system matrix
        system_matrix_test_thz = torch.randn(
            num_measurements_test_thz, num_pixels_test_thz,
            dtype=dtype_system_test, device=device_thz_recon
        )
        thz_op_instance = TerahertzOperator(
            image_shape=img_shape_test_thz,
            system_matrix=system_matrix_test_thz, # Provide the matrix
            device=device_thz_recon
        )
        print("Using actual TerahertzOperator with provided system matrix for test.")
    except Exception as e:
        print(f"Could not instantiate actual TerahertzOperator ({e}). This test might not be meaningful.")
        # Fallback to a minimal mock if absolutely necessary, but prefer actual operator
        class MockTHzOperator:
            def __init__(self, image_shape, num_measurements, is_complex, device):
                self.image_shape = image_shape
                self.num_measurements = num_measurements
                self.is_complex_system = is_complex # Store this info
                self.system_matrix_is_complex = is_complex # for reconstructor logic
                self.device = device
                # Minimal system matrix for mock
                num_pixels = np.prod(image_shape)
                dtype = torch.complex64 if is_complex else torch.float32
                self.system_matrix = torch.randn(num_measurements, num_pixels, dtype=dtype, device=device)


            def op(self, x_image):
                img_flat = x_image.reshape(-1)
                if self.is_complex_system and not img_flat.is_complex():
                    img_flat = img_flat.to(torch.complex64)
                return torch.matmul(self.system_matrix, img_flat)

            def op_adj(self, y_data):
                adj_matrix = self.system_matrix.conj().T if self.is_complex_system else self.system_matrix.T
                recon_flat = torch.matmul(adj_matrix, y_data)
                recon_img = recon_flat.reshape(self.image_shape)
                return recon_img.real if not self.is_complex_system else recon_img

        thz_op_instance = MockTHzOperator(img_shape_test_thz, num_measurements_test_thz, is_complex_system_test, device_thz_recon)
        print("Using MockTHzOperator for test due to issues with actual operator.")


    # Dummy THz data
    # Match dtype to what the operator would produce
    expected_data_dtype = dtype_system_test # If op output matches system_matrix * image (possibly complex)
    dummy_thz_data = torch.randn(num_measurements_test_thz, dtype=expected_data_dtype, device=device_thz_recon)

    try:
        recon_image_thz = tv_reconstruction_thz(
            y_thz_data=dummy_thz_data,
            thz_operator=thz_op_instance,
            lambda_tv=0.01,
            iterations=3,
            step_size=0.01,
            tv_prox_iterations=2,
            is_3d_tv=len(img_shape_test_thz) == 3,
            verbose=True
        )
        print(f"Terahertz TV reconstruction output shape: {recon_image_thz.shape}, dtype: {recon_image_thz.dtype}")
        assert recon_image_thz.shape == img_shape_test_thz
        # Check if output dtype is real if system matrix was real
        if not is_complex_system_test:
            assert not recon_image_thz.is_complex(), "Output should be real for a real system matrix."
        print("tv_reconstruction_thz basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_thz check: {e}")
        import traceback
        traceback.print_exc()
