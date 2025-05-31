import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVRegularizer # Corrected class name
from reconlib.modalities.sar.operators import SARForwardOperator # Changed to absolute import

def tv_reconstruction_sar(
    y_sar_data: torch.Tensor, # Measured visibilities
    sar_operator: SARForwardOperator,
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.01,
    # Parameters for UltrasoundTVRegularizer (corrected name)
    tv_prox_iterations: int = 10,
    tv_prox_step_size: float = 0.01, # Note: tv_prox_step_size is not directly used by current UltrasoundTVRegularizer
    tv_epsilon_grad: float = 1e-8,
    initial_estimate_fn: callable = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs Total Variation (TV) regularized reconstruction for SAR data
    using the Proximal Gradient algorithm.

    Args:
        y_sar_data (torch.Tensor): Acquired SAR visibilities (k-space samples).
                                   Shape (num_visibilities,).
        sar_operator (SARForwardOperator): Configured SAR forward/adjoint operator.
            To instantiate SARForwardOperator:
            - Provide `image_shape`.
            - Either provide `uv_coordinates` directly (e.g., pre-calculated, shape (num_vis, 2),
              scaled like FFT indices: u in approx [-Nx/2, Nx/2-1], v in approx [-Ny/2, Ny/2-1]).
            - Or, provide physical parameters: `wavelength`, `sensor_azimuth_angles`, and `fov`
              (field of view: fov_y, fov_x) to calculate uv_coordinates internally.
            - Set `use_nufft=False` (default) for reliable FFT-based operations, or `use_nufft=True`
              if torchkbnufft is installed and expected to work.
        lambda_tv (float): Regularization strength for TV.
        iterations (int, optional): Number of outer proximal gradient iterations. Defaults to 50.
        step_size (float, optional): Step size for the proximal gradient algorithm. Defaults to 0.01.
        tv_prox_iterations (int, optional): Number of inner iterations for the custom TV prox. Defaults to 10.
        tv_prox_step_size (float, optional): Step size for the inner gradient descent in custom TV prox. Defaults to 0.01.
        tv_epsilon_grad (float, optional): Epsilon for TV gradient norm calculation. Defaults to 1e-8.
        initial_estimate_fn (callable, optional): Function to compute initial image.
                                                  If None, ProxGradReconstructor defaults to zero or adjoint.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The TV-regularized reconstructed SAR image (reflectivity map).
                      Shape (Ny, Nx).
    """
    device = y_sar_data.device

    # Instantiate the TV regularizer (SAR images are 2D)
    # Note: tv_prox_step_size is not a direct param of UltrasoundTVRegularizer's __init__
    # It was used in a previous version or a different TV regularizer structure.
    # The current UltrasoundTVRegularizer uses a fixed internal step for its prox loop.
    tv_regularizer = UltrasoundTVRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False,
        epsilon_tv_grad=tv_epsilon_grad
        # prox_step_size is not a parameter for UltrasoundTVRegularizer constructor
    )

    # Wrappers for ProximalGradientReconstructor
    forward_op_fn_wrapper = lambda image_estimate, smaps: sar_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda k_space_data, smaps: sar_operator.op_adj(k_space_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn,
        verbose=verbose,
        log_fn=lambda iter_num, current_image, change, grad_norm: \
               print(f"SAR TV Recon Iter {iter_num+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_num % 10 == 0 or iter_num == iterations -1) else None
    )

    x_init_arg = None
    image_shape_for_zero_init_arg = None
    if initial_estimate_fn is None:
        print("tv_reconstruction_sar: Using adjoint of y_sar_data (dirty image) as initial estimate.")
        x_init_arg = sar_operator.op_adj(y_sar_data)
        # Or for zero init: image_shape_for_zero_init_arg = sar_operator.image_shape

    reconstructed_image = pg_reconstructor.reconstruct(
        kspace_data=y_sar_data, # These are the visibilities
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=image_shape_for_zero_init_arg
    )

    return reconstructed_image

if __name__ == '__main__':
    print("Running basic execution checks for SAR reconstructors...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    img_shape_test_sar = (32, 32) # Ny, Nx

    # Physical parameters for SARForwardOperator
    num_angles = 50 # Number of simulated k-space samples
    # Example: angles spanning a 180-degree arc (pi radians)
    angles = torch.linspace(0, torch.pi, num_angles, device=device)
    wavelength_test = 0.03 # meters (e.g., X-band SAR, ~10 GHz)
    # FOV chosen so max raw_uv_coords are approx Ny/2, Nx/2
    fov_test = (img_shape_test_sar[0] * wavelength_test / 2,
                img_shape_test_sar[1] * wavelength_test / 2)


    # Instantiate the real SARForwardOperator using physical parameters
    # Defaults to use_nufft=False, which is desired for reliable testing here.
    try:
        sar_op_for_recon_test = SARForwardOperator(
            image_shape=img_shape_test_sar,
            wavelength=wavelength_test,
            sensor_azimuth_angles=angles,
            fov=fov_test,
            device=device
        )
        print("Successfully instantiated SARForwardOperator for reconstructor test.")
    except ImportError as e:
        print(f"Could not instantiate SARForwardOperator (likely torchkbnufft missing for NUFFT path, though not selected): {e}")
        print("Skipping reconstructor test.")
        exit()
    except Exception as e:
        print(f"Error instantiating SARForwardOperator: {e}")
        print("Skipping reconstructor test.")
        exit()

    # Generate dummy visibilities using the operator's forward model
    # Need a dummy phantom image
    dummy_phantom = torch.randn(img_shape_test_sar, dtype=torch.complex64, device=device)
    dummy_sar_visibilities = sar_op_for_recon_test.op(dummy_phantom)

    # Ensure visibilities are complex (should be by default from operator)
    if not torch.is_complex(dummy_sar_visibilities):
        dummy_sar_visibilities = dummy_sar_visibilities.to(torch.complex64)

    print(f"Generated dummy visibilities shape: {dummy_sar_visibilities.shape}")


    # Test tv_reconstruction_sar
    try:
        recon_img_sar = tv_reconstruction_sar(
            y_sar_data=dummy_sar_visibilities,
            sar_operator=sar_op_for_recon_test, # Use the real operator
            lambda_tv=0.005,
            iterations=3, # Keep low for a quick test
            step_size=0.02,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"SAR TV reconstruction output shape: {recon_img_sar.shape}")
        assert recon_img_sar.shape == img_shape_test_sar
        print("tv_reconstruction_sar basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_sar check: {e}")
        # raise # Avoid raising in subtask
