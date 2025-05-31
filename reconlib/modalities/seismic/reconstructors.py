import torch
import numpy as np # For __main__ if ricker_wavelet_local uses np
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVRegularizer # Corrected name from previous subtask context
from reconlib.modalities.seismic.operators import SeismicForwardOperator # Changed to absolute import

def tv_reconstruction_seismic(
    y_seismic_traces: torch.Tensor,
    seismic_operator: SeismicForwardOperator,
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
            To instantiate SeismicForwardOperator, provide parameters like:
            - `reflectivity_map_shape` (tuple[int, int]): (Nz, Nx)
            - `wave_speed_mps` (float)
            - `time_sampling_dt_s` (float)
            - `num_time_samples` (int)
            - `source_pos_m` (tuple[float, float]): (src_x, src_z)
            - `receiver_pos_m` (torch.Tensor): Shape (num_receivers, 2) for (rec_x, rec_z)
            - `pixel_spacing_m` (float or tuple[float,float], optional)
            - `source_wavelet` (torch.Tensor, optional): 1D tensor for the source wavelet. Default: None (delta pulse).
            - `wavelet_time_offset_s` (float, optional): Time offset for the wavelet center. Default: 0.0.
            - `apply_geometrical_spreading` (bool, optional): Whether to apply 1/R spreading. Default: True.
            - `device` (str or torch.device, optional)
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

    # Instantiate the TV regularizer (Seismic reflectivity map is 2D)
    # Assuming UltrasoundTVRegularizer is the intended class (corrected from UltrasoundTVCustomRegularizer)
    # Note: tv_prox_step_size might not be used by UltrasoundTVRegularizer if its internal prox solver is fixed.
    tv_regularizer = UltrasoundTVRegularizer( # Corrected class name
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=False,
        epsilon_tv_grad=tv_epsilon_grad
        # prox_step_size=tv_prox_step_size # This was noted as unused in UltrasoundTVRegularizer
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
        log_fn=lambda iter_num, current_image, change, grad_norm: \
               print(f"Seismic TV Recon Iter {iter_num+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_num % 10 == 0 or iter_num == iterations -1) else None
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define ricker_wavelet function (can be moved to a common utils if shared)
    def ricker_wavelet_local(peak_freq, dt, num_samples, device='cpu'):
        """Generates a Ricker wavelet."""
        # Ensure num_samples is odd for a symmetric wavelet if peak is to be exact center sample
        if num_samples % 2 == 0:
            # print(f"Warning: ricker_wavelet_local num_samples {num_samples} is even, making it {num_samples+1}")
            num_samples +=1

        t_np = (np.arange(num_samples) - num_samples // 2) * dt # time vector centered at 0
        t_scaled_np = t_np * peak_freq * np.pi
        y_np = (1.0 - 2.0 * t_scaled_np**2) * np.exp(-t_scaled_np**2)
        return torch.tensor(y_np, dtype=torch.float32, device=device)

    map_shape_test_seismic = (32, 48) # Nz, Nx
    pixel_spacing_val = 1.0 # meters
    src_pos_val = (map_shape_test_seismic[1] * pixel_spacing_val / 2.0, 0.0) # Source at center surface

    num_recs_val = 10
    rec_x_np = np.linspace(0, (map_shape_test_seismic[1]-1) * pixel_spacing_val, num_recs_val)
    rec_pos_val = torch.tensor(np.stack([rec_x_np, np.zeros(num_recs_val)], axis=-1), dtype=torch.float32, device=device)

    time_sampling_dt_s_val = 0.001 # 1 ms
    num_time_samples_val = 500    # 0.5 seconds of recording

    wavelet_len_val = 31 # Odd number for symmetric peak
    test_wavelet_val = ricker_wavelet_local(25.0, time_sampling_dt_s_val, wavelet_len_val, device=device)
    # Offset for a wavelet centered at its middle sample
    test_wavelet_offset_s_val = (wavelet_len_val // 2) * time_sampling_dt_s_val

    try:
        seismic_op_for_recon_test = SeismicForwardOperator(
            reflectivity_map_shape=map_shape_test_seismic,
            wave_speed_mps=2000.0,
            time_sampling_dt_s=time_sampling_dt_s_val,
            num_time_samples=num_time_samples_val,
            source_pos_m=src_pos_val,
            receiver_pos_m=rec_pos_val,
            pixel_spacing_m=pixel_spacing_val,
            source_wavelet=test_wavelet_val,
            wavelet_time_offset_s=test_wavelet_offset_s_val,
            apply_geometrical_spreading=True, # Default, but explicit
            device=device
        )
        print("Successfully instantiated SeismicForwardOperator for reconstructor test.")

        # Generate dummy reflectivity map (real-valued for seismic)
        dummy_reflectivity_map = torch.randn(map_shape_test_seismic, device=device).float()
        # Apply some structure if desired, e.g., layers or points
        dummy_reflectivity_map[:, map_shape_test_seismic[1]//2:] *= 0.5 # Change reflectivity in one half
        dummy_reflectivity_map[map_shape_test_seismic[0]//2, map_shape_test_seismic[1]//2] = 2.0 # A point diffractor

        print("Generating dummy seismic traces using the real operator...")
        dummy_seismic_traces = seismic_op_for_recon_test.op(dummy_reflectivity_map)
        print(f"Generated dummy seismic traces shape: {dummy_seismic_traces.shape}")

        # Test tv_reconstruction_seismic
        print("Starting TV reconstruction for seismic data...")
        recon_map_seismic = tv_reconstruction_seismic(
            y_seismic_traces=dummy_seismic_traces,
            seismic_operator=seismic_op_for_recon_test,
            lambda_tv=0.01,
            iterations=3, # Keep low for a quick test
            step_size=0.05,
            tv_prox_iterations=2,
            verbose=True
        )
        print(f"Seismic TV reconstruction output shape: {recon_map_seismic.shape}")
        assert recon_map_seismic.shape == map_shape_test_seismic
        print("tv_reconstruction_seismic basic execution check PASSED.")
    except Exception as e:
        print(f"Error during tv_reconstruction_seismic check: {e}")
