import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer # Generic TV
# from .operators import PCCTProjectorOperator # For type hinting
import numpy as np # Added for np.pi in new reconstructor's fallback imports
from reconlib.operators import Operator # Added for LinearRadonOperator in new reconstructor

def tv_reconstruction_pcct_mu_ref(
    y_photon_counts_stack: torch.Tensor, # (num_bins, num_angles, num_detector_pixels)
    pcct_operator: 'PCCTProjectorOperator',
    lambda_tv: float,
    iterations: int = 50,
    step_size: float = 0.001, # May need careful tuning due to exp/log and Radon
    tv_prox_iterations: int = 10,
    initial_mu_ref_estimate: torch.Tensor | None = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Placeholder TV-regularized reconstruction for PCCT to recover a reference
    attenuation map (mu_reference_map) from multi-bin photon count data.

    This uses the simplified PCCTProjectorOperator which includes non-linear
    Beer-Lambert law. The ProximalGradientReconstructor assumes a problem of
    the form argmin_x 0.5*||Ax - y||_2^2 + lambda*R(x).
    Given the non-linearity in PCCTOperator, this is a simplification and
    the step_size might be critical or a more advanced (non-linear) solver needed.
    """
    device = y_photon_counts_stack.device

    # The image to reconstruct (mu_reference_map) is typically 2D or 3D.
    # For this operator, it's 2D.
    is_3d_map = len(pcct_operator.image_shape) == 3
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_map,
        device=device
    )

    forward_op_fn_wrapper = lambda image_estimate, smaps: pcct_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda measurement_data, smaps: pcct_operator.op_adj(measurement_data)
    regularizer_prox_fn_wrapper = lambda image, sl: tv_regularizer.proximal_operator(image, sl)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=None,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"PCCT TV Recon Iter {iter_n+1}/{iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == iterations -1) else None
    )

    x_init_arg = initial_mu_ref_estimate
    if x_init_arg is None:
        print("tv_reconstruction_pcct_mu_ref: Using adjoint of measurements as initial estimate.")
        x_init_arg = pcct_operator.op_adj(y_photon_counts_stack)
    else:
        x_init_arg = x_init_arg.to(device)

    # Ensure initial estimate is non-negative for attenuation coefficients
    x_init_arg = torch.clamp(x_init_arg, min=0.0)

    reconstructed_mu_ref = pg_reconstructor.reconstruct(
        kspace_data=y_photon_counts_stack, # 'kspace_data' is generic name for measurements
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None,
        x_init=x_init_arg,
        image_shape_for_zero_init=pcct_operator.image_shape if x_init_arg is None and initial_mu_ref_estimate is None else None
    )

    # Ensure final reconstruction is non-negative
    reconstructed_mu_ref = torch.clamp(reconstructed_mu_ref, min=0.0)
    return reconstructed_mu_ref

if __name__ == '__main__':
    print("Running basic PCCT reconstructor checks...")
    dev_recon = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_recon = (16,16)
    n_a_recon = 10
    n_d_recon = 18
    e_bins_recon = [(20,50), (50,80), (80,120)]
    n_b_recon = len(e_bins_recon)
    I0_recon = torch.tensor([1e4, 1e5, 1e4], device=dev_recon)
    e_scales_recon = torch.tensor([1.0, 0.8, 0.6], device=dev_recon)

    try:
        from .operators import PCCTProjectorOperator # Relative import
        pcct_op_inst = PCCTProjectorOperator(
            image_shape=img_s_recon,
            num_angles=n_a_recon,
            num_detector_pixels=n_d_recon,
            energy_bins_keV=e_bins_recon,
            source_photons_per_bin=I0_recon,
            energy_scaling_factors=e_scales_recon,
            device=dev_recon
        )

        # Dummy measured photon counts (stack of sinograms)
        dummy_photon_counts = torch.rand((n_b_recon, n_a_recon, n_d_recon), device=dev_recon) * (I0_recon.view(n_b_recon,1,1)*0.1)


        recon_mu = tv_reconstruction_pcct_mu_ref(
            y_photon_counts_stack=dummy_photon_counts,
            pcct_operator=pcct_op_inst,
            lambda_tv=1e-4, # May need significant tuning
            iterations=3,   # Low for quick check
            step_size=1e-5, # Very small due to potential large gradients from exp/log & Radon
            verbose=True
        )
        print(f"PCCT reconstruction output shape: {recon_mu.shape}")
        assert recon_mu.shape == img_s_recon
        print("tv_reconstruction_pcct_mu_ref basic check PASSED.")
    except Exception as e:
        print(f"Error in tv_reconstruction_pcct_mu_ref check: {e}")
        import traceback; traceback.print_exc()


# --- New Content Appended Below ---

# Assuming simple_radon_transform and simple_back_projection are accessible
# If they are in pcct.operators, we'd need to import them.
# For now, let's redefine them here for encapsulation if they are not exposed,
# or assume they will be imported if reconlib structure makes them available.
# To avoid circular dependency if they stay in pcct.operators, this might need a shared util.
# For this step, we'll assume they can be accessed or redefined if necessary.
# Let's try importing them from the operator module as they are already there.
try:
    from .operators import simple_radon_transform, simple_back_projection
except ImportError: # Fallback if running script directly or structure changes
    print("Warning: Could not import simple_radon_transform/simple_back_projection from .operators for PCCT reconstructor.")
    # Define simplified versions here if import fails (should match those in PCCTProjectorOperator)
    def simple_radon_transform(image: torch.Tensor, num_angles: int,
                               num_detector_pixels: int | None = None,
                               device='cpu') -> torch.Tensor:
        Ny, Nx = image.shape
        if num_detector_pixels is None: num_detector_pixels = max(Ny, Nx)
        image = image.to(device)
        angles = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device)
        sinogram = torch.zeros((num_angles, num_detector_pixels), device=device, dtype=image.dtype)
        x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
        y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1, num_detector_pixels, device=device)
        for i, angle in enumerate(angles):
            rot_coords = grid_x * torch.cos(angle) + grid_y * torch.sin(angle)
            for j, det_pos in enumerate(detector_coords):
                mask = (rot_coords >= det_pos - 0.5) & (rot_coords < det_pos + 0.5) # Pixel width = 1 approx
                sinogram[i, j] = torch.sum(image[mask])
        return sinogram

    def simple_back_projection(sinogram: torch.Tensor, image_shape: tuple[int,int],
                               device='cpu') -> torch.Tensor:
        num_angles, num_detector_pixels = sinogram.shape
        Ny, Nx = image_shape
        sinogram = sinogram.to(device)
        reconstructed_image = torch.zeros(image_shape, device=device, dtype=sinogram.dtype)
        angles = torch.linspace(0, np.pi, num_angles, endpoint=False, device=device)
        x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
        y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1, num_detector_pixels, device=device)
        for i, angle in enumerate(angles):
            rot_coords_pixel = grid_x * torch.cos(angle) + grid_y * torch.sin(angle)
            diffs = torch.abs(rot_coords_pixel.unsqueeze(-1) - detector_coords.view(1,1,-1))
            nearest_det_indices = torch.argmin(diffs, dim=2)
            reconstructed_image += sinogram[i, nearest_det_indices]
        return reconstructed_image / num_angles


class LinearRadonOperator(Operator):
    """A simple linear operator for Radon transform using the helper functions."""
    def __init__(self, image_shape: tuple[int,int], num_angles: int, num_detector_pixels: int, device='cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.num_angles = num_angles
        self.num_detector_pixels = num_detector_pixels
        self.device = device
        self.output_shape = (num_angles, num_detector_pixels)

    def op(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape != self.image_shape:
            raise ValueError("Input image shape mismatch for LinearRadonOperator.")
        return simple_radon_transform(image, self.num_angles, self.num_detector_pixels, self.device)

    def op_adj(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.shape != self.output_shape:
            raise ValueError("Input sinogram shape mismatch for LinearRadonOperator.")
        return simple_back_projection(sinogram, self.image_shape, self.device)


def iterative_reconstruction_pcct_bin(
    noisy_counts_sinogram_bin: torch.Tensor, # (num_angles, num_detector_pixels)
    source_photons_bin: float | torch.Tensor,
    # For shape info and Radon parameters:
    image_shape: tuple[int,int],
    num_angles: int,
    num_detector_pixels: int,
    # Regularization and PGD params:
    lambda_tv: float = 0.01,
    pgd_iterations: int = 50,
    pgd_step_size: float = 0.01,
    tv_prox_iterations: int = 10,
    initial_mu_eff_estimate: torch.Tensor | None = None,
    device: str | torch.device = 'cpu',
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs iterative TV-regularized reconstruction for a single energy bin of PCCT data.
    Reconstructs the effective attenuation map (mu_effective_bin) from linearized data.
    """
    noisy_counts_sinogram_bin = noisy_counts_sinogram_bin.to(device)
    if isinstance(source_photons_bin, (float, int)):
        source_photons_bin = torch.tensor(source_photons_bin, device=device, dtype=torch.float32)
    else:
        source_photons_bin = source_photons_bin.to(device)

    epsilon = 1e-9
    transmission = noisy_counts_sinogram_bin / (source_photons_bin + epsilon)
    transmission = torch.clamp(transmission, epsilon, 1.0 - epsilon)
    line_integrals_bin = -torch.log(transmission)

    # Define the linear Radon operator for PGD
    radon_linear_op = LinearRadonOperator(
        image_shape=image_shape,
        num_angles=num_angles,
        num_detector_pixels=num_detector_pixels,
        device=device
    )

    is_3d_map = len(image_shape) == 3 # Should be False for this 2D reconstructor
    tv_regularizer = UltrasoundTVCustomRegularizer(
        lambda_reg=lambda_tv,
        prox_iterations=tv_prox_iterations,
        is_3d=is_3d_map,
        device=device
    )

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=pgd_iterations,
        step_size=pgd_step_size,
        initial_estimate_fn=None,
        verbose=verbose,
        log_fn=lambda iter_n, img, change, grad_norm: \
               print(f"PCCT Bin Iter Recon Iter {iter_n+1}/{pgd_iterations}: Change={change:.2e}, GradNorm={grad_norm:.2e}") \
               if verbose and (iter_n % 10 == 0 or iter_n == pgd_iterations -1) else None
    )

    x_init_arg = initial_mu_eff_estimate
    if x_init_arg is None:
        if verbose: print("iterative_reconstruction_pcct_bin: Using back-projection of line_integrals as initial estimate.")
        x_init_arg = radon_linear_op.op_adj(line_integrals_bin)
    else:
        x_init_arg = x_init_arg.to(device)

    # PGD reconstructs mu_effective_bin from line_integrals_bin using Radon operator
    reconstructed_mu_effective_bin = pg_reconstructor.reconstruct(
        kspace_data=line_integrals_bin, # "kspace_data" is the measurement input to PGD
        forward_op_fn=radon_linear_op.op,
        adjoint_op_fn=radon_linear_op.op_adj,
        regularizer_prox_fn=tv_regularizer.proximal_operator,
        sensitivity_maps=None,
        x_init=x_init_arg
    )
    return reconstructed_mu_effective_bin

if __name__ == "__main__": # This will now be part of the appended content's __main__
    # Keep existing __main__ from tv_reconstruction_pcct_mu_ref if it was there
    # Add new tests for iterative_reconstruction_pcct_bin

    # The original __main__ content is already here from the file read.
    # The new __main__ content from the prompt will be appended after the original one.
    # This means there might be two if __name__ == '__main__': blocks, which is not ideal
    # but should still execute the second one if the script is run.
    # A cleaner way would be to merge them. For now, direct append as per logic.

    print("\nRunning additional PCCT iterative reconstructor checks...")
    dev_recon_iter = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s_iter = (16,16)
    n_a_iter = 10
    n_d_iter = 20
    I0_iter = 10000.0

    try:
        # Create dummy noisy counts for one bin
        dummy_mu_eff = torch.rand(img_s_iter, device=dev_recon_iter) * 0.02
        temp_radon_op = LinearRadonOperator(img_s_iter, n_a_iter, n_d_iter, dev_recon_iter)
        true_line_integrals = temp_radon_op.op(dummy_mu_eff)
        noisy_counts = torch.poisson(I0_iter * torch.exp(-true_line_integrals))

        recon_mu_eff = iterative_reconstruction_pcct_bin(
            noisy_counts_sinogram_bin=noisy_counts,
            source_photons_bin=I0_iter,
            image_shape=img_s_iter,
            num_angles=n_a_iter,
            num_detector_pixels=n_d_iter,
            lambda_tv=0.001,
            pgd_iterations=5, # Quick test
            device=dev_recon_iter,
            verbose=True
        )
        print(f"PCCT iterative bin reconstruction output shape: {recon_mu_eff.shape}")
        assert recon_mu_eff.shape == img_s_iter
        print("iterative_reconstruction_pcct_bin basic check PASSED.")

    except Exception as e:
        print(f"Error in iterative_reconstruction_pcct_bin check: {e}")
        import traceback; traceback.print_exc()
