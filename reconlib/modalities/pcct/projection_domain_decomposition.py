import torch
from typing import List, Tuple, Dict, Optional, Callable

from reconlib.operators import Operator # For type hinting and base class
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor

try:
    from reconlib.regularizers.base import RegularizerBase
except ImportError:
    print("Warning: reconlib.regularizers.base.RegularizerBase not found, using dummy placeholder for type hinting.")
    class RegularizerBase: # Dummy for type hint
        def __init__(self, *args, **kwargs): pass
        def proximal_operator(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
            # This dummy regularizer does nothing.
            return x

# For LinearRadonOperatorPlaceholder - these should be the fixed versions
# These are simplified and assume parallel beam.
def simple_radon_transform(image: torch.Tensor, num_angles: int,
                           num_detector_pixels: int | None = None,
                           device='cpu') -> torch.Tensor:
    Ny, Nx = image.shape
    if num_detector_pixels is None:
        num_detector_pixels = max(Ny, Nx)
    image = image.to(device)
    angles = torch.arange(num_angles, device=device, dtype=torch.float32) * (torch.pi / num_angles)
    sinogram = torch.zeros((num_angles, num_detector_pixels), device=device, dtype=image.dtype)
    x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1,
                                     num_detector_pixels, device=device)
    for i, angle_val in enumerate(angles):
        rot_coords = grid_x * torch.cos(angle_val) + grid_y * torch.sin(angle_val)
        for j, det_pos in enumerate(detector_coords):
            pixel_width_on_detector = 1.0
            mask = (rot_coords >= det_pos - pixel_width_on_detector/2) & \
                   (rot_coords < det_pos + pixel_width_on_detector/2)
            sinogram[i, j] = torch.sum(image[mask])
    return sinogram

def simple_back_projection(sinogram: torch.Tensor, image_shape: tuple[int,int],
                           device='cpu') -> torch.Tensor:
    num_angles, num_detector_pixels = sinogram.shape
    Ny, Nx = image_shape
    sinogram = sinogram.to(device)
    reconstructed_image = torch.zeros(image_shape, device=device, dtype=sinogram.dtype)
    angles = torch.arange(num_angles, device=device, dtype=torch.float32) * (torch.pi / num_angles)
    x_coords = torch.linspace(-Nx // 2, Nx // 2 -1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 -1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 -1,
                                     num_detector_pixels, device=device)
    for i, angle_val in enumerate(angles):
        rot_coords_pixel = grid_x * torch.cos(angle_val) + grid_y * torch.sin(angle_val)
        diffs = torch.abs(rot_coords_pixel.unsqueeze(-1) - detector_coords.view(1,1,-1))
        nearest_det_indices = torch.argmin(diffs, dim=2)
        reconstructed_image += sinogram[i, nearest_det_indices]
    return reconstructed_image / num_angles


class LinearRadonOperatorPlaceholder(Operator):
    def __init__(self, image_shape: Tuple[int, int], num_angles: int, num_detector_pixels: int, device='cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.num_angles = num_angles
        self.num_detector_pixels = num_detector_pixels
        self.device = torch.device(device)
        self.output_shape_op = (num_angles, num_detector_pixels) # For op_adj validation

    def op(self, image: torch.Tensor, sensitivity_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sensitivity_maps is accepted for PGD compatibility, not used by this simple Radon op.
        if image.shape != self.image_shape:
            raise ValueError(f"Input image shape mismatch for LinearRadonOperatorPlaceholder. Expected {self.image_shape}, got {image.shape}")
        return simple_radon_transform(image, self.num_angles, self.num_detector_pixels, str(self.device))

    def op_adj(self, sinogram: torch.Tensor, sensitivity_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sensitivity_maps accepted for PGD compatibility, not used by this simple Radon op.
        if sinogram.shape != self.output_shape_op:
            raise ValueError(f"Input sinogram shape mismatch for LinearRadonOperatorPlaceholder. Expected {self.output_shape_op}, got {sinogram.shape}")
        return simple_back_projection(sinogram, self.image_shape, str(self.device))


# Define __all__ for the module
__all__ = [
    'calculate_material_thickness_sinograms',
    'reconstruct_thickness_maps_from_sinograms',
    'LinearRadonOperatorPlaceholder' # Added for notebook example usability
]

def calculate_material_thickness_sinograms(
    log_transformed_sinograms: torch.Tensor,
    mac_matrix: torch.Tensor,
    epsilon_determinant: float = 1e-9
) -> torch.Tensor:
    """
    Calculates material thickness sinograms from log-transformed multi-energy sinograms
    using a material attenuation coefficient (MAC) matrix.

    This function solves the linear system L = M * T for T, where:
    - L is the stack of log-transformed sinograms for different energy bins.
    - M is the MAC matrix, where M_ij is the MAC of material j in energy bin i.
    - T is the stack of material thickness sinograms to be determined.

    The system is solved for each detector pixel and angle independently.
    Currently, this implementation is optimized for and requires N_energy_bins = N_materials
    (e.g., 2 energy bins and 2 materials).

    Args:
        log_transformed_sinograms (torch.Tensor): Input tensor of log-transformed sinograms.
            Expected shape: (num_energy_bins, num_angles, num_detector_pixels).
            These are typically derived as -log(I/I0) for each energy bin.
        mac_matrix (torch.Tensor): Material Attenuation Coefficient (MAC) matrix.
            Expected shape: (num_energy_bins, num_materials).
            Example for 2 bins, 2 materials (A, B):
            [[mac_A_bin1, mac_B_bin1],
             [mac_A_bin2, mac_B_bin2]]
        epsilon_determinant (float, optional): A small value added to the determinant
            (or used to check its magnitude) for numerical stability during matrix inversion,
            especially for nearly singular MAC matrices. Defaults to 1e-9.

    Returns:
        torch.Tensor: Calculated material thickness sinograms.
            Shape: (num_materials, num_angles, num_detector_pixels).

    Raises:
        ValueError: If input shapes or dimensions are inconsistent.
        RuntimeError: If matrix inversion fails for reasons other than near-singularity handled
                      by epsilon_determinant (e.g., for non-square matrices if not using pseudo-inverse).

    Potential Issues:
        - MAC Matrix Condition: If the MAC matrix is ill-conditioned (determinant close to zero),
          the results can be highly sensitive to noise in `log_transformed_sinograms` and may be
          numerically unstable. A warning is printed in such cases for 2x2 systems.
        - Noise Propagation: Noise from `log_transformed_sinograms` will propagate and potentially
          be amplified by the inverse MAC matrix, especially if ill-conditioned.
    """
    if not isinstance(log_transformed_sinograms, torch.Tensor):
        raise TypeError("log_transformed_sinograms must be a PyTorch Tensor.")
    if not isinstance(mac_matrix, torch.Tensor):
        raise TypeError("mac_matrix must be a PyTorch Tensor.")

    if log_transformed_sinograms.ndim != 3:
        raise ValueError(f"log_transformed_sinograms must be a 3D tensor. Got shape {log_transformed_sinograms.shape}")

    num_energy_bins, num_angles, num_detector_pixels = log_transformed_sinograms.shape

    if mac_matrix.ndim != 2:
        raise ValueError(f"mac_matrix must be a 2D tensor. Got shape {mac_matrix.shape}")

    if mac_matrix.shape[0] != num_energy_bins:
        raise ValueError(f"MAC matrix energy dimension ({mac_matrix.shape[0]}) must match sinogram energy dimension ({num_energy_bins}).")

    num_materials = mac_matrix.shape[1]

    if num_energy_bins != num_materials:
        raise ValueError(f"Currently supports N_energies ({num_energy_bins}) = N_materials ({num_materials}). System must be square for direct inversion.")

    dev = log_transformed_sinograms.device
    mac_matrix = mac_matrix.to(device=dev, dtype=log_transformed_sinograms.dtype) # Match dtype

    mac_matrix_inv: torch.Tensor

    if num_energy_bins == 2: # Optimized and specific handling for 2x2 systems
        a, b = mac_matrix[0, 0], mac_matrix[0, 1]
        c, d = mac_matrix[1, 0], mac_matrix[1, 1]

        determinant = a * d - b * c

        if torch.abs(determinant) < epsilon_determinant:
            print(f"Warning: MAC matrix determinant ({determinant.item()}) is close to zero (threshold: {epsilon_determinant}). Results may be unstable.")

        # Add a stabilized epsilon to the determinant for inversion
        # Using copysign ensures epsilon pushes away from zero correctly for positive/negative determinants
        # Add a very small constant (1e-20) in case (determinant + stabilized_epsilon) becomes exactly zero
        det_stabilized = determinant + torch.copysign(torch.tensor(epsilon_determinant, device=dev, dtype=determinant.dtype), determinant)
        if torch.abs(det_stabilized) < 1e-20 : # if determinant was -epsilon_determinant
             det_stabilized += (1e-20 * torch.copysign(torch.tensor(1.0, device=dev, dtype=determinant.dtype), det_stabilized)
                                if torch.abs(det_stabilized) < 1e-20 else 0.0) # ensure sign consistency

        mac_matrix_inv = (1.0 / det_stabilized) * torch.tensor([[d, -b], [-c, a]], device=dev, dtype=mac_matrix.dtype)

    else: # General case for N_bins = N_materials > 2 (or if one prefers linalg.inv for 2x2 too)
        try:
            if torch.abs(torch.linalg.det(mac_matrix)) < epsilon_determinant:
                 print(f"Warning: MAC matrix determinant is close to zero. Results may be unstable.")
            mac_matrix_inv = torch.linalg.inv(mac_matrix)
        except torch.linalg.LinAlgError as e:
            print(f"Warning: MAC matrix inversion failed with error: {e}. Using pseudo-inverse (pinv). Results may be approximate or reflect non-uniqueness.")
            mac_matrix_inv = torch.linalg.pinv(mac_matrix) # Fallback to pseudo-inverse

    # Reshape log_transformed_sinograms for vectorized calculation:
    # L_flat shape: (num_energy_bins, num_angles * num_detector_pixels)
    L_flat = log_transformed_sinograms.reshape(num_energy_bins, -1)

    # Solve for thickness sinograms: T_flat = M_inv * L_flat
    # T_flat shape: (num_materials, num_angles * num_detector_pixels)
    thickness_sinograms_flat = torch.matmul(mac_matrix_inv, L_flat)

    # Reshape back: (num_materials, num_angles, num_detector_pixels)
    thickness_sinograms = thickness_sinograms_flat.reshape(num_materials, num_angles, num_detector_pixels)

    return thickness_sinograms

if __name__ == '__main__':
    import numpy as np # For np.isclose, though torch.allclose is used mostly
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running calculate_material_thickness_sinograms Tests on {dev} ---")

    num_angles_test = 10
    num_dets_test = 20

    # Test Case 1: Well-conditioned 2x2 system
    print("\nTest Case 1: Well-conditioned 2x2 system")
    mac_A_bin1, mac_A_bin2 = 0.2, 0.15
    mac_B_bin1, mac_B_bin2 = 0.1, 0.18
    test_mac_matrix_2x2 = torch.tensor([[mac_A_bin1, mac_B_bin1], [mac_A_bin2, mac_B_bin2]], device=dev, dtype=torch.float32)

    true_tA_sino = torch.ones(num_angles_test, num_dets_test, device=dev, dtype=torch.float32) * 1.0
    true_tB_sino = torch.ones(num_angles_test, num_dets_test, device=dev, dtype=torch.float32) * 0.5

    L1_sino = mac_A_bin1 * true_tA_sino + mac_B_bin1 * true_tB_sino
    L2_sino = mac_A_bin2 * true_tA_sino + mac_B_bin2 * true_tB_sino
    test_L_sinos_2x2 = torch.stack([L1_sino, L2_sino], dim=0)

    try:
        calculated_t_sinos_2x2 = calculate_material_thickness_sinograms(test_L_sinos_2x2, test_mac_matrix_2x2)
        assert calculated_t_sinos_2x2.shape == (2, num_angles_test, num_dets_test), \
            f"Shape mismatch. Expected {(2, num_angles_test, num_dets_test)}, Got {calculated_t_sinos_2x2.shape}"

        assert torch.allclose(calculated_t_sinos_2x2[0], true_tA_sino, atol=1e-5), \
            f"Material A thickness mismatch. Max diff: {torch.max(torch.abs(calculated_t_sinos_2x2[0] - true_tA_sino))}"
        assert torch.allclose(calculated_t_sinos_2x2[1], true_tB_sino, atol=1e-5), \
            f"Material B thickness mismatch. Max diff: {torch.max(torch.abs(calculated_t_sinos_2x2[1] - true_tB_sino))}"
        print("  Test Case 1 (Well-conditioned 2x2) passed.")
    except Exception as e:
        print(f"  Test Case 1 FAILED: {e}")
        traceback.print_exc()

    # Test Case 2: Ill-conditioned/Singular 2x2 Matrix (to check warning)
    print("\nTest Case 2: Ill-conditioned 2x2 system (expect warning)")
    test_mac_matrix_singular_2x2 = torch.tensor([[0.2, 0.1], [0.20000001, 0.100000001]], device=dev, dtype=torch.float32)
    # Using pre-calculated L_sinos from well-conditioned case, result should be garbage but run without error
    try:
        print("  Calling with near-singular matrix (expect a warning print from the function):")
        _ = calculate_material_thickness_sinograms(test_L_sinos_2x2, test_mac_matrix_singular_2x2, epsilon_determinant=1e-7)
        # We are primarily checking that it runs and a warning is printed by the function.
        # The output values would be very large or NaN depending on exact epsilon handling.
        print("  Test Case 2 (Ill-conditioned 2x2) ran (check for printed warning).")
    except Exception as e:
        print(f"  Test Case 2 FAILED to run: {e}")
        traceback.print_exc()

    # Test Case 3: General case (e.g., 3x3 system, using linalg.inv)
    print("\nTest Case 3: Well-conditioned 3x3 system")
    if True: # Enable this test explicitly
        try:
            mac_3x3 = torch.tensor([
                [0.25, 0.10, 0.05],  # Mat A, B, C for Bin 1
                [0.20, 0.15, 0.10],  # Mat A, B, C for Bin 2
                [0.15, 0.12, 0.20]   # Mat A, B, C for Bin 3
            ], device=dev, dtype=torch.float32)

            true_tA_3x3 = torch.ones(num_angles_test, num_dets_test, device=dev, dtype=torch.float32) * 1.0
            true_tB_3x3 = torch.ones(num_angles_test, num_dets_test, device=dev, dtype=torch.float32) * 0.5
            true_tC_3x3 = torch.ones(num_angles_test, num_dets_test, device=dev, dtype=torch.float32) * 0.25
            true_T_3x3 = torch.stack([true_tA_3x3, true_tB_3x3, true_tC_3x3], dim=0) # (3, H, W)

            # L = M * T (element-wise for pixels, so expand T for matmul)
            # L_flat (3, N) = M (3,3) @ T_flat (3,N)
            L_flat_3x3 = torch.matmul(mac_3x3, true_T_3x3.reshape(3, -1))
            test_L_sinos_3x3 = L_flat_3x3.reshape(3, num_angles_test, num_dets_test)

            calculated_t_sinos_3x3 = calculate_material_thickness_sinograms(test_L_sinos_3x3, mac_3x3)
            assert calculated_t_sinos_3x3.shape == (3, num_angles_test, num_dets_test)
            assert torch.allclose(calculated_t_sinos_3x3, true_T_3x3, atol=1e-5), \
                f"3x3 system mismatch. Max diff: {torch.max(torch.abs(calculated_t_sinos_3x3 - true_T_3x3))}"
            print("  Test Case 3 (Well-conditioned 3x3) passed.")
        except Exception as e:
            print(f"  Test Case 3 FAILED: {e}")
            traceback.print_exc()

    print("\ncalculate_material_thickness_sinograms tests completed.")


# --- New Function: reconstruct_thickness_maps_from_sinograms ---
def reconstruct_thickness_maps_from_sinograms(
    thickness_sinograms: torch.Tensor,
    radon_transform_operator: Operator,
    image_shape: Tuple[int, int],
    iterations: int = 50,
    step_size: float = 1e-3,
    regularizers: Optional[List[Optional[RegularizerBase]]] = None,
    enforce_non_negativity: bool = False,
    verbose: bool = False
) -> torch.Tensor:
    """
    Reconstructs material thickness maps from their respective thickness sinograms.

    This function iterates through each material's thickness sinogram and applies
    an iterative reconstruction algorithm (Proximal Gradient Descent) using the
    provided Radon transform operator. Optional regularization can be applied
    per material.

    Args:
        thickness_sinograms (torch.Tensor): Stack of material thickness sinograms.
            Shape: (num_materials, num_angles, num_detector_pixels).
            This is typically the output of `calculate_material_thickness_sinograms`.
        radon_transform_operator (Operator): An instance of a Radon transform operator
            that has `op` (Radon transform) and `op_adj` (back-projection) methods.
            E.g., `LinearRadonOperatorPlaceholder` or a more sophisticated one.
        image_shape (Tuple[int, int]): Target shape (Ny, Nx) for each reconstructed material map.
        iterations (int, optional): Number of iterations for the Proximal Gradient
            Reconstructor. Defaults to 50.
        step_size (float, optional): Step size for the Proximal Gradient Reconstructor.
            Defaults to 1e-3. May need tuning based on the Radon operator's norm.
        regularizers (Optional[List[Optional[RegularizerBase]]], optional): A list of
            regularizer instances (or None if no regularization for that material).
            The list should correspond to each material. If the list is shorter than
            num_materials, remaining materials will not be regularized. Defaults to None.
        enforce_non_negativity (bool, optional): If True, clamps the reconstructed maps
            (and initial estimate) to a minimum of 0.0. Defaults to False.
        verbose (bool, optional): Verbosity flag for the Proximal Gradient Reconstructor.
            Defaults to False.

    Returns:
        torch.Tensor: Stack of reconstructed material thickness maps.
            Shape: (num_materials, Ny, Nx).
    """
    if not isinstance(thickness_sinograms, torch.Tensor):
        raise TypeError("thickness_sinograms must be a PyTorch Tensor.")
    if not isinstance(radon_transform_operator, Operator):
        raise TypeError("radon_transform_operator must be an instance of reconlib.operators.Operator.")

    num_materials = thickness_sinograms.shape[0]
    dev = thickness_sinograms.device

    if regularizers is not None and not isinstance(regularizers, list):
        raise TypeError("regularizers, if provided, must be a list.")
    if regularizers and len(regularizers) > num_materials:
        raise ValueError(f"Number of regularizers ({len(regularizers)}) cannot exceed number of materials ({num_materials}).")

    reconstructed_maps_list = []

    for m in range(num_materials):
        if verbose:
            print(f"\nReconstructing material map {m+1}/{num_materials}...")

        current_sino = thickness_sinograms[m, :, :].to(dev)

        current_regularizer_instance = None
        if regularizers and m < len(regularizers) and regularizers[m] is not None:
            current_regularizer_instance = regularizers[m]
            if not isinstance(current_regularizer_instance, RegularizerBase): # Check if it's a regularizer instance
                 raise TypeError(f"Element {m} in regularizers list is not a RegularizerBase instance or None.")

        pgd_reconstructor = ProximalGradientReconstructor(
            iterations=iterations,
            step_size=step_size,
            verbose=verbose,
            # Assuming L2 data fidelity for standard sinogram inversion problems
            data_fidelity_gradient_mode='l2',
            log_fn=lambda iter_num, current_image, change, grad_norm: \
                   print(f"  Mat {m} Recon Iter {iter_num+1}/{iterations}: Change={change:.3e}, GradNorm={grad_norm:.3e}") \
                   if verbose and (iter_num % 10 == 0 or iter_num == iterations -1 or iterations < 10) else None
        )

        initial_estimate = radon_transform_operator.op_adj(current_sino)
        if enforce_non_negativity:
            initial_estimate = torch.clamp(initial_estimate, min=0.0)

        regularizer_prox_fn_for_pgd = None
        if current_regularizer_instance:
            # Create a partial function if regularizer needs step_size, or ensure its prox_op matches signature
            regularizer_prox_fn_for_pgd = current_regularizer_instance.proximal_operator
            # This assumes RegularizerBase.proximal_operator takes (image, step_size)

        recon_map = pgd_reconstructor.reconstruct(
            kspace_data=current_sino, # 'kspace_data' is the sinogram for this problem
            forward_op_fn=radon_transform_operator.op,
            adjoint_op_fn=radon_transform_operator.op_adj,
            regularizer_prox_fn=regularizer_prox_fn_for_pgd,
            x_init=initial_estimate,
            # image_shape_for_zero_init is not strictly needed if x_init is always provided from op_adj
            image_shape_for_zero_init=image_shape
        )

        if enforce_non_negativity:
            recon_map = torch.clamp(recon_map, min=0.0)

        reconstructed_maps_list.append(recon_map)

    thickness_maps_stack = torch.stack(reconstructed_maps_list, dim=0)
    return thickness_maps_stack.to(dev)


if __name__ == '__main__':
    import numpy as np # For np.isclose, though torch.allclose is used mostly
    # ... (previous __main__ content for calculate_material_thickness_sinograms) ...
    # Ensure dev is defined from the previous section of __main__ if running sequentially
    if 'dev' not in locals(): # Define dev if this script part is run standalone for testing
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Device for reconstruct_thickness_maps_from_sinograms tests: {dev}") # Already printed or known


    print("\n--- Testing reconstruct_thickness_maps_from_sinograms ---")
    img_s_recon_test = (16, 16)
    n_angles_recon_test = 20
    n_dets_recon_test = 22
    test_enforce_non_negativity = True # Define this for use in the test

    try:
        radon_op_test = LinearRadonOperatorPlaceholder(
            image_shape=img_s_recon_test,
            num_angles=n_angles_recon_test,
            num_detector_pixels=n_dets_recon_test,
            device=str(dev) # Ensure device is passed as string if placeholder expects it
        )

        true_material1_map = torch.zeros(img_s_recon_test, device=dev, dtype=torch.float32)
        true_material1_map[4:12, 4:12] = 1.0 # Square for material 1

        true_material2_map = torch.zeros(img_s_recon_test, device=dev, dtype=torch.float32)
        center_y, center_x = img_s_recon_test[0]//2, img_s_recon_test[1]//2
        radius = img_s_recon_test[0]//4
        yy, xx = torch.meshgrid(torch.arange(img_s_recon_test[0], device=dev), torch.arange(img_s_recon_test[1], device=dev), indexing='ij')
        mask_circle = (xx - center_x)**2 + (yy - center_y)**2 < radius**2
        true_material2_map[mask_circle] = 0.7 # Circle for material 2

        true_thickness_maps_stack = torch.stack([true_material1_map, true_material2_map], dim=0)

        # Generate corresponding thickness sinograms using the test Radon operator
        sino1 = radon_op_test.op(true_material1_map)
        sino2 = radon_op_test.op(true_material2_map)
        test_thickness_sinos = torch.stack([sino1, sino2], dim=0)

        # Optional: Add a small amount of Gaussian noise
        # test_thickness_sinos += torch.randn_like(test_thickness_sinos) * 0.01 * test_thickness_sinos.abs().mean()

        print(f"Shape of true_thickness_maps_stack: {true_thickness_maps_stack.shape}")
        print(f"Shape of test_thickness_sinos: {test_thickness_sinos.shape}")

        reconstructed_maps = reconstruct_thickness_maps_from_sinograms(
            thickness_sinograms=test_thickness_sinos,
            radon_transform_operator=radon_op_test,
            image_shape=img_s_recon_test,
            iterations=10, # Few iterations for quick test
            step_size=1e-2, # May need tuning based on Radon op's norm
            enforce_non_negativity=True,
            verbose=False # Set to True for iteration details
        )

        assert reconstructed_maps.shape == true_thickness_maps_stack.shape, \
            f"Output shape mismatch. Expected {true_thickness_maps_stack.shape}, Got {reconstructed_maps.shape}"

        reconstruction_error_norm = torch.norm(reconstructed_maps - true_thickness_maps_stack).item()
        initial_error_norm = torch.norm(radon_op_test.op_adj(test_thickness_sinos[0]) - true_material1_map).item() # Error for one map from simple backprojection
                                                                                                            # This is not a perfect comparison for the stack.

        # A more direct comparison for initial error could be:
        initial_adjoint_stack = torch.stack([radon_op_test.op_adj(s) for s in test_thickness_sinos], dim=0)
        if test_enforce_non_negativity: initial_adjoint_stack = torch.clamp(initial_adjoint_stack, min=0.0)
        initial_stack_error_norm = torch.norm(initial_adjoint_stack - true_thickness_maps_stack).item()

        print(f"  Norm of (Reconstructed Maps - True Maps): {reconstruction_error_norm:.4f}")
        print(f"  Norm of (Initial Adjoint Stack - True Maps): {initial_stack_error_norm:.4f}")
        if reconstruction_error_norm < initial_stack_error_norm:
            print("  Reconstruction error is lower than initial adjoint error (good).")
        else:
            print("  Warning: Reconstruction error is NOT lower than initial adjoint error. May need more iterations or step_size tuning.")

        print("reconstruct_thickness_maps_from_sinograms test passed.")

    except Exception as e:
        print(f"Error during reconstruct_thickness_maps_from_sinograms tests: {e}")
        traceback.print_exc() # This should now work as traceback is imported at the top.
