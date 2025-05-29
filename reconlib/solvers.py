import torch
# Assuming NUFFT and other necessary components will be imported from reconlib
# For example: from .nufft import NUFFT (if NUFFT class is defined in nufft.py)
# from .voronoi_utils import compute_voronoi_density_weights (if needed directly, though the example uses it externally)
# from .utils import calculate_density_compensation (example shows default_dcf)

def iterative_reconstruction(
    kspace_data: torch.Tensor,       # Complex values at nonuniform points
    sampling_points: torch.Tensor,   # Nonuniform coordinates (N x d)
    image_shape: tuple,              # Output image shape (e.g. (128, 128))
    nufft_operator_class,            # The actual NUFFT class (e.g., reconlib.nufft.NUFFT3D)
    nufft_kwargs: dict,              # Dictionary of kwargs for NUFFT operator init (e.g., oversamp_factor, kb_J etc.)
    use_voronoi: bool = False,       # Whether to use Voronoi-based weighting
    voronoi_weights: torch.Tensor = None, # Precomputed Voronoi weights (N,)
    max_iters: int = 10,             # For iterative solver
    tol: float = 1e-6,               # Tolerance for gradient norm
    alpha_update_type: str = 'fixed', # 'fixed' or 'adaptive' (placeholder for future)
    fixed_alpha: float = 0.01        # Step size if alpha_update_type is 'fixed'
) -> torch.Tensor:
    """
    Performs iterative image reconstruction using gradient descent.

    Args:
        kspace_data (torch.Tensor): Complex-valued k-space measurements at sampling_points. Shape (N,).
        sampling_points (torch.Tensor): Coordinates of k-space samples. Shape (N, d).
        image_shape (tuple): Target image shape (e.g., (H, W) or (D, H, W)).
        nufft_operator_class: The NUFFT operator class to instantiate (e.g., from reconlib.nufft).
        nufft_kwargs (dict): Keyword arguments for initializing the NUFFT operator.
                             This should include parameters like `oversamp_factor`, `kb_J`, etc.,
                             but NOT `k_trajectory` (sampling_points) or `image_shape` which are
                             passed directly. `device` will be inferred from `kspace_data`.
        use_voronoi (bool, optional): If True, applies `voronoi_weights` to `kspace_data`
                                      and passes them to NUFFT if `density_comp` is part of `nufft_kwargs`.
                                      Defaults to False.
        voronoi_weights (torch.Tensor, optional): Precomputed Voronoi weights. Required if `use_voronoi` is True.
                                                  Shape (N,). Defaults to None.
        max_iters (int, optional): Maximum number of iterations. Defaults to 10.
        tol (float, optional): Tolerance for the L2 norm of the gradient to stop iterations. Defaults to 1e-6.
        alpha_update_type (str, optional): Method for updating step size 'alpha'.
                                           Currently, only 'fixed' is implemented. Defaults to 'fixed'.
        fixed_alpha (float, optional): The fixed step size if `alpha_update_type` is 'fixed'. Defaults to 0.01.


    Returns:
        torch.Tensor: Reconstructed image of shape `image_shape`.

    Raises:
        ValueError: If `use_voronoi` is True but `voronoi_weights` are not provided or have incorrect shape.
    """
    device = kspace_data.device
    dtype_complex = kspace_data.dtype # Should be complex
    dtype_real = sampling_points.dtype # Should be real (e.g., float32)

    if use_voronoi:
        if voronoi_weights is None:
            raise ValueError("If use_voronoi is True, voronoi_weights must be provided.")
        if voronoi_weights.shape[0] != kspace_data.shape[0]:
            raise ValueError(f"Shape mismatch: voronoi_weights ({voronoi_weights.shape[0]}) "
                             f"and kspace_data ({kspace_data.shape[0]}) must have the same length.")
        if voronoi_weights.device != device:
            voronoi_weights = voronoi_weights.to(device)
        
        # As per example: kspace_data *= sqrt(voronoi_weights)
        # Ensure voronoi_weights are positive before sqrt
        # The example `compute_voronoi_weights` implies weights are 1/area, so should be positive.
        # Clamping here for safety, though ideal weights should already be positive.
        safe_voronoi_weights = torch.clamp(voronoi_weights.to(dtype_real), min=1e-9) # Use real dtype for weights
        kspace_data_weighted = kspace_data * torch.sqrt(safe_voronoi_weights)
        
        # For passing to NUFFT operator, if it accepts density_comp
        # This assumes nufft_kwargs might contain a key like 'density_comp_weights' for the NUFFT operator
        # Or, the NUFFT operator is modified to accept 'density_comp_weights' directly.
        # The issue's example NUFFT init had `density_comp=voronoi_weights if use_voronoi else default_dcf(...)`
        # This will be handled by the NUFFT modification step. For now, we prepare it.
        density_compensation_for_nufft = voronoi_weights.to(dtype_real)
    else:
        kspace_data_weighted = kspace_data
        # density_compensation_for_nufft = default_dcf(sampling_points) # This would need default_dcf to be defined/imported
        # For now, let NUFFT handle its default if None is passed or 'density_comp_weights' is not in nufft_kwargs
        density_compensation_for_nufft = None


    # Initialize NUFFT operator
    # The actual NUFFT class should handle its own parameters like oversamp_factor, kb_J, etc., via nufft_kwargs
    # It should also accept k_trajectory (sampling_points), image_shape, and density_comp_weights.
    # We will modify NUFFT class to accept 'density_comp_weights' in the next step.
    # For now, we prepare nufft_kwargs to potentially include it.
    current_nufft_kwargs = nufft_kwargs.copy()
    if density_compensation_for_nufft is not None:
        # This key 'density_comp_weights' is what the NUFFT class will be modified to look for.
        current_nufft_kwargs['density_comp_weights'] = density_compensation_for_nufft
    
    nufft_op = nufft_operator_class(
        image_shape=image_shape,
        k_trajectory=sampling_points,
        device=device, # Pass device explicitly
        **current_nufft_kwargs # Pass other NUFFT params like J, os_factor, etc.
    )

    # Define forward and adjoint operations
    A = lambda x: nufft_op.forward(x)   # Image -> k-space
    At = lambda y: nufft_op.adjoint(y) # K-space -> image

    # Iterative reconstruction (gradient descent)
    x_recon = torch.zeros(image_shape, dtype=dtype_complex, device=device) # Initial guess

    if alpha_update_type != 'fixed':
        print(f"Warning: alpha_update_type '{alpha_update_type}' is not fully implemented. Using fixed_alpha={fixed_alpha}.")
        # Placeholder for adaptive alpha strategies (e.g., line search)

    for i in range(max_iters):
        # Gradient: At(A(x)) - At(y_weighted)
        # y_weighted is kspace_data_weighted
        grad = At(A(x_recon)) - At(kspace_data_weighted)
        
        # Update step
        # TODO: Implement adaptive alpha if needed
        alpha = fixed_alpha
        x_recon = x_recon - alpha * grad

        # Check convergence
        grad_norm = torch.linalg.norm(grad.flatten())
        if grad_norm < tol:
            print(f"Converged at iteration {i+1} with gradient norm {grad_norm:.2e} < tol {tol:.2e}.")
            break
        
        if (i + 1) % 10 == 0: # Print progress every 10 iterations
             print(f"Iteration {i+1}/{max_iters}, Gradient Norm: {grad_norm:.2e}")
    else:
        if max_iters > 0 :
            print(f"Reached max_iters ({max_iters}) without converging. Last gradient norm: {grad_norm:.2e}.")

    return x_recon
