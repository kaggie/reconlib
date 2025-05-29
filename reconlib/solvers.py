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

def fista_reconstruction(
    kspace_data: torch.Tensor,
    sampling_points: torch.Tensor,
    image_shape: tuple,
    nufft_operator_class, # e.g., reconlib.nufft.NUFFT2D or NUFFT3D
    nufft_kwargs: dict,
    regularizer, # An object with a .proximal_operator(data, step_size) method
    lambda_reg: float, # Regularization strength
    use_voronoi: bool = False,
    voronoi_weights: torch.Tensor = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    line_search_params: dict = None, # e.g., {'beta': 0.5, 'max_ls_iter': 20, 'initial_L': 1.0}
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs iterative image reconstruction using the Fast Iterative Shrinkage-Thresholding
    Algorithm (FISTA).

    Solves problems of the form: min_x 0.5 * ||A x - y||_2^2 + lambda_reg * g(x),
    where y is k-space data, A is the NUFFT operator (which internally handles density compensation
    if weights are provided), and g(x) is a regularizer with a known proximal operator.

    Args:
        kspace_data (torch.Tensor): Complex-valued k-space measurements. Shape (N,).
        sampling_points (torch.Tensor): Coordinates of k-space samples. Shape (N, d).
        image_shape (tuple): Target image shape.
        nufft_operator_class: The NUFFT operator class.
        nufft_kwargs (dict): Keyword arguments for initializing NUFFT.
        regularizer: Regularizer object with a `proximal_operator(data, step_size)` method.
        lambda_reg (float): Regularization strength.
        use_voronoi (bool, optional): If True, passes `voronoi_weights` to NUFFT.
        voronoi_weights (torch.Tensor, optional): Precomputed Voronoi weights for NUFFT.
        max_iters (int, optional): Maximum FISTA iterations. Defaults to 100.
        tol (float, optional): Tolerance for relative change in solution to stop. Defaults to 1e-6.
        line_search_params (dict, optional): Parameters for backtracking line search for Lipschitz constant.
                                             If None, a fixed step size derived from an initial L (initial_L)
                                             is used, and L might be increased if line search condition fails.
                                             Expected keys: 'beta' (L increase factor, e.g. 2.0),
                                                            'max_ls_iter' (max line search steps per FISTA iter),
                                                            'initial_L' (initial estimate for Lipschitz constant).
                                             A simple fixed step size can be used if line_search_params is None,
                                             but backtracking is more robust.
        verbose (bool, optional): If True, prints progress information. Defaults to False.


    Returns:
        torch.Tensor: Reconstructed image of shape `image_shape`.
    """
    device = kspace_data.device
    dtype_complex = kspace_data.dtype
    dtype_real = sampling_points.dtype 

    density_compensation_for_nufft = None
    if use_voronoi:
        if voronoi_weights is None:
            raise ValueError("If use_voronoi is True, voronoi_weights must be provided.")
        if voronoi_weights.shape[0] != kspace_data.shape[0]:
            raise ValueError(
                f"Shape mismatch: voronoi_weights ({voronoi_weights.shape[0]}) "
                f"and kspace_data ({kspace_data.shape[0]}) must have the same length."
            )
        if voronoi_weights.device != device:
            voronoi_weights = voronoi_weights.to(device)
        
        # Voronoi weights are passed to NUFFT, which applies them in its adjoint.
        # Unlike the iterative_reconstruction example's sqrt(w) for gradient descent,
        # here the NUFFT op itself is expected to handle density compensation if weights are given.
        density_compensation_for_nufft = torch.clamp(voronoi_weights.to(dtype_real), min=1e-9)

    # Initialize NUFFT operator
    current_nufft_kwargs = nufft_kwargs.copy()
    if density_compensation_for_nufft is not None:
        current_nufft_kwargs['density_comp_weights'] = density_compensation_for_nufft
    
    nufft_op = nufft_operator_class(
        image_shape=image_shape,
        k_trajectory=sampling_points,
        device=device,
        **current_nufft_kwargs
    )

    A_op = lambda x: nufft_op.forward(x)
    At_op = lambda y: nufft_op.adjoint(y) # At_op incorporates DCF if weights passed to NUFFT

    # FISTA Initialization
    # x_old = torch.zeros(image_shape, dtype=dtype_complex, device=device)
    # Using At_op(kspace_data) as initial guess can sometimes be better
    x_old = At_op(kspace_data) # Initial guess
    y_k = x_old.clone()
    t_old = 1.0
    
    # Line search parameters
    if line_search_params is None:
        line_search_params = {} # Use defaults if None
        
    L_k = line_search_params.get('initial_L', 1.0)
    beta_ls = line_search_params.get('beta', 2.0) # Factor to increase L_k
    max_ls_iters = line_search_params.get('max_ls_iter', 20)
    
    if verbose:
        print(f"Starting FISTA: max_iters={max_iters}, tol={tol:.1e}, lambda_reg={lambda_reg:.1e}")
        print(f"Line search params: initial_L={L_k:.1e}, beta={beta_ls}, max_ls_iters={max_ls_iters}")

    for iter_num in range(max_iters):
        grad_y_k = At_op(A_op(y_k) - kspace_data) # Gradient of data fidelity at y_k

        # Backtracking Line Search for L_k
        current_L_k_try = L_k # Start with L from previous FISTA iteration or initial_L
        
        for ls_iter in range(max_ls_iters):
            step_size = 1.0 / current_L_k_try
            # Proximal update: x_new = prox_g(y_k - (1/L_k) * grad_f(y_k), lambda_reg/L_k)
            x_new = regularizer.proximal_operator(y_k - step_size * grad_y_k, lambda_reg * step_size)
            
            # Check line search condition: F(x_new) <= Q_L(x_new, y_k)
            # F(x_new) = 0.5 * ||A(x_new) - y||^2
            Ax_new_minus_y = A_op(x_new) - kspace_data
            F_x_new = 0.5 * torch.linalg.norm(Ax_new_minus_y.flatten())**2
            
            # Q_L(x_new, y_k) = F(y_k) + <x_new - y_k, grad_f(y_k)> + (L/2) * ||x_new - y_k||^2
            # F(y_k) = 0.5 * ||A(y_k) - y||^2
            Ay_k_minus_y = A_op(y_k) - kspace_data
            F_y_k = 0.5 * torch.linalg.norm(Ay_k_minus_y.flatten())**2
            
            x_new_minus_y_k = x_new - y_k
            term_inner_prod = torch.vdot(x_new_minus_y_k.flatten(), grad_y_k.flatten()).real
            term_quadratic = (current_L_k_try / 2.0) * torch.linalg.norm(x_new_minus_y_k.flatten())**2
            
            Q_L_x_new_y_k = F_y_k + term_inner_prod + term_quadratic

            if F_x_new.real <= Q_L_x_new_y_k.real: # Condition met
                L_k = current_L_k_try # Accept this L_k for the current FISTA iteration
                break
            
            current_L_k_try = current_L_k_try * beta_ls # Increase L and retry
        else: # Line search loop finished without break (max_ls_iters reached)
            L_k = current_L_k_try # Use the last tried L_k
            if verbose:
                print(f"Warning: Line search reached max_ls_iters ({max_ls_iters}) at FISTA iter {iter_num+1}. Current L_k: {L_k:.2e}")
        
        # FISTA updates
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
        # Ensure x_new is used from the successful line search step (or last attempt)
        # x_new was already computed with the L_k that satisfied the condition (or the last L_k tried)
        
        y_k_next = x_new + ((t_old - 1.0) / t_new) * (x_new - x_old)
        
        # Check convergence: relative change in x
        delta_x_norm = torch.linalg.norm((x_new - x_old).flatten())
        x_old_norm = torch.linalg.norm(x_old.flatten())
        relative_change = delta_x_norm / (x_old_norm + 1e-9) # Add epsilon for stability if x_old is zero

        if verbose:
            cost_data = F_x_new.item() # F(x_new) from line search
            # For total cost, we need regularizer value: reg_val = lambda_reg * g(x_new)
            # This requires regularizer to have a .value(x) method, which is not specified in the interface.
            # So, we'll just print data cost and relative change.
            print(f"FISTA Iter {iter_num+1}/{max_iters}: L_k={L_k:.2e}, Step={1.0/L_k:.2e}, DataCost={cost_data:.2e}, RelChange={relative_change:.2e}")

        if relative_change < tol and iter_num > 0: # iter_num > 0 to ensure at least one step if x_old is zero initially
            if verbose:
                print(f"FISTA converged at iteration {iter_num+1} with relative change {relative_change:.2e} < tol {tol:.1e}.")
            x_old = x_new.clone() # Important to update x_old to the converged solution
            break
            
        # Update for next iteration
        x_old = x_new.clone()
        y_k = y_k_next.clone()
        t_old = t_new
        
    else: # Executed if loop completes without break
        if max_iters > 0 and verbose:
            print(f"FISTA reached max_iters ({max_iters}). Last relative change: {relative_change:.2e}.")

    return x_old # x_old holds the latest x_new

def _conjugate_gradient_for_admm_x_update(
    system_op, # Callable that applies (A^H A + rho I)
    rhs_b: torch.Tensor,
    x_init: torch.Tensor, # Initial guess for x
    max_iters: int,
    tol: float,
    verbose: bool = False # Optional verbosity for CG sub-problem
) -> torch.Tensor:
    """
    Solves (A^H A + rho I) x = b using Conjugate Gradient.
    This is a helper for the x-update step in ADMM.
    """
    x = x_init.clone()
    r = rhs_b - system_op(x)
    p = r.clone()
    rs_old = torch.vdot(r.flatten(), r.flatten()).real
    
    cg_tol_denom = 1e-12 # Small epsilon for denominators

    for i in range(max_iters):
        Ap = system_op(p)
        alpha_num = rs_old
        alpha_den = torch.vdot(p.flatten(), Ap.flatten()).real
        alpha = alpha_num / (alpha_den + cg_tol_denom)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = torch.vdot(r.flatten(), r.flatten()).real
        
        current_residual_norm_cg = torch.sqrt(rs_new)
        if current_residual_norm_cg < tol:
            if verbose and i > 0 : # Avoid print if initial guess is already good
                 print(f"    CG for x-update converged at iter {i+1}, Res: {current_residual_norm_cg:.1e}")
            break
            
        beta_num = rs_new
        beta_den = rs_old
        beta = beta_num / (beta_den + cg_tol_denom)
        
        p = r + beta * p
        rs_old = rs_new
    else: # If loop completes without break
        if max_iters > 0 and verbose:
            current_residual_norm_cg = torch.sqrt(rs_old)
            print(f"    CG for x-update reached max_iters ({max_iters}). Last Res: {current_residual_norm_cg:.1e}")
            
    return x

def admm_reconstruction(
    kspace_data: torch.Tensor,
    sampling_points: torch.Tensor,
    image_shape: tuple,
    nufft_operator_class, # e.g., reconlib.nufft.NUFFT2D or NUFFT3D
    nufft_kwargs: dict,
    regularizer, # Regularizer object with a .proximal_operator(data, step_size) method for the z-update
    lambda_reg: float, # Regularization strength (used as 'step_size' for prox: lambda_reg/rho)
    use_voronoi: bool = False,
    voronoi_weights: torch.Tensor = None,
    rho: float = 1.0, # ADMM penalty parameter
    max_iters: int = 50,
    tol_abs: float = 1e-4, # Absolute tolerance for primal and dual residuals
    tol_rel: float = 1e-3, # Relative tolerance for primal and dual residuals
    cg_max_iters_x_update: int = 10, # Max CG iterations for the x-update subproblem
    cg_tol_x_update: float = 1e-5,   # Tolerance for CG in x-update
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs iterative image reconstruction using the Alternating Direction Method of
    Multipliers (ADMM).

    Solves problems of the form: min_x 0.5 * ||A x - y||_2^2 + g(x)
    by splitting into: min_x,z 0.5 * ||A x - y||_2^2 + g(z) subject to x - z = 0.
    (Here, g(z) is handled by regularizer.proximal_operator with strength lambda_reg)

    Args:
        kspace_data (torch.Tensor): Complex-valued k-space measurements.
        sampling_points (torch.Tensor): Coordinates of k-space samples.
        image_shape (tuple): Target image shape.
        nufft_operator_class: The NUFFT operator class.
        nufft_kwargs (dict): Keyword arguments for initializing NUFFT.
        regularizer: Regularizer object with a `proximal_operator(data, step_size)` method.
        lambda_reg (float): Regularization strength. The effective step for prox is lambda_reg/rho.
        use_voronoi (bool, optional): If True, passes `voronoi_weights` to NUFFT.
        voronoi_weights (torch.Tensor, optional): Precomputed Voronoi weights for NUFFT.
        rho (float, optional): ADMM penalty parameter. Defaults to 1.0.
        max_iters (int, optional): Maximum ADMM iterations. Defaults to 50.
        tol_abs (float, optional): Absolute tolerance for convergence. Defaults to 1e-4.
        tol_rel (float, optional): Relative tolerance for convergence. Defaults to 1e-3.
        cg_max_iters_x_update (int, optional): Max CG iterations for x-update. Defaults to 10.
        cg_tol_x_update (float, optional): CG tolerance for x-update. Defaults to 1e-5.
        verbose (bool, optional): If True, prints progress information. Defaults to False.

    Returns:
        torch.Tensor: Reconstructed image of shape `image_shape`.
    """
    device = kspace_data.device
    dtype_complex = kspace_data.dtype
    # dtype_real = sampling_points.dtype # Not directly used for ADMM vars, but for weights

    density_compensation_for_nufft = None
    if use_voronoi:
        if voronoi_weights is None:
            raise ValueError("If use_voronoi is True, voronoi_weights must be provided.")
        if voronoi_weights.shape[0] != kspace_data.shape[0]:
            raise ValueError(
                f"Shape mismatch: voronoi_weights ({voronoi_weights.shape[0]}) "
                f"and kspace_data ({kspace_data.shape[0]}) must have the same length."
            )
        if voronoi_weights.device != device:
            voronoi_weights = voronoi_weights.to(device)
        density_compensation_for_nufft = torch.clamp(voronoi_weights.to(torch.float32), min=1e-9)

    # Initialize NUFFT operator
    current_nufft_kwargs = nufft_kwargs.copy()
    if density_compensation_for_nufft is not None:
        current_nufft_kwargs['density_comp_weights'] = density_compensation_for_nufft
    
    nufft_op = nufft_operator_class(
        image_shape=image_shape,
        k_trajectory=sampling_points,
        device=device,
        **current_nufft_kwargs
    )

    A_op = lambda x: nufft_op.forward(x)
    At_op = lambda y: nufft_op.adjoint(y) # Incorporates DCF if weights passed to NUFFT

    # ADMM Initialization
    x_k = At_op(kspace_data) # Initial guess for x
    z_k = x_k.clone()
    u_k = torch.zeros_like(x_k, dtype=dtype_complex) # Scaled dual variable

    At_y = At_op(kspace_data) # Precompute A^H y (or A^H W y if DCF is on)

    n_elements_x = x_k.numel() # Number of elements in x (for tolerance calculation)

    if verbose:
        print(f"Starting ADMM: max_iters={max_iters}, rho={rho:.1e}, lambda_reg={lambda_reg:.1e}")
        print(f"tol_abs={tol_abs:.1e}, tol_rel={tol_rel:.1e}")
        print(f"CG for x-update: max_iters={cg_max_iters_x_update}, tol={cg_tol_x_update:.1e}")

    for iter_num in range(max_iters):
        # x-update: Solve (A^H A + rho I) x = A^H y + rho (z_k - u_k) using CG
        rhs_x_update = At_y + rho * (z_k - u_k)
        Op_x_update = lambda v: At_op(A_op(v)) + rho * v
        
        x_k_plus_1 = _conjugate_gradient_for_admm_x_update(
            Op_x_update, rhs_x_update, x_k, # x_k as initial guess for CG
            cg_max_iters_x_update, cg_tol_x_update,
            verbose=verbose # Pass ADMM's verbose to CG for sub-problem verbosity
        )

        # z-update: z_k+1 = prox_g(x_k+1 + u_k, lambda_reg / rho)
        z_k_plus_1 = regularizer.proximal_operator(x_k_plus_1 + u_k, lambda_reg / rho)

        # u-update: u_k+1 = u_k + (x_k+1 - z_k+1)
        u_k_plus_1 = u_k + (x_k_plus_1 - z_k_plus_1)

        # Convergence checks
        r_primal_k_plus_1 = x_k_plus_1 - z_k_plus_1
        s_dual_k_plus_1 = rho * (z_k_plus_1 - z_k) # Diff in z is used for dual residual

        norm_r_primal = torch.linalg.norm(r_primal_k_plus_1.flatten())
        norm_s_dual = torch.linalg.norm(s_dual_k_plus_1.flatten())

        eps_pri = torch.sqrt(torch.tensor(n_elements_x, dtype=torch.float32, device=device)) * tol_abs + \
                  tol_rel * max(torch.linalg.norm(x_k_plus_1.flatten()), torch.linalg.norm(-z_k_plus_1.flatten()))
        
        eps_dual = torch.sqrt(torch.tensor(n_elements_x, dtype=torch.float32, device=device)) * tol_abs + \
                   tol_rel * torch.linalg.norm(rho * u_k_plus_1.flatten()) # Boyd uses u_k here, u_k_plus_1 is current

        if verbose:
            print(f"ADMM Iter {iter_num+1}/{max_iters}: "
                  f"Primal Res Norm: {norm_r_primal:.2e} (eps_pri: {eps_pri:.2e}), "
                  f"Dual Res Norm: {norm_s_dual:.2e} (eps_dual: {eps_dual:.2e})")

        if norm_r_primal < eps_pri and norm_s_dual < eps_dual:
            if verbose:
                print(f"ADMM converged at iteration {iter_num+1}.")
            break
            
        # Update variables for next iteration
        x_k = x_k_plus_1.clone()
        z_k = z_k_plus_1.clone()
        u_k = u_k_plus_1.clone()
    else: # Executed if loop completes without break
        if max_iters > 0 and verbose:
            print(f"ADMM reached max_iters ({max_iters}).")

    return x_k # Return the last x_k (which is x_k_plus_1 from the final iteration)

def conjugate_gradient_reconstruction(
    kspace_data: torch.Tensor,
    sampling_points: torch.Tensor,
    image_shape: tuple,
    nufft_operator_class, # e.g., reconlib.nufft.NUFFT2D or NUFFT3D
    nufft_kwargs: dict,
    use_voronoi: bool = False,
    voronoi_weights: torch.Tensor = None,
    max_iters: int = 10,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Performs iterative image reconstruction using the Conjugate Gradient (CG) method.

    Solves the normal equations A^H A x = A^H y, where y is the k-space data
    (potentially pre-weighted if use_voronoi is True) and A is the NUFFT operator.

    Args:
        kspace_data (torch.Tensor): Complex-valued k-space measurements. Shape (N,).
        sampling_points (torch.Tensor): Coordinates of k-space samples. Shape (N, d).
        image_shape (tuple): Target image shape (e.g., (H, W) or (D, H, W)).
        nufft_operator_class: The NUFFT operator class to instantiate.
        nufft_kwargs (dict): Keyword arguments for initializing the NUFFT operator.
                             This should include parameters like `oversamp_factor`, `kb_J`, etc.,
                             but NOT `k_trajectory` or `image_shape`. `device` will be inferred.
        use_voronoi (bool, optional): If True, applies `voronoi_weights` to `kspace_data`
                                      before forming A^H y. Defaults to False.
        voronoi_weights (torch.Tensor, optional): Precomputed Voronoi weights. Shape (N,).
                                                  Required if `use_voronoi` is True. Defaults to None.
        max_iters (int, optional): Maximum number of CG iterations. Defaults to 10.
        tol (float, optional): Tolerance for the L2 norm of the residual to stop iterations.
                               Defaults to 1e-6.

    Returns:
        torch.Tensor: Reconstructed image of shape `image_shape`.

    Raises:
        ValueError: If `use_voronoi` is True but `voronoi_weights` are not provided or have incorrect shape.
    """
    device = kspace_data.device
    dtype_complex = kspace_data.dtype
    dtype_real = sampling_points.dtype # Assuming sampling_points is real

    if use_voronoi:
        if voronoi_weights is None:
            raise ValueError("If use_voronoi is True, voronoi_weights must be provided.")
        if voronoi_weights.shape[0] != kspace_data.shape[0]:
            raise ValueError(
                f"Shape mismatch: voronoi_weights ({voronoi_weights.shape[0]}) "
                f"and kspace_data ({kspace_data.shape[0]}) must have the same length."
            )
        if voronoi_weights.device != device:
            voronoi_weights = voronoi_weights.to(device)
        
        # Apply sqrt of weights to k-space data for A^H W y formulation,
        # where W = diag(voronoi_weights). The normal equation becomes A^H W A x = A^H W y.
        # If NUFFT handles sqrt(weights) internally, then kspace_data_weighted = kspace_data.
        # The previous `iterative_reconstruction` example did k_s_w = k_s * sqrt(w) for grad = At(A(x)) - At(k_s_w).
        # For CG A^H A x = A^H y, if y is k_s_w, then rhs_b = At(k_s_w).
        # And AtA_op needs to be effectively At W A.
        # The current NUFFT classes apply density_comp_weights (which are w, not sqrt(w)) in adjoint.
        # So, if NUFFT's adjoint does At(diag(w) * k_data), then AtA is At(diag(w) * A(x)).
        # The prompt for iterative_reconstruction used:
        #   kspace_data_weighted = kspace_data * torch.sqrt(safe_voronoi_weights)
        #   density_compensation_for_nufft = voronoi_weights
        # This implies At uses `density_compensation_for_nufft` (i.e., w), and `kspace_data_weighted` is sqrt(w)*y.
        # So rhs_b = At(sqrt(w)*y). The AtA_op = At(A(x)) from nufft_op.
        # If nufft_op.adjoint internally uses `density_comp_weights` (w), then At(y) becomes At(w*y).
        # Let's ensure consistency with how NUFFT was modified.
        # NUFFT adjoints now apply self.density_comp_weights (which are w) if provided.
        # So, A_op(x) = A(x)
        # At_op(y) = At(w*y) if w is provided to NUFFT, else At(y_dcf_internal_or_none)
        # If we want to solve A^H A x = A^H y (no explicit W in problem form),
        # but NUFFT uses DCF internally, then A_effective = A and At_effective = At_dcf.
        # So AtA_op(x) = At_dcf(A(x)).
        # And rhs_b = At_dcf(kspace_data).
        # The `use_voronoi` flag here is more about whether to pass `voronoi_weights` to NUFFT.
        
        safe_voronoi_weights = torch.clamp(voronoi_weights.to(dtype_real), min=1e-9)
        # kspace_data_for_rhs = kspace_data # NUFFT adjoint will apply the weights
        density_compensation_for_nufft = safe_voronoi_weights
    else:
        # kspace_data_for_rhs = kspace_data
        density_compensation_for_nufft = None

    # Initialize NUFFT operator
    current_nufft_kwargs = nufft_kwargs.copy()
    if density_compensation_for_nufft is not None:
        current_nufft_kwargs['density_comp_weights'] = density_compensation_for_nufft
    
    nufft_op = nufft_operator_class(
        image_shape=image_shape,
        k_trajectory=sampling_points,
        device=device,
        **current_nufft_kwargs
    )

    A_op = lambda x: nufft_op.forward(x)
    At_op = lambda y: nufft_op.adjoint(y) # This At_op now incorporates DCF if density_comp_weights were passed to NUFFT.
    
    # System operator for normal equations: A^H A (or effectively A^H W A if NUFFT's At applies W)
    AtA_op = lambda x: At_op(A_op(x))

    x_recon = torch.zeros(image_shape, dtype=dtype_complex, device=device)
    
    # Right-hand side: A^H y (or A^H W y if At_op applies W)
    rhs_b = At_op(kspace_data) # Pass original kspace_data, At_op handles DCF.

    r = rhs_b - AtA_op(x_recon) # Initial residual: b - A^H A x_0
    p = r.clone()
    rs_old = torch.vdot(r.flatten(), r.flatten()).real
    
    cg_tol_denom = 1e-12 # Small epsilon for denominators

    for i in range(max_iters):
        Ap = AtA_op(p)
        
        alpha_num = rs_old
        alpha_den = torch.vdot(p.flatten(), Ap.flatten()).real
        alpha = alpha_num / (alpha_den + cg_tol_denom)
        
        x_recon = x_recon + alpha * p
        r = r - alpha * Ap
        
        rs_new = torch.vdot(r.flatten(), r.flatten()).real
        
        current_residual_norm = torch.sqrt(rs_new)
        if current_residual_norm < tol:
            print(f"CG converged at iteration {i+1} with residual norm {current_residual_norm:.2e} < tol {tol:.2e}.")
            break
        
        beta_num = rs_new
        beta_den = rs_old
        beta = beta_num / (beta_den + cg_tol_denom)
        
        p = r + beta * p
        rs_old = rs_new
        
        if (i + 1) % 10 == 0: # Print progress every 10 iterations
            print(f"CG Iteration {i+1}/{max_iters}, Residual Norm: {current_residual_norm:.2e}")
    else: # Executed if loop completes without break
        if max_iters > 0:
            current_residual_norm = torch.sqrt(rs_old) # or rs_new, should be same
            print(f"CG reached max_iters ({max_iters}). Last residual norm: {current_residual_norm:.2e}.")

    return x_recon
