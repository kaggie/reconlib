import torch
import torch.optim as optim
import math # For math.pi

def spgr_signal(T1: torch.Tensor, M0: torch.Tensor, FA_rad: torch.Tensor, TR: float) -> torch.Tensor:
    """
    Calculates the Spoiled Gradient Recalled Echo (SPGR) signal.

    Args:
        T1: Longitudinal relaxation time (ms). Tensor of shape (...spatial_dims).
        M0: Proton density or equilibrium magnetization. Tensor of shape (...spatial_dims).
        FA_rad: Flip angle in radians. Can be a scalar or a tensor of shape (...spatial_dims)
                if spatially varying (e.g., due to B1 map).
        TR: Repetition time (ms). Float.

    Returns:
        Signal intensity. Tensor of shape (...spatial_dims).
    """
    if TR <= 0:
        raise ValueError("TR must be positive.")
    # Clamp T1 to a small positive value to avoid division by zero or log(0) if T1 is an output of a fit
    # and could become zero or negative during optimization steps.
    T1_eff = torch.clamp(T1, min=1e-6) # Clamp T1 to be positive
    E1 = torch.exp(-TR / T1_eff)

    # Ensure FA_rad is processed correctly whether it's a scalar or tensor for sin/cos
    sin_FA = torch.sin(FA_rad)
    cos_FA = torch.cos(FA_rad)

    numerator = M0 * sin_FA * (1 - E1)
    denominator = (1 - cos_FA * E1)

    # Avoid division by zero if denominator is very small.
    # This can happen if cos(FA)*E1 is close to 1. (e.g. FA near 0 and E1 near 1 (long T1, short TR))
    # Or FA near pi and E1 near -1 (not physically likely for E1).
    # A small epsilon helps stabilize, or one might return 0 if M0 is also 0.
    signal = numerator / (denominator + 1e-9) # Add epsilon to denominator
    return signal

def fit_t1_vfa(
    signals: torch.Tensor,
    flip_angles_deg: torch.Tensor,
    TR: float,
    b1_map: torch.Tensor = None,
    initial_T1_ms_guess: float = 800.0,
    initial_M0_guess: float = -1.0, # Signal-derived if -1
    num_iterations: int = 100,
    learning_rate: float = 1e-1,
    min_T1_ms: float = 10.0,
    max_T1_ms: float = 5000.0,
    min_M0_scale: float = 0.01,
    max_M0_scale: float = 10.0,
    optimizer_type: str = 'adam',
    verbose: bool = False,
    device: str | torch.device = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fits T1 and M0 maps from Variable Flip Angle (VFA) SPGR signals
    using PyTorch-based iterative non-linear least squares.

    Args:
        signals: Measured signal intensities. Shape (num_flip_angles, ...spatial_dims).
        flip_angles_deg: Nominal flip angles in degrees. 1D Tensor of shape (num_flip_angles,).
        TR: Repetition time (ms). Float.
        b1_map (optional): B1+ map (actual_FA / nominal_FA).
                         Shape (...spatial_dims). If None, assumes B1=1.0.
        initial_T1_ms_guess: Initial guess for T1 (ms).
        initial_M0_guess: Initial guess for M0. If <=0, derived from max signal
                          at the smallest flip angle.
        num_iterations: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        min_T1_ms, max_T1_ms: Constraints for T1 values during optimization.
        min_M0_scale, max_M0_scale: Relative constraints for M0 based on its initial guess.
        optimizer_type: 'adam' or 'lbfgs'. LBFGS can be more robust but slower per iteration.
        verbose: If True, print loss every N iterations.
        device: Computation device ('cpu' or 'cuda').

    Returns:
        T1_map: Estimated T1 map (ms). Shape (...spatial_dims).
        M0_map: Estimated M0 map. Shape (...spatial_dims).
    """
    if signals.ndim < 1:
        raise ValueError("signals tensor must have at least one dimension (num_flip_angles).")
    if flip_angles_deg.ndim != 1 or signals.shape[0] != flip_angles_deg.shape[0]:
        raise ValueError("signals first dimension must match flip_angles_deg length.")

    _signals = signals.to(device)
    _flip_angles_rad = (flip_angles_deg * math.pi / 180.0).to(device)

    spatial_dims = _signals.shape[1:]
    # num_fas = _signals.shape[0] # Not directly used later, fa_view_shape handles it

    if b1_map is not None:
        _b1_map = b1_map.to(device)
        if _b1_map.shape != spatial_dims:
            raise ValueError(f"b1_map shape {_b1_map.shape} must match signal spatial dims {spatial_dims}")
        _b1_map_bc = _b1_map.unsqueeze(0)
    else:
        _b1_map_bc = torch.tensor(1.0, device=device, dtype=torch.float32)

    T1_map = torch.full(spatial_dims, float(initial_T1_ms_guess), device=device, dtype=torch.float32)

    if initial_M0_guess <= 0:
        m0_initial_val = torch.max(_signals, dim=0)[0]
        m0_initial_val = torch.clamp(m0_initial_val, min=1e-3)
    else:
        m0_initial_val = torch.full(spatial_dims, float(initial_M0_guess), device=device, dtype=torch.float32)

    M0_map = m0_initial_val.clone().detach()

    T1_map.requires_grad_(True)
    M0_map.requires_grad_(True)

    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([T1_map, M0_map], lr=learning_rate)
    elif optimizer_type.lower() == 'lbfgs':
        # LBFGS lr is more like a max step size, often 1.0 is used.
        # max_iter per .step() call, num_iterations is outer loops.
        optimizer = optim.LBFGS([T1_map, M0_map], lr=learning_rate, max_iter=20)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    for i in range(num_iterations):
        def closure():
            optimizer.zero_grad()

            with torch.no_grad():
                T1_clamped = torch.clamp(T1_map, min=min_T1_ms, max=max_T1_ms)
                M0_clamped = torch.clamp(M0_map,
                                         min=m0_initial_val * min_M0_scale,
                                         max=m0_initial_val * max_M0_scale)
                T1_map.data.copy_(T1_clamped.data)
                M0_map.data.copy_(M0_clamped.data)

            fa_view_shape = (-1,) + (1,) * len(spatial_dims)
            current_fas_rad = _flip_angles_rad.view(*fa_view_shape) * _b1_map_bc

            predicted_s = spgr_signal(
                T1_map.unsqueeze(0),
                M0_map.unsqueeze(0),
                current_fas_rad,
                TR
            )

            loss = torch.sum((predicted_s - _signals)**2)
            loss.backward()

            if verbose and (i % max(1, num_iterations // 10) == 0 or i == num_iterations - 1):
                print(f"Iter {i}/{num_iterations}, Loss: {loss.item():.4e}")
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        T1_map.data.copy_(torch.clamp(T1_map.data, min=min_T1_ms, max=max_T1_ms))
        M0_map.data.copy_(torch.clamp(M0_map.data,
                                      min=m0_initial_val * min_M0_scale,
                                      max=m0_initial_val * max_M0_scale))

    return T1_map.detach(), M0_map.detach()
