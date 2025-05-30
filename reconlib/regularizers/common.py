import torch
import torch.nn as nn # For functional.py, if it uses it, though not directly here
from .base import Regularizer # IMPORTANT: Inherit from the new base class
from .functional import l1_norm, l2_norm_squared, total_variation, huber_penalty, charbonnier_penalty # Added charbonnier_penalty

class L1Regularizer(Regularizer):
    """L1 Norm Regularizer: R(x) = lambda_reg * ||x||_1.

    This regularizer promotes sparsity in the solution `x` by penalizing the
    sum of the absolute values of its elements. It is widely used in compressed
    sensing and feature selection.
    The L1 norm is applied element-wise and summed. For complex numbers,
    the absolute value (magnitude) is used.
    """
    def __init__(self, lambda_reg: float):
        """Initializes the L1 Regularizer.

        Args:
            lambda_reg (float): The regularization strength parameter.
                Must be non-negative.
        """
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        self.lambda_reg = lambda_reg

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the L1 regularization value: lambda_reg * ||x||_1.

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization value.
        """
        return self.lambda_reg * l1_norm(x)

    def _soft_threshold_complex(self, x: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
        """
        Complex-aware soft-thresholding: sgn(x_i) * max(|x_i| - threshold, 0) for each element x_i.
        If x is real, sgn(x_i) is equivalent to sign(x_i).
        If x is complex, sgn(x_i) is x_i / |x_i| (or 0 if x_i is 0).
        """
        abs_x = torch.abs(x)
        shrinkage = torch.maximum(abs_x - threshold, torch.zeros_like(abs_x))

        # Handle x_i = 0 case for complex sgn to avoid division by zero (0/0 -> NaN)
        # torch.sgn handles this correctly for complex numbers (sgn(0) = 0).
        if x.is_complex():
            return torch.sgn(x) * shrinkage
        else: # Real case
            return torch.sign(x) * shrinkage

    def proximal_operator(self, x: torch.Tensor, steplength: float | torch.Tensor) -> torch.Tensor:
        """Computes the proximal operator of the L1 regularizer.

        Solves: `argmin_u { lambda_reg * ||u||_1 + (1/(2*steplength)) * ||u - x||_2^2 }`
        This is equivalent to element-wise complex soft-thresholding:
        `sgn(x_i) * max(|x_i| - lambda_reg * steplength, 0)`.

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.
            steplength (float | torch.Tensor): The step length parameter (often
                denoted as gamma or tau in optimization algorithms, sometimes
                referred to as `t` or `alpha`). This scales the influence of the
                quadratic term. Can be a float or a tensor broadcastable with `x`.

        Returns:
            torch.Tensor: The result of the proximal operation on `x`.
            Has the same shape and dtype as `x`.
        """
        if isinstance(steplength, torch.Tensor) and steplength.numel() > 1:
            if not steplength.shape == x.shape and not x.shape == steplength.shape: # Basic check
                 # A more robust check would be to try broadcasting and catch error, or use torch.broadcast_shapes
                 try:
                     torch.broadcast_shapes(x.shape, steplength.shape)
                 except RuntimeError:
                     raise ValueError(f"If steplength is a tensor ({steplength.shape}), its shape must be broadcastable to x ({x.shape}).")
        
        threshold_val = self.lambda_reg * steplength
        return self._soft_threshold_complex(x, threshold_val)

class L2Regularizer(Regularizer):
    """Squared L2 Norm Regularizer: R(x) = 0.5 * lambda_reg * ||x||_2^2.

    This regularizer, also known as Tikhonov regularization or Ridge regression,
    penalizes large values in `x`, promoting solutions with smaller magnitudes.
    The L2 norm is the sum of squares of the elements. For complex numbers,
    it's the sum of squares of their magnitudes: `0.5 * lambda_reg * sum(|x_i|^2)`.
    """
    def __init__(self, lambda_reg: float):
        """Initializes the L2 Regularizer.

        Args:
            lambda_reg (float): The regularization strength parameter.
                Must be non-negative.
        """
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        self.lambda_reg = lambda_reg

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the L2 regularization value: 0.5 * lambda_reg * ||x||_2^2.

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization value.
        """
        return 0.5 * self.lambda_reg * l2_norm_squared(x)

    def proximal_operator(self, x: torch.Tensor, steplength: float | torch.Tensor) -> torch.Tensor:
        """Computes the proximal operator of the squared L2 regularizer.

        Solves: `argmin_u { 0.5 * lambda_reg * ||u||_2^2 + (1/(2*steplength)) * ||u - x||_2^2 }`
        The solution is `x / (1 + lambda_reg * steplength)`.

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.
            steplength (float | torch.Tensor): The step length parameter.
                Can be a float or a tensor broadcastable with `x`.

        Returns:
            torch.Tensor: The result of the proximal operation on `x`.
            Has the same shape and dtype as `x`.
        """
        return x / (1 + self.lambda_reg * steplength)

class TVRegularizer(Regularizer):
    """Total Variation (TV) Regularizer: R(x) = lambda_param * TV(x).

    This regularizer promotes piece-wise constant solutions by penalizing the
    sum of the magnitudes of the gradients (or finite differences) of `x`.
    It is commonly used for image denoising and reconstruction to preserve edges
    while smoothing flat regions. This implementation assumes isotropic TV:
    `TV(x) = sum_i sqrt( (grad_x x)_i^2 + (grad_y x)_i^2 + ... )`.

    The proximal operator is solved using Chambolle's projection algorithm (for 2D/3D)
    or its variants.
    """
    def __init__(self, 
                 lambda_param: float, 
                 max_chambolle_iter: int = 50, 
                 tol_chambolle: float = 1e-5, 
                 verbose_chambolle: bool = False):
        """Initializes the Total Variation (TV) Regularizer.

        Args:
            lambda_param (float): The regularization strength parameter.
                Must be non-negative.
            max_chambolle_iter (int, optional): Maximum number of iterations for
                Chambolle's algorithm in the proximal operator. Defaults to 50.
            tol_chambolle (float, optional): Tolerance for convergence of
                Chambolle's algorithm. Defaults to 1e-5.
            verbose_chambolle (bool, optional): If True, prints convergence
                information from Chambolle's algorithm. Defaults to False.
        """
        super().__init__()
        if lambda_param < 0:
            raise ValueError("lambda_param must be non-negative.")
        self.lambda_param = lambda_param
        self.max_iter = max_chambolle_iter
        self.tol = tol_chambolle
        self.verbose = verbose_chambolle

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the TV regularization value: lambda_param * TV(x).

        Assumes isotropic TV. For complex data, TV is typically applied to the
        magnitude or to real and imaginary parts separately. This implementation
        passes the complex data to `functional.total_variation` which may
        handle it by summing TV of real and imaginary parts or by using
        complex-valued gradients.

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.
                Expected to be 2D (H,W), 3D (D,H,W), or higher with leading
                batch/channel dimensions.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization value.
        """
        return self.lambda_param * total_variation(x, isotropic=True)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        spatial_ndim = 0
        if x.ndim == 2: # H, W
            spatial_ndim = 2
        elif x.ndim == 3: # D, H, W (or C, H, W - assume spatial if C is not tiny)
            spatial_ndim = 3 if x.shape[0] > 4 or x.ndim -1 ==2  else 2 # Heuristic, x.ndim-1 ==2 means shape is (D,H,W) basically
            if x.shape[0] <=4 and x.ndim ==3 : spatial_ndim =2 # (C,H,W)
        elif x.ndim >= 4: 
            # N, C, H, W (spatial_ndim=2, last two dims)
            # N, C, D, H, W (spatial_ndim=3, last three dims)
            # Heuristic: if shape[-3] is large, likely D for NCDHW
            is_NCDHW_like = x.ndim == 5 and x.shape[-3] > 4 
            is_NCHW_like_batch_gt_1 = x.ndim ==4 and x.shape[0] > 1 # N,C,H,W
            
            if is_NCDHW_like : spatial_ndim = 3
            elif is_NCHW_like_batch_gt_1 : spatial_ndim = 2
            elif x.ndim == 4 and x.shape[0] == 1: # 1,C,H,W -> C,H,W for processing
                 spatial_ndim = 2 # Process last two as spatial
            elif x.ndim == 5 and x.shape[0] == 1: # 1,C,D,H,W -> C,D,H,W for processing
                 spatial_ndim = 3 # Process last three as spatial
            else: # Default for safety, or could raise error
                 spatial_ndim = 2 if x.ndim - 2 >=0 else x.ndim # last two if possible
        else:
            raise ValueError(f"Unsupported tensor ndim for gradient: {x.ndim}")

        grads = []
        for d in range(spatial_ndim):
            axis = x.ndim - spatial_ndim + d
            slicers_curr = [slice(None)] * x.ndim
            slicers_next = [slice(None)] * x.ndim
            slicers_curr[axis] = slice(None, -1)
            slicers_next[axis] = slice(1, None)
            grad_d = x[tuple(slicers_next)] - x[tuple(slicers_curr)]
            padding_config = [0] * (2 * x.ndim)
            dim_pair_idx_in_pad_config = 2 * (x.ndim - 1 - axis)
            padding_config[dim_pair_idx_in_pad_config + 1] = 1 
            grads.append(torch.nn.functional.pad(grad_d, tuple(padding_config)))
        return torch.stack(grads, dim=0) 

    def _divergence(self, grad_field: torch.Tensor) -> torch.Tensor:
        num_spatial_dims = grad_field.shape[0]
        # Output shape is spatial part of input to _gradient, e.g. (H,W) or (D,H,W)
        # grad_field shape is (num_spatial_dims, *original_spatial_shape)
        # So div output shape is grad_field.shape[1:]
        div = torch.zeros_like(grad_field[0], dtype=grad_field.dtype) 

        for d in range(num_spatial_dims):
            component_d = grad_field[d] 
            axis = component_d.ndim - num_spatial_dims + d
            
            current_div_comp = component_d.clone()
            shifted_comp = torch.roll(component_d, shifts=1, dims=axis)
            first_slice = [slice(None)] * component_d.ndim
            first_slice[axis] = 0
            shifted_comp[tuple(first_slice)] = 0.0
            current_div_comp -= shifted_comp
            div += current_div_comp
        return div

    def proximal_operator(self, x_tensor: torch.Tensor, steplength: float | torch.Tensor) -> torch.Tensor:
        if x_tensor.is_complex():
            if self.verbose: print("TVProx: Processing complex data (real and imaginary parts separately).")
            x_real = self.proximal_operator(x_tensor.real.contiguous(), steplength) # Ensure contiguous
            x_imag = self.proximal_operator(x_tensor.imag.contiguous(), steplength) # Ensure contiguous
            return torch.complex(x_real, x_imag)

        original_shape = x_tensor.shape
        is_squeezed_batch = False
        is_squeezed_channel = False
        x_proc = x_tensor

        # Handle batch and channel dimensions for processing
        # Goal: process a 2D (H,W) or 3D (D,H,W) tensor for TV
        if x_tensor.ndim == 4: # N,C,H,W or C,D,H,W (if N=1) or D,H,W,C (unlikely for torch)
            if original_shape[0] == 1: # Squeeze batch dim
                x_proc = x_tensor.squeeze(0) 
                is_squeezed_batch = True
            else: # True batch N>1, C, H, W
                 # Process each item in batch, less efficient.
                 # For now, let's assume solver handles iterating over batch if needed,
                 # or this regularizer is applied per-batch-item.
                 # If not, this would need a loop:
                 return torch.stack([self.proximal_operator(x_tensor[i], steplength) for i in range(original_shape[0])])

            # After squeezing batch (if N=1), x_proc might be C,H,W or C,D,H,W
            if x_proc.ndim == 3 and x_proc.shape[0] == 1: # C=1, H, W
                x_proc = x_proc.squeeze(0) # Now H,W
                is_squeezed_channel = True
            elif x_proc.ndim == 4 and x_proc.shape[0] == 1: # C=1, D, H, W
                 x_proc = x_proc.squeeze(0) # Now D,H,W
                 is_squeezed_channel = True
            # If C > 1, it's multichannel TV, _gradient and _divergence need to handle this.
            # The current _gradient assumes last 2 or 3 dims are spatial.
            
        elif x_tensor.ndim == 5: # N,C,D,H,W
            if original_shape[0] == 1: # Squeeze batch dim
                x_proc = x_tensor.squeeze(0)
                is_squeezed_batch = True
            else: # True batch N>1, C, D, H, W
                return torch.stack([self.proximal_operator(x_tensor[i], steplength) for i in range(original_shape[0])])
            
            # After squeezing batch (if N=1), x_proc is C,D,H,W
            if x_proc.ndim == 4 and x_proc.shape[0] == 1: # C=1, D,H,W
                x_proc = x_proc.squeeze(0) # Now D,H,W
                is_squeezed_channel = True


        effective_lambda = self.lambda_param * steplength 
        
        # p_spatial_dims: number of dimensions in the gradient vector (e.g., 2 for 2D image, 3 for 3D)
        # This should match the number of spatial dimensions of x_proc.
        if x_proc.ndim == 2: p_spatial_dims = 2 # H,W
        elif x_proc.ndim == 3: p_spatial_dims = 3 # D,H,W (assuming not C,H,W with C>1 here)
                                               # If C,H,W with C>1, it's multichannel TV.
                                               # User's _gradient heuristic might set p_spatial_dims=2 for C,H,W.
                                               # Let's be explicit: if x_proc is 3D and first dim is small (<=4), assume it's channels.
        elif x_proc.ndim > 3:
             raise ValueError(f"x_proc has too many dimensions ({x_proc.ndim}) after squeezing for TV prox. Shape: {x_proc.shape}")
        else:
             raise ValueError(f"Unsupported x_proc.ndim ({x_proc.ndim}) for TV prox. Shape: {x_proc.shape}")


        p = torch.zeros((p_spatial_dims,) + x_proc.shape, device=x_proc.device, dtype=x_proc.dtype)
        # Tau selection: For L2 norm (isotropic TV), tau <= 1/ (2*num_spatial_dims) for stability of standard Chambolle.
        # However, the formulation used (dual of FGP or similar) might have different stability constraints.
        # The MIRT value 0.120 is likely a safe empirical value.
        tau = 0.120 # Step size for the dual update (p)

        # Effective lambda for this specific prox computation
        # prox_{steplength * R(x)} where R(x) = lambda_param * TV(x)
        # So, we are solving prox_{steplength * lambda_param * TV(x)}
        effective_lambda_prox = self.lambda_param * steplength
        if effective_lambda_prox == 0: # No regularization
            return x_tensor # Return original tensor if lambda is zero

        for i in range(self.max_iter):
            # Update dual variable p using Chambolle's algorithm for isotropic TV (L2 norm of gradient)
            # This corresponds to a fixed-point iteration for the dual problem.
            # The term (div_p - x_proc / effective_lambda_prox) is related to the gradient of the Fenchel conjugate.
            div_p = self._divergence(p)
            grad_term = self._gradient(div_p - x_proc / effective_lambda_prox)
            p_candidate = p + tau * grad_term
            
            # Projection step: p_new = p_candidate / max(1, ||p_candidate||_vec)
            # where ||.||_vec is the L2 norm computed for each gradient vector (over spatial dimensions component)
            norm_p_candidate_vectors = torch.sqrt(torch.sum(p_candidate**2, dim=0, keepdim=True)) # keepdim for broadcasting
            p_new = p_candidate / torch.maximum(torch.ones_like(norm_p_candidate_vectors), norm_p_candidate_vectors)

            # Convergence check
            diff_p_norm = torch.linalg.norm(p_new.flatten() - p.flatten())
            p_norm = torch.linalg.norm(p.flatten()) + 1e-9 # Avoid division by zero
            relative_diff_p = diff_p_norm / p_norm
            
            p = p_new

            if self.verbose and (i % 10 == 0 or i == self.max_iter -1):
                # For verbose output, calculate current estimate and its TV
                current_estimate = x_proc - effective_lambda_prox * self._divergence(p)
                tv_val = total_variation(current_estimate, isotropic=True) # Using functional for consistency
                print(f"TV Prox iter {i+1}/{self.max_iter}, rel_diff_p: {relative_diff_p.item():.2e}, est. TV: {tv_val.item():.2e}")

            if relative_diff_p < self.tol:
                if self.verbose: print(f"TV Prox converged at iter {i+1}, rel_diff_p: {relative_diff_p.item():.2e}")
                break
        
        # Primal update: x_denoised = x_input - effective_lambda_prox * div(p_final)
        x_denoised = x_proc - effective_lambda_prox * self._divergence(p)

        # Restore original shape if squeezed
        if is_squeezed_channel:
            x_denoised = x_denoised.unsqueeze(0)
        if is_squeezed_batch:
            x_denoised = x_denoised.unsqueeze(0)
        
        # Final check if shape matches original_shape if any squeezing happened
        if (is_squeezed_batch or is_squeezed_channel) and x_denoised.shape != original_shape:
             x_denoised = x_denoised.reshape(original_shape)


        return x_denoised

# The __main__ part from user code for L1/L2/TV can be added here for testing this file directly.
# Note: TVRegularizer._gradient and _divergence might need adjustments based on final decision
# on how to handle various input shapes (esp. with channels/batches).
# The provided TV code also had a forward method, which is inherited from base.Regularizer now.


class HuberRegularizer(Regularizer):
    """Huber Regularizer: R(x) = lambda_reg * sum_i H_delta(x_i).

    The Huber penalty H_delta(a) is defined as:
    - `0.5 * a^2` if `|a| <= delta` (quadratic region)
    - `delta * (|a| - 0.5 * delta)` if `|a| > delta` (linear region)

    It combines the properties of L2 (smoothness for small errors) and L1
    (robustness to outliers for large errors). It is convex and continuously
    differentiable.
    """
    def __init__(self, lambda_reg: float, delta: float):
        """Initializes the Huber Regularizer.

        Args:
            lambda_reg (float): The regularization strength. Must be non-negative.
            delta (float): The threshold parameter that separates the quadratic
                and linear regions of the Huber penalty. Must be positive.
        """
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if delta <= 0:
            raise ValueError("delta must be positive.")
        self.lambda_reg = lambda_reg
        self.delta = delta

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Huber regularization value: lambda_reg * sum_i H_delta(x_i).

        Args:
            x (torch.Tensor): The input tensor. Typically real-valued.
                If complex, the Huber penalty is usually applied to the magnitude
                or real/imaginary parts separately (current functional.huber_penalty
                applies to elements as if they are real).

        Returns:
            torch.Tensor: A scalar tensor representing the regularization value.
        """
        return self.lambda_reg * huber_penalty(x, self.delta)

    def proximal_operator(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        """Computes the proximal operator of the Huber regularizer.

        Solves element-wise:
        `argmin_u { lambda_reg * H_delta(u) + (1/(2*steplength)) * (u - x)^2 }`

        The solution is:
        - `x / (1 + lambda_reg * steplength)` if `|x / (1 + lambda_reg * steplength)| <= delta`
        - `x - lambda_reg * steplength * delta * sgn(x)` if `|x - lambda_reg * steplength * delta * sgn(x)| > delta`
        - This means the solution `u` is `x / (1 + gamma_eff)` if `|u| <= delta`,
          and `x - gamma_eff * delta * sgn(x)` if `|u| > delta`, where `gamma_eff = lambda_reg * steplength`.

        Args:
            x (torch.Tensor): The input tensor. Typically real-valued.
                If complex, this prox might not be standard; usually Huber is
                applied to magnitude or real/imaginary parts. Current
                implementation is element-wise, so complex numbers are processed
                with their real/imaginary parts potentially following different regimes.
            steplength (float): The step length parameter.

        Returns:
            torch.Tensor: The result of the proximal operation on `x`.
            Same shape and dtype as `x`.
        """
        gamma_eff = self.lambda_reg * steplength # Effective coefficient for Huber in prox objective

        # Solution for u:
        # Case 1: If |u_quadratic| <= delta, then u = u_quadratic = x / (1 + gamma_eff)
        # Case 2: If |u_linear| > delta, then u = u_linear = x - gamma_eff * delta * sgn(x)
        
        # Compute potential solution from quadratic region shrinkage
        u_quadratic_candidate = x / (1 + gamma_eff)
        
        # Compute potential solution from linear region shrinkage
        u_linear_candidate = x - gamma_eff * self.delta * torch.sign(x) # sign(x) works for real and complex sgn(0)=0
        
        # Determine which case applies for each element
        # An element is in Case 1 if its quadratic solution u_quadratic_candidate satisfies |u_quadratic_candidate| <= delta.
        # Otherwise, it's in Case 2 (linear region).
        
        # For complex numbers, torch.abs is magnitude.
        # torch.sign(complex) = complex_val / abs(complex_val) or 0 if complex_val is 0.
        
        condition_case1 = torch.abs(u_quadratic_candidate) <= self.delta

        out = torch.where(condition_case1, u_quadratic_candidate, u_linear_candidate)
        
        return out


class CharbonnierRegularizer(Regularizer):
    """Charbonnier Regularizer: R(x) = lambda_reg * sum_i (sqrt(x_i^2 + epsilon^2)).

    This is a smooth approximation of the L1 norm, also known as the
    L2-L1 norm or pseudo-Huber loss (related, but not identical).
    It is continuously differentiable and promotes sparsity while being less
    sensitive to very small values compared to L1.
    The form `sqrt(x_i^2 + epsilon^2) - epsilon` is sometimes used to ensure R(0)=0;
    this implementation uses `sqrt(x_i^2 + epsilon^2)`.
    """
    def __init__(self, lambda_reg: float, epsilon: float, newton_iter: int = 5):
        """Initializes the Charbonnier Regularizer.

        Args:
            lambda_reg (float): The regularization strength. Must be non-negative.
            epsilon (float): A small positive constant that controls the smoothness
                near zero. Must be positive.
            newton_iter (int, optional): Number of Newton-Raphson iterations used
                to solve the scalar non-linear equation in the proximal operator.
                Defaults to 5.
        """
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.newton_iter = newton_iter

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes lambda_reg * sum_i (sqrt(x_i^2 + epsilon^2)).

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.
                If complex, `x_i^2` is typically `|x_i|^2`.
                The `functional.charbonnier_penalty` handles this.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization value.
        """
        return self.lambda_reg * charbonnier_penalty(x, self.epsilon)

    def _solve_charbonnier_prox_scalar(self, v_abs: torch.Tensor, gamma_eff: float) -> torch.Tensor:
        """
        Solves y * (1 + gamma_eff / sqrt(y^2 + epsilon^2)) = v_abs for y >= 0.
        This is for a single absolute value v_abs.
        Uses Newton-Raphson method.
        """
        # Initial guess: y = v_abs / (1 + gamma_eff / self.epsilon) (approx for small y)
        # or y = v_abs (approx for large y, where sqrt term approaches 1)
        # A simple and often effective initial guess is y = v_abs for y >=0.
        # Or, more carefully, if gamma_eff is large, u is small. If gamma_eff is small, u approx v_abs.
        # Let's initialize y to v_abs. If v_abs is 0, y is 0.
        y = v_abs.clone()
        
        # Mask for non-zero v_abs to avoid division by zero or unnecessary computation
        active_mask = y > 1e-9 
        if not torch.any(active_mask):
            return y

        # Apply updates only to active elements
        y_active = y[active_mask]
        v_abs_active = v_abs[active_mask]

        for _ in range(self.newton_iter):
            sqrt_term = torch.sqrt(y_active**2 + self.epsilon**2)
            g_u = y_active * (1 + gamma_eff / sqrt_term) - v_abs_active
            # g_prime_u = 1 + gamma_eff * self.epsilon**2 / (sqrt_term**3) # Derivative of g(u) w.r.t u
            # Avoid issues if sqrt_term is zero, though epsilon > 0 should prevent this.
            # Derivative of (gamma_eff * y_active / sqrt_term) is (gamma_eff * epsilon^2 / sqrt_term^3)
            g_prime_u = 1 + gamma_eff * (self.epsilon**2) / (y_active**2 + self.epsilon**2).pow(1.5)

            # Newton step: y_new = y - g(y) / g'(y)
            y_active = y_active - g_u / (g_prime_u + 1e-9) # Add epsilon for stability
            y_active = torch.relu(y_active) # Ensure y remains non-negative

        y[active_mask] = y_active
        return y


    def proximal_operator(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        """
        Computes prox_R(x, steplength) for R(u) = lambda_reg * sum(sqrt(u_i^2 + epsilon^2)).
        Solves u_i + (lambda_reg * steplength) * u_i / sqrt(u_i^2 + epsilon^2) = x_i for each element.
        """
        if x.is_complex():
            # For complex data, apply prox to magnitude and restore phase.
            # This is a common way to handle Charbonnier for complex values,
            # effectively applying it to |x_i|.
            # prox(|x|, params) * (x / |x|)
            x_abs = torch.abs(x)
            # Handle x=0 case for phase to avoid NaN (x / abs_x results in NaN if abs_x is 0)
            phase = torch.where(x_abs > 1e-9, x / (x_abs + 1e-9), torch.zeros_like(x))

            x_abs_processed = self.proximal_operator(x_abs, steplength) # Recursive call for real part
            return x_abs_processed * phase

        gamma_eff = self.lambda_reg * steplength
        if gamma_eff == 0: # No regularization
            return x

        # The solution u will have the same sign as x.
        # Let u_abs = |u| and x_abs = |x|. We solve for u_abs.
        # u_abs * (1 + gamma_eff / sqrt(u_abs^2 + epsilon^2)) = x_abs
        x_abs = torch.abs(x)
        u_abs = self._solve_charbonnier_prox_scalar(x_abs, gamma_eff)
        
        return torch.sign(x) * u_abs


class NonnegativityConstraint(Regularizer):
    """Non-negativity Constraint Regularizer.

    This acts as an indicator function for the set of non-negative numbers.
    R(x) = 0 if all elements of x are >= 0.
    R(x) = +infinity if any element of x is < 0.

    The proximal operator for this regularizer is a projection onto the
    non-negative orthant, which means setting negative values to zero.
    """
    def __init__(self):
        """Initializes the NonnegativityConstraint regularizer."""
        super().__init__()

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the value of the non-negativity constraint.

        Returns 0 if all elements of x are non-negative, otherwise conceptually
        returns infinity. For practical purposes in optimization, this function
        might return 0 if the constraint is satisfied, assuming the proximal
        operator enforces the constraint. A large penalty could be returned if
        violated, but typically this is handled by the prox.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A scalar tensor. Returns 0.0 if all elements of `x`
            (or `x.real` if complex) are >= 0. For simplicity in typical proximal
            algorithms, this often returns 0, as the enforcement is done by the prox.
        """
        # Check based on real part if complex, as non-negativity typically applies to real quantities.
        data_to_check = x.real if x.is_complex() else x
        if torch.all(data_to_check >= -1e-9): # Allow for small numerical errors
            return torch.tensor(0.0, device=x.device, dtype=x.dtype if x.is_floating_point() else torch.float32)
        else:
            # Representing infinity can be problematic for some solvers if not handled explicitly.
            # Returning a very large number or relying on the prox is common.
            # For now, returning 0 and assuming prox enforces it.
            return torch.tensor(0.0, device=x.device, dtype=x.dtype if x.is_floating_point() else torch.float32)


    def proximal_operator(self, x: torch.Tensor, steplength: float | torch.Tensor) -> torch.Tensor:
        """Computes the proximal operator (projection onto the non-negative set).

        For real `x`, this is `max(x, 0)`.
        For complex `x`, this implementation applies non-negativity to the real
        part and zeros out the imaginary part. This behavior is chosen assuming
        the underlying physical quantity (e.g., image intensity) must be real
        and non-negative. Other behaviors for complex data might be valid
        depending on the application (e.g., preserving the imaginary part if
        only the real part is constrained).

        The `steplength` parameter is not used for projection onto a convex set.

        Args:
            x (torch.Tensor): The input tensor. Can be real or complex.
            steplength (float | torch.Tensor): The step length parameter (ignored).

        Returns:
            torch.Tensor: The tensor `x` projected onto the non-negative set,
            with the same shape and dtype as `x`.
        """
        if x.is_complex():
            # Behavior for complex: Apply ReLu to real part, zero imaginary part.
            # Consider if a warning is appropriate if this is not always desired.
            # print("Warning: NonnegativityConstraint applied to complex tensor. "
            #       "Applying to real part and zeroing imaginary part.")
            return torch.complex(torch.relu(x.real), torch.zeros_like(x.imag))
        else:
            return torch.relu(x)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Enforces non-negativity on the input tensor.

        This is a convenience method that calls the proximal operator.
        The `steplength` argument to the proximal operator is irrelevant for
        this projection.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The image tensor with non-negativity enforced.
        """
        return self.proximal_operator(image, steplength=0.0) # steplength value doesn't matter
