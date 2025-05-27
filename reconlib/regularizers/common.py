import torch
import torch.nn as nn # For functional.py, if it uses it, though not directly here
from .base import Regularizer # IMPORTANT: Inherit from the new base class
from .functional import l1_norm, l2_norm_squared, total_variation, huber_penalty, charbonnier_penalty # Added charbonnier_penalty

class L1Regularizer(Regularizer):
    """
    L1 Norm Regularizer: R(x) = lambda_reg * ||x||_1.
    """
    def __init__(self, lambda_reg: float):
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        self.lambda_reg = lambda_reg

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes lambda_reg * ||x||_1."""
        return self.lambda_reg * l1_norm(x)

    def _soft_threshold_complex(self, x: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
        """
        Complex-aware soft-thresholding: sign(x) * max(|x| - threshold, 0).
        """
        abs_x = torch.abs(x)
        # threshold should be broadcastable with abs_x if it's a tensor
        shrinkage = torch.maximum(abs_x - threshold, torch.zeros_like(abs_x))

        if x.is_complex():
            return torch.sgn(x) * shrinkage
        else: # Real case
            return torch.sign(x) * shrinkage

    def proximal_operator(self, x: torch.Tensor, steplength: float | torch.Tensor) -> torch.Tensor:
        """
        Computes prox_R(x, steplength) = argmin_u { lambda_reg * ||u||_1 + (1/(2*steplength)) * ||u - x||_2^2 }
        This is equivalent to soft_threshold(x, lambda_reg * steplength).
        """
        # Note: The 'steplength' here is the gamma in (1/(2*gamma)) * ||u-x||^2.
        # The threshold for soft-thresholding is typically lambda_reg_true * gamma_optimizer.
        # If self.lambda_reg is the true lambda, then the threshold is self.lambda_reg * steplength.
        if isinstance(steplength, torch.Tensor) and steplength.numel() > 1:
            if not steplength.shape == x.shape: # Ensure broadcastable or same shape
                 raise ValueError("If steplength is a tensor, its shape must be broadcastable to x.")
        
        threshold_val = self.lambda_reg * steplength
        return self._soft_threshold_complex(x, threshold_val)

class L2Regularizer(Regularizer):
    """
    Squared L2 Norm Regularizer: R(x) = 0.5 * lambda_reg * ||x||_2^2.
    """
    def __init__(self, lambda_reg: float):
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        self.lambda_reg = lambda_reg

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes 0.5 * lambda_reg * ||x||_2^2."""
        return 0.5 * self.lambda_reg * l2_norm_squared(x)

    def proximal_operator(self, x: torch.Tensor, steplength: float | torch.Tensor) -> torch.Tensor:
        """
        Computes prox_R(x, steplength) = argmin_u { 0.5 * lambda_reg * ||u||_2^2 + (1/(2*steplength)) * ||u - x||_2^2 }
        This simplifies to x / (1 + lambda_reg * steplength).
        """
        # Here, steplength is gamma. The factor in prox is (lambda_reg * gamma).
        return x / (1 + self.lambda_reg * steplength)

class TVRegularizer(Regularizer):
    """
    Total Variation (TV) Regularizer: R(x) = lambda_param * TV(x).
    The proximal operator is solved using Chambolle's algorithm.
    Assumes isotropic TV.
    """
    def __init__(self, 
                 lambda_param: float, 
                 max_chambolle_iter: int = 50, 
                 tol_chambolle: float = 1e-5, 
                 verbose_chambolle: bool = False):
        super().__init__()
        if lambda_param < 0:
            raise ValueError("lambda_param must be non-negative.")
        self.lambda_param = lambda_param # lambda_reg for consistency with L1/L2
        self.max_iter = max_chambolle_iter
        self.tol = tol_chambolle
        self.verbose = verbose_chambolle

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes lambda_param * TV(x) using isotropic TV."""
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
        tau = 0.120 

        for i in range(self.max_iter):
            # x_bar = x_proc - effective_lambda * self._divergence(p)
            # grad_x_bar = self._gradient(x_bar) # grad_x_bar is (p_spatial_dims, *x_proc.shape)
            # p_candidate = p + tau * grad_x_bar
            # norm_p_candidate_vec = torch.sqrt(torch.sum(p_candidate**2, dim=0, keepdim=True))
            # p_new = p_candidate / torch.maximum(torch.ones_like(norm_p_candidate_vec), norm_p_candidate_vec)
            
            # Simpler equivalent from Chambolle's paper (equation 11 / alg 3.1)
            # This version avoids recomputing gradient of x_bar each time.
            # It updates p based on grad_div_p_minus_f where f = x_proc / effective_lambda
            # p_tilde = p + tau * self._gradient(self._divergence(p) - x_proc / effective_lambda)
            # p_new = p_tilde / (1 + tau * torch.abs(p_tilde_over_tau_grad_component_wise_not_vector_norm)) <- this is for anisotropic
            # For isotropic TV (L2 norm of gradient):
            # p_tilde = p + tau * self._gradient(self._divergence(p) - x_proc / effective_lambda)
            # norm_p_tilde_vec = torch.sqrt(torch.sum(p_tilde**2, dim=0, keepdim=True))
            # p_new = p_tilde / torch.maximum(torch.ones_like(norm_p_tilde_vec), norm_p_tilde_vec)

            # Let's use the formulation from the user's code which is more common:
            # grad_of_term_in_prox = self._gradient(self._divergence(p) - (x_proc / effective_lambda) )
            # p_temp = p + tau * grad_of_term_in_prox
            # norm_p_temp_vec = torch.sqrt(torch.sum(p_temp**2, dim=0, keepdim=True))
            # p_new = p_temp / torch.maximum(torch.ones_like(norm_p_temp_vec), norm_p_temp_vec)

            # The provided code has:
            # grad_x_p = self._gradient(x_proc - effective_lambda * self._divergence(p))
            # p_candidate = p + tau * grad_x_p
            # norm_p_candidate_vec = torch.sqrt(torch.sum(p_candidate**2, dim=0, keepdim=True))
            # p_new = p_candidate / torch.maximum(torch.ones_like(norm_p_candidate_vec), norm_p_candidate_vec)
            # This is correct for prox_{lambda*TV}(x_proc) where TV is isotropic.

            div_p = self._divergence(p)
            grad_term = self._gradient(div_p - x_proc / effective_lambda) #Matches MIRT if lambda is eff_lambda
            p_candidate = p + tau * grad_term
            
            # Projection of each vector p(i,j) onto the unit L2 ball
            norm_p_candidate_vectors = torch.sqrt(torch.sum(p_candidate**2, dim=0, keepdim=True))
            p_new = p_candidate / torch.maximum(torch.ones_like(norm_p_candidate_vectors), norm_p_candidate_vectors)

            diff_p_norm = torch.norm(p_new.flatten() - p.flatten())
            p_norm = torch.norm(p.flatten()) + 1e-9
            diff_p = diff_p_norm / p_norm
            
            p = p_new

            if self.verbose and (i % 10 == 0 or i == self.max_iter -1):
                # For verbose, calculate current objective value or TV of current estimate
                current_estimate = x_proc - effective_lambda * self._divergence(p)
                tv_val = total_variation(current_estimate, isotropic=True)
                data_fidelity = 0.5 * torch.sum((current_estimate - x_proc)**2)
                obj = effective_lambda * tv_val + data_fidelity # This is not quite right.
                                                            # Objective is lambda*TV(u) + 1/(2*step)*||u-x||^2
                print(f"TV Prox iter {i+1}/{self.max_iter}, diff_p: {diff_p.item():.2e}, current TV(u_k): {tv_val:.2e}")

            if diff_p < self.tol:
                if self.verbose: print(f"TV Prox converged at iter {i+1}, diff_p: {diff_p.item():.2e}")
                break
        
        x_denoised = x_proc - effective_lambda * self._divergence(p)

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
    """
    Huber Regularizer: R(x) = lambda_reg * sum(H_delta(x_i)),
    where H_delta(a) = 0.5 * a^2 if |a| <= delta,
                     = delta * (|a| - 0.5 * delta) if |a| > delta.
    """
    def __init__(self, lambda_reg: float, delta: float):
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if delta <= 0:
            raise ValueError("delta must be positive.")
        self.lambda_reg = lambda_reg
        self.delta = delta

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes lambda_reg * sum(H_delta(x_i))."""
        # huber_penalty from functional.py should compute sum(H_delta(x_i))
        return self.lambda_reg * huber_penalty(x, self.delta)

    def proximal_operator(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        """
        Computes prox_R(x, steplength) for R(u) = lambda_reg * sum(H_delta(u_i)).
        The steplength parameter here is the gamma in (1/(2*gamma)) * ||u-x||_2^2.
        The effective coefficient for the regularizer in the prox problem is lambda_reg * steplength.
        Let sigma = self.lambda_reg * steplength.
        The prox of sigma * H_delta(u) is:
        y = x / (1 + sigma)
        result = torch.where(torch.abs(y) <= self.delta, y, x - sigma * self.delta * torch.sign(x))
        This can be simplified:
        result_i = x_i if |x_i| <= self.delta * (sigma + 1), else x_i - sigma * self.delta * torch.sign(x_i)
        This is equivalent to:
        x_abs = torch.abs(x)
        result = torch.where(x_abs <= self.delta * (1 + sigma),
                             x / (1 + sigma), # This part is actually x if |x| <= delta * sigma
                                             # and x / (1+sigma) if delta*sigma < |x| <= delta*(1+sigma)
                                             # This is getting tricky.
                             x - sigma * self.delta * torch.sign(x)
                            )
        Let's use a standard formulation for prox_{gamma * H_delta}(x):
        x_abs = abs(x)
        idx_quadratic = x_abs <= delta * (1 + gamma)
        idx_linear = x_abs > delta * (1 + gamma)

        out = torch.zeros_like(x)
        gamma = self.lambda_reg * steplength

        # Quadratic region: x / (1 + gamma) -- this is only if |x| <= delta
        # More general form (Combettes, Pesquet, "Proximal Splitting Methods in Signal Processing"):
        # prox_{gamma*H_delta}(x) = x - gamma * prox_{H_delta/gamma*}(x/gamma)
        # prox_{gamma*H_delta}(x)_i = x_i - gamma * P_B(x_i/gamma)
        # where P_B(y)_i = sign(y_i) * min(|y_i|, delta)
        # So, prox(x)_i = x_i - (lambda_reg * steplength) * sign(x_i/ (lambda_reg*steplength)) * min(|x_i/(lambda_reg*steplength)|, delta)
        # prox(x)_i = x_i - sign(x_i) * min(|x_i|, delta * lambda_reg * steplength)

        gamma_eff = self.lambda_reg * steplength # This is the 'sigma' or 'tau' coefficient of Huber in prox objective
        
        # Moreau's identity: prox_f(x) = x - prox_f*(x) if f is proper, convex, lsc
        # For f(u) = gamma_eff * H_delta(u)
        # prox_f*(x)_i = gamma_eff * delta * ( (x_i/(gamma_eff*delta)) - proj_unitball(x_i/(gamma_eff*delta)) )
        # This is not simple.

        # Let's use the element-wise solution:
        # solve u + gamma_eff * grad(H_delta(u)) = x
        # grad H_delta(u) = u if |u| <= delta
        # grad H_delta(u) = delta * sign(u) if |u| > delta
        # Case 1: |u| <= delta. Then u + gamma_eff * u = x  => u = x / (1 + gamma_eff).
        #   This case holds if |x / (1 + gamma_eff)| <= delta.
        # Case 2: |u| > delta. Then u + gamma_eff * delta * sign(u) = x.
        #   If u > delta, then u = x - gamma_eff * delta. This holds if x - gamma_eff * delta > delta.
        #   If u < -delta, then u = x + gamma_eff * delta. This holds if x + gamma_eff * delta < -delta.
        # Combining these:
        
        out = torch.zeros_like(x)
        
        # Condition for u = x / (1 + gamma_eff)
        # Check if |x / (1 + gamma_eff)| <= delta
        u_case1 = x / (1 + gamma_eff)
        idx_case1 = (torch.abs(u_case1) <= self.delta)
        out[idx_case1] = u_case1[idx_case1]
        
        # Condition for u = x - gamma_eff * delta (implies u > 0)
        # Check if (x - gamma_eff * delta) > delta AND not already in case 1
        u_case2_pos = x - gamma_eff * self.delta
        idx_case2_pos = (u_case2_pos > self.delta) & (~idx_case1)
        out[idx_case2_pos] = u_case2_pos[idx_case2_pos]
        
        # Condition for u = x + gamma_eff * delta (implies u < 0)
        # Check if (x + gamma_eff * delta) < -delta AND not already in case 1
        u_case2_neg = x + gamma_eff * self.delta
        idx_case2_neg = (u_case2_neg < -self.delta) & (~idx_case1)
        out[idx_case2_neg] = u_case2_neg[idx_case2_neg]
        
        return out


class CharbonnierRegularizer(Regularizer):
    """
    Charbonnier Regularizer: R(x) = lambda_reg * sum(sqrt(x_i^2 + epsilon^2) - epsilon).
    The "- epsilon" term is sometimes included so R(0)=0, but often omitted.
    Here, we use R(x) = lambda_reg * sum(sqrt(x_i^2 + epsilon^2)).
    """
    def __init__(self, lambda_reg: float, epsilon: float, newton_iter: int = 5):
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.newton_iter = newton_iter # Iterations for solving the prox subproblem

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Computes lambda_reg * sum(sqrt(x_i^2 + epsilon^2))."""
        # charbonnier_penalty from functional.py should compute sum(sqrt(x_i^2 + epsilon^2))
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
