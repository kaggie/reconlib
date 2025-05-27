"""Module for defining Optimizer classes for MRI reconstruction."""

import torch
from abc import ABC, abstractmethod
# Import GradientMatchingRegularizer for type hinting if needed, though not strictly necessary for runtime
# from reconlib.regularizers import GradientMatchingRegularizer # Example for clarity

class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Defines the interface for the solve method.
    """
    @abstractmethod
    def solve(self, k_space_data, forward_op, regularizer, initial_guess=None):
        """
        Solves the optimization problem.

        Args:
            k_space_data: The k-space data (PyTorch tensor).
            forward_op: The forward operator (instance of reconlib.operators.Operator).
            regularizer: An instance of a class adhering to the Regularizer interface
                         (e.g., from reconlib.regularizers.base.Regularizer),
                         expected to have a `proximal_operator` method.
                         For ADMM, this is the `prox_regularizer`.
            initial_guess: An optional initial guess for the solution (PyTorch tensor, default None).

        Returns:
            The optimized solution (PyTorch tensor).
        """
        pass

class FISTA(Optimizer):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    """
    def __init__(self, max_iter=100, tol=1e-6, verbose=False, 
                 line_search_beta=0.5, max_line_search_iter=20, initial_step_size=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.line_search_beta = line_search_beta 
        self.max_line_search_iter = max_line_search_iter
        self.initial_L_k_estimate = initial_step_size 

    def solve(self, k_space_data, forward_op, regularizer, initial_guess=None):
        device = k_space_data.device 
        image_shape = forward_op.image_shape

        if initial_guess is None:
            x_old = torch.zeros(image_shape, dtype=torch.complex64, device=device)
        else:
            x_old = initial_guess.clone().to(device=device, dtype=torch.complex64)

        y_k = x_old.clone()
        t_old = 1.0
        L_k = self.initial_L_k_estimate 

        if self.verbose:
            print(f"FISTA Optimizer Started. Device: {device}")
            print(f"Params: max_iter={self.max_iter}, tol={self.tol}, initial_L_k={L_k}, line_search_beta (for L_k increase factor)={1/self.line_search_beta}")

        for iter_num in range(self.max_iter):
            grad_y_k = forward_op.op_adj(forward_op.op(y_k.to(torch.complex64)) - k_space_data)

            current_L_k = L_k 
            for ls_iter in range(self.max_line_search_iter):
                step = 1.0 / current_L_k 
                x_new = regularizer.proximal_operator(y_k - step * grad_y_k, step)
                
                f_yk_op_output = forward_op.op(y_k.to(torch.complex64))
                f_x_new_op_output = forward_op.op(x_new.to(torch.complex64))

                f_yk = 0.5 * torch.linalg.norm(f_yk_op_output - k_space_data)**2
                f_x_new = 0.5 * torch.linalg.norm(f_x_new_op_output - k_space_data)**2
                
                diff_x_y = x_new - y_k 
                grad_term = torch.real(torch.vdot(grad_y_k.flatten(), diff_x_y.flatten()))
                quadratic_term = (current_L_k / 2.0) * torch.linalg.norm(diff_x_y)**2
                
                if f_x_new <= f_yk + grad_term + quadratic_term:
                    L_k = current_L_k 
                    break 
                
                current_L_k = current_L_k / self.line_search_beta 
            else: 
                L_k = current_L_k 
                if self.verbose:
                    print(f"FISTA: Line search reached max_iter ({self.max_line_search_iter}) at main iter {iter_num+1}. Using L_k={L_k:.2e} (step={1.0/L_k:.2e}).")
            
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
            
            delta_x_num = torch.linalg.norm(x_new - x_old)
            delta_x_den = torch.linalg.norm(x_old) + 1e-9 
            delta_x = delta_x_num / delta_x_den
            
            if self.verbose:
                obj_val_approx = f_x_new 
                print(f"FISTA Iter {iter_num+1}/{self.max_iter}, L_k: {L_k:.2e}, Step: {1.0/L_k:.2e}, RelChange: {delta_x:.2e}, ApproxObjective: {obj_val_approx:.2e}")

            if delta_x < self.tol:
                if self.verbose:
                    print(f"FISTA converged at iter {iter_num+1} with relative change {delta_x:.2e} < tol {self.tol}.")
                x_old = x_new 
                break
            
            y_k_next = x_new + ((t_old - 1.0) / t_new) * (x_new - x_old)
            x_old = x_new.clone() 
            y_k = y_k_next.clone() 
            t_old = t_new
        
        else: 
            if self.verbose:
                print(f"FISTA reached max_iter ({self.max_iter}) without converging to tol {self.tol}. Last relative change: {delta_x:.2e}.")

        return x_old

class ADMM(Optimizer):
    """
    Alternating Direction Method of Multipliers (ADMM) optimizer.
    Solves problems of the form: 
        min_x 0.5 * ||Ax - y||_2^2 + g_prox(x) + sum_i g_quad_i(x)
    where g_prox is handled by a proximal operator, and g_quad_i are quadratic 
    (or have easily computable gradients and Hessian products for CG), such as GradientMatchingRegularizer.
    Standard ADMM splitting: 
        min_x,z 0.5 * ||Ax - y||_2^2 + g_prox(z) + sum_i g_quad_i(x) 
        subject to x - z = 0.
    """
    def __init__(self, rho=1.0, max_iter=50, tol_abs=1e-4, tol_rel=1e-3, 
                 verbose=False, cg_max_iter=10, cg_tol=1e-5):
        self.rho = rho
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.verbose = verbose
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol

    def _solve_x_update(self, system_op_for_cg, b_vector, initial_x):
        """ Solves system_op_for_cg(x) = b_vector using Conjugate Gradient (CG). """
        x_img_shape = initial_x.shape
        x = initial_x.clone().flatten() 
        b_flat = b_vector.flatten()
        
        def system_op_flat(v_flat):
            # Ensure input to system_op_for_cg is image-shaped and complex
            v_img_shaped = v_flat.reshape(x_img_shape).to(dtype=torch.complex64)
            return system_op_for_cg(v_img_shaped).flatten()

        r = b_flat - system_op_flat(x)
        p = r.clone()
        rs_old = torch.vdot(r, r).real 

        for i in range(self.cg_max_iter):
            Ap = system_op_flat(p)
            
            alpha_num = rs_old
            alpha_den = torch.vdot(p, Ap).real + 1e-12 
            alpha = alpha_num / alpha_den
            
            x = x + alpha * p
            r_new = r - alpha * Ap
            rs_new = torch.vdot(r_new, r_new).real

            if torch.sqrt(rs_new) < self.cg_tol:
                if self.verbose and i > 0: 
                     pass 
                break
            
            beta_num = rs_new
            beta_den = rs_old + 1e-12
            beta = beta_num / beta_den
            
            p = r_new + beta * p
            r = r_new
            rs_old = rs_new
        else: 
            if self.verbose:
                pass
        
        return x.reshape(x_img_shape)

    # Adapted solve method signature
    def solve(self, k_space_data, forward_op, prox_regularizer, initial_guess=None, 
              quadratic_plus_prox_regularizers=None):
        A_op = forward_op 
        device = k_space_data.device if hasattr(k_space_data, 'device') else A_op.device
        image_shape = A_op.image_shape

        if initial_guess is None:
            # Ensure x_k is complex, as forward_op typically expects complex input for MRI
            x_k = torch.zeros(image_shape, dtype=torch.complex64, device=device)
        else:
            x_k = initial_guess.clone().to(device=device, dtype=torch.complex64)
        
        z_k = x_k.clone()
        u_k = torch.zeros_like(x_k, dtype=torch.complex64, device=device) 

        At_y = A_op.op_adj(k_space_data) 

        if quadratic_plus_prox_regularizers is None:
            quadratic_plus_prox_regularizers = []

        if self.verbose:
            print(f"ADMM Optimizer Started. Device: {device}")
            print(f"Params: rho={self.rho}, max_iter={self.max_iter}, tol_abs={self.tol_abs}, tol_rel={self.tol_rel}")
            print(f"CG Params: max_iter={self.cg_max_iter}, tol={self.cg_tol}")
            if quadratic_plus_prox_regularizers:
                print(f"Using {len(quadratic_plus_prox_regularizers)} quadratic_plus_prox regularizer(s).")

        for iter_num in range(self.max_iter):
            # x-update: solve (A^H A + rho I + sum_qr H_qr_i) x = A^H y + rho(z_k - u_k) + sum_qr b_qr_i
            
            # Construct RHS for x-update
            b_x_update = At_y + self.rho * (z_k - u_k)
            for qr_reg in quadratic_plus_prox_regularizers:
                if hasattr(qr_reg, 'get_rhs_term'):
                    # Ensure terms are compatible (e.g. complex + real results in complex)
                    b_x_update = b_x_update + qr_reg.get_rhs_term().to(b_x_update.dtype)
                else: # Should be GradientMatchingRegularizer type
                    raise AttributeError(f"Regularizer {type(qr_reg)} in quadratic_plus_prox_regularizers "
                                         "must have a 'get_rhs_term' method.")

            # Construct system operator for CG: v -> (A^H A + rho I + sum_qr H_qr_i)v
            def current_system_op_cg(v_img):
                # Base operator: (A^H A + rho I) v
                # Ensure v_img is complex for forward_op
                res = A_op.op_adj(A_op.op(v_img.to(torch.complex64))) + self.rho * v_img.to(torch.complex64)
                
                # Add terms from quadratic regularizers: sum_qr (lambda_qr * G^T G v)
                for qr_reg_lhs in quadratic_plus_prox_regularizers:
                    if hasattr(qr_reg_lhs, 'get_lhs_operator_product'):
                         # get_lhs_operator_product typically expects float and returns float.
                         # If res is complex, ensure result is compatible.
                         # GradientMatchingRegularizer's product is real.
                        lhs_prod = qr_reg_lhs.get_lhs_operator_product(v_img) # v_img is complex here
                        res = res + lhs_prod.to(res.dtype) # Cast to complex if needed
                    else:
                        raise AttributeError(f"Regularizer {type(qr_reg_lhs)} in quadratic_plus_prox_regularizers "
                                             "must have a 'get_lhs_operator_product' method.")
                return res
            
            x_k_plus_1 = self._solve_x_update(current_system_op_cg, b_x_update, x_k)

            # z-update: z_k+1 = prox_g(x_k+1 + u_k, 1.0 / rho)
            if prox_regularizer is not None:
                z_k_plus_1 = prox_regularizer.proximal_operator(x_k_plus_1 + u_k, 1.0 / self.rho)
            else: 
                z_k_plus_1 = x_k_plus_1 + u_k # If no prox_regularizer, g_prox(z) = 0

            # u-update (scaled dual variable):
            u_k_plus_1 = u_k + x_k_plus_1 - z_k_plus_1

            # Convergence Checks
            r_primal_norm = torch.linalg.norm((x_k_plus_1 - z_k_plus_1).flatten())
            s_dual_norm = torch.linalg.norm((self.rho * (z_k_plus_1 - z_k)).flatten()) 

            sqrt_n_term = torch.sqrt(torch.tensor(x_k_plus_1.numel(), dtype=torch.float32, device=device))
            
            eps_pri_term_max = torch.maximum(torch.linalg.norm(x_k_plus_1.flatten()), torch.linalg.norm(-z_k_plus_1.flatten()))
            eps_pri = sqrt_n_term * self.tol_abs + self.tol_rel * eps_pri_term_max
            
            eps_dual = sqrt_n_term * self.tol_abs + self.tol_rel * torch.linalg.norm((self.rho * u_k_plus_1).flatten())

            if self.verbose:
                print(f"ADMM Iter {iter_num+1}/{self.max_iter}: "
                      f"r_primal={r_primal_norm:.2e} (eps_pri={eps_pri:.2e}), "
                      f"s_dual={s_dual_norm:.2e} (eps_dual={eps_dual:.2e})")

            if r_primal_norm < eps_pri and s_dual_norm < eps_dual:
                if self.verbose:
                    print(f"ADMM converged at iter {iter_num+1}.")
                x_k = x_k_plus_1 
                break
            
            x_k = x_k_plus_1.clone() 
            z_k = z_k_plus_1.clone()
            u_k = u_k_plus_1.clone()
        else: 
            if self.verbose:
                print(f"ADMM reached max_iter ({self.max_iter}) without converging.")
                
        return x_k
