"""Module for defining Optimizer classes for MRI reconstruction."""

import torch
from abc import ABC, abstractmethod

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
            regularizer: The regularizer (instance of reconlib.regularizers.Regularizer).
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
        """
        Initializes the FISTA optimizer.

        Args:
            max_iter: Maximum number of iterations.
            tol: Tolerance for stopping criterion based on relative change in solution.
            verbose: If True, prints progress information.
            line_search_beta: Factor to modify L_k during line search (L_k_new = L_k / line_search_beta).
                              Typically > 1 for increasing L_k, or < 1 if L_k_new = L_k * beta.
                              Prompt implies L_k / 0.5, so L_k * 2.0.
            max_line_search_iter: Maximum iterations for backtracking line search.
            initial_step_size: Initial estimate for L_k (Lipschitz constant of grad_f).
                               A smaller L_k means a larger initial step (1/L_k).
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.line_search_beta = line_search_beta # Should be > 1 if L_k_new = L_k * beta, or < 1 if L_k_new = L_k / beta for increase
                                                 # Given prompt L_k = L_k / self.line_search_beta (0.5) -> L_k * 2
        self.max_line_search_iter = max_line_search_iter
        self.initial_L_k_estimate = initial_step_size # This is L_k, not 1/L_k

    def solve(self, k_space_data, forward_op, regularizer, initial_guess=None):
        """
        Solves the optimization problem using FISTA.
        min_x { 0.5 * || A(x) - y ||_2^2 + g(x) }
        where A is forward_op, y is k_space_data, g is regularization term handled by regularizer.prox.
        """
        device = k_space_data.device # Assuming k_space_data is a tensor on the target device
                                     # Or forward_op.device if available

        # Initialization
        if initial_guess is None:
            # MRI images are typically complex. forward_op.op expects complex.
            x_old = torch.zeros(forward_op.image_shape, dtype=torch.complex64, device=device)
        else:
            x_old = initial_guess.clone().to(device=device, dtype=torch.complex64)

        y_k = x_old.clone()
        t_old = 1.0
        L_k = self.initial_L_k_estimate # Current estimate of Lipschitz constant for grad_f

        if self.verbose:
            print(f"FISTA Optimizer Started. Device: {device}")
            print(f"Params: max_iter={self.max_iter}, tol={self.tol}, initial_L_k={L_k}, line_search_beta (for L_k increase factor)={1/self.line_search_beta}")


        for iter_num in range(self.max_iter):
            # Gradient of data fidelity term: grad_f(y_k) = A^H(A(y_k) - k_space_data)
            # Ensure y_k is complex for forward_op
            grad_y_k = forward_op.op_adj(forward_op.op(y_k.to(torch.complex64)) - k_space_data)

            # Backtracking Line Search for step size (1/L_k)
            current_L_k = L_k # Use a temporary L_k for this iteration's line search
            for ls_iter in range(self.max_line_search_iter):
                step = 1.0 / current_L_k # step_size for prox is 1/L_k
                
                # Proximal update: x_new = prox_g(y_k - step * grad_y_k, step)
                # The 'step' (1/L_k) is passed to regularizer.prox, which uses it with its lambda_param.
                x_new = regularizer.prox(y_k - step * grad_y_k, step)

                # Line search condition: f(x_new) <= f(y_k) + <grad_f(y_k), x_new - y_k> + (L_k/2) * ||x_new - y_k||_2^2
                # where f(z) = 0.5 * ||A(z) - k_space_data||_2^2
                
                # Ensure inputs to forward_op are complex
                f_yk_op_output = forward_op.op(y_k.to(torch.complex64))
                f_x_new_op_output = forward_op.op(x_new.to(torch.complex64))

                f_yk = 0.5 * torch.linalg.norm(f_yk_op_output - k_space_data)**2
                f_x_new = 0.5 * torch.linalg.norm(f_x_new_op_output - k_space_data)**2
                
                diff_x_y = x_new - y_k # No need to flatten for norm, but useful for dot product structure
                
                # grad_term = <grad_f(y_k), x_new - y_k>
                grad_term = torch.real(torch.vdot(grad_y_k.flatten(), diff_x_y.flatten()))
                
                # quadratic_term = (L_k/2) * ||x_new - y_k||_2^2
                quadratic_term = (current_L_k / 2.0) * torch.linalg.norm(diff_x_y)**2
                
                if f_x_new <= f_yk + grad_term + quadratic_term:
                    L_k = current_L_k # Persist L_k if step size is accepted
                    break # Found suitable L_k (step size)
                
                current_L_k = current_L_k / self.line_search_beta # Increase L_k (decrease step size)
                                                                  # e.g. current_L_k * (1/0.5) = current_L_k * 2
            else: # Max line search iterations reached
                L_k = current_L_k # Use the last L_k tried
                if self.verbose:
                    print(f"FISTA: Line search reached max_line_search_iter ({self.max_line_search_iter}) at main iter {iter_num+1}. Using L_k={L_k:.2e} (step={1.0/L_k:.2e}).")
            
            # Update momentum terms
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
            
            # Check for convergence using relative change in x
            # Use norm of x_old.flatten() for denominator for stability
            delta_x_num = torch.linalg.norm(x_new - x_old)
            delta_x_den = torch.linalg.norm(x_old) + 1e-9 # Avoid division by zero
            delta_x = delta_x_num / delta_x_den
            
            if self.verbose:
                obj_val_approx = f_x_new # This is data fidelity part. Full objective would include reg term.
                print(f"FISTA Iter {iter_num+1}/{self.max_iter}, L_k: {L_k:.2e}, Step: {1.0/L_k:.2e}, RelChange: {delta_x:.2e}, ApproxObjective: {obj_val_approx:.2e}")

            if delta_x < self.tol:
                if self.verbose:
                    print(f"FISTA converged at iter {iter_num+1} with relative change {delta_x:.2e} < tol {self.tol}.")
                x_old = x_new # Final update before breaking
                break
            
            # Prepare for next iteration
            y_k_next = x_new + ((t_old - 1.0) / t_new) * (x_new - x_old)
            x_old = x_new.clone() # x_new is from prox, can be used directly
            y_k = y_k_next.clone() # y_k_next is derived, clone for safety
            t_old = t_new
        
        else: # Loop finished without break (i.e., max_iter reached)
            if self.verbose:
                print(f"FISTA reached max_iter ({self.max_iter}) without converging to tol {self.tol}. Last relative change: {delta_x:.2e}.")

        return x_old
