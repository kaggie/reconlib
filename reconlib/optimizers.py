"""Module for defining Optimizer classes for MRI reconstruction."""

import torch
from abc import ABC, abstractmethod
import numpy as np # Added for OSEM
from typing import Optional # Added for OSEM
# Import GradientMatchingRegularizer for type hinting if needed, though not strictly necessary for runtime
# from reconlib.regularizers import GradientMatchingRegularizer # Example for clarity
from reconlib.geometry import SystemMatrix, ScannerGeometry # Added ScannerGeometry for OSEM subset SM
from reconlib.operators import Operator
from reconlib.regularizers.base import Regularizer
from reconlib.regularizers.common import NonnegativityConstraint # Added for OSEM

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


class OrderedSubsetsExpectationMaximization(Optimizer):
    """
    Ordered Subsets Expectation Maximization (OSEM) algorithm for PET reconstruction.
    """
    def __init__(self,
                 system_matrix: SystemMatrix,
                 num_subsets: int,
                 num_iterations: int,
                 device: str = 'cpu',
                 nonnegativity_constraint: Optional[NonnegativityConstraint] = None,
                 epsilon: float = 1e-9,
                 verbose: bool = False):
        """
        Initializes the OSEM optimizer.

        Args:
            system_matrix (SystemMatrix): The system matrix modeling the PET scanner physics.
            num_subsets (int): The number of ordered subsets to divide the projection data into.
            num_iterations (int): The total number of iterations to perform.
            device (str): The computational device ('cpu' or 'cuda').
            nonnegativity_constraint (Optional[NonnegativityConstraint]): Non-negativity constraint operator.
                                                                        If None, one is created.
            epsilon (float): Small constant for numerical stability in divisions.
            verbose (bool): If True, prints progress information.
        """
        self.system_matrix = system_matrix
        self.num_subsets = num_subsets
        self.num_iterations = num_iterations
        self.device = device
        self.epsilon = epsilon
        self.verbose = verbose

        if nonnegativity_constraint is None:
            self.nonnegativity_constraint = NonnegativityConstraint()
        else:
            self.nonnegativity_constraint = nonnegativity_constraint

        # Ensure system_matrix components are on the correct device
        if hasattr(self.system_matrix, 'to') and callable(getattr(self.system_matrix, 'to')):
            self.system_matrix.to(self.device)
        elif hasattr(self.system_matrix, 'projector_op') and \
             hasattr(self.system_matrix.projector_op, 'to') and \
             callable(getattr(self.system_matrix.projector_op, 'to')):
            self.system_matrix.projector_op.to(self.device)

    def _get_subset_system_matrix(self, subset_angle_indices: np.ndarray) -> SystemMatrix:
        """
        Creates a new SystemMatrix for a given subset of angles.
        This is a temporary approach due to current SystemMatrix limitations.
        """
        original_scanner_geom = self.system_matrix.scanner_geometry
        
        # Create new ScannerGeometry for the subset
        subset_scanner_geom = ScannerGeometry(
            detector_positions=original_scanner_geom.detector_positions, # Assuming this doesn't change per subset
            angles=original_scanner_geom.angles[subset_angle_indices],
            detector_size=original_scanner_geom.detector_size,
            geometry_type=original_scanner_geom.geometry_type,
            n_detector_pixels=original_scanner_geom.n_detector_pixels
        )
        
        # Create new SystemMatrix for this subset geometry
        subset_system_matrix = SystemMatrix(
            scanner_geometry=subset_scanner_geom,
            img_size=self.system_matrix.img_size,
            device=self.device
        )
        return subset_system_matrix

    def _calculate_sensitivity_image_subset(self, subset_system_matrix: SystemMatrix, subset_projection_shape: tuple) -> torch.Tensor:
        """
        Calculates the sensitivity image for a given subset by backprojecting ones.
        subset_projection_shape is (batch, channels, num_subset_angles, num_detectors)
        """
        # Create a sinogram of ones matching the subset's projection data shape
        # The shape required by SystemMatrix.backward_project might depend on its internal operator.
        # For PETForwardProjection, it expects (batch, 1, n_angles, n_detectors)
        # For IRadon, it expects (batch, 1, n_angles, n_rays_per_proj)
        # Assuming subset_projection_shape[0] is batch, subset_projection_shape[2] is num_subset_angles,
        # and subset_projection_shape[3] is num_detectors/rays.
        
        # We need the expected shape by the projector, not necessarily the input `subset_projection_shape`
        # if it has extra channels. For sensitivity, it's usually a single channel.
        num_subset_angles = subset_system_matrix.scanner_geometry.angles.shape[0]
        num_detectors = subset_system_matrix.scanner_geometry.n_detector_pixels
        
        # Assuming batch size of 1 for sensitivity calculation, or it should match current_image_estimate's batch
        # Let's assume the sensitivity image is calculated for a single batch item, or it broadcasts.
        # For now, let's take the batch size from the subset_projection_shape passed.
        batch_size = subset_projection_shape[0]
        
        # The number of channels for the 'ones' tensor should typically be 1 for sensitivity.
        # The projector op inside system_matrix will handle this.
        # If system_matrix.backward_project expects (B, C, H, W) for projections,
        # then C should usually be 1 for the 'ones' tensor.
        # Let's assume the projector can handle a single channel input for this.
        ones_sinogram = torch.ones(batch_size, 1, num_subset_angles, num_detectors,
                                   dtype=torch.float32, device=self.device) # Use float32 for calculations

        sensitivity_image_subset = subset_system_matrix.backward_project(ones_sinogram)
        sensitivity_image_subset = sensitivity_image_subset + self.epsilon # Avoid division by zero
        return sensitivity_image_subset

    def reconstruct(self, projection_data: torch.Tensor, initial_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs OSEM reconstruction.

        The OSEM algorithm iteratively updates the image estimate using subsets of projection data.
        A typical update rule for a subset 's' is:
        image_new = image_old * (system_matrix_subset_s^T * (projection_data_subset_s / (system_matrix_subset_s * image_old + epsilon)))
                    / (sensitivity_image_subset + epsilon)
        where sensitivity_image_subset = system_matrix_subset_s^T * 1.

        Args:
            projection_data (torch.Tensor): The full set of projection data (sinogram).
                                            Expected shape (batch, channels, num_total_angles, num_detectors).
            initial_image (Optional[torch.Tensor]): An initial guess for the image.
                                                    If None, a uniform positive image is created.
                                                    Expected shape (batch, channels, height, width).
        Returns:
            torch.Tensor: The reconstructed image.
        """
        # --- Input Validation and Initialization ---
        if not isinstance(projection_data, torch.Tensor):
            raise ValueError("projection_data must be a PyTorch tensor.")
        projection_data = projection_data.to(self.device)

        if initial_image is not None:
            if not isinstance(initial_image, torch.Tensor):
                raise ValueError("initial_image must be a PyTorch tensor if provided.")
            current_image_estimate = initial_image.clone().to(self.device)
        else:
            # Determine image_shape from system_matrix
            try:
                img_h, img_w = self.system_matrix.img_size
                # Assume batch size and channels from projection_data if possible, else default
                batch_size = projection_data.shape[0]
                num_channels = 1 # PET images are typically single-channel
                current_image_estimate = torch.ones(batch_size, num_channels, img_h, img_w,
                                                    dtype=torch.float32, device=self.device)
            except AttributeError as e:
                raise ValueError(f"Could not determine image size from system_matrix: {e}. "
                                 "Ensure system_matrix.img_size is set or provide initial_image.")

        # Ensure initial image is positive
        current_image_estimate[current_image_estimate <= 0] = self.epsilon

        # --- Subset Definition ---
        all_angles = self.system_matrix.scanner_geometry.angles
        num_total_angles = len(all_angles)
        if self.num_subsets <= 0 or self.num_subsets > num_total_angles:
            raise ValueError(f"Number of subsets ({self.num_subsets}) must be positive and not exceed total angles ({num_total_angles}).")

        subset_angle_indices_list = [[] for _ in range(self.num_subsets)]
        for i in range(num_total_angles):
            subset_angle_indices_list[i % self.num_subsets].append(i)
        
        # Convert lists to numpy arrays for easier indexing later if needed by ScannerGeometry
        subset_angle_indices_list = [np.array(indices) for indices in subset_angle_indices_list]

        if self.verbose:
            print(f"Starting OSEM Reconstruction: {self.num_iterations} iterations, {self.num_subsets} subsets.")
            print(f"Initial image estimate shape: {current_image_estimate.shape}, device: {current_image_estimate.device}")
            print(f"Projection data shape: {projection_data.shape}, device: {projection_data.device}")

        # --- Main Iteration Loop ---
        for iter_num in range(self.num_iterations):
            if self.verbose:
                print(f"--- Iteration {iter_num + 1}/{self.num_iterations} ---")

            # --- Subset Loop ---
            for subset_idx in range(self.num_subsets):
                current_subset_angle_indices = subset_angle_indices_list[subset_idx]
                if len(current_subset_angle_indices) == 0:
                    if self.verbose: print(f"Skipping empty subset {subset_idx + 1}/{self.num_subsets}")
                    continue
                
                if self.verbose:
                    print(f"  Processing subset {subset_idx + 1}/{self.num_subsets} with {len(current_subset_angle_indices)} angles.")

                # a. Get current subset's projection data
                # Assuming projection_data is (batch, channels, num_total_angles, num_detectors)
                # And angles are the 3rd dimension (index 2)
                measured_projections_subset = projection_data.index_select(2, torch.tensor(current_subset_angle_indices, device=self.device))

                # b. Create SystemMatrix for the current subset
                subset_system_matrix = self._get_subset_system_matrix(current_subset_angle_indices)
                
                # c. Calculate Sensitivity Image for the subset
                # The shape of measured_projections_subset is (batch, channels, num_subset_angles, num_detectors)
                sensitivity_image_subset = self._calculate_sensitivity_image_subset(subset_system_matrix, measured_projections_subset.shape)

                # d. Forward Projection
                # Ensure image estimate is positive before projection (though non-negativity is applied later too)
                current_image_estimate[current_image_estimate <= self.epsilon] = self.epsilon
                estimated_projections_subset = subset_system_matrix.forward_project(current_image_estimate)
                estimated_projections_subset = estimated_projections_subset + self.epsilon # Avoid division by zero

                # e. Ratio and Backprojection
                ratio_subset = measured_projections_subset / estimated_projections_subset
                correction_factor_subset = subset_system_matrix.backward_project(ratio_subset)

                # f. Image Update
                current_image_estimate = current_image_estimate * (correction_factor_subset / sensitivity_image_subset)
                
                # g. Apply Non-negativity
                current_image_estimate = self.nonnegativity_constraint.apply(current_image_estimate)


            if self.verbose:
                img_norm = torch.linalg.norm(current_image_estimate.flatten()).item()
                print(f"End of Iteration {iter_num + 1}. Image norm: {img_norm:.4e}")
        
        if self.verbose:
            print("OSEM Reconstruction Finished.")
        return current_image_estimate

    def solve(self, k_space_data: torch.Tensor, forward_op: SystemMatrix, regularizer: Optional[Regularizer] = None, initial_guess: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calls the OSEM reconstruct method.
        Adapts OSEM to the general Optimizer interface.
        'k_space_data' is projection_data, 'forward_op' is the SystemMatrix.
        """
        if regularizer is not None:
            print("Warning: Basic OSEM (via `solve`) does not typically use a regularizer. It will be ignored.")
        
        if forward_op is not self.system_matrix:
            print("Warning: The `forward_op` passed to OSEM.solve() is different from the "
                  "`system_matrix` it was initialized with. Using the initialized `system_matrix`.")

        return self.reconstruct(projection_data=k_space_data, initial_image=initial_guess)


class PenalizedLikelihoodReconstruction(Optimizer):
    """
    Framework for Penalized Likelihood Reconstruction using iterative optimizers
    like FISTA or ADMM, but with a custom data fidelity term (e.g., Poisson likelihood).
    """
    def __init__(self,
                 system_matrix: SystemMatrix,
                 regularizer: Regularizer,
                 optimizer_choice: str = 'fista',
                 optimizer_params: dict = None,
                 data_fidelity: str = 'poisson', # 'poisson' or 'gaussian'
                 device: str = 'cpu',
                 epsilon: float = 1e-9): # Epsilon for Poisson gradient
        """
        Initializes the PenalizedLikelihoodReconstruction optimizer.
        """
        self.system_matrix = system_matrix
        self.regularizer = regularizer
        self.optimizer_choice = optimizer_choice.lower()
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.data_fidelity = data_fidelity.lower()
        self.device = device
        self.epsilon = epsilon

        if hasattr(self.system_matrix, 'to') and callable(getattr(self.system_matrix, 'to')):
            self.system_matrix.to(self.device)
        elif hasattr(self.system_matrix, 'projector_op') and \
             hasattr(self.system_matrix.projector_op, 'to') and \
             callable(getattr(self.system_matrix.projector_op, 'to')):
            self.system_matrix.projector_op.to(self.device)

        if hasattr(self.regularizer, 'to') and callable(getattr(self.regularizer, 'to')):
            self.regularizer.to(self.device)

        if self.optimizer_choice == 'fista':
            self.internal_optimizer = FISTA(**self.optimizer_params)
        elif self.optimizer_choice == 'admm':
            self.internal_optimizer = ADMM(**self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer_choice: {optimizer_choice}. Choose 'fista' or 'admm'.")

    def _data_fidelity_gradient(self, current_image: torch.Tensor, projection_data: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradient of the data fidelity (negative log-likelihood) term.
        This is the gradient of -log(L(projection_data | current_image)).

        Args:
            current_image (torch.Tensor): The current image estimate.
            projection_data (torch.Tensor): The measured projection data.

        Returns:
            torch.Tensor: The gradient of the negative log-likelihood w.r.t. current_image.

        Formulas:
        - Poisson: grad = system_matrix.op_adj(1 - projection_data / (system_matrix.op(current_image) + epsilon))
        - Gaussian: grad = system_matrix.op_adj(system_matrix.op(current_image) - projection_data)
        """
        if current_image.device.type != self.device:
            current_image = current_image.to(self.device)
        if projection_data.device.type != self.device:
            projection_data = projection_data.to(self.device)
        
        # Ensure internal components of system_matrix are on device (if applicable)
        if hasattr(self.system_matrix, 'projector_op') and \
           hasattr(self.system_matrix.projector_op, 'device') and \
           self.system_matrix.projector_op.device != self.device:
             if hasattr(self.system_matrix.projector_op, 'to') and \
                callable(getattr(self.system_matrix.projector_op, 'to')):
                 self.system_matrix.projector_op.to(self.device)

        print(f"Placeholder: Would calculate data fidelity gradient for '{self.data_fidelity}' type.")
        raise NotImplementedError("`_data_fidelity_gradient` is not yet implemented. "
                                  "The specific gradient formula needs to be applied based on self.data_fidelity.")

    def solve(self, k_space_data: torch.Tensor, 
              forward_op: SystemMatrix, 
              regularizer: Regularizer = None, 
              initial_guess: torch.Tensor = None) -> torch.Tensor:
        """
        Solves the penalized likelihood reconstruction problem.
        This method needs to integrate `_data_fidelity_gradient` with the chosen
        `self.internal_optimizer` and `self.regularizer`.
        """
        if forward_op is not self.system_matrix:
            print("Warning: `forward_op` in solve() differs from `system_matrix` in __init__. Using initialized one.")
        if regularizer is not None and regularizer is not self.regularizer:
            # Note: The regularizer passed to solve() is often the one FISTA/ADMM directly use.
            # Here, self.regularizer is passed to FISTA/ADMM during their solve call.
            print("Warning: `regularizer` in solve() might conflict with `regularizer` in __init__. "
                  "The one from __init__ will be used by the internal optimizer normally.")

        print(f"Placeholder: Would use {self.optimizer_choice} with {self.data_fidelity} likelihood.")
        raise NotImplementedError(
            "Integrating custom data fidelity gradient with FISTA/ADMM's `solve` method is non-trivial "
            "and requires careful adaptation of the optimizer's internal gradient computation. "
            "This `solve` method for PenalizedLikelihoodReconstruction is not yet implemented."
        )

# TODO: Further refine PenalizedLikelihoodReconstruction, especially the solve method's integration
# with FISTA/ADMM and the custom gradient.
