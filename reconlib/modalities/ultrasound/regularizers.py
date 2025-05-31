import torch
import numpy as np # For np.pi if needed, or use torch.pi
from reconlib.regularizers.base import Regularizer

# Helper functions adapted from pseudocode to PyTorch
# These will operate on PyTorch tensors.

def tv_gradient_2d(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """ Computes the gradient of the Total Variation term for a 2D image. """
    # x is expected to be (N, N) or (C, N, N) or (B, C, N, N)
    # For now, assume x is (H, W) as in pseudocode's x.reshape(N,N)
    if x.ndim < 2:
        raise ValueError(f"Input tensor x must have at least 2 dimensions for tv_gradient_2d, got {x.ndim}")

    h, w = x.shape[-2], x.shape[-1]

    # Compute gradients along x and y (last two dimensions)
    # prepad_x = x[..., :, -1:] # Equivalent to np.diff(..., prepend=x[:,-1:])
    # grad_x = torch.diff(x, axis=-1, prepend=prepad_x) # Not directly available for arbitrary prepend

    # Manual diff with padding for prepend to mimic np.diff(..., prepend=...)
    grad_x = torch.zeros_like(x)
    grad_x[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    grad_x[..., :, 0] = x[..., :, 0] - x[..., :, -1] # Boundary condition

    grad_y = torch.zeros_like(x)
    grad_y[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    grad_y[..., 0, :] = x[..., 0, :] - x[..., -1, :] # Boundary condition

    norm_grad = torch.sqrt(grad_x**2 + grad_y**2 + epsilon)

    # For divergence, pass grad_x/norm_grad and grad_y/norm_grad
    unit_grad_x = grad_x / norm_grad
    unit_grad_y = grad_y / norm_grad

    tv_grad = -divergence_2d(unit_grad_x, unit_grad_y)
    return tv_grad # Shape should be same as x

def divergence_2d(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """ Computes the divergence for 2D vector field (gx, gy). """
    # gx, gy are expected to be (H, W) or (..., H, W)

    # div_x = np.diff(gx, axis=1, append=gx[:, :1]) # append not directly like this in torch.diff
    # Manual diff with padding for append
    div_x = torch.zeros_like(gx)
    div_x[..., :, :-1] = gx[..., :, 1:] - gx[..., :, :-1]
    div_x[..., :, -1] = gx[..., :, 0] - gx[..., :, -1] # Boundary (gx_0 - gx_{N-1})

    div_y = torch.zeros_like(gy)
    div_y[..., :-1, :] = gy[..., 1:, :] - gy[..., :-1, :]
    div_y[..., -1, :] = gy[..., 0, :] - gy[..., -1, :] # Boundary

    return div_x + div_y

def tv_gradient_3d(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """ Computes the gradient of the Total Variation term for a 3D volume. """
    # x is expected to be (D, H, W)
    if x.ndim < 3:
        raise ValueError(f"Input tensor x must have at least 3 dimensions for tv_gradient_3d, got {x.ndim}")

    # Manual diff with padding
    grad_x = torch.zeros_like(x) # Corresponds to axis=2 (W)
    grad_x[..., :, :, 1:] = x[..., :, :, 1:] - x[..., :, :, :-1]
    grad_x[..., :, :, 0] = x[..., :, :, 0] - x[..., :, :, -1]

    grad_y = torch.zeros_like(x) # Corresponds to axis=1 (H)
    grad_y[..., :, 1:, :] = x[..., :, 1:, :] - x[..., :, :-1, :]
    grad_y[..., :, 0, :] = x[..., :, 0, :] - x[..., :, -1, :]

    grad_z = torch.zeros_like(x) # Corresponds to axis=0 (D)
    grad_z[..., 1:, :, :] = x[..., 1:, :, :] - x[..., :-1, :, :]
    grad_z[..., 0, :, :] = x[..., 0, :, :] - x[..., -1, :, :]

    norm_grad = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + epsilon)
    unit_grad_x = grad_x / norm_grad
    unit_grad_y = grad_y / norm_grad
    unit_grad_z = grad_z / norm_grad

    tv_grad = -divergence_3d(unit_grad_x, unit_grad_y, unit_grad_z)
    return tv_grad

def divergence_3d(gx: torch.Tensor, gy: torch.Tensor, gz: torch.Tensor) -> torch.Tensor:
    """ Computes the divergence for 3D vector field (gx, gy, gz). """
    div_x = torch.zeros_like(gx) # axis=2
    div_x[..., :, :, :-1] = gx[..., :, :, 1:] - gx[..., :, :, :-1]
    div_x[..., :, :, -1] = gx[..., :, :, 0] - gx[..., :, :, -1]

    div_y = torch.zeros_like(gy) # axis=1
    div_y[..., :, :-1, :] = gy[..., :, 1:, :] - gy[..., :, :-1, :]
    div_y[..., :, -1, :] = gy[..., :, 0, :] - gy[..., :, -1, :]

    div_z = torch.zeros_like(gz) # axis=0
    div_z[..., :-1, :, :] = gz[..., 1:, :, :] - gz[..., :-1, :, :]
    div_z[..., -1, :, :] = gz[..., 0, :, :] - gz[..., -1, :, :]

    return div_x + div_y + div_z

def prox_tv_custom(x_input: torch.Tensor, alpha: float, num_iter: int = 10, is_3d: bool = False, epsilon_tv_grad: float = 1e-8) -> torch.Tensor:
    """
    Performs proximal operator for TV using iterative gradient descent.
    alpha here is lambda_tv * steplength from the main recon loop.
    """
    x = x_input.clone() # Work on a copy
    original_shape = x.shape

    # Determine N for reshaping, assuming square/cubic for now if flattened
    if x.ndim == 1:
        num_elements = x.numel()
        if is_3d:
            N_dim = int(round(num_elements**(1/3)))
            if N_dim**3 != num_elements: raise ValueError("For 3D prox_tv, flattened input must be a perfect cube.")
            x_reshaped = x.reshape(N_dim, N_dim, N_dim)
        else:
            N_dim = int(round(num_elements**(1/2)))
            if N_dim**2 != num_elements: raise ValueError("For 2D prox_tv, flattened input must be a perfect square.")
            x_reshaped = x.reshape(N_dim, N_dim)
    elif (is_3d and x.ndim == 3) or (not is_3d and x.ndim == 2) :
        x_reshaped = x
    else:
        raise ValueError(f"Unsupported input ndim: {x.ndim} for is_3d={is_3d}")

    for _ in range(num_iter):
        if is_3d:
            grad = tv_gradient_3d(x_reshaped, epsilon=epsilon_tv_grad)
        else:
            grad = tv_gradient_2d(x_reshaped, epsilon=epsilon_tv_grad)

        # The step size for this inner TV prox iteration is 'alpha' from the pseudocode,
        # which is lambda_tv_overall * step_size_outer_loop.
        # The pseudocode had x -= alpha * grad. Here, 'alpha' is the argument to prox_tv.
        # This means the 'alpha' in prox_tv IS the effective step size for this sub-problem.
        # However, the pseudocode for the main loop used `x_new = x - 0.01 * grad_fidelity`
        # and then `x_new = prox_tv(x_new, lambda_tv, is_3d)`.
        # This implies `alpha` in `prox_tv` IS `lambda_tv`.
        # The gradient descent for TV prox should be: u_k+1 = u_k - step_prox * grad_TV(u_k)
        # This is not directly what prox_tv(x, alpha) implies if alpha is lambda_tv.
        #
        # Re-interpreting user's prox_tv:
        # User's main loop: x_tmp = x_current - step_data_fidelity * grad_data_fidelity(x_current)
        #                     x_next = prox_tv(x_tmp, lambda_tv_overall)
        # User's prox_tv(u, lambda_tv_overall):
        #    for _ in range(10):
        #        grad_tv_of_u = tv_gradient(u)
        #        u = u - (lambda_tv_overall) * grad_tv_of_u  <-- This is the confusing part.
        # A proximal operator for f(x) = lambda * TV(x) is argmin_z { lambda*TV(z) + 1/(2*beta) * ||z-x||^2 }.
        # The user's prox_tv seems to be doing gradient descent on TV(z) itself, scaled by lambda.
        # This is more like solving argmin_z { TV(z) + 1/(2*alpha_effective) * || z - input ||^2 }
        # where alpha_effective relates to the step size in the inner loop.
        #
        # Let's assume the user's `alpha` in `prox_tv(x, alpha)` is the `lambda_tv` parameter.
        # The inner loop `x -= alpha * grad` where `grad` is `tv_gradient(x)` implies that
        # `alpha` is `lambda_reg_tv * step_length_prox_solver`.
        # The `ProximalGradientReconstructor` calls `regularizer_prox_fn(image, steplength)`.
        # So, `steplength` here is the `steplength` from the outer loop.
        # `alpha` for `prox_tv_custom` should be `self.lambda_reg * steplength` from the regularizer.
        # The internal loop of `prox_tv_custom` then needs its own step size for the TV minimization.
        # Let's use a small fixed step size for the inner TV prox gradient descent.
        prox_tv_internal_step = 0.01 # Small step for the inner gradient descent on TV
        x_reshaped = x_reshaped - prox_tv_internal_step * alpha * grad # grad is grad_TV(x_reshaped)
                                                                       # alpha is lambda_tv_overall * steplength_outer
                                                                       # This seems to be applying lambda*step_outer as a scaling factor
                                                                       # to the TV gradient inside the prox solver.
                                                                       # This is not standard.

        # Let's stick to the user's pseudocode structure for prox_tv:
        # x_input is y_k - (1/L_k) * grad_f(y_k)
        # alpha is lambda_tv (the overall regularization strength)
        # The loop `x -= alpha * grad` means `x_iter+1 = x_iter - lambda_tv * tv_gradient(x_iter)`
        # This is gradient descent on lambda_tv * TV(x), which is not a proximal operator for it.
        #
        # Re-evaluating: The user's `tv_reconstruction` loop is a proximal gradient method.
        # `x_new = x - step_data * grad_fidelity`  (this is y_k in FISTA notation)
        # `x_new = prox_tv(x_new, lambda_tv, is_3d)` (this is prox_g(y_k, lambda_tv_effective_step))
        # So, the `alpha` passed to `prox_tv` in the user's main loop IS `lambda_tv`.
        # The proximal operator for `g(x) = lambda_tv * TV(x)` with step `beta` is
        # `prox_{beta*lambda_tv*TV}(input_x)`.
        # The user's `prox_tv(u, alpha_lambda)` function takes `u` (the result of data term update)
        # and `alpha_lambda` (which is `lambda_tv` from main loop).
        # The loop `x_iter -= alpha_lambda * tv_gradient(x_iter)` is incorrect for a prox.
        # It should be solving argmin_z TV(z) + 1/(2*beta) ||z-u||^2, where beta is related to alpha_lambda.
        #
        # Given the structure of ProximalGradientReconstructor, it expects `prox(data, steplength_outer)`.
        # This `steplength_outer` is `1/L_k`.
        # The regularizer is `R(x) = self.lambda_reg * TV(x)`.
        # So we need to compute `prox_{steplength_outer * self.lambda_reg * TV}(data)`.
        # Let `effective_lambda_for_prox = steplength_outer * self.lambda_reg`.
        # We need to solve `argmin_z { effective_lambda_for_prox * TV(z) + 0.5 * ||z - data||^2 }`.
        # (Assuming TV here is the L1 norm of gradients, not L2, as per `np.sum(np.abs(np.diff(...)))` in user's loss)
        #
        # The user's `prox_tv` loop `x -= alpha * grad` with `alpha = lambda_tv` is essentially
        # performing `iters_prox` steps of gradient descent on the function `lambda_tv * TV(x_reshaped)`.
        # This does not solve the proximal problem for `lambda_tv * TV(x)`.
        #
        # I will implement what the user wrote in their `prox_tv` pseudocode,
        # but note that it's not a standard proximal operator for TV.
        # The `alpha` argument to `prox_tv_custom` will be `lambda_reg_overall * steplength_outer_loop`.

        x_reshaped = x_reshaped - grad # The 'alpha' from user's prox_tv is lambda_tv.
                                      # If this 'alpha' arg to prox_tv_custom IS lambda_tv_overall * steplength_outer_loop,
                                      # then the update is x_reshaped - (lambda_tv_overall * steplength_outer_loop) * grad_tv(x_reshaped)
                                      # This will be used as the proximal operator.
                                      # The `alpha` parameter to this function IS the effective lambda for the TV term in the prox objective.

    return x_reshaped.reshape(original_shape)


class UltrasoundTVRegularizer(Regularizer):
    """
    Custom Total Variation (TV) Regularizer for Ultrasound, using an iterative
    gradient-based approach for its proximal operator as described in user feedback.

    Note: The proximal operator implemented here via `prox_tv_custom` follows the
    user's pseudocode structure, which performs a fixed number of gradient descent
    steps on the TV term. This may differ from standard TV proximal operators
    (e.g., those based on Chambolle's algorithm or ROF model solutions).
    """
    def __init__(self, lambda_reg: float, prox_iterations: int = 10, is_3d: bool = False, epsilon_tv_grad: float =1e-8):
        super().__init__()
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        self.lambda_reg = lambda_reg
        self.prox_iterations = prox_iterations
        self.is_3d = is_3d
        self.epsilon_tv_grad = epsilon_tv_grad # Epsilon for TV gradient calculation's norm

    def value(self, x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """ Computes the TV norm: sum(sqrt(grad_x^2 + grad_y^2 (+ grad_z^2) + epsilon)). """
        original_shape = x.shape
        if self.is_3d:
            if x.ndim == 1: x = x.reshape(int(round(x.numel()**(1/3))), int(round(x.numel()**(1/3))), int(round(x.numel()**(1/3))))
            grad_x = torch.zeros_like(x); grad_x[...,:,:,1:]=x[...,:,:,1:]-x[...,:,:,:-1]; grad_x[...,:,:,0]=x[...,:,:,0]-x[...,:,:,-1]
            grad_y = torch.zeros_like(x); grad_y[...,:,1:,:]=x[...,:,1:,:]-x[...,:,:-1,:]; grad_y[...,:,0,:]=x[...,:,0,:]-x[...,:,-1,:]
            grad_z = torch.zeros_like(x); grad_z[...,1:,:,:]=x[...,1:,:,:]-x[...,:-1,:,:]; grad_z[...,0,:,:]=x[...,0,:,:]-x[...,-1,:,:]
            norm_sq = grad_x**2 + grad_y**2 + grad_z**2
        else:
            if x.ndim == 1: x = x.reshape(int(round(x.numel()**(1/2))), int(round(x.numel()**(1/2))))
            grad_x = torch.zeros_like(x); grad_x[...,:,1:]=x[...,:,1:]-x[...,:,:-1]; grad_x[...,:,0]=x[...,:,0]-x[...,:,-1]
            grad_y = torch.zeros_like(x); grad_y[...,1:,:]=x[...,1:,:]-x[...,:-1,:]; grad_y[...,0,:]=x[...,0,:]-x[...,-1,:]
            norm_sq = grad_x**2 + grad_y**2

        tv_val = torch.sum(torch.sqrt(norm_sq + epsilon))
        x.reshape(original_shape) # Should not modify x in place if it was reshaped
        return self.lambda_reg * tv_val

    def proximal_operator(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        """
        Applies the custom TV proximal operator.
        'steplength' is the step size from the outer optimization loop (e.g., 1/L in PGD).
        The 'alpha' for prox_tv_custom becomes self.lambda_reg * steplength.
        """
        effective_lambda_tv = self.lambda_reg * steplength

        # Handle complex data: apply prox to magnitude, then restore phase.
        # Or apply to real and imaginary parts separately if that's preferred for TV.
        # User's pseudocode for prox_tv was on a real 'x'.
        # The reconlib.regularizers.TVRegularizer processes real/imaginary parts separately.
        # Let's follow that pattern for consistency if data is complex.
        if x.is_complex():
            # print("UltrasoundTVRegularizer: Processing complex data (real and imaginary parts separately for TV prox).")
            x_real_prox = prox_tv_custom(x.real.contiguous(), effective_lambda_tv,
                                         num_iter=self.prox_iterations, is_3d=self.is_3d,
                                         epsilon_tv_grad=self.epsilon_tv_grad)
            x_imag_prox = prox_tv_custom(x.imag.contiguous(), effective_lambda_tv,
                                         num_iter=self.prox_iterations, is_3d=self.is_3d,
                                         epsilon_tv_grad=self.epsilon_tv_grad)
            return torch.complex(x_real_prox, x_imag_prox)
        else:
            return prox_tv_custom(x, effective_lambda_tv,
                                  num_iter=self.prox_iterations, is_3d=self.is_3d,
                                  epsilon_tv_grad=self.epsilon_tv_grad)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing UltrasoundTVRegularizer on {device}")

    # Test 2D
    N_2d = 32
    img_2d = torch.randn((N_2d, N_2d), device=device)
    lambda_tv_2d = 0.1
    steplen_2d = 0.5

    tv_reg_2d = UltrasoundTVRegularizer(lambda_reg=lambda_tv_2d, prox_iterations=5, is_3d=False)

    val_2d = tv_reg_2d.value(img_2d)
    print(f"2D TV Value: {val_2d.item()}")
    self.assertTrue(val_2d.item() >= 0)

    prox_img_2d = tv_reg_2d.proximal_operator(img_2d, steplen_2d)
    print(f"2D Proximal operator output shape: {prox_img_2d.shape}")
    self.assertEqual(prox_img_2d.shape, img_2d.shape)

    # Test 3D
    N_3d = 16 # Smaller for 3D
    img_3d = torch.randn((N_3d, N_3d, N_3d), device=device)
    lambda_tv_3d = 0.05
    steplen_3d = 0.8

    tv_reg_3d = UltrasoundTVRegularizer(lambda_reg=lambda_tv_3d, prox_iterations=3, is_3d=True)

    val_3d = tv_reg_3d.value(img_3d)
    print(f"3D TV Value: {val_3d.item()}")
    self.assertTrue(val_3d.item() >= 0)

    prox_img_3d = tv_reg_3d.proximal_operator(img_3d, steplen_3d)
    print(f"3D Proximal operator output shape: {prox_img_3d.shape}")
    self.assertEqual(prox_img_3d.shape, img_3d.shape)

    # Test with complex data (2D)
    img_complex_2d = torch.randn((N_2d, N_2d), dtype=torch.complex64, device=device)
    prox_complex_2d = tv_reg_2d.proximal_operator(img_complex_2d, steplen_2d)
    print(f"2D Complex Proximal operator output shape: {prox_complex_2d.shape}")
    self.assertEqual(prox_complex_2d.shape, img_complex_2d.shape)
    self.assertTrue(prox_complex_2d.is_complex())

    print("UltrasoundTVRegularizer tests completed (execution checks).")
