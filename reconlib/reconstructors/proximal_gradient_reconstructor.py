import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

class ProximalGradientReconstructor(nn.Module):
    """
    A generic Proximal Gradient algorithm reconstructor.

    This module performs iterative reconstruction using a proximal gradient algorithm,
    suitable for problems of the form:
        argmin_x { || A(x) - y ||_2^2 + lambda * R(x) }
    where A is the forward operator, y is the measured k-space data,
    and R is a regularizer with a known proximal operator. The strength lambda
    is assumed to be incorporated into the regularizer_prox_fn or the regularizer object itself.
    """
    def __init__(self, 
                 iterations: int = 10, 
                 step_size: float = 0.1, 
                 initial_estimate_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], Callable], torch.Tensor]] = None,
                 verbose: bool = False, 
                 log_fn: Optional[Callable[[int, torch.Tensor, float, float], None]] = None,
                 data_fidelity_gradient_mode: str = 'l2', # New parameter
                 poisson_epsilon: float = 1e-9):           # New parameter
        """
        Args:
            iterations: Number of iterations to perform.
            step_size: Step size (learning rate) for the gradient descent update.
                       This step_size is also passed as `steplength` to the regularizer's proximal operator.
            initial_estimate_fn: An optional function to compute an initial image estimate.
                                 Signature: `fn(kspace_data, sensitivity_maps, adjoint_op_fn) -> initial_image`.
            verbose: If True, print iteration progress.
            log_fn: An optional function to log iteration metrics.
                    Signature: `fn(iter_num, current_image, change_norm, grad_norm)`.
            data_fidelity_gradient_mode: Mode for calculating data fidelity gradient.
                                         Options: 'l2', 'poisson_likelihood'.
            poisson_epsilon: Epsilon value for stabilizing Poisson likelihood gradient.
        """
        super().__init__()
        self.iterations = iterations
        self.step_size = step_size
        self.initial_estimate_fn = initial_estimate_fn
        self.verbose = verbose
        self.log_fn = log_fn
        self.data_fidelity_gradient_mode = data_fidelity_gradient_mode
        self.poisson_epsilon = poisson_epsilon

    def reconstruct(self, 
                    kspace_data: torch.Tensor, 
                    forward_op_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor], 
                    adjoint_op_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor], 
                    regularizer_prox_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]], # Made Optional
                    sensitivity_maps: Optional[torch.Tensor] = None, 
                    x_init: Optional[torch.Tensor] = None,
                    image_shape_for_zero_init: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Performs the proximal gradient reconstruction.

        Args:
            kspace_data: Measured k-space data. Shape depends on forward_op_fn.
            forward_op_fn: Callable for the forward operation A(x).
                           Signature: `fn(image_estimate, sensitivity_maps) -> kspace_coils`.
            adjoint_op_fn: Callable for the adjoint operation A^H(y).
                           Signature: `fn(kspace_coils, sensitivity_maps) -> image_estimate`.
            regularizer_prox_fn: Callable for the proximal operator of the regularizer R.
                                 The regularizer's strength (lambda) should be encapsulated within this function
                                 or the object providing this function. It will be called as `fn(image, steplength)`,
                                 where `steplength` is `self.step_size`.
            sensitivity_maps: Coil sensitivity maps. Optional.
            x_init: An optional initial estimate for the image.
            image_shape_for_zero_init: Required if x_init and initial_estimate_fn are None,
                                       and sensitivity_maps is also None or doesn't provide spatial dims.

        Returns:
            The reconstructed image.
        """
        device = kspace_data.device
        x_current: torch.Tensor

        if sensitivity_maps is not None:
            sensitivity_maps = sensitivity_maps.to(device)

        if x_init is not None:
            x_current = x_init.clone().to(device)
        elif self.initial_estimate_fn is not None:
            x_current = self.initial_estimate_fn(kspace_data, sensitivity_maps, adjoint_op_fn).to(device)
        else:
            _image_shape_inferred: Optional[Tuple[int, ...]] = None
            if image_shape_for_zero_init is not None:
                _image_shape_inferred = image_shape_for_zero_init
            elif sensitivity_maps is not None:
                _image_shape_inferred = sensitivity_maps.shape[1:]
            else:
                try:
                    if kspace_data.ndim > 1: 
                        dummy_k_shape = (kspace_data.shape[0], kspace_data.shape[-1])
                    else: 
                        dummy_k_shape = kspace_data.shape
                    dummy_k = torch.zeros(dummy_k_shape, dtype=kspace_data.dtype, device=device)
                    _image_shape_inferred = adjoint_op_fn(dummy_k, sensitivity_maps).shape
                except Exception as e:
                    raise ValueError(
                        "Cannot determine image shape for zero initialization. "
                        "Provide x_init, initial_estimate_fn, sensitivity_maps, or image_shape_for_zero_init. "
                        f"Error during adjoint call: {e}"
                    )
            
            if _image_shape_inferred is None:
                 raise ValueError("Image shape for zero initialization could not be determined.")

            x_current = torch.zeros(_image_shape_inferred, dtype=kspace_data.dtype, device=device)
            if torch.is_complex(kspace_data) and not torch.is_complex(x_current):
                 x_current = x_current.to(torch.complex64)

        for iter_num in range(self.iterations):
            estimated_mean_data = forward_op_fn(x_current, sensitivity_maps)
            
            if self.data_fidelity_gradient_mode == 'l2':
                k_resid = estimated_mean_data - kspace_data
                grad_data_fidelity = adjoint_op_fn(k_resid, sensitivity_maps)
            elif self.data_fidelity_gradient_mode == 'poisson_likelihood':
                # grad_likelihood_component: (1 - y/Ax)
                # Note: kspace_data here is the measured data 'y'
                grad_likelihood_component = (1.0 - kspace_data / (estimated_mean_data + self.poisson_epsilon))
                grad_data_fidelity = adjoint_op_fn(grad_likelihood_component, sensitivity_maps)
            else:
                raise ValueError(f"Unknown data_fidelity_gradient_mode: {self.data_fidelity_gradient_mode}")

            x_gradient_updated = x_current - self.step_size * grad_data_fidelity
            
            # Regularization Step: Pass self.step_size as steplength to the prox operator
            if regularizer_prox_fn is not None:
                x_next = regularizer_prox_fn(x_gradient_updated, self.step_size)
            else:
                x_next = x_gradient_updated.clone() # If no regularizer, proceed with gradient updated image
            
            if self.verbose or self.log_fn:
                with torch.no_grad():
                    current_norm = torch.norm(x_current)
                    if current_norm.item() == 0: 
                        change = torch.norm(x_next - x_current)
                    else:
                        change = torch.norm(x_next - x_current) / (current_norm + 1e-9)
                    grad_norm_val = torch.norm(grad_data_fidelity) # Use grad_data_fidelity here

                    if self.verbose:
                        print(f"Iter {iter_num + 1}/{self.iterations}, Step Size: {self.step_size:.2e}, "
                              f"Change: {change.item():.2e}, Grad Norm: {grad_norm_val.item():.2e}")
                    
                    if self.log_fn:
                        self.log_fn(iter_num=iter_num, current_image=x_next.clone(), 
                                    change=change.item(), grad_norm=grad_norm_val.item())
            
            x_current = x_next
            
        return x_current


if __name__ == '__main__':
    print("--- Testing ProximalGradientReconstructor (modified steplength handling) ---")
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {test_device}")

    image_shape_test = (32, 32)
    num_coils_test = 4
    k_len_test = 100

    def mock_forward_op_sense(image_2d: torch.Tensor, smaps_3d: Optional[torch.Tensor]) -> torch.Tensor:
        if smaps_3d is None:
            return torch.fft.fft2(image_2d).flatten()[:k_len_test].unsqueeze(0)
        coil_images = image_2d.unsqueeze(0) * smaps_3d
        kspace_coils = torch.stack([torch.fft.fft2(ci).flatten()[:k_len_test] for ci in coil_images])
        return kspace_coils

    def mock_adjoint_op_sense(kspace_coils_2d: torch.Tensor, smaps_3d: Optional[torch.Tensor]) -> torch.Tensor:
        C = kspace_coils_2d.shape[0] if kspace_coils_2d.ndim > 1 else 1
        H, W = image_shape_test
        if smaps_3d is None:
            k_data = kspace_coils_2d.squeeze(0) if kspace_coils_2d.ndim > 1 else kspace_coils_2d
            temp_k = torch.zeros(H*W, dtype=k_data.dtype, device=k_data.device)
            temp_k[:len(k_data)] = k_data
            return torch.fft.ifft2(temp_k.view(H,W))
        coil_images_adj = []
        for i in range(C):
            temp_k = torch.zeros(H*W, dtype=kspace_coils_2d.dtype, device=kspace_coils_2d.device)
            temp_k[:kspace_coils_2d.shape[1]] = kspace_coils_2d[i]
            coil_images_adj.append(torch.fft.ifft2(temp_k.view(H,W)))
        return torch.sum(torch.stack(coil_images_adj) * smaps_3d.conj(), dim=0)

    def mock_regularizer_prox_updated(image: torch.Tensor, steplength: float) -> torch.Tensor:
        # Assume this prox uses steplength and an internal lambda (e.g., lambda_val_internal = 0.01)
        # For this mock, let lambda_val_internal be effectively 0.01 for the example.
        # Threshold = steplength * lambda_val_internal
        threshold = steplength * 0.01 
        return torch.sign(image) * torch.clamp(torch.abs(image) - threshold, min=0.0)


    def mock_initial_estimate_fn(kspace_data_loc, sensitivity_maps_loc, adjoint_op_fn_loc):
        return adjoint_op_fn_loc(kspace_data_loc, sensitivity_maps_loc)

    reconstructor = ProximalGradientReconstructor(
        iterations=5, step_size=0.1, verbose=True, initial_estimate_fn=mock_initial_estimate_fn
    )
    
    kspace_data_test = torch.randn(num_coils_test, k_len_test, dtype=torch.complex64, device=test_device)
    sensitivity_maps_test = torch.ones(num_coils_test, *image_shape_test, dtype=torch.complex64, device=test_device)
    sensitivity_maps_test = sensitivity_maps_test / torch.sqrt(torch.sum(sensitivity_maps_test.abs()**2, dim=0, keepdim=True))

    reconstructed_image = reconstructor.reconstruct(
        kspace_data=kspace_data_test,
        sensitivity_maps=sensitivity_maps_test,
        forward_op_fn=mock_forward_op_sense,
        adjoint_op_fn=mock_adjoint_op_sense,
        regularizer_prox_fn=mock_regularizer_prox_updated # No regularizer_strength here
    )
    
    assert reconstructed_image.shape == image_shape_test
    print("\nProximalGradientReconstructor basic test (modified steplength) passed.")

    # Test zero initialization path (no regularizer_strength)
    reconstructor_zero_init = ProximalGradientReconstructor(iterations=2, step_size=0.1, verbose=False)
    reconstructed_image_zero_init = reconstructor_zero_init.reconstruct(
        kspace_data=kspace_data_test,
        sensitivity_maps=sensitivity_maps_test,
        forward_op_fn=mock_forward_op_sense,
        adjoint_op_fn=mock_adjoint_op_sense,
        regularizer_prox_fn=mock_regularizer_prox_updated
    )
    assert reconstructed_image_zero_init.shape == image_shape_test
    print("Zero initialization test (modified steplength) passed.")
