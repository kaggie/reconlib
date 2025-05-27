import torch
import torch.nn as nn
from typing import List, Callable, Optional, Dict, Any, Tuple

def project_onto_support(image: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    """
    Projects an image onto a given support mask.

    Args:
        image: The input image tensor.
        support_mask: A tensor defining the support. Should be broadcastable
                      to the image's shape. Typically boolean or 0s and 1s.
    Returns:
        The projected image (image * support_mask).
    """
    if not isinstance(support_mask, torch.Tensor):
        raise TypeError("support_mask must be a PyTorch tensor.")
    if image.shape[-support_mask.ndim:] != support_mask.shape: # Check if trailing dims match
        try:
            # Attempt to make mask broadcastable, e.g. if image is (D,H,W) and mask is (H,W)
            # This specific unsqueeze might not cover all cases but is common for 2D mask on 3D image
            if image.ndim > support_mask.ndim and image.shape[-support_mask.ndim:] == support_mask.shape:
                 num_unsqueeze = image.ndim - support_mask.ndim
                 support_mask = support_mask.view((1,) * num_unsqueeze + support_mask.shape)
            else: # Other shape mismatches
                 raise ValueError(f"Shape mismatch: Image shape {image.shape}, support_mask shape {support_mask.shape}. "
                                 "Mask cannot be directly broadcast.")
        except ValueError as e: # Catch potential view errors too
            raise ValueError(f"Shape mismatch: Image shape {image.shape}, support_mask shape {support_mask.shape}. "
                             f"Mask cannot be broadcast. Original error: {e}")


    return image * support_mask.to(dtype=image.dtype, device=image.device) # Ensure mask is same dtype and device


class POCSENSEreconstructor(nn.Module):
    """
    Implements a Projection Onto Convex Sets (POCS) based SENSE reconstruction.
    It iteratively applies data consistency and a series of other projection operators.
    """
    def __init__(self, 
                 iterations: int = 10, 
                 data_consistency_step_size: float = 0.1, 
                 verbose: bool = False, 
                 log_fn: Optional[Callable[[int, torch.Tensor, float, float], None]] = None):
        """
        Args:
            iterations: Number of POCS iterations.
            data_consistency_step_size: Step size for the SENSE data consistency update.
            verbose: If True, print iteration progress.
            log_fn: An optional function to log iteration metrics.
                    Signature: `fn(iter_num, current_image, change_norm, grad_norm_dc_step)`.
        """
        super().__init__()
        self.iterations = iterations
        self.data_consistency_step_size = data_consistency_step_size
        self.verbose = verbose
        self.log_fn = log_fn
        self.projectors: List[Callable[..., torch.Tensor]] = []
        self.projector_names: List[str] = []

    def add_projector(self, projector_fn: Callable[..., torch.Tensor], name: Optional[str] = None):
        """
        Adds a projector function to the POCS sequence.

        Args:
            projector_fn: A callable that takes an image tensor as its first argument,
                          and potentially other keyword arguments, and returns a projected image tensor.
                          Example signature: `proj_fn(image: torch.Tensor, **kwargs) -> torch.Tensor`.
            name: An optional name for the projector (for logging/debugging).
        """
        self.projectors.append(projector_fn)
        if name is None:
            name = f"projector_{len(self.projectors)}"
        self.projector_names.append(name)

    def reconstruct(self, 
                    kspace_data: torch.Tensor, 
                    sensitivity_maps: Optional[torch.Tensor], 
                    forward_op_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor], 
                    adjoint_op_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor], 
                    initial_estimate: Optional[torch.Tensor] = None,
                    projector_kwargs_list: Optional[List[Dict[str, Any]]] = None,
                    image_shape_for_zero_init: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Performs POCS-SENSE reconstruction.

        Args:
            kspace_data: Measured k-space data.
            sensitivity_maps: Coil sensitivity maps.
            forward_op_fn: Callable for the forward SENSE operation A(x).
            adjoint_op_fn: Callable for the adjoint SENSE operation A^H(y).
            initial_estimate: Optional initial estimate for the image.
            projector_kwargs_list: Optional list of dictionaries, where each dictionary
                                   contains keyword arguments for the corresponding projector
                                   in self.projectors. If None, empty dicts are used.
            image_shape_for_zero_init: Required if initial_estimate is None and shape cannot be inferred.

        Returns:
            The reconstructed image.
        """
        device = kspace_data.device
        x_current: torch.Tensor

        if sensitivity_maps is not None:
            sensitivity_maps = sensitivity_maps.to(device)

        if initial_estimate is not None:
            x_current = initial_estimate.clone().to(device)
        else:
            _image_shape_inferred: Optional[Tuple[int, ...]] = None
            if image_shape_for_zero_init is not None:
                _image_shape_inferred = image_shape_for_zero_init
            elif sensitivity_maps is not None:
                _image_shape_inferred = sensitivity_maps.shape[1:]
            else:
                try:
                    dummy_k_shape = (kspace_data.shape[0], kspace_data.shape[-1]) if kspace_data.ndim > 1 else kspace_data.shape
                    dummy_k = torch.zeros(dummy_k_shape, dtype=kspace_data.dtype, device=device)
                    _image_shape_inferred = adjoint_op_fn(dummy_k, sensitivity_maps).shape
                except Exception as e:
                    raise ValueError(
                        "POCS: Cannot determine image shape for zero initialization. "
                        f"Provide initial_estimate, sensitivity_maps, or image_shape_for_zero_init. Error: {e}"
                    )
            if _image_shape_inferred is None:
                 raise ValueError("POCS: Image shape for zero initialization could not be determined.")
            
            x_current = torch.zeros(_image_shape_inferred, dtype=kspace_data.dtype, device=device)
            if torch.is_complex(kspace_data) and not torch.is_complex(x_current):
                 x_current = x_current.to(torch.complex64)


        if projector_kwargs_list is None:
            projector_kwargs_list = [{} for _ in self.projectors]
        elif len(projector_kwargs_list) != len(self.projectors):
            raise ValueError("Length of projector_kwargs_list must match the number of added projectors.")

        for iter_num in range(self.iterations):
            # SENSE Data Consistency (Gradient Descent Step)
            k_pred = forward_op_fn(x_current, sensitivity_maps)
            k_resid = k_pred - kspace_data
            grad_data_term = adjoint_op_fn(k_resid, sensitivity_maps)
            
            x_dc = x_current - self.data_consistency_step_size * grad_data_term
            
            # Apply other projectors sequentially
            x_after_projections = x_dc
            for p_idx, proj_fn in enumerate(self.projectors):
                current_kwargs = projector_kwargs_list[p_idx]
                x_after_projections = proj_fn(x_after_projections, **current_kwargs)
            
            if self.verbose or self.log_fn:
                with torch.no_grad():
                    current_norm = torch.norm(x_current)
                    change = torch.norm(x_after_projections - x_current) / (current_norm + 1e-9) if current_norm > 0 else torch.norm(x_after_projections - x_current)
                    grad_norm_dc_step = torch.norm(grad_data_term) # Norm of gradient from DC step

                    if self.verbose:
                        projector_names_str = ", ".join(self.projector_names)
                        print(f"Iter {iter_num + 1}/{self.iterations}, DC Step Size: {self.data_consistency_step_size:.2e}, "
                              f"Change: {change.item():.2e}, Grad Norm (DC): {grad_norm_dc_step.item():.2e}, Proj: [{projector_names_str}]")
                    
                    if self.log_fn:
                        self.log_fn(iter_num=iter_num, current_image=x_after_projections.clone(), 
                                    change=change.item(), grad_norm=grad_norm_dc_step.item())
            
            x_current = x_after_projections
            
        return x_current

if __name__ == '__main__':
    print("--- Testing POCSENSEreconstructor ---")
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {test_device}")

    image_shape_test = (16, 16)
    num_coils_test = 2
    k_len_test = 50

    def mock_forward_op_sense(image_2d, smaps_3d):
        coil_images = image_2d.unsqueeze(0) * smaps_3d
        return torch.stack([torch.fft.fft2(ci).flatten()[:k_len_test] for ci in coil_images])

    def mock_adjoint_op_sense(kspace_coils_2d, smaps_3d):
        C, H, W = smaps_3d.shape
        coil_images_adj = []
        for i in range(C):
            temp_k = torch.zeros(H*W, dtype=kspace_coils_2d.dtype, device=kspace_coils_2d.device)
            temp_k[:k_len_test] = kspace_coils_2d[i]
            coil_images_adj.append(torch.fft.ifft2(temp_k.view(H,W)))
        return torch.sum(torch.stack(coil_images_adj) * smaps_3d.conj(), dim=0)

    # Test project_onto_support
    test_img_proj = torch.ones(image_shape_test, device=test_device)
    test_mask_proj = torch.zeros(image_shape_test, device=test_device, dtype=torch.bool)
    test_mask_proj[0:8, 0:8] = True
    projected_img = project_onto_support(test_img_proj, test_mask_proj)
    assert projected_img[0,0].item() == 1.0 and projected_img[8,8].item() == 0.0, "project_onto_support basic test failed."
    print("project_onto_support function test passed.")
    
    # Test broadcast in project_onto_support
    test_img_3d_proj = torch.ones((3, *image_shape_test), device=test_device) # (D,H,W)
    projected_img_3d = project_onto_support(test_img_3d_proj, test_mask_proj) # 2D mask on 3D image
    assert projected_img_3d.shape == test_img_3d_proj.shape
    assert projected_img_3d[0,0,0].item() == 1.0 and projected_img_3d[0,8,8].item() == 0.0
    print("project_onto_support broadcast test passed.")


    # Instantiate Reconstructor
    pocs_reconstructor = POCSENSEreconstructor(
        iterations=5, 
        data_consistency_step_size=0.1, 
        verbose=True
    )
    
    # Add a projector
    support_m = torch.ones(image_shape_test, device=test_device, dtype=torch.bool) # Trivial all-pass mask
    pocs_reconstructor.add_projector(project_onto_support, name="FullSupport")
    projector_args_test = [{'support_mask': support_m}]

    print("POCSENSEreconstructor instantiated and projector added.")

    # Create dummy data
    kspace_data_test = torch.randn(num_coils_test, k_len_test, dtype=torch.complex64, device=test_device)
    sensitivity_maps_test = torch.rand(num_coils_test, *image_shape_test, dtype=torch.complex64, device=test_device) + 1e-3 # Avoid zero smaps
    sensitivity_maps_test = sensitivity_maps_test / torch.sqrt(torch.sum(sensitivity_maps_test.abs()**2, dim=0, keepdim=True))
    initial_estimate_test = mock_adjoint_op_sense(kspace_data_test, sensitivity_maps_test)


    print(f"K-space data shape: {kspace_data_test.shape}")
    print(f"Sensitivity maps shape: {sensitivity_maps_test.shape}")
    print(f"Initial estimate shape: {initial_estimate_test.shape}")

    # Test reconstruction
    try:
        reconstructed_image = pocs_reconstructor.reconstruct(
            kspace_data=kspace_data_test,
            sensitivity_maps=sensitivity_maps_test,
            forward_op_fn=mock_forward_op_sense,
            adjoint_op_fn=mock_adjoint_op_sense,
            initial_estimate=initial_estimate_test.clone(),
            projector_kwargs_list=projector_args_test
        )
        print("Reconstruction call completed.")
    except Exception as e:
        print(f"Error during reconstructor.reconstruct call: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    assert reconstructed_image.shape == image_shape_test
    assert reconstructed_image.dtype == kspace_data_test.dtype
    assert reconstructed_image.device == test_device
    assert not torch.isnan(reconstructed_image).any()
    
    print(f"\nReconstructed image shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    print("POCSENSEreconstructor basic test passed successfully.")

```
