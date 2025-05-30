import torch
import torch.nn as nn
from typing import Tuple, List, Optional

# Attempt to import actual modules with fallbacks to Mocks for standalone testing / dev
try:
    from ..wavelets_scratch import WaveletTransform
except ImportError:
    print("Warning: reconlib.wavelets_scratch.WaveletTransform not found. Using MockWaveletTransform for LRI.")
    class MockWaveletTransform:
        def __init__(self, device, level=3, wavelet_name='mock_wavelet'): # Added wavelet_name for denoiser init
            self.device = device
            self.level = level
            self.wavelet_name = wavelet_name # Denoiser might need this
        def forward(self, image_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
            dummy_cA = image_tensor.clone().to(self.device)
            dummy_details_list = []
            # Simple mock: just pass image_tensor as "coefficients"
            # This structure needs to be compatible with SimpleWaveletDenoiser and inverse
            # For SimpleWaveletDenoiser, detail coeffs are tuples (cH,cV,cD)
            for _ in range(self.level):
                # Create dummy detail tensors with plausible downsampled shapes
                # This is highly simplified. Real shapes depend on filter & image size.
                h, w = dummy_cA.shape[-2]//2, dummy_cA.shape[-1]//2
                if h==0 or w==0: # Stop if too small
                    # If level is too high for image size, store fewer levels
                    # This mock doesn't perfectly replicate this behavior, but aims for compatibility
                    break 
                dummy_details_list.append((
                    torch.zeros((h, w), device=self.device, dtype=image_tensor.dtype),
                    torch.zeros((h, w), device=self.device, dtype=image_tensor.dtype),
                    torch.zeros((h, w), device=self.device, dtype=image_tensor.dtype)
                ))
                dummy_cA = torch.zeros((h,w), device=self.device, dtype=image_tensor.dtype) # Update cA shape for next level

            # Ensure coeffs_list has level+1 elements if possible
            # If dummy_details_list is shorter than self.level due to small image size, adjust
            actual_levels_mocked = len(dummy_details_list)
            coeffs_list_mock = [dummy_cA] + dummy_details_list
            # Pad with Nones if necessary for SimpleWaveletDenoiser init, though it expects full list
            # For this mock, this structure should be okay if SimpleWaveletDenoiser handles it.
            # Or ensure the test image for mock is large enough.

            # Slices info: also simplified
            slices_info_mock = [image_tensor.shape] * (actual_levels_mocked + 1)
            return coeffs_list_mock, slices_info_mock

        def inverse(self, coeffs_list: List[torch.Tensor], slices_info: List[Tuple[int, int]]) -> torch.Tensor:
            # Just return the first element (approximation coeffs) for simplicity
            return coeffs_list[0].to(self.device)
    WaveletTransform = MockWaveletTransform # Use mock if actual not found


try:
    from .denoisers import SimpleWaveletDenoiser
except ImportError:
    print("Warning: reconlib.deeplearning.denoisers.SimpleWaveletDenoiser not found. Using MockSimpleWaveletDenoiser for LRI.")
    class MockSimpleWaveletDenoiser(nn.Module):
        def __init__(self, wavelet_transform_op: WaveletTransform): # Takes WT op
            super().__init__()
            self.device = wavelet_transform_op.device # Inherit device
            self.num_levels = wavelet_transform_op.level
             # Mock layers for compatibility if forward expects them
            self.process_cA = nn.Identity() 
            self.process_cH_levels = nn.ModuleList([nn.Identity() for _ in range(self.num_levels)])
            self.process_cV_levels = nn.ModuleList([nn.Identity() for _ in range(self.num_levels)])
            self.process_cD_levels = nn.ModuleList([nn.Identity() for _ in range(self.num_levels)])
            self.to(self.device)

        def forward(self, coeffs_list: List[torch.Tensor]) -> List[torch.Tensor]:
            # Simple pass-through for mock, ensuring device consistency
            processed_coeffs = [coeffs_list[0].to(self.device)] # cA
            for details_tuple in coeffs_list[1:]:
                processed_coeffs.append(tuple(d.to(self.device) for d in details_tuple))
            return processed_coeffs
    SimpleWaveletDenoiser = MockSimpleWaveletDenoiser


try:
    from ..nufft_multi_coil import MultiCoilNUFFTOperator
except ImportError:
    print("Warning: reconlib.nufft_multi_coil.MultiCoilNUFFTOperator not found. Using MockMultiCoilNUFFTOp for LRI.")
    class MockMultiCoilNUFFTOp:
        def __init__(self, image_shape: Tuple[int,...], k_points_shape: Tuple[int,...], device: torch.device):
            self.image_shape = image_shape # e.g. (H,W)
            self.k_points_shape = k_points_shape # e.g. (num_coils, num_kpoints)
            self.device = device
        def op(self, coil_images: torch.Tensor) -> torch.Tensor: # coil_images: (num_coils, H, W)
            num_coils = coil_images.shape[0]
            # Ensure num_kpoints is from k_points_shape[1]
            return torch.zeros((num_coils, self.k_points_shape[1]), dtype=torch.complex64, device=self.device)
        def op_adj(self, coil_kspace: torch.Tensor) -> torch.Tensor: # coil_kspace: (num_coils, num_kpoints)
            num_coils = coil_kspace.shape[0]
            return torch.zeros((num_coils,) + self.image_shape, dtype=torch.complex64, device=self.device)
    MultiCoilNUFFTOperator = MockMultiCoilNUFFTOp


class LearnedRegularizationIteration(nn.Module):
    """
    Implements one iteration of an unrolled learned regularization scheme.
    This iteration includes a data consistency step and a denoising step
    performed in the wavelet domain using a SimpleWaveletDenoiser.
    """
    def __init__(self, 
                 nufft_op: MultiCoilNUFFTOperator, 
                 wavelet_transform_op: WaveletTransform, 
                 denoiser_module: SimpleWaveletDenoiser, 
                 eta_init: float = 0.1):
        """
        Args:
            nufft_op: The multi-coil NUFFT operator.
            wavelet_transform_op: The WaveletTransform operator.
            denoiser_module: The SimpleWaveletDenoiser module.
            eta_init: Initial value for the learnable step size parameter eta.
        """
        super().__init__()
        self.nufft_op = nufft_op
        self.wavelet_transform_op = wavelet_transform_op
        self.denoiser_module = denoiser_module
        
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        
        if hasattr(nufft_op, 'device'):
            self.device = nufft_op.device
        elif hasattr(wavelet_transform_op, 'device'): # Fallback
            self.device = wavelet_transform_op.device
        else: # Further fallback
            self.device = torch.device('cpu')
            print("Warning: Device not found on nufft_op or wavelet_transform_op. Defaulting LRI device to CPU.")
        
        # Move eta to the correct device. Other modules are assumed to be on correct device.
        self.eta.data = self.eta.data.to(self.device)


    def forward(self, 
                x_k: torch.Tensor, 
                y_kspace: torch.Tensor, 
                sensitivity_maps: torch.Tensor) -> torch.Tensor:
        """
        Performs one iteration of the learned regularization.

        Args:
            x_k: Current image estimate. Shape: (H, W) or (D, H, W). Assumed to be complex or real.
            y_kspace: Measured multi-coil k-space data. Shape: (num_coils, num_kpoints).
            sensitivity_maps: Coil sensitivity maps. Shape: (num_coils, H, W) or (num_coils, D, H, W).

        Returns:
            Updated image estimate x_k_plus_1. Shape: same as x_k.
        """
        x_k = x_k.to(self.device)
        y_kspace = y_kspace.to(self.device)
        sensitivity_maps = sensitivity_maps.to(self.device)
        
        # Data Consistency (Gradient Step)
        # x_k is (H,W), sensitivity_maps is (C,H,W) -> x_k_coils is (C,H,W)
        x_k_coils = x_k.unsqueeze(0) * sensitivity_maps  
        
        # k_pred: (C, K)
        k_pred = self.nufft_op.op(x_k_coils) 
        
        k_resid = k_pred - y_kspace # (C, K)
        
        # adj_coils: (C, H, W)
        adj_coils = self.nufft_op.op_adj(k_resid) 
        
        # grad_img_data_consistency: (H, W)
        grad_img_data_consistency = torch.sum(adj_coils * sensitivity_maps.conj(), dim=0) 
        
        # Gradient descent update
        # Ensure eta is used correctly (it's a scalar parameter)
        x_gradient_updated = x_k - self.eta * grad_img_data_consistency

        # Denoising Step (in Wavelet Domain)
        # WaveletTransform and SimpleWaveletDenoiser handle internal datatypes (typically float32)
        # If x_gradient_updated is complex, apply to real and imag parts separately
        if torch.is_complex(x_gradient_updated):
            real_part_gd = x_gradient_updated.real
            imag_part_gd = x_gradient_updated.imag

            coeffs_real, slices_info_real = self.wavelet_transform_op.forward(real_part_gd)
            denoised_coeffs_real = self.denoiser_module(coeffs_real)
            denoised_real_part = self.wavelet_transform_op.inverse(denoised_coeffs_real, slices_info_real)

            coeffs_imag, slices_info_imag = self.wavelet_transform_op.forward(imag_part_gd)
            denoised_coeffs_imag = self.denoiser_module(coeffs_imag)
            denoised_imag_part = self.wavelet_transform_op.inverse(denoised_coeffs_imag, slices_info_imag)
            
            x_k_plus_1 = torch.complex(denoised_real_part, denoised_imag_part)
        else: # Real image
            coeffs_list, slices_info = self.wavelet_transform_op.forward(x_gradient_updated)
            denoised_coeffs_list = self.denoiser_module(coeffs_list)
            x_k_plus_1 = self.wavelet_transform_op.inverse(denoised_coeffs_list, slices_info)
            
        return x_k_plus_1


if __name__ == '__main__':
    print("--- Testing LearnedRegularizationIteration ---")
    
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {test_device}")

    image_shape_test = (32, 32)
    num_coils_test = 4
    num_kpoints_test = 100
    wavelet_level_test = 2 # Keep level low for mock WT

    # Instantiate Mocks
    # Mock NUFFT op - k_points_shape is (num_coils, num_kpoints)
    mock_nufft_op = MultiCoilNUFFTOperator(image_shape_test, (num_coils_test, num_kpoints_test), test_device)
    
    # Mock WaveletTransform - or actual if available and configured
    mock_wt_op = WaveletTransform(device=test_device, level=wavelet_level_test, wavelet_name='haar') # using 'haar' as it's always in ALL_WAVELET_FILTERS
                                                                                                # in wavelets_scratch.py
    
    # Mock SimpleWaveletDenoiser - or actual
    # If using actual SimpleWaveletDenoiser, it needs a valid WaveletTransform op
    mock_denoiser_op = SimpleWaveletDenoiser(wavelet_transform_op=mock_wt_op)
    mock_denoiser_op.to(test_device) # Ensure denoiser is on device

    # Instantiate the module to be tested
    try:
        iteration_module = LearnedRegularizationIteration(
            nufft_op=mock_nufft_op,
            wavelet_transform_op=mock_wt_op,
            denoiser_module=mock_denoiser_op,
            eta_init=0.05
        )
        iteration_module.to(test_device) # Move the main module to device
        print("LearnedRegularizationIteration instantiated successfully.")
        print(f"  Eta parameter: {iteration_module.eta.item()}, device: {iteration_module.eta.device}")

    except Exception as e:
        print(f"Error instantiating LearnedRegularizationIteration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create dummy input tensors
    x_k_input_test = torch.randn(image_shape_test, dtype=torch.complex64, device=test_device)
    y_kspace_input_test = torch.randn(num_coils_test, num_kpoints_test, dtype=torch.complex64, device=test_device)
    sensitivity_maps_input_test = torch.randn(num_coils_test, *image_shape_test, dtype=torch.complex64, device=test_device)
    
    print(f"  Input x_k shape: {x_k_input_test.shape}")
    print(f"  Input y_kspace shape: {y_kspace_input_test.shape}")
    print(f"  Input sensitivity_maps shape: {sensitivity_maps_input_test.shape}")

    # Perform a forward pass
    try:
        x_k_plus_1_output = iteration_module.forward(x_k_input_test, y_kspace_input_test, sensitivity_maps_input_test)
        print("Forward pass completed.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Assertions
    assert x_k_plus_1_output.shape == image_shape_test, \
        f"Output shape mismatch. Expected {image_shape_test}, got {x_k_plus_1_output.shape}"
    assert x_k_plus_1_output.dtype == x_k_input_test.dtype, \
        f"Output dtype mismatch. Expected {x_k_input_test.dtype}, got {x_k_plus_1_output.dtype}"
    assert x_k_plus_1_output.device == test_device, \
        f"Output device mismatch. Expected {test_device}, got {x_k_plus_1_output.device}"
    assert not torch.isnan(x_k_plus_1_output).any(), "NaNs found in output."

    print("LearnedRegularizationIteration basic test passed successfully.")
