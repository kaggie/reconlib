import torch
import torch.nn as nn
from typing import List, Tuple

# Assuming wavelets_scratch is in the parent directory 'reconlib'
# Adjust if the directory structure is different or if using a proper package install
try:
    from ..wavelets_scratch import WaveletTransform
except ImportError:
    # Fallback for direct execution or if reconlib is not installed in a way that supports relative import from script
    # This might happen if you run this file directly without `python -m reconlib.deeplearning.denoisers`
    # For robust package structure, the relative import should work when reconlib is a package.
    print("Could not perform relative import for WaveletTransform. Attempting direct import (may fail if not in sys.path).")
    from reconlib.wavelets_scratch import WaveletTransform


class SimpleWaveletDenoiser(nn.Module):
    """
    A simple denoiser that processes wavelet coefficients using learnable 1x1 convolutions.
    """
    def __init__(self, wavelet_transform_op: WaveletTransform):
        """
        Args:
            wavelet_transform_op: An instance of WaveletTransform that defines
                                  the wavelet, level, and device to be used.
        """
        super().__init__()
        self.wavelet_transform_op = wavelet_transform_op
        self.device = wavelet_transform_op.device
        self.num_levels = wavelet_transform_op.level

        # Learnable layer for the approximation coefficients (cA_n)
        self.process_cA = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # Learnable layers for detail coefficients at each level
        self.process_cH_levels = nn.ModuleList()
        self.process_cV_levels = nn.ModuleList()
        self.process_cD_levels = nn.ModuleList()

        for _ in range(self.num_levels):
            self.process_cH_levels.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
            self.process_cV_levels.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
            self.process_cD_levels.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
        
        # Move the entire module (including all submodules) to the specified device
        self.to(self.device)

    def forward(self, coeffs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Processes a list of wavelet coefficients.

        Args:
            coeffs_list: A list of wavelet coefficients, structured as
                         [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)].
                         cA_n is a tensor.
                         Each detail entry is a tuple of 3 tensors (cH_i, cV_i, cD_i).

        Returns:
            A list of processed wavelet coefficients with the same structure.
        """
        if not isinstance(coeffs_list, list) or len(coeffs_list) != self.num_levels + 1:
            raise ValueError(f"coeffs_list must be a list of length num_levels+1 ({self.num_levels+1}). "
                             f"Got length {len(coeffs_list)}.")

        processed_coeffs_list = []

        # Process cA_n (approximation coefficients at the coarsest level)
        cA_n = coeffs_list[0].to(self.device)
        # Add batch and channel dimensions for Conv2d, then remove them
        cA_n_processed = self.process_cA(cA_n.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        processed_coeffs_list.append(cA_n_processed)

        # Process detail coefficients (cH_i, cV_i, cD_i) for each level
        # coeffs_list[1] contains (cH_n, cV_n, cD_n) - coarsest details
        # coeffs_list[num_levels] contains (cH_1, cV_1, cD_1) - finest details
        # ModuleLists are indexed 0 to num_levels-1.
        # We want self.process_..._levels[0] to process coarsest (cH_n, etc.)
        # and self.process_..._levels[num_levels-1] to process finest (cH_1, etc.)
        for i in range(self.num_levels):
            level_idx_in_coeffs_list = i + 1 # Accesses details from coeffs_list[1] to coeffs_list[num_levels]
            
            details_tuple = coeffs_list[level_idx_in_coeffs_list]
            if not isinstance(details_tuple, tuple) or len(details_tuple) != 3:
                raise ValueError(f"Detail coefficients at index {level_idx_in_coeffs_list} must be a tuple of 3 tensors.")
            
            cH_i, cV_i, cD_i = details_tuple
            cH_i = cH_i.to(self.device)
            cV_i = cV_i.to(self.device)
            cD_i = cD_i.to(self.device)

            # The layer index `i` corresponds to processing from coarsest to finest detail levels.
            # Example: num_levels = 3
            # i = 0: processes coeffs_list[1] (level 3 details, coarsest) using layers[0]
            # i = 1: processes coeffs_list[2] (level 2 details) using layers[1]
            # i = 2: processes coeffs_list[3] (level 1 details, finest) using layers[2]
            layer_module_idx = i 

            cH_processed = self.process_cH_levels[layer_module_idx](cH_i.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            cV_processed = self.process_cV_levels[layer_module_idx](cV_i.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            cD_processed = self.process_cD_levels[layer_module_idx](cD_i.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            
            processed_coeffs_list.append((cH_processed, cV_processed, cD_processed))
            
        return processed_coeffs_list


if __name__ == '__main__':
    print("--- Testing SimpleWaveletDenoiser ---")
    
    # Setup parameters for testing
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wavelet_name_test = 'haar' # Haar is simple and guaranteed in ALL_WAVELET_FILTERS
    test_level = 3
    img_size = (64, 64)

    print(f"Device: {test_device}, Wavelet: {wavelet_name_test}, Level: {test_level}, Image Size: {img_size}")

    # 1. Instantiate WaveletTransform
    # Need to ensure WaveletTransform can be imported.
    # The try-except for relative import handles if run directly.
    try:
        wavelet_op = WaveletTransform(wavelet_name=wavelet_name_test, level=test_level, device=test_device)
        print("WaveletTransform instantiated.")
    except Exception as e:
        print(f"Error instantiating WaveletTransform: {e}")
        print("Ensure reconlib.wavelets_scratch.WaveletTransform is accessible.")
        sys.exit(1) # Exit if WaveletTransform cannot be created, as denoiser depends on it.

    # 2. Instantiate SimpleWaveletDenoiser
    try:
        denoiser_module = SimpleWaveletDenoiser(wavelet_transform_op=wavelet_op)
        print(f"SimpleWaveletDenoiser instantiated on device: {next(denoiser_module.parameters()).device}")
    except Exception as e:
        print(f"Error instantiating SimpleWaveletDenoiser: {e}")
        sys.exit(1)


    # 3. Create a dummy 2D test image and get its wavelet coefficients
    # WaveletTransform.forward expects float32 data internally due to filter casting and data casting
    test_image = torch.randn(*img_size, device=test_device, dtype=torch.float32) 
    print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
    
    coeffs_input_list, slices_info = wavelet_op.forward(test_image)
    print(f"Input coeffs_list length: {len(coeffs_input_list)}")
    print(f"  cA_n input shape: {coeffs_input_list[0].shape}")
    for i, details_tuple in enumerate(coeffs_input_list[1:]):
        print(f"  Level {test_level-i} details (cH,cV,cD) input shapes: {details_tuple[0].shape}, {details_tuple[1].shape}, {details_tuple[2].shape}")


    # 4. Pass coefficients to the denoiser module
    try:
        coeffs_output_list = denoiser_module.forward(coeffs_input_list)
        print("Denoiser forward pass completed.")
    except Exception as e:
        print(f"Error during denoiser forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Check output structure and shapes
    assert len(coeffs_output_list) == len(coeffs_input_list), \
        f"Output list length mismatch: expected {len(coeffs_input_list)}, got {len(coeffs_output_list)}"
    print(f"Output coeffs_list length: {len(coeffs_output_list)}")

    # Check cA_n shape
    assert coeffs_output_list[0].shape == coeffs_input_list[0].shape, \
        f"cA_n shape mismatch: input {coeffs_input_list[0].shape}, output {coeffs_output_list[0].shape}"
    print(f"  cA_n output shape: {coeffs_output_list[0].shape}")
    assert coeffs_output_list[0].device == test_device, f"cA_n output device mismatch."


    # Check detail coefficient shapes and device
    for i in range(test_level):
        input_details_tuple = coeffs_input_list[i+1]
        output_details_tuple = coeffs_output_list[i+1]
        
        assert isinstance(output_details_tuple, tuple) and len(output_details_tuple) == 3, \
            f"Details at level index {i} in output are not a tuple of 3 tensors."
            
        assert output_details_tuple[0].shape == input_details_tuple[0].shape, \
            f"cH shape mismatch at level index {i}: input {input_details_tuple[0].shape}, output {output_details_tuple[0].shape}"
        assert output_details_tuple[1].shape == input_details_tuple[1].shape, \
            f"cV shape mismatch at level index {i}: input {input_details_tuple[1].shape}, output {output_details_tuple[1].shape}"
        assert output_details_tuple[2].shape == input_details_tuple[2].shape, \
            f"cD shape mismatch at level index {i}: input {input_details_tuple[2].shape}, output {output_details_tuple[2].shape}"
        
        assert output_details_tuple[0].device == test_device, f"cH output device mismatch at level index {i}."
        assert output_details_tuple[1].device == test_device, f"cV output device mismatch at level index {i}."
        assert output_details_tuple[2].device == test_device, f"cD output device mismatch at level index {i}."

        print(f"  Level {test_level-i} details (cH,cV,cD) output shapes: {output_details_tuple[0].shape}, {output_details_tuple[1].shape}, {output_details_tuple[2].shape}")

    print("\nSimpleWaveletDenoiser test passed: Output structure and shapes are consistent.")
    print("All parameters should be on device:", next(denoiser_module.parameters()).device)
