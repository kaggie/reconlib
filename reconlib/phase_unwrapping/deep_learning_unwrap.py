# reconlib/phase_unwrapping/deep_learning_unwrap.py
"""Deep Learning based phase unwrapping using a pre-trained U-Net model."""

import torch
import os
from typing import Optional

# Attempt to import UNet, handle error if not found
try:
    from reconlib.deeplearning.models.unet_denoiser import UNet
    _UNET_MODEL_AVAILABLE = True
except ImportError as e:
    _UNET_MODEL_AVAILABLE = False
    _UNET_IMPORT_ERROR = e

def unwrap_phase_deep_learning(
    wrapped_phase: torch.Tensor,
    model_path: str,
    unet_in_channels: int = 1,
    unet_out_channels: int = 1,
    unet_num_levels: int = 4,
    unet_initial_features: int = 64,
    unet_bilinear_upsampling: bool = True,
    input_is_2d: bool = False,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Performs phase unwrapping using a pre-trained 2D U-Net model.

    The function loads a U-Net model, processes the input wrapped phase
    (either as a single 2D image or slice-by-slice for 3D data),
    and returns the model's output, assumed to be the unwrapped phase.

    Args:
        wrapped_phase (torch.Tensor): Input wrapped phase image (in radians).
            Shape: (H, W) for 2D, or (D, H, W) for 3D.
        model_path (str): Path to the pre-trained U-Net model state_dict (.pth file).
        unet_in_channels (int, optional): Number of input channels for the U-Net.
            Defaults to 1.
        unet_out_channels (int, optional): Number of output channels for the U-Net.
            Defaults to 1.
        unet_num_levels (int, optional): Number of levels in the U-Net encoder/decoder.
            Defaults to 4.
        unet_initial_features (int, optional): Number of features in the first
            convolutional layer of the U-Net. Defaults to 64.
        unet_bilinear_upsampling (bool, optional): Whether to use bilinear upsampling
            in the U-Net decoder. If False, transposed convolutions are used.
            Defaults to True.
        input_is_2d (bool, optional):
            - If True: Input `wrapped_phase` is treated as a single 2D image (H,W).
            - If False (default):
                - If `wrapped_phase` is 2D (H,W), it's processed as a single 2D image.
                - If `wrapped_phase` is 3D (D,H,W), it's processed slice-by-slice
                  along the depth dimension (dim 0) using the 2D U-Net.
        device (Optional[str], optional): Target device for model and tensor operations
            (e.g., "cpu", "cuda:0"). If None, uses the device of `wrapped_phase`.
            Defaults to None.

    Returns:
        torch.Tensor: Unwrapped phase image. Shape matches input `wrapped_phase`.

    Raises:
        NotImplementedError: If the UNet model definition cannot be imported.
        FileNotFoundError: If `model_path` does not exist.
        RuntimeError: If model loading fails (e.g., architecture mismatch, corrupted file).
        ValueError: If input tensor dimensions are unsupported.
    """
    if not _UNET_MODEL_AVAILABLE:
        raise NotImplementedError(
            "UNet model definition from reconlib.deeplearning.models.unet_denoiser "
            f"could not be imported. Original error: {_UNET_IMPORT_ERROR}"
        )

    # 1. Device Handling
    if device is None:
        target_device = wrapped_phase.device
    else:
        target_device = torch.device(device)

    # 2. Model Loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = UNet(
        in_channels=unet_in_channels,
        out_channels=unet_out_channels,
        num_levels=unet_num_levels,
        initial_features=unet_initial_features,
        bilinear_upsampling=unet_bilinear_upsampling
    )
    model.to(target_device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=target_device))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}. Ensure model architecture "
                           f"matches the state_dict. Original error: {e}")
    
    model.eval()

    # 3. Input Preprocessing & Inference
    input_tensor = wrapped_phase.to(target_device, dtype=torch.float32)
    
    original_ndim = input_tensor.ndim
    if not (original_ndim == 2 or original_ndim == 3):
        raise ValueError(f"Unsupported input_tensor ndim: {original_ndim}. Expected 2D (H,W) or 3D (D,H,W).")

    with torch.no_grad():
        if (original_ndim == 2) or (original_ndim == 3 and input_is_2d):
            # This branch handles 2D inputs (H,W) and 3D inputs (D,H,W) that should be treated as 2D.
            # The 3D input (D,H,W) could mean D is the channel for a multi-channel 2D U-Net,
            # or we take the first slice if unet_in_channels=1.

            if original_ndim == 2: # Input is (H,W)
                # Reshape to (1, 1, H, W)
                processed_input = input_tensor.unsqueeze(0).unsqueeze(0)
                if unet_in_channels > 1:
                    # If model expects C_in > 1, repeat the single channel.
                    # This assumes the single 2D slice should be fed to all input channels.
                    processed_input = processed_input.repeat(1, unet_in_channels, 1, 1)
            
            else: # original_ndim == 3 and input_is_2d is True. Input is (D,H,W)
                if input_tensor.shape[0] == unet_in_channels:
                    # Treat D as the channel dimension for the 2D U-Net.
                    processed_input = input_tensor.unsqueeze(0) # (1, D, H, W)
                elif unet_in_channels == 1:
                    # Take the first slice D[0] and make it (1, 1, H, W)
                    if input_tensor.shape[0] > 1:
                        print(f"Warning: input_is_2d=True, unet_in_channels=1, but input tensor is 3D with depth {input_tensor.shape[0]}."
                              " Only the first slice (index 0) will be processed.")
                    processed_input = input_tensor[0, :, :].unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                else: # input_tensor.shape[0] != unet_in_channels AND unet_in_channels > 1
                    # Ambiguous case: take the first slice and repeat it to match unet_in_channels.
                    if input_tensor.shape[0] > 1:
                         print(f"Warning: input_is_2d=True, input tensor depth {input_tensor.shape[0]} != unet_in_channels {unet_in_channels}."
                              f" Using the first slice (index 0) and repeating it to {unet_in_channels} channels.")
                    slice_to_process = input_tensor[0, :, :] # (H,W)
                    processed_input = slice_to_process.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
                    processed_input = processed_input.repeat(1, unet_in_channels, 1, 1) # (1, unet_in_channels, H,W)

            # Perform inference
            output_tensor = model(processed_input) # Output shape: (1, unet_out_channels, H, W)
            
            # Squeeze output back to match expected output format
            if unet_out_channels == 1:
                # If original was (D,H,W) but processed as 2D (e.g. first slice), output should be (H,W)
                # If original was (H,W), output should be (H,W)
                unwrapped_phase = output_tensor.squeeze(0).squeeze(0) # (H,W)
            else: # Multi-channel output from U-Net
                # If original was (D,H,W) with D=unet_in_channels, output (unet_out_channels, H,W) is appropriate
                # If original was (H,W) fed into multi-channel in/out, output (unet_out_channels, H,W)
                unwrapped_phase = output_tensor.squeeze(0) # (unet_out_channels, H,W)

        elif original_ndim == 3 and not input_is_2d: # Process 3D input slice-by-slice
            slices_out_list = []
            for i in range(input_tensor.shape[0]): # Iterate over depth D
                slice_2d = input_tensor[i, :, :] # Current slice (H,W)
                
                # Prepare slice for 2D U-Net: (1, unet_in_channels, H, W)
                processed_slice = slice_2d.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                if unet_in_channels > 1:
                    processed_slice = processed_slice.repeat(1, unet_in_channels, 1, 1)
                
                output_slice = model(processed_slice) # Output: (1, unet_out_channels, H, W)
                
                # Squeeze output slice appropriately before appending
                if unet_out_channels == 1:
                    slices_out_list.append(output_slice.squeeze(0).squeeze(0)) # (H,W)
                else:
                    slices_out_list.append(output_slice.squeeze(0)) # (unet_out_channels, H,W)
            
            unwrapped_phase = torch.stack(slices_out_list, dim=0) # Stack to (D,H,W) or (D,unet_out_channels,H,W)
        else:
             raise ValueError(f"Unhandled input configuration: ndim={original_ndim}, input_is_2d={input_is_2d}")

    return unwrapped_phase.to(wrapped_phase.device) # Ensure output is on original device
