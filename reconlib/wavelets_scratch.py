import torch
import torch.nn as nn
import math
import pywt # For obtaining standard filter coefficients
from typing import List, Tuple, Optional
import numpy as np # Ensure numpy is imported for the test code

# Attempt to import actual base classes, fall back to placeholders if not found
try:
    from reconlib.regularizers.base import Regularizer
except ImportError:
    print("Warning: reconlib.regularizers.base.Regularizer not found. Using placeholder.")
    class Regularizer:
        def __init__(self):
            pass
        def proximal_operator(self, x, steplength):
            raise NotImplementedError
        def value(self, x):
            raise NotImplementedError

try:
    from reconlib.operators import NUFFTOperator
except ImportError:
    print("Warning: reconlib.operators.NUFFTOperator not found. Using placeholder.")
    NUFFTOperator = object # Placeholder type


# --- Wavelet Filter Definitions ---
ALL_WAVELET_FILTERS = {}

# Haar Wavelet (manual definition, float32 for consistency with existing tests)
ALL_WAVELET_FILTERS['haar'] = {
    'dec_lo': torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.float32),
    'dec_hi': torch.tensor([-1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.float32),
    'rec_lo': torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.float32),
    'rec_hi': torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)], dtype=torch.float32),
    'pywt_name': 'haar' # For reference
}

# Other wavelet families from PyWavelets (stored as float64 for precision)
WAVELET_NAMES_TO_LOAD = {
    'db4': 'db4', 'db8': 'db8',
    'sym4': 'sym4', 'sym8': 'sym8',
    'coif1': 'coif1', 'coif2': 'coif2',
    'bior2.2': 'bior2.2', 'bior4.4': 'bior4.4',
}

for key_name, pywt_name_str in WAVELET_NAMES_TO_LOAD.items():
    try:
        wavelet = pywt.Wavelet(pywt_name_str)
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
        ALL_WAVELET_FILTERS[key_name] = {
            'dec_lo': torch.tensor(dec_lo, dtype=torch.float64),
            'dec_hi': torch.tensor(dec_hi, dtype=torch.float64),
            'rec_lo': torch.tensor(rec_lo, dtype=torch.float64),
            'rec_hi': torch.tensor(rec_hi, dtype=torch.float64),
            'pywt_name': pywt_name_str # For reference
        }
    except Exception as e:
        print(f"Could not load wavelet {pywt_name_str}: {e}. Skipping.")


# --- DWT and IDWT Implementations ---
def _dwt1d(data: torch.Tensor, filter_lo: torch.Tensor, filter_hi: torch.Tensor):
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a PyTorch tensor.")
    if not isinstance(filter_lo, torch.Tensor) or not isinstance(filter_hi, torch.Tensor):
        raise TypeError("Filters must be PyTorch tensors.")
    if data.ndim != 1:
        if data.ndim == 0 and data.numel() == 1:
            data = data.view(1)
        elif data.numel() == 0: 
            approx_out_len = max(0, (0 + filter_lo.shape[0] -1 )//2)
            return torch.empty(approx_out_len, dtype=data.dtype, device=data.device), \
                   torch.empty(approx_out_len, dtype=data.dtype, device=data.device)
        else:
            raise ValueError(f"Input data must be a 1D tensor or scalar, got {data.ndim}D with shape {data.shape}")

    device = data.device
    filter_lo = filter_lo.to(device=device, dtype=data.dtype)
    filter_hi = filter_hi.to(device=device, dtype=data.dtype)

    filter_len = filter_lo.shape[0]
    if filter_len != filter_hi.shape[0]:
        raise ValueError("Low-pass and high-pass filters must have the same length.")

    pad_left = (filter_len - 1) // 2
    pad_right = (filter_len - 1) - pad_left
    
    data_reshaped = data.view(1, 1, -1)
    data_padded = torch.nn.functional.pad(data_reshaped, (pad_left, pad_right), mode='reflect')

    filter_lo_reshaped = filter_lo.view(1, 1, -1)
    filter_hi_reshaped = filter_hi.view(1, 1, -1)

    cA = torch.nn.functional.conv1d(data_padded, filter_lo_reshaped, stride=2).squeeze(0).squeeze(0) 
    cD = torch.nn.functional.conv1d(data_padded, filter_hi_reshaped, stride=2).squeeze(0).squeeze(0) 
    
    if cA.ndim == 0: cA = cA.view(1)
    if cD.ndim == 0: cD = cD.view(1)

    return cA, cD

def _idwt1d(cA: torch.Tensor, cD: torch.Tensor, filter_lo_recon: torch.Tensor, filter_hi_recon: torch.Tensor, original_length: int = -1):
    if not isinstance(cA, torch.Tensor) or not isinstance(cD, torch.Tensor):
        raise TypeError("Input coefficients must be PyTorch tensors.")
    if not isinstance(filter_lo_recon, torch.Tensor) or not isinstance(filter_hi_recon, torch.Tensor):
        raise TypeError("Filters must be PyTorch tensors.")
    
    if cA.ndim == 0 and cA.numel() == 1: cA = cA.view(1)
    if cD.ndim == 0 and cD.numel() == 1: cD = cD.view(1)

    if cA.ndim != 1 or cD.ndim != 1:
        if cA.numel() == 0 and cD.numel() == 0:
            return torch.empty(original_length if original_length >=0 else 0, dtype=cA.dtype, device=cA.device)
        raise ValueError(f"Input coeffs must be 1D tensors or scalars. Got cA shape {cA.shape}, cD shape {cD.shape}")

    device = cA.device
    filter_lo_recon = filter_lo_recon.to(device=device, dtype=cA.dtype)
    filter_hi_recon = filter_hi_recon.to(device=device, dtype=cA.dtype)

    filter_len = filter_lo_recon.shape[0]
    if filter_len != filter_hi_recon.shape[0]:
        raise ValueError("Reconstruction low-pass and high-pass filters must have the same length.")

    len_cA = cA.shape[0]
    len_cD = cD.shape[0] 
    
    cA_up_len = len_cA * 2
    cD_up_len = len_cD * 2

    cA_up = torch.zeros(cA_up_len, device=device, dtype=cA.dtype)
    if len_cA > 0: cA_up[0::2] = cA
    cD_up = torch.zeros(cD_up_len, device=device, dtype=cD.dtype)
    if len_cD > 0: cD_up[0::2] = cD
    
    cA_up_reshaped = cA_up.view(1, 1, -1)
    cD_up_reshaped = cD_up.view(1, 1, -1)

    filter_lo_recon_reshaped = filter_lo_recon.view(1, 1, -1)
    filter_hi_recon_reshaped = filter_hi_recon.view(1, 1, -1)

    padding_conv = filter_len - 1

    y_lo = torch.nn.functional.conv1d(cA_up_reshaped, filter_lo_recon_reshaped, padding=padding_conv)
    y_hi = torch.nn.functional.conv1d(cD_up_reshaped, filter_hi_recon_reshaped, padding=padding_conv)

    if y_lo.shape[2] != y_hi.shape[2]:
        diff = y_lo.shape[2] - y_hi.shape[2]
        if diff > 0: 
            y_hi = torch.nn.functional.pad(y_hi, (0, diff), mode='constant', value=0)
        else: 
            y_lo = torch.nn.functional.pad(y_lo, (0, -diff), mode='constant', value=0)
            
    reconstructed_signal = (y_lo + y_hi).squeeze(0).squeeze(0) 
    if reconstructed_signal.ndim == 0 and reconstructed_signal.numel() == 1: 
        reconstructed_signal = reconstructed_signal.view(1)
    elif reconstructed_signal.numel() == 0 and original_length == 0 : 
         reconstructed_signal = torch.empty(0, dtype=reconstructed_signal.dtype, device=reconstructed_signal.device)

    target_len = -1
    if original_length != -1:
        target_len = original_length
    else:
        if len_cA == 0 and len_cD == 0 : 
             target_len = 0
        elif filter_len % 2 == 0: 
             target_len = 2 * len_cA 
        else: 
             target_len = 2 * len_cA - (filter_len -1) 

    current_len = reconstructed_signal.shape[0]
    if target_len != -1 and current_len != target_len:
        delta = current_len - target_len
        if delta < 0: 
            reconstructed_signal = torch.nn.functional.pad(reconstructed_signal, (0, -delta), mode='constant', value=0)
        else: 
            if current_len - (delta - delta // 2) >= 0 and current_len > 0 : # Ensure slice is valid
                crop_start = delta // 2
                crop_end = delta - crop_start 
                reconstructed_signal = reconstructed_signal[crop_start : current_len - crop_end]
            elif target_len == 0 : 
                 reconstructed_signal = torch.empty(0, dtype=reconstructed_signal.dtype, device=reconstructed_signal.device)
            
    return reconstructed_signal

# --- WaveletTransform Class ---
class WaveletTransform:
    def __init__(self, wavelet_name: str = 'db4', level: int = 3, device: str = 'cpu'):
        self.wavelet_name = wavelet_name
        self.level = level
        self.device = torch.device(device)

        if wavelet_name not in ALL_WAVELET_FILTERS:
            raise KeyError(f"Wavelet '{wavelet_name}' not found in ALL_WAVELET_FILTERS. "
                           f"Available wavelets: {list(ALL_WAVELET_FILTERS.keys())}")

        filters = ALL_WAVELET_FILTERS[wavelet_name]
        self.dec_lo = filters['dec_lo'].to(device=self.device, dtype=torch.float32)
        self.dec_hi = filters['dec_hi'].to(device=self.device, dtype=torch.float32)
        self.rec_lo = filters['rec_lo'].to(device=self.device, dtype=torch.float32)
        self.rec_hi = filters['rec_hi'].to(device=self.device, dtype=torch.float32)

    def _apply_dwt1d_to_axis(self, data_tensor: torch.Tensor, axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if axis == 0: data_permuted = data_tensor.permute(1, 0)
        elif axis == 1: data_permuted = data_tensor
        else: raise ValueError("Axis must be 0 (columns) or 1 (rows)")

        coeffs_a_list, coeffs_d_list = [], []
        for i in range(data_permuted.shape[0]):
            cA_i, cD_i = _dwt1d(data_permuted[i, :], self.dec_lo, self.dec_hi)
            coeffs_a_list.append(cA_i)
            coeffs_d_list.append(cD_i)
        
        def _stack_or_pad(coeffs_list, current_dim_size):
            if not coeffs_list: 
                if current_dim_size > 0: 
                    dummy_input = torch.zeros(current_dim_size, device=self.device, dtype=self.dec_lo.dtype)
                    dummy_ca, _ = _dwt1d(dummy_input, self.dec_lo, self.dec_hi)
                    expected_len = dummy_ca.shape[0]
                else: 
                    expected_len = 0
                return torch.empty((0, expected_len), device=self.device, dtype=self.dec_lo.dtype)
            try:
                return torch.stack(coeffs_list, dim=0)
            except RuntimeError:
                max_len = max(c.shape[0] for c in coeffs_list) if coeffs_list else 0
                padded_list = [torch.nn.functional.pad(c, (0, max_len - c.shape[0])) for c in coeffs_list]
                return torch.stack(padded_list, dim=0)

        stacked_cA = _stack_or_pad(coeffs_a_list, data_permuted.shape[1] if data_permuted.ndim > 1 else 0)
        stacked_cD = _stack_or_pad(coeffs_d_list, data_permuted.shape[1] if data_permuted.ndim > 1 else 0)

        if axis == 0: return stacked_cA.permute(1, 0), stacked_cD.permute(1, 0)
        return stacked_cA, stacked_cD

    def _apply_idwt1d_to_axis(self, cA_tensor: torch.Tensor, cD_tensor: torch.Tensor, axis: int, original_length: int) -> torch.Tensor:
        if axis == 0: 
            cA_permuted, cD_permuted = cA_tensor.permute(1, 0), cD_tensor.permute(1, 0)
        elif axis == 1: 
            cA_permuted, cD_permuted = cA_tensor, cD_tensor
        else: raise ValueError("Axis must be 0 (columns) or 1 (rows)")

        reconstructed_list = []
        num_vectors = cA_permuted.shape[0] # Assume cA and cD have compatible number of vectors for IDWT
        if cD_permuted.shape[0] != num_vectors:
            # This might occur if one of them is empty due to very small input to DWT
            # For safety, use min, though ideally DWT output handling should prevent this for valid inputs.
            num_vectors = min(cA_permuted.shape[0], cD_permuted.shape[0])

        for i in range(num_vectors):
            cA_i = cA_permuted[i, :]
            # Ensure cD_i matches cA_i if shapes are slightly off due to earlier padding/empty handling
            cD_i = cD_permuted[i, :cD_permuted.shape[1]] if i < cD_permuted.shape[0] else torch.zeros_like(cA_i)

            recon_i = _idwt1d(cA_i, cD_i, self.rec_lo, self.rec_hi, original_length)
            reconstructed_list.append(recon_i)
        
        if not reconstructed_list : 
             return torch.empty((0, original_length) if axis == 1 else (original_length,0), device=self.device, dtype=self.rec_lo.dtype)

        stacked_recon = torch.stack(reconstructed_list, dim=0)

        if axis == 0: return stacked_recon.permute(1, 0)
        return stacked_recon

    def _wavedec2_scratch(self, data_tensor: torch.Tensor, level: int) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        if data_tensor.ndim != 2: raise ValueError("Input data_tensor must be 2D.")
        data_tensor = data_tensor.to(device=self.device, dtype=torch.float32)

        coeffs_list: List[torch.Tensor] = [] 
        shapes_list: List[Tuple[int, int]] = [] 

        current_cA = data_tensor
        for _ in range(level):
            if current_cA.numel() == 0 or current_cA.shape[0] == 0 or current_cA.shape[1] == 0: 
                shapes_list.append(current_cA.shape) 
                break 
            shapes_list.append(current_cA.shape)
            cA_rows, cD_rows = self._apply_dwt1d_to_axis(current_cA, axis=1)
            cA_level_i, cV_level_i = self._apply_dwt1d_to_axis(cA_rows, axis=0)
            cH_level_i, cD_level_i = self._apply_dwt1d_to_axis(cD_rows, axis=0)
            coeffs_list.insert(0, (cH_level_i, cV_level_i, cD_level_i))
            current_cA = cA_level_i
        
        coeffs_list.insert(0, current_cA)
        shapes_list.reverse() 
        return coeffs_list, shapes_list

    def _waverec2_scratch(self, coeffs_list_with_cA_n_first: List[torch.Tensor], shapes_list: List[Tuple[int, int]]):
        if not coeffs_list_with_cA_n_first: raise ValueError("Coefficient list cannot be empty.")
        
        current_cA = coeffs_list_with_cA_n_first[0].to(device=self.device, dtype=torch.float32)
        
        num_detail_levels_in_coeffs = len(coeffs_list_with_cA_n_first) - 1

        for i in range(num_detail_levels_in_coeffs):
            if i >= len(shapes_list): 
                print("Warning: Mismatch between coefficient levels and shapes_list length during waverec.")
                break
            
            detail_coeffs_tuple = coeffs_list_with_cA_n_first[i + 1]
            if not isinstance(detail_coeffs_tuple, tuple) or len(detail_coeffs_tuple) != 3:
                raise ValueError("Detail coefficients must be a tuple of (cH, cV, cD).")
            
            cH_level_i, cV_level_i, cD_level_i = detail_coeffs_tuple
            cH_level_i = cH_level_i.to(device=self.device, dtype=torch.float32)
            cV_level_i = cV_level_i.to(device=self.device, dtype=torch.float32)
            cD_level_i = cD_level_i.to(device=self.device, dtype=torch.float32)

            target_shape_current_level = shapes_list[i] 
            original_rows_at_this_level = target_shape_current_level[0]
            original_cols_at_this_level = target_shape_current_level[1]
            
            # Ensure current_cA and detail bands are compatible for _apply_idwt1d_to_axis
            # This involves potentially padding if DWT of small/empty inputs led to empty subbands
            def _ensure_compatible_for_idwt(approx_band, detail_band, target_rows_for_idwt_input):
                if approx_band.shape[0] != detail_band.shape[0] and approx_band.numel() > 0 and detail_band.numel() > 0:
                    # If row counts differ for non-empty bands, this is problematic.
                    # Fallback: use min rows. This might lose data or indicate earlier issue.
                    min_rows = min(approx_band.shape[0], detail_band.shape[0])
                    approx_band = approx_band[:min_rows, :]
                    detail_band = detail_band[:min_rows, :]
                elif approx_band.numel() == 0 and detail_band.numel() > 0: # Approx is empty, detail is not
                    approx_band = torch.zeros_like(detail_band)
                elif detail_band.numel() == 0 and approx_band.numel() > 0: # Detail is empty, approx is not
                    detail_band = torch.zeros_like(approx_band)
                elif approx_band.numel() == 0 and detail_band.numel() == 0: # Both empty
                    # Create empty tensors with expected column dim for IDWT input
                    # Column dim for column-wise IDWT is based on output of row-wise DWT.
                    # It's tricky to get this perfectly if all is empty.
                    # Let _apply_idwt1d_to_axis handle it with its empty input logic.
                    pass
                return approx_band, detail_band

            current_cA, cV_level_i = _ensure_compatible_for_idwt(current_cA, cV_level_i, original_rows_at_this_level)
            cH_level_i, cD_level_i = _ensure_compatible_for_idwt(cH_level_i, cD_level_i, original_rows_at_this_level)

            cA_rows_recon = self._apply_idwt1d_to_axis(current_cA, cV_level_i, axis=0, original_length=original_rows_at_this_level)
            cD_rows_recon = self._apply_idwt1d_to_axis(cH_level_i, cD_level_i, axis=0, original_length=original_rows_at_this_level)
            
            cA_rows_recon, cD_rows_recon = _ensure_compatible_for_idwt(cA_rows_recon, cD_rows_recon, original_cols_at_this_level)
            current_cA = self._apply_idwt1d_to_axis(cA_rows_recon, cD_rows_recon, axis=1, original_length=original_cols_at_this_level)
            
        return current_cA

    def forward(self, img_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        if img_tensor.ndim != 2:
            if img_tensor.ndim == 3: 
                coeffs_batch, shapes_list_for_batch = [], None
                for i in range(img_tensor.shape[0]):
                    coeffs, shapes = self._wavedec2_scratch(img_tensor[i], self.level)
                    coeffs_batch.append(coeffs)
                    if i == 0: shapes_list_for_batch = shapes 
                return coeffs_batch, shapes_list_for_batch 
            else:
                raise ValueError(f"Input img_tensor must be 2D (H,W) or 3D (B,H,W), got {img_tensor.ndim}D.")
        return self._wavedec2_scratch(img_tensor, self.level)

    def inverse(self, coeffs_list: List[torch.Tensor], shapes_info: List[Tuple[int, int]]) -> torch.Tensor:
        if isinstance(coeffs_list, list) and len(coeffs_list)>0 and \
           coeffs_list[0] is not None and isinstance(coeffs_list[0], list):
            # Batch of coefficient sets
            recon_batch = [self._waverec2_scratch(coeffs_list[i], shapes_info) for i in range(len(coeffs_list))]
            return torch.stack(recon_batch, dim=0)
        # Single coefficient set
        return self._waverec2_scratch(coeffs_list, shapes_info)

# --- WaveletRegularizationTerm Class ---
class WaveletRegularizationTerm(Regularizer):
    def __init__(self, lambda_reg: float, wavelet_name: str = 'db4', level: int = 3, device: str = 'cpu'):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.wavelet_transform = WaveletTransform(wavelet_name, level, device)
        self.device = torch.device(device) 

    def proximal_operator(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        orig_dtype = x.dtype
        # Proximal operator expects float32 for wavelet transform internal consistency
        x = x.to(dtype=torch.float32) 

        if x.ndim == 2: 
            x = x.to(self.device)
            coeffs_list, slices_info = self.wavelet_transform.forward(x)
            
            threshold = float(self.lambda_reg * steplength)
            thresholded_coeffs_list = []

            if coeffs_list[0] is not None:
                thresholded_coeffs_list.append(coeffs_list[0].to(self.device))
            else: 
                thresholded_coeffs_list.append(None) 

            for details_tuple in coeffs_list[1:]:
                if details_tuple is None : 
                    thresholded_coeffs_list.append(None)
                    continue
                cH_i, cV_i, cD_i = details_tuple
                cH_thresh = torch.sign(cH_i) * torch.clamp(torch.abs(cH_i) - threshold, min=0.0)
                cV_thresh = torch.sign(cV_i) * torch.clamp(torch.abs(cV_i) - threshold, min=0.0)
                cD_thresh = torch.sign(cD_i) * torch.clamp(torch.abs(cD_i) - threshold, min=0.0)
                thresholded_coeffs_list.append((cH_thresh.to(self.device), cV_thresh.to(self.device), cD_thresh.to(self.device)))
            
            x_thresholded = self.wavelet_transform.inverse(thresholded_coeffs_list, slices_info)
            return x_thresholded.to(orig_dtype)

        elif x.ndim == 3: 
            x_thresholded_batch = []
            coeffs_batch, shapes_info_shared = self.wavelet_transform.forward(x) 
            
            for i in range(x.shape[0]): 
                coeffs_list_single_image = coeffs_batch[i]
                threshold = float(self.lambda_reg * steplength)
                thresholded_coeffs_list_single = []
                if coeffs_list_single_image[0] is not None:
                    thresholded_coeffs_list_single.append(coeffs_list_single_image[0].to(self.device))
                else:
                    thresholded_coeffs_list_single.append(None)
                for details_tuple in coeffs_list_single_image[1:]:
                    if details_tuple is None:
                        thresholded_coeffs_list_single.append(None)
                        continue
                    cH_i, cV_i, cD_i = details_tuple
                    cH_thresh = torch.sign(cH_i) * torch.clamp(torch.abs(cH_i) - threshold, min=0.0)
                    cV_thresh = torch.sign(cV_i) * torch.clamp(torch.abs(cV_i) - threshold, min=0.0)
                    cD_thresh = torch.sign(cD_i) * torch.clamp(torch.abs(cD_i) - threshold, min=0.0)
                    thresholded_coeffs_list_single.append((cH_thresh.to(self.device), cV_thresh.to(self.device), cD_thresh.to(self.device)))
                
                x_thresh_single = self.wavelet_transform._waverec2_scratch(thresholded_coeffs_list_single, shapes_info_shared)
                x_thresholded_batch.append(x_thresh_single)
            
            return torch.stack(x_thresholded_batch, dim=0).to(orig_dtype)
        else:
            raise ValueError(f"Input x must be 2D or 3D tensor, got {x.ndim}D.")

    def value(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32) # Value calculation based on float32 representation

        if x.ndim == 2: 
            x = x.to(self.device)
            coeffs_list, _ = self.wavelet_transform.forward(x)
            sum_abs_detail_coeffs = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for details_tuple in coeffs_list[1:]:
                if details_tuple is None: continue 
                cH_i, cV_i, cD_i = details_tuple
                sum_abs_detail_coeffs += torch.sum(torch.abs(cH_i)) + \
                                         torch.sum(torch.abs(cV_i)) + \
                                         torch.sum(torch.abs(cD_i))
            return (self.lambda_reg * sum_abs_detail_coeffs).to(orig_dtype)
        elif x.ndim == 3: 
            total_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            coeffs_batch, _ = self.wavelet_transform.forward(x) 
            for i in range(x.shape[0]): 
                coeffs_list_single_image = coeffs_batch[i]
                sum_abs_detail_coeffs_single = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                for details_tuple in coeffs_list_single_image[1:]:
                    if details_tuple is None: continue
                    cH_i, cV_i, cD_i = details_tuple
                    sum_abs_detail_coeffs_single += torch.sum(torch.abs(cH_i)) + \
                                                   torch.sum(torch.abs(cV_i)) + \
                                                   torch.sum(torch.abs(cD_i))
                total_value += sum_abs_detail_coeffs_single
            return (self.lambda_reg * total_value).to(orig_dtype)
        else:
            raise ValueError(f"Input x must be 2D or 3D tensor, got {x.ndim}D.")

# --- NUFFTWaveletRegularizedReconstructor Class ---
class NUFFTWaveletRegularizedReconstructor(nn.Module):
    def __init__(self, 
                 nufft_op: NUFFTOperator, 
                 wavelet_regularizer: WaveletRegularizationTerm, 
                 n_iter: int = 10, 
                 step_size: float = 1.0):
        super().__init__()
        self.nufft_op = nufft_op
        self.wavelet_regularizer = wavelet_regularizer
        self.n_iter = n_iter
        self.step_size = step_size
        
        if not hasattr(nufft_op, 'device'):
            # Attempt to infer device from regularizer if nufft_op doesn't have it
            # Or default to a common device like CPU if regularizer also doesn't specify.
            # This part depends on actual nufft_op and WaveletRegularizationTerm having .device
            print("Warning: nufft_op does not have a 'device' attribute. Trying wavelet_regularizer.device.")
            if hasattr(wavelet_regularizer, 'device'):
                 self.device = wavelet_regularizer.device
            else: # Fallback if neither has it, though WaveletRegularizationTerm should.
                 self.device = torch.device('cpu')
                 print("Warning: Defaulting Reconstructor device to CPU as it couldn't be inferred.")
        else:
            self.device = nufft_op.device


    def forward(self, 
                kspace_data: torch.Tensor, 
                sensitivity_maps: torch.Tensor, 
                initial_image_estimate: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        kspace_data = kspace_data.to(self.device)
        sensitivity_maps = sensitivity_maps.to(self.device)

        # Assuming sensitivity_maps: (num_coils, H, W) or (num_coils, D, H, W)
        # img_dims = sensitivity_maps.shape[1:] # H, W or D, H, W

        if initial_image_estimate is None:
            # SENSE-like initial estimate: sum_coils( smaps_conj * NUFFT_adj(k_coil) )
            # This assumes nufft_op.op_adj can handle kspace_data shaped (num_coils, ...) 
            # and returns images shaped (num_coils, H, W)
            img_estimate = torch.sum(self.nufft_op.op_adj(kspace_data) * sensitivity_maps.conj(), dim=0)
        else:
            img_estimate = initial_image_estimate.clone().to(self.device)

        for _ in range(self.n_iter):
            # Forward Model (SENSE-like)
            # img_estimate: (H, W) -> unsqueeze to (1, H, W) then multiply by smaps (C, H, W) -> (C, H, W)
            coils_img_estimate = sensitivity_maps * img_estimate.unsqueeze(0) 
            
            # k_pred: (num_coils, num_kpoints_total) assuming nufft_op.op handles batch of coils
            k_pred = self.nufft_op.op(coils_img_estimate)
            
            # Residue
            k_resid = k_pred - kspace_data 
            
            # Adjoint for Gradient (SENSE-like)
            # coils_grad_img: (num_coils, H, W)
            coils_grad_img = self.nufft_op.op_adj(k_resid)
            # grad_img: (H, W)
            grad_img = torch.sum(coils_grad_img * sensitivity_maps.conj(), dim=0)
            
            # Gradient Descent Update
            img_estimate = img_estimate - self.step_size * grad_img
            
            # Wavelet Regularization (Proximal Operator)
            # Ensure img_estimate is complex if it became real after GD, but prox handles floats.
            # WaveletRegularizationTerm currently expects float32 internally for wavelet transform.
            # If img_estimate is complex, prox might need to handle real/imag parts or magnitude.
            # For now, assume img_estimate is complex and prox will handle it (e.g. by processing real/imag separately or magnitude)
            # The current WaveletRegularizationTerm is designed for real-valued images.
            # If img_estimate is complex, this step needs careful handling.
            # Common approach: apply prox to real and imag parts separately if regularizer is for real values.
            # Or, if problem assumes real image, ensure img_estimate is real.
            # For now, let's assume the problem implies img_estimate could be complex and
            # wavelet_regularizer.proximal_operator is robust or we handle it here.
            
            if torch.is_complex(img_estimate):
                real_part = self.wavelet_regularizer.proximal_operator(img_estimate.real, self.step_size)
                imag_part = self.wavelet_regularizer.proximal_operator(img_estimate.imag, self.step_size) # Potentially regularize imag too
                # Or, if only real part is regularized (common for magnitude images where phase is less structured):
                # imag_part = img_estimate.imag 
                img_estimate = torch.complex(real_part, imag_part)
            else: # Real image
                img_estimate = self.wavelet_regularizer.proximal_operator(img_estimate, self.step_size)
            
        return img_estimate


if __name__ == '__main__':
    # --- Previous tests (condensed) ---
    print("--- Running Basic Sanity Checks for 1D and 2D Transforms ---")
    haar_filters = ALL_WAVELET_FILTERS['haar']
    L_d_haar, H_d_haar, L_r_haar, H_r_haar = haar_filters['dec_lo'], haar_filters['dec_hi'], haar_filters['rec_lo'], haar_filters['rec_hi']
    data_even = torch.arange(1, 11, dtype=torch.float32) 
    cA_e, cD_e = _dwt1d(data_even, L_d_haar, H_d_haar)
    reconstructed_even = _idwt1d(cA_e, cD_e, L_r_haar, H_r_haar, original_length=data_even.shape[0])
    print(f"1D Even Rec Test Passed: {torch.allclose(reconstructed_even, data_even, atol=1e-6)}")
    
    device_test = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_image = torch.rand(16, 16, dtype=torch.float32).to(device_test)
    transformer = WaveletTransform(wavelet_name='haar', level=2, device=device_test)
    coeffs_list, shapes_list_info = transformer.forward(original_image)
    reconstructed_image = transformer.inverse(coeffs_list, shapes_list_info)
    print(f"2D Haar Rec Test Passed: {torch.allclose(reconstructed_image, original_image, atol=1e-5)}")

    print("\n--- Testing WaveletRegularizationTerm Class (condensed) ---")
    regularizer_test_instance = WaveletRegularizationTerm(lambda_reg=0.1, wavelet_name='haar', device=device_test)
    sample_tensor_reg = torch.rand(8, 8, dtype=torch.float32).to(device_test)
    prox_output_reg = regularizer_test_instance.proximal_operator(sample_tensor_reg.clone(), 1.0)
    value_output_reg = regularizer_test_instance.value(sample_tensor_reg.clone())
    print(f"Regularizer prox output shape: {prox_output_reg.shape}, value: {value_output_reg.item()}")
    assert prox_output_reg.shape == sample_tensor_reg.shape
    assert value_output_reg >= 0.0

    # --- Test for NUFFTWaveletRegularizedReconstructor class ---
    print("\n--- Testing NUFFTWaveletRegularizedReconstructor Instantiation ---")
    
    class MockNUFFTOp:
        def __init__(self, device, shape=(16,16), num_coils=1, num_kpoints=256):
            self.device = device
            self.shape = shape # H, W for image
            self.num_coils = num_coils
            self.num_kpoints = num_kpoints

        def op(self, x: torch.Tensor) -> torch.Tensor: # x: (C, H, W)
            # Output: (C, K)
            if x.ndim == 2: # Allow (H,W) for single coil case test
                x = x.unsqueeze(0)
            return torch.randn(x.shape[0], self.num_kpoints, dtype=torch.complex64, device=self.device)

        def op_adj(self, y: torch.Tensor) -> torch.Tensor: # y: (C, K)
            # Output: (C, H, W)
            if y.ndim == 1: # Allow (K,) for single coil case test
                y = y.unsqueeze(0)
            return torch.randn(y.shape[0], *self.shape, dtype=torch.complex64, device=self.device)

    mock_nufft = MockNUFFTOp(device=torch.device(device_test))
    wave_reg = WaveletRegularizationTerm(lambda_reg=0.01, wavelet_name='haar', device=torch.device(device_test))
    
    try:
        reconstructor = NUFFTWaveletRegularizedReconstructor(nufft_op=mock_nufft, wavelet_regularizer=wave_reg)
        print("NUFFTWaveletRegularizedReconstructor instantiated successfully.")

        # Test forward pass with mock data
        print("\n--- Testing NUFFTWaveletRegularizedReconstructor Forward Pass ---")
        num_coils_test = 2
        img_h, img_w = 16, 16
        num_kpoints_test = 256

        mock_nufft_mc = MockNUFFTOp(device=torch.device(device_test), shape=(img_h, img_w), num_coils=num_coils_test, num_kpoints=num_kpoints_test)
        wave_reg_mc = WaveletRegularizationTerm(lambda_reg=0.01, wavelet_name='haar', level=2, device=torch.device(device_test))
        reconstructor_mc = NUFFTWaveletRegularizedReconstructor(nufft_op=mock_nufft_mc, wavelet_regularizer=wave_reg_mc, n_iter=3, step_size=0.5)

        # Create mock inputs
        kspace_data_test = torch.randn(num_coils_test, num_kpoints_test, dtype=torch.complex64).to(device_test)
        sensitivity_maps_test = torch.randn(num_coils_test, img_h, img_w, dtype=torch.complex64).to(device_test)
        
        print(f"kspace_data_test shape: {kspace_data_test.shape}")
        print(f"sensitivity_maps_test shape: {sensitivity_maps_test.shape}")

        # Test with no initial estimate
        print("Testing forward pass without initial estimate...")
        recon_image_no_init = reconstructor_mc.forward(kspace_data_test, sensitivity_maps_test)
        print(f"Reconstructed image shape (no init): {recon_image_no_init.shape}")
        assert recon_image_no_init.shape == (img_h, img_w), "Shape mismatch (no init)."
        assert recon_image_no_init.device == torch.device(device_test), "Device mismatch (no init)."

        # Test with initial estimate
        print("Testing forward pass with initial estimate...")
        initial_estimate_test = torch.randn(img_h, img_w, dtype=torch.complex64).to(device_test)
        recon_image_with_init = reconstructor_mc.forward(kspace_data_test, sensitivity_maps_test, initial_image_estimate=initial_estimate_test)
        print(f"Reconstructed image shape (with init): {recon_image_with_init.shape}")
        assert recon_image_with_init.shape == (img_h, img_w), "Shape mismatch (with init)."
        assert recon_image_with_init.device == torch.device(device_test), "Device mismatch (with init)."
        
        print("NUFFTWaveletRegularizedReconstructor forward pass tests completed (mock data).")

    except Exception as e:
        print(f"Error during NUFFTWaveletRegularizedReconstructor testing: {e}")
        import traceback
        traceback.print_exc()
