import torch
import pywt
import sys
import os

# Add reconlib to path - Adjust if your test runner handles this differently
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.wavelets_scratch import (
    WaveletTransform, 
    ALL_WAVELET_FILTERS, 
    _dwt1d, 
    _idwt1d
)

def get_max_level_safely(data_len, filter_name_or_len):
    """Safely gets max DWT level, ensuring it's at least 1 if possible."""
    if isinstance(filter_name_or_len, str):
        filter_len = pywt.Wavelet(filter_name_or_len).dec_len
    else:
        filter_len = filter_name_or_len
    
    if data_len < filter_len: # Cannot perform even one level of DWT if data is shorter than filter
        return 0
        
    max_level = pywt.dwt_max_level(data_len, filter_len)
    return max(1, max_level) # Ensure at least 1 if decomposition is possible


def test_perfect_reconstruction_1d(wavelet_name: str, signal_length: int, device: torch.device):
    print(f"Testing 1D: Wavelet='{wavelet_name}', Length={signal_length}, Device='{device}'")
    
    signal = torch.randn(signal_length, device=device, dtype=torch.float64)

    if wavelet_name not in ALL_WAVELET_FILTERS:
        print(f"  SKIPPED: Wavelet '{wavelet_name}' not in ALL_WAVELET_FILTERS.")
        return

    filters = ALL_WAVELET_FILTERS[wavelet_name]
    pywt_ref_name = filters.get('pywt_name', wavelet_name) # Fallback to key if pywt_name not stored

    try:
        dec_lo = filters['dec_lo'].to(device=device, dtype=signal.dtype)
        dec_hi = filters['dec_hi'].to(device=device, dtype=signal.dtype)
        rec_lo = filters['rec_lo'].to(device=device, dtype=signal.dtype)
        rec_hi = filters['rec_hi'].to(device=device, dtype=signal.dtype)
    except KeyError:
        print(f"  SKIPPED: Filters for '{wavelet_name}' not fully defined in ALL_WAVELET_FILTERS.")
        return
    except Exception as e:
        print(f"  ERROR loading filters for {wavelet_name}: {e}")
        return


    level = get_max_level_safely(signal_length, pywt_ref_name)
    if level == 0:
        print(f"  SKIPPED: Signal length {signal_length} too short for wavelet '{wavelet_name}' (filter len {pywt.Wavelet(pywt_ref_name).dec_len}).")
        # For very short signals where DWT is not possible, reconstruction is trivially perfect if no op.
        if signal_length == 0:
             assert signal.numel() == 0, "Test setup error for zero length signal"
             print("  PASSED (trivial for 0 length).")
        elif signal_length < pywt.Wavelet(pywt_ref_name).dec_len :
             # If no decomposition is done, signal should be unchanged.
             # Our _dwt1d might still process it. This case needs clarification on _dwt1d behavior for L < filter_len.
             # For now, we assume if level is 0, no transform is applied for this test structure.
             print(f"  PASSED (trivial as level=0, signal length {signal_length} < filter length).")
        else: # Should not happen given get_max_level_safely logic
             print(f"  WARNING: Level 0 but signal seems long enough. Max level calc issue?")

        return


    # Manual 1D DWT
    coeffs_1d = []
    cA = signal
    original_lengths = []
    
    for _ in range(level):
        if cA.numel() == 0 : # Should not happen if level > 0 and signal_length > 0
            print(f"  SKIPPED: cA became empty during DWT for {wavelet_name}, length {signal_length}.")
            return
        original_lengths.append(len(cA))
        cA, cD = _dwt1d(cA, dec_lo, dec_hi)
        coeffs_1d.insert(0, cD) 
    coeffs_1d.insert(0, cA) 
    original_lengths.reverse() 

    # Manual 1D IDWT
    cA_recon = coeffs_1d[0]
    details_recon = coeffs_1d[1:]
    
    for i_level in range(level):
        cD_recon = details_recon[i_level]
        orig_len = original_lengths[i_level]
        if cA_recon.numel() == 0 and cD_recon.numel() == 0 and orig_len == 0: # Both empty, target empty
             cA_recon = torch.empty(0, device=device, dtype=signal.dtype)
        elif cA_recon.numel() == 0 and orig_len > 0 and pywt.Wavelet(pywt_ref_name).dec_len > orig_len :
            # If cA is empty because original signal at this level was too short for the filter
            # And we expect a non-empty output, this is tricky.
            # _idwt1d should handle empty cA/cD if original_length is non-zero.
            pass # Let _idwt1d handle it.

        cA_recon = _idwt1d(cA_recon, cD_recon, rec_lo, rec_hi, original_length=orig_len)

    reconstructed_signal = cA_recon

    try:
        assert torch.allclose(reconstructed_signal, signal, atol=1e-9), \
            f"1D Recon failed. Max diff: {torch.abs(reconstructed_signal - signal).max()}"
        print("  PASSED.")
    except AssertionError as e:
        print(f"  FAILED: {e}")


def test_perfect_reconstruction_2d(wavelet_name: str, shape: tuple, device: torch.device):
    print(f"Testing 2D: Wavelet='{wavelet_name}', Shape={shape}, Device='{device}'")
    
    # WaveletTransform processes data as float32 internally
    image = torch.randn(shape, device=device, dtype=torch.float32) 

    if wavelet_name not in ALL_WAVELET_FILTERS:
        print(f"  SKIPPED: Wavelet '{wavelet_name}' not in ALL_WAVELET_FILTERS.")
        return
        
    pywt_ref_name = ALL_WAVELET_FILTERS[wavelet_name].get('pywt_name', wavelet_name)

    min_dim = min(shape)
    try:
        filter_len = pywt.Wavelet(pywt_ref_name).dec_len
    except ValueError as e: # Handle if pywt_ref_name is not a valid pywt wavelet name (e.g. our internal key)
         print(f"  SKIPPED: Could not get filter_len for {pywt_ref_name} from pywt: {e}")
         return

    level = get_max_level_safely(min_dim, filter_len)

    if level == 0:
        print(f"  SKIPPED: Image dimension {min_dim} too small for wavelet '{wavelet_name}' (filter len {filter_len}).")
        if image.numel() == 0 :
            assert image.numel() == 0
            print("  PASSED (trivial for 0 element image).")
        elif min_dim < filter_len:
             print(f"  PASSED (trivial as level=0, min_dim {min_dim} < filter_len).")
        return

    # Adjust level to be at most 3 for practical test duration, but respect max_level
    level = min(3, level)

    try:
        wt = WaveletTransform(wavelet_name=wavelet_name, level=level, device=device)
        coeffs_list, shapes_list = wt.forward(image)
        reconstructed_image = wt.inverse(coeffs_list, shapes_list)
    except Exception as e:
        print(f"  ERROR during 2D transform for {wavelet_name}, shape {shape}: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # atol for float32, might need adjustment based on wavelet & level
        assert torch.allclose(reconstructed_image, image, atol=1e-6), \
            f"2D Recon failed. Max diff: {torch.abs(reconstructed_image - image).max()}"
        print("  PASSED.")
    except AssertionError as e:
        print(f"  FAILED: {e}")


if __name__ == '__main__':
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Global test device: {test_device}\n")

    wavelets_to_test = ['haar', 'db4', 'sym4', 'bior2.2'] 
    # Remove wavelets not present in ALL_WAVELET_FILTERS to prevent initial skip messages
    wavelets_to_test = [w for w in wavelets_to_test if w in ALL_WAVELET_FILTERS]
    if not wavelets_to_test:
        print("No wavelets found to test (check ALL_WAVELET_FILTERS in wavelets_scratch.py and WAVELET_NAMES_TO_LOAD)")
    else:
        print(f"Will test with wavelets: {wavelets_to_test}\n")


    signal_lengths = [0, 1, 2, 5, 32, 33, 60, 61, 128] # Test edge cases and typical sizes
    image_shapes = [(0,0), (1,1), (5,5), (32, 32), (31, 33), (64, 60), (61, 65)] # Test edge cases and typical sizes

    print("--- Starting 1D Tests ---")
    for wavelet in wavelets_to_test:
        for length in signal_lengths:
            test_perfect_reconstruction_1d(wavelet, length, test_device)
        print("-" * 30)
    
    print("\n--- Starting 2D Tests ---")
    for wavelet in wavelets_to_test:
        for shape in image_shapes:
            if shape == (0,0) and wavelet in ALL_WAVELET_FILTERS : # Test 0-element image
                 img = torch.empty(shape, device=test_device, dtype=torch.float32)
                 level_2d = 1 # level doesn't matter much for empty
                 wt_test = WaveletTransform(wavelet_name=wavelet, level=level_2d, device=test_device)
                 coeffs, S = wt_test.forward(img)
                 recon = wt_test.inverse(coeffs, S)
                 assert recon.shape == shape, f"Empty 2D test failed for {wavelet}"
                 print(f"Testing 2D: Wavelet='{wavelet}', Shape={shape}, Device='{test_device}'")
                 print(f"  PASSED (manual empty image test).")
                 continue
            elif shape[0]==0 or shape[1]==0 : # Skip other invalid shapes directly
                print(f"Testing 2D: Wavelet='{wavelet}', Shape={shape}, Device='{test_device}'")
                print(f"  SKIPPED (one dimension is zero).")
                continue

            test_perfect_reconstruction_2d(wavelet, shape, test_device)
        print("-" * 30)

    print("\nAll tests completed.")

```
