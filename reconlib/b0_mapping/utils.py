# reconlib/b0_mapping/utils.py
"""Utility functions for B0 mapping."""

import numpy as np

def create_mask_from_magnitude(magnitude_image, threshold_factor=0.1):
    """
    Creates a simple binary mask by thresholding a magnitude image.

    Args:
        magnitude_image (np.ndarray): Input magnitude image.
        threshold_factor (float): Factor of the maximum intensity to use as threshold.
                                  Defaults to 0.1.

    Returns:
        np.ndarray: Boolean mask.
    """
    if not isinstance(magnitude_image, np.ndarray):
        raise TypeError("Magnitude image must be a NumPy array.")
    if magnitude_image.size == 0:
        return np.array([], dtype=bool) # Handle empty array
        
    max_val = np.max(magnitude_image)
    if max_val == 0: # Avoid division by zero or issues with all-zero images
        return np.zeros_like(magnitude_image, dtype=bool)
        
    threshold = threshold_factor * max_val
    mask = magnitude_image > threshold
    return mask
