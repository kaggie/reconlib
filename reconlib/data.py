"""Module for MRI data representation."""

import numpy as np

class MRIData:
    """
    Represents MRI data, including k-space data, trajectory, coil sensitivities,
    image shape, and optional reference image and density compensation weights.
    """
    def __init__(self, 
                 k_space_data: np.ndarray, 
                 k_trajectory: np.ndarray, 
                 image_shape: tuple, 
                 coil_sensitivities: np.ndarray = None, 
                 reference_image: np.ndarray = None, 
                 density_comp_weights: np.ndarray = None):
        """
        Initializes an MRIData object.

        Args:
            k_space_data: The k-space data (e.g., NumPy array).
            k_trajectory: The k-space trajectory (e.g., NumPy array, 
                          shape (num_points, dimensions) where dimensions 
                          can be 2 for 2D or 3 for 3D).
            image_shape: The shape of the image (tuple, e.g., (H, W) or (D, H, W)).
            coil_sensitivities: Optional coil sensitivity maps (e.g., NumPy array, default None).
            reference_image: An optional reference image (default None).
            density_comp_weights: Optional density compensation weights (default None).
        """
        self.k_space_data = k_space_data
        self.k_trajectory = k_trajectory
        self.image_shape = image_shape
        self.coil_sensitivities = coil_sensitivities
        self.reference_image = reference_image
        self.density_comp_weights = density_comp_weights
