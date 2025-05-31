import torch
import numpy as np
from typing import Tuple, Optional
from reconlib.operators import Operator # Assuming Operator base class exists

# --- Basic Radon Transform Utilities (adapted from pcct.operators) ---
# These are simplified and assume parallel beam.

def _simple_radon_transform(image: torch.Tensor,
                            angles_rad: torch.Tensor,
                            num_detector_pixels: int,
                            device: str = 'cpu') -> torch.Tensor:
    """
    Internal helper function for a simplified parallel-beam Radon transform.

    This function performs a basic forward projection of a 2D image onto a
    sinogram. It iterates through each projection angle, calculates the rotated
    coordinates of image pixels, and sums pixel values that project onto each
    detector bin. This is a simplified model, primarily for testing and basic
    demonstration, and does not account for more complex physics like beam hardening
    or scatter.

    Args:
        image (torch.Tensor): The input 2D image tensor of shape (Ny, Nx).
        angles_rad (torch.Tensor): A 1D tensor containing the projection angles
            in radians.
        num_detector_pixels (int): The number of detector pixels (bins) in the sinogram.
        device (str): The computation device ('cpu' or 'cuda') for tensor operations.

    Returns:
        torch.Tensor: The computed 2D sinogram tensor of shape (num_angles, num_detector_pixels).
    """
    Ny, Nx = image.shape
    image = image.to(device)
    angles_rad = angles_rad.to(device)
    num_angles = angles_rad.shape[0]

    sinogram = torch.zeros((num_angles, num_detector_pixels), device=device, dtype=image.dtype)

    # Create a grid of coordinates for the image
    x_coords = torch.linspace(-Nx // 2, Nx // 2 - 1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 - 1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 - 1,
                                     num_detector_pixels, device=device)

    for i, angle in enumerate(angles_rad):
        # Rotated coordinates: t = x*cos(theta) + y*sin(theta)
        rot_coords = grid_x * torch.cos(angle) + grid_y * torch.sin(angle)

        for j, det_pos in enumerate(detector_coords):
            pixel_width_on_detector = 1.0
            mask = (rot_coords >= det_pos - pixel_width_on_detector/2) & \
                   (rot_coords < det_pos + pixel_width_on_detector/2)
            sinogram[i, j] = torch.sum(image[mask])
    return sinogram

def _simple_back_projection(sinogram: torch.Tensor,
                            image_shape: tuple[int,int],
                            angles_rad: torch.Tensor,
                            device: str ='cpu') -> torch.Tensor:
    """
    Simplified parallel-beam back-projection.
    Output image shape: image_shape.

    Args:
        sinogram (torch.Tensor): The input 2D sinogram tensor of shape
            (num_angles, num_detector_pixels).
        image_shape (tuple[int,int]): The target shape (Ny, Nx) for the
            reconstructed image.
        angles_rad (torch.Tensor): A 1D tensor containing the projection angles
            in radians, corresponding to the sinogram.
        device (str): The computation device ('cpu' or 'cuda') for tensor operations.

    Returns:
        torch.Tensor: The reconstructed 2D image tensor of shape `image_shape`.
    """
    num_angles_sino, num_detector_pixels = sinogram.shape
    Ny, Nx = image_shape
    sinogram = sinogram.to(device)
    angles_rad = angles_rad.to(device)

    if angles_rad.shape[0] != num_angles_sino:
        raise ValueError(f"Number of angles in sinogram ({num_angles_sino}) must match angles_rad ({angles_rad.shape[0]}).")

    reconstructed_image = torch.zeros(image_shape, device=device, dtype=sinogram.dtype)

    x_coords = torch.linspace(-Nx // 2, Nx // 2 - 1 , Nx, device=device)
    y_coords = torch.linspace(-Ny // 2, Ny // 2 - 1 , Ny, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    detector_coords = torch.linspace(-num_detector_pixels // 2, num_detector_pixels // 2 - 1,
                                     num_detector_pixels, device=device)

    for i, angle in enumerate(angles_rad):
        rot_coords_pixel = grid_x * torch.cos(angle) + grid_y * torch.sin(angle)
        diffs = torch.abs(rot_coords_pixel.unsqueeze(-1) - detector_coords.view(1,1,-1))
        nearest_det_indices = torch.argmin(diffs, dim=2)
        reconstructed_image += sinogram[i, nearest_det_indices]

    return reconstructed_image / num_angles_sino if num_angles_sino > 0 else reconstructed_image

# --- End of Basic Radon Transform Utilities ---

class CTProjectorOperator(Operator):
    """
    A linear operator for Computed Tomography (CT) forward and adjoint projections.

    This class encapsulates a simplified parallel-beam Radon transform (`op` method)
    and its corresponding back-projection (`op_adj` method). It is designed to be
    used within iterative reconstruction algorithms where repeated applications of
    the forward and adjoint models are necessary.

    The underlying projection model is pixel-based and does not incorporate
    advanced physical effects like beam hardening, scatter, or detector response.
    It serves as a basic building block or for educational purposes.

    Inherits from `reconlib.operators.Operator`.
    """
    def __init__(self,
                 image_shape: Tuple[int, int],
                 angles_rad: torch.Tensor,
                 detector_pixels: int,
                 device: Optional[str] = None):
        """
        Initializes the CTProjectorOperator.

        Args:
            image_shape (Tuple[int, int]): The shape of the input image (Ny, Nx)
                that this operator will work with.
            angles_rad (torch.Tensor): A 1D tensor containing the projection angles
                in radians. These angles define the geometry of the Radon transform.
            detector_pixels (int): The number of detector pixels or bins for each
                projection angle in the sinogram.
            device (Optional[str]): The computation device ('cpu' or 'cuda') for
                tensor operations. If None, defaults to 'cpu'.
        """
        super().__init__()
        self.image_shape = image_shape
        self.angles_rad = angles_rad
        self.detector_pixels = detector_pixels
        self.device = device if device is not None else 'cpu'

        self.num_angles = angles_rad.shape[0]
        self.sinogram_shape = (self.num_angles, self.detector_pixels)

    def op(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward projection (Radon transform) of an image.

        Args:
            image (torch.Tensor): The input 2D image tensor, expected to match
                `self.image_shape`.

        Returns:
            torch.Tensor: The resulting 2D sinogram tensor, of shape
                (num_angles, detector_pixels) as defined by `self.sinogram_shape`.
        """
        if image.shape != self.image_shape:
            raise ValueError(f"Input image shape {image.shape} must match operator's image_shape {self.image_shape}.")
        return _simple_radon_transform(image, self.angles_rad, self.detector_pixels, self.device)

    def op_adj(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Performs the adjoint operation (Back-projection) of a sinogram.

        Args:
            sinogram (torch.Tensor): The input 2D sinogram tensor, expected to match
                `self.sinogram_shape`.

        Returns:
            torch.Tensor: The reconstructed 2D image tensor, of shape `self.image_shape`.
        """
        if sinogram.shape != self.sinogram_shape:
            raise ValueError(f"Input sinogram shape {sinogram.shape} must match operator's sinogram_shape {self.sinogram_shape}.")
        return _simple_back_projection(sinogram, self.image_shape, self.angles_rad, self.device)

__all__ = ['CTProjectorOperator']
