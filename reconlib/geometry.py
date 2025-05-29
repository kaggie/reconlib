"""Module for defining scanner geometry and system matrices for PET/CT reconstruction."""

import numpy as np
import torch
from abc import ABC, abstractmethod

from reconlib.operators import PETForwardProjection, IRadon
# Placeholder for image size, will be refined later
DEFAULT_IMG_SIZE = (128, 128)

class ScannerGeometry:
    """Defines the geometry of a PET or CT scanner."""

    def __init__(self,
                 detector_positions: np.ndarray,
                 angles: np.ndarray,
                 detector_size: np.ndarray, # e.g., [width, height] or [spacing_x, spacing_y]
                 geometry_type: str, # e.g., 'fanbeam', 'cone-beam', 'cylindrical_pet'
                 n_detector_pixels: int, # Number of detector pixels along one dimension
                 ):
        """
        Initializes the ScannerGeometry.

        Args:
            detector_positions (np.ndarray): Positions of the detector elements.
                                             For PET, this could be crystal positions.
                                             For CT, this could be detector array positions over angles.
            angles (np.ndarray): Projection angles in radians.
            detector_size (np.ndarray): Physical size of a detector element or pixel spacing.
            geometry_type (str): Type of scanner geometry (e.g., 'fanbeam', 'cone-beam', 'cylindrical_pet').
            n_detector_pixels (int): Number of detector elements/pixels in one dimension of a projection.
        """
        self.detector_positions = detector_positions
        self.angles = angles
        self.detector_size = detector_size
        self.geometry_type = geometry_type
        self.n_detector_pixels = n_detector_pixels # Added based on common needs for operators

    def generate_rays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates ray origin and direction vectors for the scanner geometry.

        Returns:
            A tuple containing two numpy arrays:
            - ray_origins: The origin points of the rays.
            - ray_directions: The direction vectors of the rays.
        """
        raise NotImplementedError("Ray generation is not yet implemented for this geometry.")

    def visualize_geometry(self) -> None:
        """
        Visualizes the scanner geometry.
        (e.g., using matplotlib or other plotting libraries)
        """
        raise NotImplementedError("Geometry visualization is not yet implemented.")


class SystemMatrix(ABC):
    """
    Represents the system matrix for tomographic reconstruction,
    encapsulating the forward and backward projection operations.
    """

    def __init__(self, scanner_geometry: ScannerGeometry, img_size: tuple[int, int] = DEFAULT_IMG_SIZE, device: str = 'cpu'):
        """
        Initializes the SystemMatrix.

        Args:
            scanner_geometry (ScannerGeometry): The scanner geometry definition.
            img_size (tuple[int, int]): The size of the image to be reconstructed (height, width).
            device (str): The computational device ('cpu' or 'cuda').
        """
        self.scanner_geometry = scanner_geometry
        self.img_size = img_size
        self.device = device
        self.projector_op = None

        if self.scanner_geometry.geometry_type == 'cylindrical_pet':
            # For PET, n_angles is len(angles), n_detectors is related to n_detector_pixels
            # This is a simplification; PETForwardProjection might need more specific params
            self.projector_op = PETForwardProjection(
                n_subsets=1, # Default, can be configured
                n_angles=len(self.scanner_geometry.angles),
                n_detectors=self.scanner_geometry.n_detector_pixels,
                img_size=self.img_size
            )
            # PET projectors typically don't use 'op_adj' for projection, but 'op'
            self._forward_is_op = True
        elif self.scanner_geometry.geometry_type in ['fanbeam', 'parallelbeam']:
            # IRadon takes angles and number of detector pixels (n_rays_per_proj)
            self.projector_op = IRadon(
                angles=self.scanner_geometry.angles,
                n_rays_per_proj=self.scanner_geometry.n_detector_pixels,
                img_size=self.img_size,
                filter_type=None # No filtering for basic projection
            )
            # IRadon's op is FBP (backward), op_adj is Radon transform (forward)
            self._forward_is_op = False
        else:
            raise ValueError(f"Unsupported geometry type: {self.scanner_geometry.geometry_type}")

        # Ensure the operator is on the correct device if it has a to() method
        if hasattr(self.projector_op, 'to') and callable(getattr(self.projector_op, 'to')):
            self.projector_op.to(self.device)

    def forward_project(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs forward projection of an image to projection data (sinogram).

        Args:
            image (torch.Tensor): The image to project.

        Returns:
            torch.Tensor: The resulting projection data.
        """
        if self.projector_op is None:
            raise RuntimeError("Projector operator not initialized.")
        if self._forward_is_op:
            return self.projector_op.op(image)
        else:
            return self.projector_op.op_adj(image)

    def backward_project(self, projection_data: torch.Tensor) -> torch.Tensor:
        """
        Performs back-projection of projection data to image space.

        Args:
            projection_data (torch.Tensor): The projection data to back-project.

        Returns:
            torch.Tensor: The resulting image.
        """
        if self.projector_op is None:
            raise RuntimeError("Projector operator not initialized.")
        if self._forward_is_op:
            return self.projector_op.op_adj(projection_data)
        else:
            return self.projector_op.op(projection_data)

    def op(self, image: torch.Tensor) -> torch.Tensor:
        """Performs forward projection using the configured projector.
        This method fulfills the Operator abstract base class requirement.
        """
        return self.forward_project(image)

    def op_adj(self, projection_data: torch.Tensor) -> torch.Tensor:
        """Performs back-projection using the configured projector.
        This method fulfills the Operator abstract base class requirement.
        """
        return self.backward_project(projection_data)

# Example usage (commented out, for testing/illustration if run directly)
# if __name__ == '__main__':
#     # PET Example
#     angles_pet = np.linspace(0, np.pi, 180, endpoint=False)
#     detectors_pet = np.arange(200) # Simplified
#     detector_size_pet = np.array([4.0, 4.0]) # mm
#     n_pixels_pet = 200
#     pet_geom = ScannerGeometry(detector_positions=detectors_pet,
#                                angles=angles_pet,
#                                detector_size=detector_size_pet,
#                                geometry_type='cylindrical_pet',
#                                n_detector_pixels=n_pixels_pet)
#     pet_sys_matrix = SystemMatrix(scanner_geometry=pet_geom, img_size=(128,128))
#     dummy_image_pet = torch.randn(1, 1, 128, 128)
#     sinogram_pet = pet_sys_matrix.forward_project(dummy_image_pet)
#     back_projected_pet = pet_sys_matrix.backward_project(sinogram_pet)
#     print("PET Sinogram shape:", sinogram_pet.shape)
#     print("PET Back-projected shape:", back_projected_pet.shape)

    # CT Example (Fanbeam)
    # angles_ct = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # n_detectors_ct = 256
    # # For fanbeam, detector_positions might be more complex or implicit in IRadon
    # fan_geom = ScannerGeometry(detector_positions=np.zeros((n_detectors_ct, 2)), # Placeholder
    #                            angles=angles_ct,
    #                            detector_size=np.array([1.0, 1.0]), # mm
    #                            geometry_type='fanbeam',
    #                            n_detector_pixels=n_detectors_ct)
    # ct_sys_matrix = SystemMatrix(scanner_geometry=fan_geom, img_size=(256,256))
    # dummy_image_ct = torch.randn(1, 1, 256, 256)
    # sinogram_ct = ct_sys_matrix.forward_project(dummy_image_ct)
    # back_projected_ct = ct_sys_matrix.backward_project(sinogram_ct)
    # print("CT Sinogram shape:", sinogram_ct.shape) # Should match (batch, 1, n_angles, n_detectors_ct) for IRadon
    # print("CT Back-projected shape:", back_projected_ct.shape)
