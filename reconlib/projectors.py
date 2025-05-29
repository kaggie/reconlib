"""Module for defining Forward and Backward Projector classes for PET/CT."""

import torch
from reconlib.geometry import ScannerGeometry, SystemMatrix
# The following imports are not strictly needed if SystemMatrix handles operator selection,
# but good for context or if these classes were to be more complex directly using them.
# from reconlib.operators import PETForwardProjection, IRadon

# Default image size, can be overridden in constructor
DEFAULT_IMG_SIZE = (128, 128)

class ForwardProjector:
    """Encapsulates the forward projection operation for a given scanner geometry."""

    def __init__(self,
                 scanner_geometry: ScannerGeometry,
                 img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
                 device: str = 'cpu'):
        """
        Initializes the ForwardProjector.

        Args:
            scanner_geometry (ScannerGeometry): The scanner geometry definition.
            img_size (tuple[int, int]): The size of the image to be reconstructed (height, width).
            device (str): The computational device ('cpu' or 'cuda').
        """
        self.scanner_geometry = scanner_geometry
        self.img_size = img_size
        self.device = device
        self.system_matrix = SystemMatrix(scanner_geometry=self.scanner_geometry,
                                          img_size=self.img_size,
                                          device=self.device)

    def project(self, image: torch.Tensor) -> torch.Tensor:
        """
        Projects an image to sinogram/projection data.

        Args:
            image (torch.Tensor): The image to project.
                                  Expected shape: (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The resulting projection data.
        """
        if image.device.type != self.device:
            image = image.to(self.device)
        return self.system_matrix.forward_project(image)


class BackwardProjector:
    """Encapsulates the backward projection operation for a given scanner geometry."""

    def __init__(self,
                 scanner_geometry: ScannerGeometry,
                 img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
                 device: str = 'cpu'):
        """
        Initializes the BackwardProjector.

        Args:
            scanner_geometry (ScannerGeometry): The scanner geometry definition.
            img_size (tuple[int, int]): The size of the image space for back-projection (height, width).
            device (str): The computational device ('cpu' or 'cuda').
        """
        self.scanner_geometry = scanner_geometry
        self.img_size = img_size
        self.device = device
        self.system_matrix = SystemMatrix(scanner_geometry=self.scanner_geometry,
                                          img_size=self.img_size,
                                          device=self.device)

    def backproject(self, projection_data: torch.Tensor) -> torch.Tensor:
        """
        Back-projects sinogram/projection data to image space.

        Args:
            projection_data (torch.Tensor): The projection data to back-project.
                                           Expected shape depends on the geometry,
                                           e.g., (batch_size, channels, num_angles, num_detectors).

        Returns:
            torch.Tensor: The resulting back-projected image.
        """
        if projection_data.device.type != self.device:
            projection_data = projection_data.to(self.device)
        return self.system_matrix.backward_project(projection_data)

# Example Usage (commented out)
# if __name__ == '__main__':
#     import numpy as np
#     from reconlib.geometry import DEFAULT_IMG_SIZE as GEO_DEFAULT_IMG_SIZE

    # Setup a dummy PET geometry
#     angles_pet = np.linspace(0, np.pi, 180, endpoint=False)
#     n_pixels_pet = 128 # Number of detector pixels
#     pet_geom = ScannerGeometry(detector_positions=np.zeros((n_pixels_pet, 2)), # Dummy
#                                angles=angles_pet,
#                                detector_size=np.array([4.0, 4.0]),
#                                geometry_type='cylindrical_pet',
#                                n_detector_pixels=n_pixels_pet)

    # Test ForwardProjector
#     img_size_pet = (128, 128)
#     forward_projector_pet = ForwardProjector(scanner_geometry=pet_geom, img_size=img_size_pet)
#     dummy_image_pet = torch.randn(1, 1, *img_size_pet) # batch, channels, H, W
#     sinogram = forward_projector_pet.project(dummy_image_pet)
#     print(f"PET Forward Projection - Image shape: {dummy_image_pet.shape}, Sinogram shape: {sinogram.shape}")

    # Test BackwardProjector
#     backward_projector_pet = BackwardProjector(scanner_geometry=pet_geom, img_size=img_size_pet)
#     # Sinogram shape for PETForwardProjection is (batch, 1, num_angles, num_detectors)
#     dummy_sinogram_pet = torch.randn(1, 1, len(angles_pet), n_pixels_pet)
#     back_image = backward_projector_pet.backproject(dummy_sinogram_pet)
#     print(f"PET Backward Projection - Sinogram shape: {dummy_sinogram_pet.shape}, Image shape: {back_image.shape}")

    # Setup a dummy CT (fanbeam) geometry
#     angles_ct = np.linspace(0, 2 * np.pi, 360, endpoint=False)
#     n_detectors_ct = 256
#     ct_geom = ScannerGeometry(detector_positions=np.zeros((n_detectors_ct, 2)), # Dummy
#                               angles=angles_ct,
#                               detector_size=np.array([1.0, 1.0]),
#                               geometry_type='fanbeam',
#                               n_detector_pixels=n_detectors_ct)
#     img_size_ct = (256, 256)

    # Test ForwardProjector for CT
#     forward_projector_ct = ForwardProjector(scanner_geometry=ct_geom, img_size=img_size_ct)
#     dummy_image_ct = torch.randn(1, 1, *img_size_ct)
#     projection_ct = forward_projector_ct.project(dummy_image_ct)
#     print(f"CT Forward Projection - Image shape: {dummy_image_ct.shape}, Projection shape: {projection_ct.shape}")
#     # Expected for IRadon op_adj (forward): (batch, 1, num_angles, n_rays_per_proj)


    # Test BackwardProjector for CT
#     backward_projector_ct = BackwardProjector(scanner_geometry=ct_geom, img_size=img_size_ct)
    # Projection shape for IRadon op (backward) is (batch, 1, num_angles, n_rays_per_proj)
#     dummy_projection_ct = torch.randn(1, 1, len(angles_ct), n_detectors_ct)
#     back_image_ct = backward_projector_ct.backproject(dummy_projection_ct)
#     print(f"CT Backward Projection - Projection shape: {dummy_projection_ct.shape}, Image shape: {back_image_ct.shape}")
