"""Module for defining scanner geometry and system matrices for PET/CT reconstruction."""

import numpy as np
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt # Added for visualization
from typing import Tuple, Optional, Union, List, Dict # Added for type hinting

from reconlib.operators import PETForwardProjection, IRadon
# Placeholder for image size, will be refined later
DEFAULT_IMG_SIZE = (128, 128)

class ScannerGeometry:
    """Defines the geometry of a PET or CT scanner."""

    def __init__(self,
                 geometry_type: str,
                 angles: np.ndarray,
                 n_detector_pixels: int,
                 detector_size: np.ndarray, # e.g., [width, height] or [spacing_x, spacing_y]
                 detector_positions: Optional[np.ndarray] = None, # Mandatory for PET, can be derived for some CT
                 # CT specific (especially fanbeam/conebeam)
                 source_to_detector_distance: Optional[float] = None,
                 source_to_isocenter_distance: Optional[float] = None,
                 # PET specific (can also be relevant for conebeam CT)
                 detector_radius: Optional[float] = None # For cylindrical PET
                 ):
        """
        Initializes the ScannerGeometry.

        Args:
            geometry_type (str): Type of scanner geometry (e.g., 'parallelbeam_ct', 'fanbeam_ct', 'cylindrical_pet').
            angles (np.ndarray): Projection angles in radians. Shape (num_angles,).
            n_detector_pixels (int): Number of detector elements/pixels in one dimension of a projection.
            detector_size (np.ndarray): Physical size of a single detector element [width] or [width, height] or pixel spacing.
            detector_positions (Optional[np.ndarray]):
                - For 'cylindrical_pet': Positions of detector crystals, e.g., shape (num_crystals, 2 or 3).
                                         If None and geometry is 'cylindrical_pet', can be initialized based on detector_radius.
                - For 'parallelbeam_ct'/'fanbeam_ct': Defines the extent of the detector array at angle 0.
                                                Shape (n_detector_pixels, 2 for 2D).
                                                If None, can be initialized as a flat array centered at isocenter.
            source_to_detector_distance (Optional[float]): For fan-beam/cone-beam CT. Distance from source to detector array.
            source_to_isocenter_distance (Optional[float]): For fan-beam/cone-beam CT. Distance from source to isocenter.
            detector_radius (Optional[float]): For 'cylindrical_pet', radius of the detector ring.
        """
        self.geometry_type = geometry_type.lower()
        self.angles = angles
        self.n_detector_pixels = n_detector_pixels
        self.detector_size = detector_size # Assuming this is [pixel_width, pixel_height] or [pixel_spacing]

        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_distance = source_to_isocenter_distance
        self.detector_radius = detector_radius

        # Initialize detector_positions based on geometry type if not provided
        if detector_positions is None:
            if self.geometry_type == 'cylindrical_pet':
                if self.detector_radius is None:
                    raise ValueError("detector_radius must be provided for cylindrical_pet if detector_positions is None.")
                # Arrange detectors on a circle
                det_angles = np.linspace(0, 2 * np.pi, self.n_detector_pixels, endpoint=False)
                self.detector_positions = np.stack(
                    [self.detector_radius * np.cos(det_angles), self.detector_radius * np.sin(det_angles)], axis=-1
                )
            elif self.geometry_type in ['parallelbeam_ct', 'fanbeam_ct']:
                # Create a flat linear detector array centered at isocenter for angle 0
                # Total width of detector array: n_detector_pixels * detector_size[0]
                detector_total_width = self.n_detector_pixels * self.detector_size[0]
                # Positions relative to center of detector array
                self.detector_positions = np.zeros((self.n_detector_pixels, 2))
                self.detector_positions[:, 0] = np.linspace(-detector_total_width / 2 + self.detector_size[0] / 2,
                                                            detector_total_width / 2 - self.detector_size[0] / 2,
                                                            self.n_detector_pixels)
                # For fanbeam, this initial array is often placed at y = -source_to_detector_distance (if source at origin)
                # or related to source_to_isocenter and source_to_detector distances.
                # For now, assume it's centered at (0,0) for angle 0 and will be rotated/translated.
                # If fanbeam, it's often defined relative to the source.
                if self.geometry_type == 'fanbeam_ct':
                    if self.source_to_detector_distance is None or self.source_to_isocenter_distance is None:
                        raise ValueError("source_to_detector_distance and source_to_isocenter_distance are required for fanbeam_ct.")
                    # For fanbeam, detector_positions are often defined on an arc or flat array relative to source.
                    # Here, we assume the flat array defined above is at y = source_to_isocenter_distance - source_to_detector_distance
                    # when source is at (0, -source_to_isocenter_distance) for angle 0.
                    # This is a simplification; precise definition depends on coordinate system.
                    # Let's assume the self.detector_positions are the *local* coordinates on the detector array.
            else:
                raise ValueError(f"detector_positions must be provided for geometry_type '{self.geometry_type}' or logic to derive them must exist.")
        else:
            self.detector_positions = detector_positions
        
        # Validation
        if self.geometry_type == 'fanbeam_ct' and (self.source_to_detector_distance is None or self.source_to_isocenter_distance is None):
            raise ValueError("source_to_detector_distance and source_to_isocenter_distance must be provided for fanbeam_ct.")
        if self.geometry_type == 'cylindrical_pet' and self.detector_radius is None and detector_positions is None:
             raise ValueError("detector_radius or detector_positions must be provided for cylindrical_pet.")


    def generate_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates ray origin and direction vectors for the scanner geometry.
        This is primarily for 2D geometries.

        Returns:
            A tuple containing two numpy arrays:
            - ray_origins (np.ndarray): Shape (num_angles * n_detector_pixels, 2).
            - ray_directions (np.ndarray): Shape (num_angles * n_detector_pixels, 2).
        """
        num_angles = len(self.angles)
        total_rays = num_angles * self.n_detector_pixels
        ray_origins = np.zeros((total_rays, 2))
        ray_directions = np.zeros((total_rays, 2))
        
        # Detector width/spacing for parallel/fan beam
        # Assuming self.detector_positions stores the x-coordinates for angle 0 if 1D array, or (x,y) if 2D array.
        # For simplicity, assume self.detector_positions are the local x-coords on the detector array for CT.
        # If detector_positions is (N_det, 2), then it contains (x,y) local coords.
        local_detector_coords_x = self.detector_positions[:, 0] # Should be (n_detector_pixels,)

        if self.geometry_type == 'parallelbeam_ct':
            # For each angle, rays are parallel.
            # Detector line rotates around isocenter (0,0).
            # Ray direction is perpendicular to detector line.
            # Ray origin starts far from object. Assume FOV centered at (0,0), radius R.
            # Start rays at -R in direction of ray, or from edge of a large bounding box.
            # For simplicity, start from a fixed distance along the negative direction of the ray from detector center.
            fov_radius_approx = np.max(np.abs(local_detector_coords_x)) * 1.5 # Approximate FOV extent
            
            idx = 0
            for i, angle_rad in enumerate(self.angles):
                # Rotation matrix for current angle
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                # Ray direction (normal to detector array)
                ray_dir = np.array([sin_a, -cos_a]) # Rotated unit vector along y-axis if angle=0
                
                for j in range(self.n_detector_pixels):
                    # Detector point position in world coordinates
                    # Detector array is rotated by 'angle_rad'
                    # Local x-coord of detector pixel j: local_detector_coords_x[j]
                    # This point is on the rotated detector line.
                    # If detector line passes through origin:
                    # det_x = local_detector_coords_x[j] * cos_a
                    # det_y = local_detector_coords_x[j] * sin_a
                    # Simpler: rotate the detector array positions
                    rotated_det_x = local_detector_coords_x[j] * cos_a # if detector_positions are just x-coords
                    rotated_det_y = local_detector_coords_x[j] * sin_a # if detector_positions are just x-coords

                    # Ray origin: start "far" from this detector point along the negative ray direction
                    ray_origins[idx, :] = np.array([rotated_det_x, rotated_det_y]) - ray_dir * fov_radius_approx
                    ray_directions[idx, :] = ray_dir
                    idx += 1
            return ray_origins, ray_directions

        elif self.geometry_type == 'fanbeam_ct':
            # Source rotates around isocenter. Detector array also rotates with source.
            # SID = source_to_isocenter_distance
            # SDD = source_to_detector_distance
            # Detector array is typically flat or on an arc relative to the source.
            # Let's assume self.detector_positions are local x-coordinates on a flat detector.
            
            if self.source_to_isocenter_distance is None or self.source_to_detector_distance is None:
                raise ValueError("Fanbeam CT requires source_to_isocenter_distance and source_to_detector_distance.")

            idx = 0
            for i, angle_rad in enumerate(self.angles):
                # Source position for current angle
                source_x = self.source_to_isocenter_distance * np.cos(angle_rad + np.pi/2) # Source starts at (0, SID) for angle=0 if gantry rotates CCW
                source_y = self.source_to_isocenter_distance * np.sin(angle_rad + np.pi/2) 
                # Or more standard: source at (SID*cos(angle), SID*sin(angle)) and detector array rotates with it
                # Let's use the common convention: source rotates, detector array rotates with it.
                # Source at (D_so * cos(beta), D_so * sin(beta)) where D_so = source_to_isocenter_distance
                # Detector center at ( (D_so - D_sd) * cos(beta), (D_so - D_sd) * sin(beta) )
                # For angle = 0, source is at (SID, 0) or (0, SID) depending on convention. Let's assume (SID * cos(angle), SID * sin(angle)).
                # No, typically source is at a fixed distance from isocenter, and its angular position defines the gantry angle.
                # Source position: (R_s * cos(alpha_gantry), R_s * sin(alpha_gantry)) where R_s = source_to_isocenter_distance
                # Angle of ray from source to isocenter is alpha_gantry + pi
                
                # Simpler: source is at (0, SID) for angle=0, then rotates.
                # Source position: (SID * sin(angle), SID * cos(angle)) if angle is from y-axis
                # Or (SID * cos(angle_rad_eff), SID * sin(angle_rad_eff))
                # Let source be at x_s = D_s * sin(theta), y_s = -D_s * cos(theta) where D_s = source_to_isocenter_distance
                source_pos = np.array([
                    self.source_to_isocenter_distance * np.sin(angle_rad),
                   -self.source_to_isocenter_distance * np.cos(angle_rad)
                ]) # Source moves in a circle

                # Detector array center relative to source, along the line from source to isocenter.
                # For angle_rad=0, source at (0, -SID). Detector center at (0, -SID + SDD) or (0, -SOD) where SOD = SID - SDD
                # Angle of detector normal is angle_rad.
                # Detector local x-coordinates: self.detector_positions[:,0]
                
                for j in range(self.n_detector_pixels):
                    # Position of detector element j in world coords
                    # Local detector x-coord: u = local_detector_coords_x[j]
                    # Detector point in canonical system (source at (0, D_s), detector plane at y=0): (u, 0)
                    # World detector position: Rotate (u, -D_d) by angle_rad around source, then translate by source_pos.
                    # This is getting complex. A standard way:
                    # Source pos: S = (R_s cos(beta), R_s sin(beta))
                    # Detector center: C_d = ( (R_s-R_d)cos(beta), (R_s-R_d)sin(beta) )
                    # Detector point j: P_j = C_d + u_j * (-sin(beta), cos(beta)) where u_j is local coord
                    
                    # Let's use the definition where detector array is flat and centered at a distance SDD from source,
                    # and this whole assembly rotates.
                    # Angle of the central ray from source: angle_rad
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    
                    # Detector element local x-coordinate u_j
                    u_j = local_detector_coords_x[j]
                    
                    # Detector element position in a system where source is at origin, central ray along X-axis:
                    # (SDD, u_j)
                    # Now rotate this detector point by angle_rad and add source_pos
                    # This is one way:
                    det_local_x = self.source_to_detector_distance
                    det_local_y = u_j
                    
                    rotated_det_local_x = det_local_x * cos_a - det_local_y * sin_a
                    rotated_det_local_y = det_local_x * sin_a + det_local_y * cos_a
                    
                    detector_world_pos = source_pos + np.array([rotated_det_local_x, rotated_det_local_y])

                    ray_origins[idx, :] = source_pos
                    ray_dir = detector_world_pos - source_pos
                    ray_directions[idx, :] = ray_dir / (np.linalg.norm(ray_dir) + self.epsilon) # Normalize
                    idx += 1
            return ray_origins, ray_directions

        elif self.geometry_type == 'cylindrical_pet':
            # For PET, rays are Lines of Response (LORs) between pairs of detectors.
            # This requires a different approach, often directly generating LORs.
            # The output format (ray_origins, ray_directions) might mean center of LOR and its direction.
            raise NotImplementedError("Ray generation for 'cylindrical_pet' is complex and involves LORs. "
                                      "It should typically produce LOR start/end points or center+direction.")
        else:
            raise ValueError(f"Unsupported geometry_type for ray generation: {self.geometry_type}")

    def visualize_geometry(self) -> None:
        """
        Visualizes the scanner geometry (basic 2D top-down view).
        """
        plt.figure(figsize=(7, 7))
        ax = plt.gca()

        if self.geometry_type == 'cylindrical_pet':
            if self.detector_positions is not None:
                ax.plot(self.detector_positions[:, 0], self.detector_positions[:, 1], 'o', label="Detectors")
                if self.detector_radius: # Draw the circle if radius is known
                    circle = plt.Circle((0,0), self.detector_radius, color='blue', fill=False, linestyle='--')
                    ax.add_artist(circle)
            ax.set_title("Cylindrical PET Geometry")

        elif self.geometry_type == 'parallelbeam_ct':
            # Plot detector array at a few angles
            num_angles_to_show = min(3, len(self.angles))
            angles_to_show = self.angles[np.linspace(0, len(self.angles) - 1, num_angles_to_show, dtype=int)]
            
            detector_xs = self.detector_positions[:, 0] # Assuming 1D local coords for simplicity here
            min_x, max_x = np.min(detector_xs), np.max(detector_xs)

            for i, angle_rad in enumerate(angles_to_show):
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                # Rotate detector array
                det_line_x = np.array([min_x * cos_a, max_x * cos_a])
                det_line_y = np.array([min_x * sin_a, max_x * sin_a])
                ax.plot(det_line_x, det_line_y, label=f"Detector Angle {np.rad2deg(angle_rad):.0f}°", alpha=0.7)
            ax.set_title("Parallel Beam CT Geometry")

        elif self.geometry_type == 'fanbeam_ct':
            # Plot source and detector arc for a few angles
            if self.source_to_isocenter_distance is None:
                raise ValueError("source_to_isocenter_distance is needed for fanbeam_ct visualization.")

            num_angles_to_show = min(3, len(self.angles))
            angles_to_show = self.angles[np.linspace(0, len(self.angles) - 1, num_angles_to_show, dtype=int)]
            
            local_detector_xs = self.detector_positions[:, 0] # Assuming local x-coords on detector
            
            for i, angle_rad in enumerate(angles_to_show):
                # Source position (simplified: source rotates around isocenter)
                source_x = self.source_to_isocenter_distance * np.sin(angle_rad)
                source_y = -self.source_to_isocenter_distance * np.cos(angle_rad)
                ax.plot(source_x, source_y, 'o', markersize=8, label=f"Source Angle {np.rad2deg(angle_rad):.0f}°", alpha=0.7)

                # Detector array (simplified: flat detector centered opposite source, distance SDD)
                if self.source_to_detector_distance:
                    # Central ray direction from source
                    central_ray_dx, central_ray_dy = -np.sin(angle_rad), np.cos(angle_rad)
                    # Detector center
                    detector_center_x = source_x + self.source_to_detector_distance * central_ray_dx
                    detector_center_y = source_y + self.source_to_detector_distance * central_ray_dy
                    
                    # Detector orientation (perpendicular to central ray)
                    det_orient_dx, det_orient_dy = np.cos(angle_rad), np.sin(angle_rad)
                    
                    det_end1_x = detector_center_x + local_detector_xs[0] * det_orient_dx
                    det_end1_y = detector_center_y + local_detector_xs[0] * det_orient_dy
                    det_end2_x = detector_center_x + local_detector_xs[-1] * det_orient_dx
                    det_end2_y = detector_center_y + local_detector_xs[-1] * det_orient_dy
                    ax.plot([det_end1_x, det_end2_x], [det_end1_y, det_end2_y], '-', alpha=0.7, color=plt.gca().lines[-1].get_color())
            ax.set_title("Fanbeam CT Geometry")
        else:
            ax.set_title(f"Geometry: {self.geometry_type} (Vis not fully implemented)")

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def get_subset_angles(self, subset_indices: np.ndarray) -> np.ndarray:
        """
        Returns a subset of scanner angles based on provided indices.

        Args:
            subset_indices (np.ndarray): An array of integer indices.

        Returns:
            np.ndarray: The subset of angles from self.angles.
        """
        if not isinstance(subset_indices, np.ndarray) or subset_indices.ndim != 1:
            raise ValueError("subset_indices must be a 1D NumPy array of integers.")
        if np.any(subset_indices < 0) or np.any(subset_indices >= len(self.angles)):
            max_idx = len(self.angles) -1
            raise IndexError(f"Subset indices out of bounds. Must be between 0 and {max_idx}.")
            
        return self.angles[subset_indices]


class SystemMatrix(ABC):
    """
    Represents the system matrix for tomographic reconstruction,
    encapsulating the forward and backward projection operations.
    It can be configured to operate on the full set of projection angles
    or a specified subset of angles.
    """

    def __init__(self, scanner_geometry: ScannerGeometry, img_size: Tuple[int, int] = DEFAULT_IMG_SIZE, device: str = 'cpu'):
        """
        Initializes the SystemMatrix for the full set of angles.

        Args:
            scanner_geometry (ScannerGeometry): The scanner geometry definition.
            img_size (Tuple[int, int]): The size of the image to be reconstructed (height, width).
            device (str): The computational device ('cpu' or 'cuda').
        """
        self.full_scanner_geometry = scanner_geometry # Store the original geometry
        self.img_size = img_size
        self.device = device
        
        self._projector_op_full = None # Operator for the full set of angles
        self._projector_op_subset = None # Operator for a subset of angles (if configured)
        self._current_angles_subset_indices: Optional[np.ndarray] = None # Indices of angles in current subset
        self._is_subset_configured = False 
        self._forward_is_op_full = True # For the full operator
        self._forward_is_op_subset = True # For the subset operator, will be set in configure_for_subset

        self._initialize_projector(angles=self.full_scanner_geometry.angles, is_subset=False)

    def _initialize_projector(self, angles: np.ndarray, is_subset: bool):
        """Helper to initialize a projector_op for a given set of angles."""
        temp_scanner_geometry: ScannerGeometry
        if is_subset:
            # Create a temporary ScannerGeometry for the subset of angles
            temp_scanner_geometry = ScannerGeometry(
                geometry_type=self.full_scanner_geometry.geometry_type,
                angles=angles, # Subset angles
                n_detector_pixels=self.full_scanner_geometry.n_detector_pixels,
                detector_size=self.full_scanner_geometry.detector_size,
                detector_positions=self.full_scanner_geometry.detector_positions, # Use original, assumes it's not angle dependent
                source_to_detector_distance=self.full_scanner_geometry.source_to_detector_distance,
                source_to_isocenter_distance=self.full_scanner_geometry.source_to_isocenter_distance,
                detector_radius=self.full_scanner_geometry.detector_radius
            )
        else:
            temp_scanner_geometry = self.full_scanner_geometry

        projector_op = None
        forward_is_op_flag = True

        if temp_scanner_geometry.geometry_type == 'cylindrical_pet':
            projector_op = PETForwardProjection(
                # n_subsets=1, # PETForwardProjection doesn't take n_subsets directly for this purpose.
                # n_angles refers to total angles for the operator instance.
                n_angles=len(angles),
                n_detectors=temp_scanner_geometry.n_detector_pixels,
                img_size=self.img_size,
                device=self.device # Pass device to operator
            )
            forward_is_op_flag = True
        elif temp_scanner_geometry.geometry_type in ['fanbeam', 'parallelbeam']:
            projector_op = IRadon(
                angles=angles,
                # n_rays_per_proj is equivalent to n_detector_pixels for IRadon
                n_rays_per_proj=temp_scanner_geometry.n_detector_pixels,
                img_size=self.img_size,
                filter_type=None, # No filtering for basic projection
                device=self.device # Pass device to operator
            )
            forward_is_op_flag = False # IRadon's op is FBP (backward), op_adj is Radon transform (forward)
        else:
            raise ValueError(f"Unsupported geometry type: {temp_scanner_geometry.geometry_type}")

        if hasattr(projector_op, 'to') and callable(getattr(projector_op, 'to')):
             projector_op.to(self.device) # Ensure operator is on device, though init should handle it.

        if is_subset:
            self._projector_op_subset = projector_op
            self._forward_is_op_subset = forward_is_op_flag
        else:
            self._projector_op_full = projector_op
            self._forward_is_op_full = forward_is_op_flag
            
    def configure_for_subset(self, angle_indices: np.ndarray):
        """
        Configures the SystemMatrix to operate on a subset of angles.

        Args:
            angle_indices (np.ndarray): Indices of the angles (from the full set) to be used in this subset.
        """
        if not isinstance(angle_indices, np.ndarray):
            raise TypeError("angle_indices must be a NumPy array.")
        
        self._current_angles_subset_indices = angle_indices
        subset_angles = self.full_scanner_geometry.get_subset_angles(angle_indices)
        self._initialize_projector(angles=subset_angles, is_subset=True)
        self._is_subset_configured = True

    def configure_for_full_set(self):
        """
        Resets the SystemMatrix to operate on the full set of angles.
        """
        self._current_angles_subset_indices = None
        self._projector_op_subset = None # Clear subset operator
        self._is_subset_configured = False

    def _get_current_projector_config(self) -> Tuple[Any, bool]:
        """Returns the current projector operator and its forward_is_op flag."""
        if self._is_subset_configured and self._projector_op_subset is not None:
            return self.projector_op_subset, self._forward_is_op_subset
        elif self._projector_op_full is not None:
            return self._projector_op_full, self._forward_is_op_full
        else:
            raise RuntimeError("Projector operator not initialized. Call __init__ correctly.")
            
    @property
    def projector_op(self): # For backward compatibility or direct access if needed
        """Returns the currently active projector operator (full or subset)."""
        op, _ = self._get_current_projector_config()
        return op

    def forward_project(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs forward projection of an image to projection data (sinogram)
        using the currently configured (full or subset) projector.
        """
        current_projector, forward_is_op = self._get_current_projector_config()
        
        if image.device != self.device: # Ensure image is on the correct device
            image = image.to(self.device)

        if forward_is_op:
            return current_projector.op(image)
        else:
            return current_projector.op_adj(image)

    def backward_project(self, projection_data: torch.Tensor) -> torch.Tensor:
        """
        Performs back-projection of projection data to image space
        using the currently configured (full or subset) projector.
        """
        current_projector, forward_is_op = self._get_current_projector_config()

        if projection_data.device != self.device: # Ensure data is on the correct device
            projection_data = projection_data.to(self.device)

        if forward_is_op:
            return current_projector.op_adj(projection_data)
        else:
            return current_projector.op(projection_data)

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
