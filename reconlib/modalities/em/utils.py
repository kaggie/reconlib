import torch
import torch.nn.functional as F

def rotate_volume_z_axis(volume: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """
    Rotates a 3D volume around the Z-axis (slice-by-slice 2D rotation).
    Assumes volume is (D, H, W). Rotation is in the H-W plane.
    Args:
        volume (torch.Tensor): Input 3D volume.
        angle_rad (float): Rotation angle in radians.
    Returns:
        torch.Tensor: Rotated 3D volume.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D (D, H, W).")

    device = volume.device
    dtype = volume.dtype
    D, H, W = volume.shape

    # Create affine grid for 2D rotation for each slice
    cos_a = torch.cos(torch.tensor(angle_rad, device=device, dtype=dtype))
    sin_a = torch.sin(torch.tensor(angle_rad, device=device, dtype=dtype))

    # Rotation matrix for 2D (H-W plane)
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    # grid_sample expects theta that transforms output coords to input coords
    # So, for rotating image, we use inverse rotation matrix on grid
    # Theta: [[cos, sin, 0], [-sin, cos, 0]]
    theta = torch.tensor([
        [cos_a, sin_a, 0],
        [-sin_a, cos_a, 0]
    ], dtype=dtype, device=device).unsqueeze(0).repeat(D, 1, 1) # Repeat for each slice

    # Create a normalized grid of coordinates for grid_sample
    grid = F.affine_grid(theta, torch.Size((D, 1, H, W)), align_corners=False) # Add channel dim for grid_sample

    # Reshape volume to (D, 1, H, W) to act as batch of 2D images
    volume_reshaped = volume.unsqueeze(1)

    rotated_volume = F.grid_sample(volume_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return rotated_volume.squeeze(1) # Remove channel dim

# For simplicity in this first pass, we will only implement Z-axis rotation.
# Full 3D rotation (e.g., using Euler angles or rotation matrices) is more complex
# to implement from scratch without scipy or similar.
# The pseudocode's `rotate_volume(x, theta)` was generic.
# We can add more sophisticated rotation if needed later.

def project_volume(volume: torch.Tensor, projection_axis: int = 0) -> torch.Tensor:
    """
    Projects a 3D volume onto a 2D plane by summing along a specified axis.
    Args:
        volume (torch.Tensor): Input 3D volume (e.g., D, H, W).
        projection_axis (int): Axis along which to sum for projection.
                               0 for projection onto H-W plane (sum over D).
                               (Other axes can be used if volume is permuted first)
    Returns:
        torch.Tensor: 2D projection.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D.")
    return torch.sum(volume, dim=projection_axis)

def backproject_2d_to_3d(projection_2d: torch.Tensor, target_volume_shape: tuple,
                         projection_axis: int = 0, angle_rad: float = 0.0,
                         is_rotated_projection: bool = False) -> torch.Tensor:
    """
    Backprojects a 2D projection into a 3D volume.
    If is_rotated_projection is True, it applies inverse rotation before backprojection.
    Args:
        projection_2d (torch.Tensor): The 2D projection image.
        target_volume_shape (tuple): The desired shape of the 3D volume (D, H, W).
        projection_axis (int): The axis along which the original projection was made.
        angle_rad (float): The rotation angle (in radians) that was applied *before* projection.
                           The inverse of this rotation will be applied here.
        is_rotated_projection (bool): If True, assumes projection_2d was from a rotated volume.
    Returns:
        torch.Tensor: 3D volume with the projection backprojected.
    """
    D, H, W = target_volume_shape
    device = projection_2d.device
    dtype = projection_2d.dtype

    # Expand the 2D projection to 3D by repeating it along the projection axis
    if projection_axis == 0: # Projected along D, so repeat along D
        if projection_2d.shape != (H, W):
            raise ValueError(f"Projection shape {projection_2d.shape} mismatch for axis 0 projection from {(D,H,W)}")
        backprojected_slice = projection_2d.unsqueeze(0).repeat(D, 1, 1)
    elif projection_axis == 1: # Projected along H, so repeat along H
        if projection_2d.shape != (D, W):
            raise ValueError(f"Projection shape {projection_2d.shape} mismatch for axis 1 projection from {(D,H,W)}")
        backprojected_slice = projection_2d.unsqueeze(1).repeat(1, H, 1)
    elif projection_axis == 2: # Projected along W, so repeat along W
        if projection_2d.shape != (D, H):
            raise ValueError(f"Projection shape {projection_2d.shape} mismatch for axis 2 projection from {(D,H,W)}")
        backprojected_slice = projection_2d.unsqueeze(2).repeat(1, 1, W)
    else:
        raise ValueError(f"Invalid projection_axis: {projection_axis}")

    if is_rotated_projection and angle_rad != 0.0:
        # Apply inverse rotation. For Z-axis rotation, inverse is rotation by -angle.
        # This assumes the backprojected_slice is oriented as if it were a slice
        # from the un-rotated volume, and now we need to rotate it back.
        # This is complex. A simpler adjoint for op(rotate(vol)) is rotate_adj(op_adj(proj)).
        # The current `rotate_volume_z_axis` is its own adjoint if mode='bilinear'
        # due to how grid_sample works with inverse transforms.
        # So, to invert `project(rotate(vol))`, we do `rotate(backproject_slice_from_unrotated_frame)`.
        # This means we rotate the `backprojected_slice` by -angle_rad.

        # This part is tricky: if projection was sum(rotate(V)), adjoint is rotate_adj(sum_adj(P)).
        # sum_adj is backproject (repeat). rotate_adj is rotate by -angle.
        backprojected_volume = rotate_volume_z_axis(backprojected_slice, -angle_rad)
    else:
        backprojected_volume = backprojected_slice

    return backprojected_volume
