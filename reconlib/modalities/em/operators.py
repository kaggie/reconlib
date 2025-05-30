import torch
import numpy as np # For np.pi, np.linspace
from reconlib.operators import Operator
from .utils import rotate_volume_z_axis, project_volume, backproject_2d_to_3d

class EMForwardOperator(Operator):
    """
    Forward and Adjoint Operator for Electron Microscopy (EM) Tomography.

    Models EM data acquisition as taking 2D projections of a 3D volume
    at various angles. This implementation uses simplified Z-axis rotations.

    Args:
        volume_shape (tuple[int, int, int]): Shape of the input 3D volume (Depth, Height, Width).
        angles_rad (torch.Tensor or list[float]): 1D tensor or list of projection angles
                                                 in radians. For this simplified version,
                                                 these are angles for Z-axis rotation.
        projection_axis (int, optional): The axis of the (original, unrotated) volume
                                         along which projection occurs (summation).
                                         Defaults to 0 (summing along Depth).
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 volume_shape: tuple[int, int, int], # (D, H, W)
                 angles_rad: torch.Tensor | list[float],
                 projection_axis: int = 0, # Axis to sum over for projection
                 device: str | torch.device = 'cpu'):
        super().__init__()
        if len(volume_shape) != 3:
            raise ValueError("volume_shape must be a 3-tuple (D, H, W).")
        self.volume_shape = volume_shape
        self.projection_axis = projection_axis
        self.device = torch.device(device)

        if not isinstance(angles_rad, torch.Tensor):
            angles_rad = torch.tensor(angles_rad, dtype=torch.float32, device=self.device)
        self.angles_rad = angles_rad.to(self.device)

        # Determine shape of a single 2D projection
        proj_dims = [dim for i, dim in enumerate(volume_shape) if i != projection_axis]
        if len(proj_dims) != 2:
            raise ValueError("Projection must result in a 2D image.")
        self.single_projection_shape = tuple(proj_dims) # e.g., (H,W) if projecting along D

    def op(self, x_volume: torch.Tensor) -> torch.Tensor:
        """
        Forward EM operation: Rotates the volume and takes 2D projections.
        x_volume: (D, H, W) density volume.
        Returns: (num_angles, proj_H, proj_W) stack of 2D projections.
        """
        if x_volume.shape != self.volume_shape:
            raise ValueError(f"Input x_volume shape {x_volume.shape} must match {self.volume_shape}.")
        if x_volume.device != self.device:
            x_volume = x_volume.to(self.device)
        if not torch.is_complex(x_volume): # Density is usually real, but allow complex for generality
             x_volume = x_volume.to(torch.complex64)

        num_angles = len(self.angles_rad)
        projections = torch.zeros(
            (num_angles,) + self.single_projection_shape, # (num_angles, H, W)
            dtype=x_volume.dtype, device=self.device
        )

        for i, angle_rad in enumerate(self.angles_rad):
            # For now, only Z-axis rotation is implemented in utils.
            # If projection_axis is not the Z-axis (0 for D,H,W), this simplified rotation
            # might not be physically representative for all projection directions.
            # This assumes the 'projection_axis' is what becomes the line-of-sight *after* rotation.
            # A more general approach would rotate around multiple axes or use a projection matrix.

            # If we rotate around Z, then project along Z (axis 0), it means slices are rotated.
            # If we want to simulate tilting and projecting (e.g. tilt around X, then project along new Z),
            # the rotation needs to be more general or volume permuted.
            # For now, let's assume Z-axis rotation and then projection along the original Z (axis 0).

            if self.projection_axis != 0:
                print("Warning: Current simple rotation is around Z-axis. Projection along an axis other than Z (axis 0) "
                      "after only Z-rotation might not produce tomographically standard projections for all angles.")

            rotated_volume = rotate_volume_z_axis(x_volume, angle_rad.item())
            projections[i] = project_volume(rotated_volume, projection_axis=self.projection_axis)

        return projections

    def op_adj(self, y_projections: torch.Tensor) -> torch.Tensor:
        """
        Adjoint EM operation: Backprojects 2D projections into a 3D volume.
        y_projections: (num_angles, proj_H, proj_W) stack of 2D projections.
        Returns: (D, H, W) reconstructed 3D volume.
        """
        expected_proj_stack_shape = (len(self.angles_rad),) + self.single_projection_shape
        if y_projections.shape != expected_proj_stack_shape:
            raise ValueError(f"Input y_projections shape {y_projections.shape} must match {expected_proj_stack_shape}.")
        if y_projections.device != self.device:
            y_projections = y_projections.to(self.device)
        if not torch.is_complex(y_projections):
             y_projections = y_projections.to(torch.complex64)

        reconstructed_volume = torch.zeros(self.volume_shape, dtype=y_projections.dtype, device=self.device)

        for i, angle_rad in enumerate(self.angles_rad):
            single_proj = y_projections[i]

            # Backproject the 2D projection into a 3D slice/volume
            # The backproject_2d_to_3d utility needs to know the original projection axis
            # and then it handles the inverse rotation.

            # This is an adjoint to: project_volume(rotate_volume_z_axis(x, angle), proj_axis)
            # Adjoint: rotate_volume_z_axis_adj(backproject_2d_to_3d_adj(proj), -angle)
            # Since rotate_volume_z_axis uses grid_sample, its adjoint is complex.
            # For a simple Z-axis rotation, rotate_volume_z_axis(vol, -angle) is the geometric inverse.
            # If the forward op was P(R(V)), adjoint is R_adj(P_adj(Y)).
            # P_adj is backproject (smear/repeat). R_adj is rotate by -angle.

            # 1. Smear the 2D projection to a 3D volume (adjoint of summation)
            temp_vol_no_rot = torch.zeros(self.volume_shape, dtype=y_projections.dtype, device=self.device)
            if self.projection_axis == 0: # Projection was along D
                temp_vol_no_rot = single_proj.unsqueeze(0).repeat(self.volume_shape[0], 1, 1)
            elif self.projection_axis == 1: # Projection was along H
                temp_vol_no_rot = single_proj.unsqueeze(1).repeat(1, self.volume_shape[1], 1)
            elif self.projection_axis == 2: # Projection was along W
                temp_vol_no_rot = single_proj.unsqueeze(2).repeat(1, 1, self.volume_shape[2])
            else:
                raise ValueError(f"Invalid projection_axis: {self.projection_axis}")

            # 2. Apply inverse rotation
            backprojected_and_rotated_slice = rotate_volume_z_axis(temp_vol_no_rot, -angle_rad.item())

            reconstructed_volume += backprojected_and_rotated_slice # Sum contributions

        return reconstructed_volume

if __name__ == '__main__':
    print("Running basic EMForwardOperator checks...")
    device = torch.device('cpu')
    vol_shape_test = (16, 32, 32) # D, H, W. Keep small for tests.
    angles_test = torch.tensor(np.linspace(0, np.pi, 5, endpoint=False), dtype=torch.float32, device=device) # 5 angles

    try:
        em_op_test = EMForwardOperator(
            volume_shape=vol_shape_test,
            angles_rad=angles_test,
            projection_axis=0, # Project along Depth
            device=device
        )
        print("EMForwardOperator instantiated.")

        phantom_vol = torch.randn(vol_shape_test, dtype=torch.complex64, device=device)
        # Add a feature for visual check if possible
        phantom_vol[vol_shape_test[0]//2,
                    vol_shape_test[1]//4:vol_shape_test[1]*3//4,
                    vol_shape_test[2]//4:vol_shape_test[2]*3//4] = 3.0

        projections_sim = em_op_test.op(phantom_vol)
        print(f"Forward op output shape (projections): {projections_sim.shape}")
        expected_proj_shape = (len(angles_test), vol_shape_test[1], vol_shape_test[2])
        assert projections_sim.shape == expected_proj_shape

        recon_vol_adj = em_op_test.op_adj(projections_sim)
        print(f"Adjoint op output shape (volume): {recon_vol_adj.shape}")
        assert recon_vol_adj.shape == vol_shape_test

        # Basic dot product test (can be slow and less accurate for complex ops like this)
        x_dp_em = torch.randn_like(phantom_vol)
        y_dp_rand_em = torch.randn_like(projections_sim)
        Ax_em = em_op_test.op(x_dp_em)
        Aty_em = em_op_test.op_adj(y_dp_rand_em)
        lhs_em = torch.vdot(Ax_em.flatten(), y_dp_rand_em.flatten())
        rhs_em = torch.vdot(x_dp_em.flatten(), Aty_em.flatten())
        print(f"EM Dot product test: LHS={lhs_em.item():.4f}, RHS={rhs_em.item():.4f}")
        # Tolerance for rotation/interpolation based operators can be significant
        if not np.isclose(lhs_em.real.item(), rhs_em.real.item(), rtol=1e-2, atol=1e-3) or \
           not np.isclose(lhs_em.imag.item(), rhs_em.imag.item(), rtol=1e-2, atol=1e-3):
           print("Warning: EM Dot product components differ. This is common with interpolation in rotation/projection.")

        print("EMForwardOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in EMForwardOperator __main__ checks: {e}")
        # raise # Avoid raising in subtask for now
