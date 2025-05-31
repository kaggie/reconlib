import torch
from reconlib.operators import Operator
import numpy as np
from scipy.signal import convolve # For a more robust PSF generation if needed

class FluorescenceMicroscopyOperator(Operator):
    """
    Forward and Adjoint Operator for Fluorescence Microscopy.

    Models the image formation process, typically as a convolution of the
    true fluorescence distribution with the microscope's Point Spread Function (PSF).
    Y = PSF * X + Noise, where Y is observed, X is true, * is convolution.

    This is relevant for deconvolution problems. For super-resolution microscopy
    (STORM, PALM), the forward model might be different (e.g., localizing sparse emitters).
    This placeholder focuses on the deconvolution scenario.
    """
    def __init__(self, image_shape: tuple[int, int] | tuple[int, int, int], # (Ny, Nx) or (Nz, Ny, Nx)
                 psf: torch.Tensor, # Point Spread Function
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.is_3d = len(image_shape) == 3
        self.device = torch.device(device)

        if psf.ndim != len(image_shape):
            raise ValueError(f"PSF dimensionality ({psf.ndim}D) must match image dimensionality ({len(image_shape)}D).")

        self.psf = psf.to(self.device)
        # Normalize PSF (optional, but common for it to sum/integrate to 1)
        # self.psf = self.psf / torch.sum(self.psf)

        # Determine padding for 'same' convolution
        self.padding = []
        for dim_size in self.psf.shape:
            self.padding.extend([(dim_size - 1) // 2, dim_size // 2]) # Pad for each dim (before, after)
        # PyTorch convnd padding is (pad_dimN_before, pad_dimN_after, pad_dimN-1_before, ...)
        # So we need to reverse the per-dimension padding list.
        self.padding.reverse()


        print(f"FluorescenceMicroscopyOperator initialized for image shape {self.image_shape} "
              f"with PSF shape {self.psf.shape}. 3D: {self.is_3d}. Padding: {self.padding}")

    def op(self, true_fluorescence_map: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: True fluorescence map to observed (blurred) image.

        Args:
            true_fluorescence_map (torch.Tensor): The true fluorescence distribution.
                                                  Shape: self.image_shape.

        Returns:
            torch.Tensor: Simulated observed microscope image (blurred).
                          Shape: self.image_shape.
        """
        if true_fluorescence_map.shape != self.image_shape:
            raise ValueError(f"Input map shape {true_fluorescence_map.shape} must match {self.image_shape}.")
        if true_fluorescence_map.device != self.device:
            true_fluorescence_map = true_fluorescence_map.to(self.device)

        # Convolution: Y = PSF * X
        # PyTorch conv_nd expects input: (batch, channels, D, H, W) or (batch, channels, H, W)
        # PSF (weights): (out_channels, in_channels/groups, kD, kH, kW) or (kH, kW)

        x_expanded = true_fluorescence_map.unsqueeze(0).unsqueeze(0) # (1, 1, [D], H, W)
        psf_expanded = self.psf.unsqueeze(0).unsqueeze(0)          # (1, 1, [kD], kH, kW)

        if self.is_3d:
            blurred_image = torch.nn.functional.conv3d(x_expanded, psf_expanded, padding='same') # Requires PyTorch 1.10+ for string padding
            # blurred_image = torch.nn.functional.conv3d(x_expanded, psf_expanded, padding=self.padding) # Manual padding
        else: # 2D
            blurred_image = torch.nn.functional.conv2d(x_expanded, psf_expanded, padding='same')
            # blurred_image = torch.nn.functional.conv2d(x_expanded, psf_expanded, padding=self.padding)


        return blurred_image.squeeze()

    def op_adj(self, observed_image: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Observed image to a map in the true fluorescence domain.
        For deconvolution, this is convolution with the flipped PSF.
        Y_adj = PSF_flipped * X_observed

        Args:
            observed_image (torch.Tensor): The observed (blurred) microscope image.
                                           Shape: self.image_shape.

        Returns:
            torch.Tensor: Image processed by adjoint operation.
                          Shape: self.image_shape.
        """
        if observed_image.shape != self.image_shape:
            raise ValueError(f"Input image shape {observed_image.shape} must match {self.image_shape}.")
        if observed_image.device != self.device:
            observed_image = observed_image.to(self.device)

        y_expanded = observed_image.unsqueeze(0).unsqueeze(0)

        # Flipped PSF for adjoint convolution
        # For a symmetric PSF, flipped_psf = psf.
        # torch.flip is easy for this.
        dims_to_flip = list(range(self.psf.ndim)) # Flip all spatial/depth dimensions of PSF
        psf_flipped = torch.flip(self.psf, dims=dims_to_flip).unsqueeze(0).unsqueeze(0)

        if self.is_3d:
            adj_conv_image = torch.nn.functional.conv3d(y_expanded, psf_flipped, padding='same')
            # adj_conv_image = torch.nn.functional.conv3d(y_expanded, psf_flipped, padding=self.padding)
        else: # 2D
            adj_conv_image = torch.nn.functional.conv2d(y_expanded, psf_flipped, padding='same')
            # adj_conv_image = torch.nn.functional.conv2d(y_expanded, psf_flipped, padding=self.padding)

        return adj_conv_image.squeeze()

# Helper to generate a simple PSF (e.g., Gaussian)
def generate_gaussian_psf(shape: tuple, sigma: float | tuple[float,...], device='cpu') -> torch.Tensor:
    is_3d_psf = len(shape) == 3
    if isinstance(sigma, (float, int)):
        sigma = [float(sigma)] * len(shape)
    if len(sigma) != len(shape):
        raise ValueError("Sigma tuple length must match shape dimensionality.")

    coords = []
    for i, s in enumerate(shape):
        coords.append(torch.linspace(-s // 2 + 1, s // 2, s, device=device))

    if is_3d_psf:
        zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        psf = torch.exp(-((zz / sigma[0])**2 + (yy / sigma[1])**2 + (xx / sigma[2])**2) / 2.0)
    else: # 2D
        yy, xx = torch.meshgrid(coords[0], coords[1], indexing='ij')
        psf = torch.exp(-((yy / sigma[0])**2 + (xx / sigma[1])**2) / 2.0)

    psf = psf / torch.sum(psf) # Normalize
    return psf


if __name__ == '__main__':
    print("Running basic FluorescenceMicroscopyOperator checks...")
    device_fm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test 2D
    img_shape_2d_fm = (64, 64)
    psf_shape_2d = (7, 7)
    psf_2d_fm = generate_gaussian_psf(psf_shape_2d, sigma=1.5, device=device_fm)

    try:
        fm_op_test_2d = FluorescenceMicroscopyOperator(
            image_shape=img_shape_2d_fm, psf=psf_2d_fm, device=device_fm
        )
        print("FluorescenceMicroscopyOperator (2D) instantiated.")

        phantom_2d_fm = torch.zeros(img_shape_2d_fm, device=device_fm)
        phantom_2d_fm[img_shape_2d_fm[0]//4:img_shape_2d_fm[0]*3//4,
                      img_shape_2d_fm[1]//4:img_shape_2d_fm[1]*3//4] = 1.0 # A square

        blurred_2d_fm = fm_op_test_2d.op(phantom_2d_fm)
        print(f"Forward op (2D) output shape (blurred): {blurred_2d_fm.shape}")
        assert blurred_2d_fm.shape == img_shape_2d_fm

        adj_conv_2d_fm = fm_op_test_2d.op_adj(blurred_2d_fm)
        print(f"Adjoint op (2D) output shape: {adj_conv_2d_fm.shape}")
        assert adj_conv_2d_fm.shape == img_shape_2d_fm

        # Dot product test for 2D
        x_dp_2d = torch.randn_like(phantom_2d_fm)
        y_dp_rand_2d = torch.randn_like(blurred_2d_fm)
        Ax_2d = fm_op_test_2d.op(x_dp_2d)
        Aty_2d = fm_op_test_2d.op_adj(y_dp_rand_2d)
        lhs_2d = torch.dot(Ax_2d.flatten(), y_dp_rand_2d.flatten())
        rhs_2d = torch.dot(x_dp_2d.flatten(), Aty_2d.flatten())
        print(f"FM (2D) Dot product test: LHS={lhs_2d.item():.4f}, RHS={rhs_2d.item():.4f}")
        assert np.isclose(lhs_2d.item(), rhs_2d.item(), rtol=1e-3), "Dot product test failed for FM 2D operator."

    except Exception as e:
        print(f"Error in FM Operator (2D) __main__ checks: {e}")
        import traceback; traceback.print_exc()

    # Test 3D
    img_shape_3d_fm = (32, 32, 16) # Nz, Ny, Nx
    psf_shape_3d = (5, 5, 5)
    psf_3d_fm = generate_gaussian_psf(psf_shape_3d, sigma=(1.0,1.0,1.5), device=device_fm)
    try:
        fm_op_test_3d = FluorescenceMicroscopyOperator(
            image_shape=img_shape_3d_fm, psf=psf_3d_fm, device=device_fm
        )
        print("FluorescenceMicroscopyOperator (3D) instantiated.")

        phantom_3d_fm = torch.zeros(img_shape_3d_fm, device=device_fm)
        phantom_3d_fm[img_shape_3d_fm[0]//4:img_shape_3d_fm[0]*3//4,
                      img_shape_3d_fm[1]//4:img_shape_3d_fm[1]*3//4,
                      img_shape_3d_fm[2]//4:img_shape_3d_fm[2]*3//4] = 1.0 # A cuboid

        blurred_3d_fm = fm_op_test_3d.op(phantom_3d_fm)
        print(f"Forward op (3D) output shape (blurred): {blurred_3d_fm.shape}")
        assert blurred_3d_fm.shape == img_shape_3d_fm

        adj_conv_3d_fm = fm_op_test_3d.op_adj(blurred_3d_fm)
        print(f"Adjoint op (3D) output shape: {adj_conv_3d_fm.shape}")
        assert adj_conv_3d_fm.shape == img_shape_3d_fm

        # Dot product test for 3D
        x_dp_3d = torch.randn_like(phantom_3d_fm)
        y_dp_rand_3d = torch.randn_like(blurred_3d_fm)
        Ax_3d = fm_op_test_3d.op(x_dp_3d)
        Aty_3d = fm_op_test_3d.op_adj(y_dp_rand_3d)
        lhs_3d = torch.dot(Ax_3d.flatten(), y_dp_rand_3d.flatten())
        rhs_3d = torch.dot(x_dp_3d.flatten(), Aty_3d.flatten())
        print(f"FM (3D) Dot product test: LHS={lhs_3d.item():.4f}, RHS={rhs_3d.item():.4f}")
        assert np.isclose(lhs_3d.item(), rhs_3d.item(), rtol=1e-3), "Dot product test failed for FM 3D operator."


        print("FluorescenceMicroscopyOperator __main__ checks completed (2D and 3D).")
    except Exception as e:
        print(f"Error in FM Operator (3D) __main__ checks: {e}")
        import traceback; traceback.print_exc()
