import torch
import numpy as np # For np.pi
from reconlib.operators import Operator

class OCTForwardOperator(Operator):
    """
    Forward and Adjoint Operator for Optical Coherence Tomography (OCT).

    Models the OCT signal acquisition process, primarily as a Fourier Transform
    along the depth axis for each A-scan, based on simplified principles.

    Args:
        image_shape (tuple[int, int]): Shape of the input reflectivity image (num_ascan, depth_pixels).
                                       e.g., (num_bscans_lines, depth_resolution_axial).
        lambda_w (float): Center wavelength of the light source in meters (e.g., 850e-9 for 850 nm).
        z_max_m (float): Maximum imaging depth in meters (e.g., 0.002 for 2mm).
                         This defines the range over which the Fourier transform is effectively performed.
        n_refractive_index (float, optional): Refractive index of the medium. Defaults to 1.35 (typical for tissue).
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (num_ascan_lines, num_depth_pixels)
                 lambda_w: float, # Wavelength in meters
                 z_max_m: float,  # Max imaging depth in meters
                 n_refractive_index: float = 1.35,
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape # (num_ascan, depth_pixels)
        self.num_ascan_lines = image_shape[0]
        self.depth_pixels = image_shape[1]

        self.lambda_w = lambda_w
        self.z_max_m = z_max_m
        self.n_refractive_index = n_refractive_index
        self.device = torch.device(device)

    def op(self, x_reflectivity: torch.Tensor) -> torch.Tensor:
        """
        Forward OCT operation: Transforms reflectivity profile to k-space signal (spectral interferogram).
        x_reflectivity: (num_ascan_lines, depth_pixels) - Represents R(z) for each A-scan.
        Returns: (num_ascan_lines, depth_pixels) - Represents S(k) for each A-scan.
        """
        if x_reflectivity.shape != self.image_shape:
            raise ValueError(f"Input x_reflectivity shape {x_reflectivity.shape} must match {self.image_shape}.")
        if x_reflectivity.device != self.device:
            x_reflectivity = x_reflectivity.to(self.device)
        if not torch.is_complex(x_reflectivity): # Reflectivity can be complex
             x_reflectivity = x_reflectivity.to(torch.complex64)

        # Perform 1D FFT along the depth dimension (axis=1) for each A-scan line
        y_spectral_signal = torch.fft.fft(x_reflectivity, dim=1, norm='ortho')

        return y_spectral_signal

    def op_adj(self, y_spectral_signal: torch.Tensor) -> torch.Tensor:
        """
        Adjoint OCT operation: Transforms k-space signal back to reflectivity profile.
        y_spectral_signal: (num_ascan_lines, depth_pixels) - S(k) for each A-scan.
        Returns: (num_ascan_lines, depth_pixels) - R(z) for each A-scan.
        """
        if y_spectral_signal.shape != self.image_shape:
            raise ValueError(f"Input y_spectral_signal shape {y_spectral_signal.shape} must match {self.image_shape}.")
        if y_spectral_signal.device != self.device:
            y_spectral_signal = y_spectral_signal.to(self.device)
        if not torch.is_complex(y_spectral_signal):
             y_spectral_signal = y_spectral_signal.to(torch.complex64)

        # Perform 1D IFFT along the depth dimension (axis=1) for each A-scan line
        x_reflectivity_adj = torch.fft.ifft(y_spectral_signal, dim=1, norm='ortho')

        return x_reflectivity_adj

if __name__ == '__main__':
    # This block will not be executed by this subtask script.
    # It's kept here for users who might run the file directly in a stable environment.
    print("Running basic OCTForwardOperator checks (if run directly)...")
    device = torch.device('cpu') # Force CPU for subtask __main__ to avoid potential CUDA init delays
    img_shape_test = (16, 32) # Smaller shapes

    try:
        oct_op_test = OCTForwardOperator(
            image_shape=img_shape_test,
            lambda_w=850e-9,
            z_max_m=0.002,
            device=device
        )
        print("OCTForwardOperator instantiated.")

        phantom_test = torch.randn(img_shape_test, dtype=torch.complex64, device=device)

        k_space_data_test = oct_op_test.op(phantom_test)
        print(f"Forward op output shape: {k_space_data_test.shape}")

        recon_adj_test = oct_op_test.op_adj(k_space_data_test)
        print(f"Adjoint op output shape: {recon_adj_test.shape}")

        x_dp_test = torch.randn_like(phantom_test)
        y_dp_rand_test = torch.randn_like(k_space_data_test)
        Ax_test = oct_op_test.op(x_dp_test)
        Aty_test = oct_op_test.op_adj(y_dp_rand_test)
        lhs_test = torch.vdot(Ax_test.flatten(), y_dp_rand_test.flatten())
        rhs_test = torch.vdot(x_dp_test.flatten(), Aty_test.flatten())
        print(f"Dot product test: LHS={lhs_test.item():.4f}, RHS={rhs_test.item():.4f}")

        print("OCTForwardOperator __main__ checks completed (if run directly).")
    except Exception as e:
        print(f"Error in OCTForwardOperator __main__ checks (if run directly): {e}")
