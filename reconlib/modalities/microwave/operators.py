import torch
from reconlib.operators import Operator
import numpy as np

class MicrowaveImagingOperator(Operator):
    """
    Forward and Adjoint Operator for Microwave Imaging (MWI).

    Models the scattering of microwave signals by an object, relating its
    dielectric properties (e.g., permittivity, conductivity) to measurements
    made by an array of antennas.

    Microwave imaging is often a non-linear inverse scattering problem.
    Common linearized approximations (e.g., Born or Rytov approximations)
    can lead to a linear system Ax = y, where x represents the contrast
    in dielectric properties and y is the scattered field data.

    This placeholder will use a system matrix 'A' to model this relationship,
    similar to the Terahertz placeholder.
    """
    def __init__(self, image_shape: tuple[int, int] | tuple[int,int,int],
                 system_matrix: torch.Tensor, # (num_measurements, num_image_pixels)
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape  # (Ny, Nx) or (Nz, Ny, Nx) for dielectric property map
        self.device = torch.device(device)

        self.system_matrix = system_matrix.to(self.device)
        # num_image_pixels = np.prod(image_shape)
        # num_measurements = self.system_matrix.shape[0]

        if self.system_matrix.shape[1] != np.prod(self.image_shape):
            raise ValueError(
                f"System matrix columns ({self.system_matrix.shape[1]}) "
                f"must match total number of image pixels ({np.prod(self.image_shape)})."
            )
        self.num_measurements = self.system_matrix.shape[0]

        # Microwave data is often complex (amplitude and phase)
        if not self.system_matrix.is_complex():
            print("Warning: MicrowaveImagingOperator system_matrix is real. Microwave data is typically complex.")
            # Forcing it to be complex for this placeholder, as it's common.
            # In a real scenario, the matrix would be derived from physics and could be complex.
            self.system_matrix = self.system_matrix.to(torch.complex64)


        print(f"MicrowaveImagingOperator initialized for image shape {self.image_shape}, "
              f"system matrix shape {self.system_matrix.shape} (dtype: {self.system_matrix.dtype}).")

    def op(self, dielectric_contrast_map: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Dielectric contrast map to scattered field data.

        Args:
            dielectric_contrast_map (torch.Tensor): The map of dielectric property contrast.
                                                   Shape: self.image_shape.

        Returns:
            torch.Tensor: Simulated scattered field data (complex-valued).
                          Shape: (num_measurements,).
        """
        if dielectric_contrast_map.shape != self.image_shape:
            raise ValueError(f"Input map shape {dielectric_contrast_map.shape} must match {self.image_shape}.")
        if dielectric_contrast_map.device != self.device:
            dielectric_contrast_map = dielectric_contrast_map.to(self.device)

        image_vector = dielectric_contrast_map.reshape(-1) # Flatten image

        # Ensure image_vector is complex if system_matrix is complex
        if self.system_matrix.is_complex() and not image_vector.is_complex():
            image_vector = image_vector.to(torch.complex64)

        # Forward operation: y = A * x
        scattered_data = torch.matmul(self.system_matrix, image_vector)
        return scattered_data

    def op_adj(self, scattered_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Scattered field data to image domain (dielectric contrast map).
        This is the conjugate transpose of the system matrix.

        Args:
            scattered_data (torch.Tensor): Microwave scattered field data (complex-valued).
                                           Shape: (num_measurements,).

        Returns:
            torch.Tensor: Image reconstructed by adjoint operation.
                          Shape: self.image_shape.
                          Should be complex if system matrix is complex.
        """
        if scattered_data.ndim != 1 or scattered_data.shape[0] != self.num_measurements:
            raise ValueError(f"Input data has invalid shape {scattered_data.shape}. "
                             f"Expected 1D tensor of length {self.num_measurements}.")
        if scattered_data.device != self.device:
            scattered_data = scattered_data.to(self.device)

        if not scattered_data.is_complex():
            print("Warning: op_adj received real scattered_data. Microwave data is typically complex.")
            scattered_data = scattered_data.to(torch.complex64) # Ensure complex for A^H y

        # Adjoint operation: x_adj = A^H * y
        # system_matrix.H is conjugate transpose
        system_matrix_adj = self.system_matrix.conj().T

        reconstructed_vector = torch.matmul(system_matrix_adj, scattered_data)
        reconstructed_image = reconstructed_vector.reshape(self.image_shape)

        return reconstructed_image # Output is generally complex

if __name__ == '__main__':
    print("Running basic MicrowaveImagingOperator checks...")
    device_mwi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape_mwi = (32, 32) # Dielectric property map (e.g., permittivity contrast)
    num_pixels_mwi = np.prod(img_shape_mwi)
    # num_antennas = 16
    # num_measurements_mwi = num_antennas * (num_antennas -1) # Example for multi-static setup
    num_measurements_mwi = num_pixels_mwi // 2 # Simplified for placeholder

    # System matrix A for MWI is typically complex
    system_matrix_mwi = torch.randn(num_measurements_mwi, num_pixels_mwi, dtype=torch.complex64, device=device_mwi)

    try:
        mwi_op_test = MicrowaveImagingOperator(
            image_shape=img_shape_mwi,
            system_matrix=system_matrix_mwi,
            device=device_mwi
        )
        print("MicrowaveImagingOperator instantiated.")

        # Create a simple phantom (dielectric contrast)
        # This phantom should be complex if we are reconstructing complex permittivity,
        # or real if reconstructing only real part or magnitude.
        # For simplicity with complex system matrix, let's use complex phantom.
        phantom_contrast_real = torch.zeros(img_shape_mwi, device=device_mwi)
        phantom_contrast_real[img_shape_mwi[0]//4:img_shape_mwi[0]//4*3, img_shape_mwi[1]//4:img_shape_mwi[1]//4*3] = 1.0 # Real part
        phantom_contrast_imag = torch.zeros(img_shape_mwi, device=device_mwi)
        phantom_contrast_imag[img_shape_mwi[0]//3:img_shape_mwi[0]//3*2, img_shape_mwi[1]//3:img_shape_mwi[1]//3*2] = 0.5 # Imaginary part (loss)
        phantom_contrast_mwi = torch.complex(phantom_contrast_real, phantom_contrast_imag)


        simulated_scatter_mwi = mwi_op_test.op(phantom_contrast_mwi)
        print(f"Forward op output shape (scattered data): {simulated_scatter_mwi.shape}")
        assert simulated_scatter_mwi.shape == (num_measurements_mwi,)
        assert simulated_scatter_mwi.is_complex()

        reconstructed_map_mwi = mwi_op_test.op_adj(simulated_scatter_mwi)
        print(f"Adjoint op output shape (reconstructed map): {reconstructed_map_mwi.shape}")
        assert reconstructed_map_mwi.shape == img_shape_mwi
        assert reconstructed_map_mwi.is_complex()


        # Basic dot product test
        x_dp_mwi_real = torch.randn(img_shape_mwi, device=device_mwi)
        x_dp_mwi_imag = torch.randn(img_shape_mwi, device=device_mwi)
        x_dp_mwi = torch.complex(x_dp_mwi_real, x_dp_mwi_imag)

        y_dp_rand_mwi = torch.randn(num_measurements_mwi, dtype=torch.complex64, device=device_mwi)

        Ax_mwi = mwi_op_test.op(x_dp_mwi)
        Aty_mwi = mwi_op_test.op_adj(y_dp_rand_mwi)

        lhs_mwi = torch.vdot(Ax_mwi.flatten(), y_dp_rand_mwi.flatten())
        rhs_mwi = torch.vdot(x_dp_mwi.flatten(), Aty_mwi.flatten())

        print(f"MWI Dot product test: LHS={lhs_mwi.item():.4f}, RHS={rhs_mwi.item():.4f}")
        assert np.isclose(lhs_mwi.real.item(), rhs_mwi.real.item(), rtol=1e-3), "Real parts of dot product differ"
        assert np.isclose(lhs_mwi.imag.item(), rhs_mwi.imag.item(), rtol=1e-3), "Imaginary parts of dot product differ"

        print("MicrowaveImagingOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in MicrowaveImagingOperator __main__ checks: {e}")
        import traceback
        traceback.print_exc()
