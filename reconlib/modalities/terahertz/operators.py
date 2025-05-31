import torch
from reconlib.operators import Operator
import numpy as np

class TerahertzOperator(Operator):
    """
    Forward and Adjoint Operator for Terahertz (THz) Imaging.

    Models THz wave interaction with a sample. This can vary greatly depending
    on the THz imaging mode (e.g., transmission, reflection, pulsed, continuous wave).
    A common mode is THz Computed Tomography (CT), which might use a Radon-like transform,
    or THz pulsed imaging which might involve deconvolution.

    This placeholder will assume a generic imaging scenario where the forward
    operator applies some system matrix (e.g., related to Fourier sampling or a point spread function)
    and the adjoint is its conjugate transpose.
    """
    def __init__(self, image_shape: tuple[int, int], system_matrix: torch.Tensor | None = None, device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape  # (Ny, Nx) or (Nz, Ny, Nx)
        self.device = torch.device(device)

        if system_matrix is not None:
            self.system_matrix = system_matrix.to(self.device)
            # Expected shape for system_matrix: (num_measurements, num_image_pixels)
            # where num_image_pixels = np.prod(image_shape)
            if self.system_matrix.shape[1] != np.prod(self.image_shape):
                raise ValueError(
                    f"System matrix columns ({self.system_matrix.shape[1]}) "
                    f"must match total number of image pixels ({np.prod(self.image_shape)})."
                )
            self.num_measurements = self.system_matrix.shape[0]
        else:
            # Placeholder: if no system matrix, create a dummy one (e.g., identity or random)
            # This would correspond to a very simple direct mapping for measurements
            print("Warning: No system_matrix provided to TerahertzOperator. Using a placeholder (random).")
            self.num_measurements = np.prod(self.image_shape) // 2 # Example: half the number of pixels
            self.system_matrix = torch.randn(
                self.num_measurements, np.prod(self.image_shape),
                dtype=torch.complex64 if np.random.rand() > 0.5 else torch.float32, # Randomly complex or real
                device=self.device
            ) * 0.1

        print(f"TerahertzOperator initialized for image shape {self.image_shape}, "
              f"system matrix shape {self.system_matrix.shape}.")

    def op(self, image_estimate: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Image estimate to THz measurement data.

        Args:
            image_estimate (torch.Tensor): The material property map (e.g., refractive index, absorption).
                                           Shape: self.image_shape.

        Returns:
            torch.Tensor: Simulated THz measurement data.
                          Shape: (num_measurements,).
        """
        if image_estimate.shape != self.image_shape:
            raise ValueError(f"Input image_estimate shape {image_estimate.shape} must match {self.image_shape}.")
        if image_estimate.device != self.device:
            image_estimate = image_estimate.to(self.device)

        image_vector = image_estimate.reshape(-1) # Flatten image

        # Ensure type consistency for matrix multiplication
        if self.system_matrix.is_complex() and not image_vector.is_complex():
            image_vector = image_vector.to(torch.complex64)
        elif not self.system_matrix.is_complex() and image_vector.is_complex():
            # This case might need careful handling depending on the physics
            # For simplicity, we'll cast system_matrix if image is complex
            print("Warning: Complex image with real system matrix. Casting system matrix to complex for matmul.")
            self.system_matrix = self.system_matrix.to(torch.complex64)


        # Forward operation: y = A * x
        # A is system_matrix, x is image_vector
        measurement_data = torch.matmul(self.system_matrix, image_vector)
        return measurement_data

    def op_adj(self, measurement_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: THz measurement data to image domain.
        This is typically the conjugate transpose of the system matrix.

        Args:
            measurement_data (torch.Tensor): THz measurement data.
                                             Shape: (num_measurements,).

        Returns:
            torch.Tensor: Image reconstructed by adjoint operation.
                          Shape: self.image_shape.
        """
        if measurement_data.ndim != 1 or measurement_data.shape[0] != self.num_measurements:
            raise ValueError(f"Input measurement_data has invalid shape {measurement_data.shape}. "
                             f"Expected 1D tensor of length {self.num_measurements}.")
        if measurement_data.device != self.device:
            measurement_data = measurement_data.to(self.device)

        # Adjoint operation: x_adj = A^H * y
        # A^H is the conjugate transpose of system_matrix

        # Ensure type consistency
        if self.system_matrix.is_complex() and not measurement_data.is_complex():
            measurement_data = measurement_data.to(torch.complex64)
        elif not self.system_matrix.is_complex() and measurement_data.is_complex() and self.system_matrix.dtype != measurement_data.dtype:
             # If system matrix is real and data is complex, ensure system_matrix is also complex for matmul
            self.system_matrix = self.system_matrix.to(torch.complex64)


        # system_matrix_adj = self.system_matrix.H # .H is conjugate transpose
        if self.system_matrix.is_complex():
            system_matrix_adj = self.system_matrix.conj().T
        else:
            system_matrix_adj = self.system_matrix.T # Just transpose for real matrix

        reconstructed_vector = torch.matmul(system_matrix_adj, measurement_data)
        reconstructed_image = reconstructed_vector.reshape(self.image_shape)

        # If original image was likely real, take real part of adjoint result
        # This depends on how system_matrix was defined.
        # For this placeholder, if system_matrix was real, output should be real.
        if not self.system_matrix.is_complex() and reconstructed_image.is_complex():
             reconstructed_image = reconstructed_image.real


        return reconstructed_image

if __name__ == '__main__':
    print("Running basic TerahertzOperator checks...")
    device_thz = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape_thz = (32, 32) # Ny, Nx
    num_pixels_thz = np.prod(img_shape_thz)
    num_measurements_thz = num_pixels_thz // 2 # Example

    # Create a dummy system matrix (e.g., random Gaussian)
    # Can be real or complex
    is_complex_system = np.random.rand() > 0.5
    dtype_system = torch.complex64 if is_complex_system else torch.float32

    system_matrix_thz = torch.randn(num_measurements_thz, num_pixels_thz, dtype=dtype_system, device=device_thz)

    try:
        thz_op_test = TerahertzOperator(
            image_shape=img_shape_thz,
            system_matrix=system_matrix_thz,
            device=device_thz
        )
        print("TerahertzOperator instantiated.")

        # Create a simple phantom image
        phantom_thz_image_real = torch.zeros(img_shape_thz, device=device_thz)
        phantom_thz_image_real[img_shape_thz[0]//4:img_shape_thz[0]//4*3, img_shape_thz[1]//4:img_shape_thz[1]//4*3] = 1.0

        # Decide if phantom should be complex based on system matrix for simplicity in test
        phantom_thz_image = phantom_thz_image_real.to(dtype_system) if is_complex_system else phantom_thz_image_real


        simulated_measurements_thz = thz_op_test.op(phantom_thz_image)
        print(f"Forward op output shape (measurements): {simulated_measurements_thz.shape}")
        assert simulated_measurements_thz.shape == (num_measurements_thz,)

        reconstructed_image_thz = thz_op_test.op_adj(simulated_measurements_thz)
        print(f"Adjoint op output shape (reconstructed image): {reconstructed_image_thz.shape}")
        assert reconstructed_image_thz.shape == img_shape_thz

        # Basic dot product test
        # Create x_dp that matches the dtype of the system_matrix or image for consistency
        x_dp_thz_real = torch.randn(img_shape_thz, device=device_thz)
        x_dp_thz = x_dp_thz_real.to(dtype_system) if is_complex_system else x_dp_thz_real

        # Create y_dp_rand that matches the dtype of the measurements
        y_dp_rand_thz = torch.randn(num_measurements_thz, dtype=simulated_measurements_thz.dtype, device=device_thz)

        Ax_thz = thz_op_test.op(x_dp_thz)
        Aty_thz = thz_op_test.op_adj(y_dp_rand_thz)

        # Dot product: <Ax, y> vs <x, A^H y>
        if Ax_thz.is_complex() or y_dp_rand_thz.is_complex():
            lhs_thz = torch.vdot(Ax_thz.flatten(), y_dp_rand_thz.flatten())
        else:
            lhs_thz = torch.dot(Ax_thz.flatten(), y_dp_rand_thz.flatten())

        if x_dp_thz.is_complex() or Aty_thz.is_complex():
            rhs_thz = torch.vdot(x_dp_thz.flatten(), Aty_thz.flatten())
        else:
            rhs_thz = torch.dot(x_dp_thz.flatten(), Aty_thz.flatten())

        print(f"THz Dot product test: LHS={lhs_thz.item():.4f}, RHS={rhs_thz.item():.4f}")
        # This test should pass if op and op_adj are correctly implemented as A and A^H
        assert np.isclose(lhs_thz.real.item(), rhs_thz.real.item(), rtol=1e-3), "Real parts of dot product differ"
        if is_complex_system: # Only check imaginary part if system is complex
            assert np.isclose(lhs_thz.imag.item(), rhs_thz.imag.item(), rtol=1e-3), "Imaginary parts of dot product differ"


        print("TerahertzOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in TerahertzOperator __main__ checks: {e}")
        import traceback
        traceback.print_exc()
