import torch
from reconlib.operators import Operator
import numpy as np

class HyperspectralImagingOperator(Operator):
    """
    Forward and Adjoint Operator for Hyperspectral Imaging (HSI).

    Models the acquisition process in HSI, often in the context of
    compressed sensing or spectral unmixing.

    - For Compressed HSI (Snapshot Spectral Imaging):
      A common model is y = Hx, where x is the flattened hyperspectral cube (X_vector),
      H is the sensing matrix (representing encoding, dispersion, etc.),
      and y is the set of measurements from a 2D detector.
      The image_shape would be (Ny, Nx, N_bands).
      The measurement_shape would be (N_detector_pixels_y, N_detector_pixels_x) or flattened.

    - For Spectral Unmixing (less of an 'operator' in this sense, but related):
      If x is the abundance maps and A contains endmember spectra, then M = Ax,
      where M is the observed hyperspectral cube. This is a different problem structure.

    This placeholder will focus on the compressed HSI model y = Hx.
    The 'image' to reconstruct is the full hyperspectral cube.
    """
    def __init__(self, image_shape: tuple[int, int, int], # (Ny, Nx, N_bands)
                 sensing_matrix: torch.Tensor, # (num_measurements, Ny*Nx*N_bands)
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape  # (Ny, Nx, N_bands)
        self.device = torch.device(device)

        self.sensing_matrix = sensing_matrix.to(self.device)
        # num_image_elements = np.prod(image_shape)
        # num_measurements = self.sensing_matrix.shape[0]

        if self.sensing_matrix.shape[1] != np.prod(self.image_shape):
            raise ValueError(
                f"Sensing matrix columns ({self.sensing_matrix.shape[1]}) "
                f"must match total number of hyperspectral image elements ({np.prod(self.image_shape)})."
            )
        self.num_measurements = self.sensing_matrix.shape[0]

        print(f"HyperspectralImagingOperator initialized for image cube shape {self.image_shape}, "
              f"sensing matrix shape {self.sensing_matrix.shape}.")

    def op(self, hyperspectral_cube: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Hyperspectral data cube to sensor measurements.

        Args:
            hyperspectral_cube (torch.Tensor): The full hyperspectral data cube.
                                               Shape: self.image_shape (Ny, Nx, N_bands).

        Returns:
            torch.Tensor: Simulated sensor measurement data.
                          Shape: (num_measurements,).
        """
        if hyperspectral_cube.shape != self.image_shape:
            raise ValueError(f"Input cube shape {hyperspectral_cube.shape} must match {self.image_shape}.")
        if hyperspectral_cube.device != self.device:
            hyperspectral_cube = hyperspectral_cube.to(self.device)

        image_vector = hyperspectral_cube.reshape(-1) # Flatten the cube (Ny*Nx*N_bands)

        # Ensure type consistency, though HSI data and matrix are usually real
        if self.sensing_matrix.is_complex() and not image_vector.is_complex():
            image_vector = image_vector.to(torch.complex64)
        elif not self.sensing_matrix.is_complex() and image_vector.is_complex():
            # This case is unlikely for standard HSI sensing models
            print("Warning: Complex image with real sensing matrix. Taking real part of image.")
            image_vector = image_vector.real

        # Forward operation: y = H * x
        measurement_data = torch.matmul(self.sensing_matrix, image_vector)
        return measurement_data

    def op_adj(self, measurement_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Sensor measurements to hyperspectral data cube domain.
        This is the transpose (or conjugate transpose if complex) of the sensing matrix.

        Args:
            measurement_data (torch.Tensor): Sensor measurement data.
                                             Shape: (num_measurements,).

        Returns:
            torch.Tensor: Hyperspectral data cube reconstructed by adjoint operation.
                          Shape: self.image_shape.
        """
        if measurement_data.ndim != 1 or measurement_data.shape[0] != self.num_measurements:
            raise ValueError(f"Input data has invalid shape {measurement_data.shape}. "
                             f"Expected 1D tensor of length {self.num_measurements}.")
        if measurement_data.device != self.device:
            measurement_data = measurement_data.to(self.device)

        # Adjoint operation: x_adj = H^T * y (or H^H * y if complex)
        if self.sensing_matrix.is_complex():
            sensing_matrix_adj = self.sensing_matrix.conj().T
            if not measurement_data.is_complex(): # Ensure data is complex for matmul
                 measurement_data = measurement_data.to(torch.complex64)
        else:
            sensing_matrix_adj = self.sensing_matrix.T
            if measurement_data.is_complex(): # Ensure data is real if matrix is real
                 measurement_data = measurement_data.real


        reconstructed_vector = torch.matmul(sensing_matrix_adj, measurement_data)
        reconstructed_cube = reconstructed_vector.reshape(self.image_shape)

        return reconstructed_cube

if __name__ == '__main__':
    print("Running basic HyperspectralImagingOperator checks...")
    device_hsi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image cube: Ny, Nx, N_bands
    img_shape_hsi = (32, 32, 10) # 10 spectral bands
    num_elements_hsi = np.prod(img_shape_hsi)

    # Number of measurements (e.g., pixels on a 2D detector in a CASSI-like system)
    # For compressed sensing, num_measurements < num_elements_hsi
    num_measurements_hsi = num_elements_hsi // 3 # Compression factor of 3

    # Sensing matrix H (usually real for HSI)
    sensing_matrix_hsi = torch.randn(num_measurements_hsi, num_elements_hsi, dtype=torch.float32, device=device_hsi)

    try:
        hsi_op_test = HyperspectralImagingOperator(
            image_shape=img_shape_hsi,
            sensing_matrix=sensing_matrix_hsi,
            device=device_hsi
        )
        print("HyperspectralImagingOperator instantiated.")

        # Create a simple phantom hyperspectral cube
        phantom_hsi_cube = torch.zeros(img_shape_hsi, device=device_hsi)
        # Add a 'square' feature that has a specific spectrum
        square_region = (slice(img_shape_hsi[0]//4, img_shape_hsi[0]*3//4),
                         slice(img_shape_hsi[1]//4, img_shape_hsi[1]*3//4))
        spectrum1 = torch.sin(torch.linspace(0, np.pi*2, img_shape_hsi[2], device=device_hsi)) * 0.5 + 0.5
        phantom_hsi_cube[square_region[0], square_region[1], :] = spectrum1.unsqueeze(0).unsqueeze(0)
        # Add another feature
        circle_center_y, circle_center_x = img_shape_hsi[0]*2//3, img_shape_hsi[1]*2//3
        radius = img_shape_hsi[0]//5
        yy,xx = torch.meshgrid(torch.arange(img_shape_hsi[0], device=device_hsi), torch.arange(img_shape_hsi[1], device=device_hsi), indexing='ij')
        mask = (xx - circle_center_x)**2 + (yy - circle_center_y)**2 < radius**2
        spectrum2 = torch.cos(torch.linspace(0, np.pi*2, img_shape_hsi[2], device=device_hsi)) * 0.5 + 0.5
        phantom_hsi_cube[mask,:] = spectrum2.unsqueeze(0)


        simulated_measurements_hsi = hsi_op_test.op(phantom_hsi_cube)
        print(f"Forward op output shape (measurements): {simulated_measurements_hsi.shape}")
        assert simulated_measurements_hsi.shape == (num_measurements_hsi,)

        reconstructed_cube_hsi = hsi_op_test.op_adj(simulated_measurements_hsi)
        print(f"Adjoint op output shape (reconstructed cube): {reconstructed_cube_hsi.shape}")
        assert reconstructed_cube_hsi.shape == img_shape_hsi

        # Basic dot product test
        x_dp_hsi = torch.randn_like(phantom_hsi_cube)
        y_dp_rand_hsi = torch.randn_like(simulated_measurements_hsi)

        Ax_hsi = hsi_op_test.op(x_dp_hsi)
        Aty_hsi = hsi_op_test.op_adj(y_dp_rand_hsi)

        # Assuming real data and matrix for HSI
        lhs_hsi = torch.dot(Ax_hsi.flatten(), y_dp_rand_hsi.flatten())
        rhs_hsi = torch.dot(x_dp_hsi.flatten(), Aty_hsi.flatten())

        print(f"HSI Dot product test: LHS={lhs_hsi.item():.4f}, RHS={rhs_hsi.item():.4f}")
        assert np.isclose(lhs_hsi.item(), rhs_hsi.item(), rtol=1e-3), "Dot product test failed for HSI operator."

        print("HyperspectralImagingOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in HyperspectralImagingOperator __main__ checks: {e}")
        import traceback
        traceback.print_exc()
