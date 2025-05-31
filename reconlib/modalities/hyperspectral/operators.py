import torch
from reconlib.operators import Operator
import numpy as np

class HyperspectralImagingOperator(Operator):
    """
    Forward and Adjoint Operator for Hyperspectral Imaging (HSI).

    Models the acquisition process in HSI, particularly for Compressed HSI (CS-HSI)
    where measurements are obtained via a sensing matrix: y = Hx.
    'x' is the flattened hyperspectral cube (X_vector).
    'H' is the sensing matrix, which can be dense or sparse. This implementation
    allows for a generic H.
    'y' is the set of measurements.
    """
    def __init__(self,
                 image_shape: tuple[int, int, int], # (Ny, Nx, N_bands)
                 sensing_matrix: torch.Tensor, # (num_measurements, Ny*Nx*N_bands)
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape  # (Ny, Nx, N_bands)
        self.Ny, self.Nx, self.N_bands = self.image_shape
        self.device = torch.device(device)

        if not isinstance(sensing_matrix, torch.Tensor):
            raise TypeError("sensing_matrix must be a PyTorch Tensor.")

        self.sensing_matrix = sensing_matrix.to(self.device)

        num_image_elements = np.prod(self.image_shape)
        if self.sensing_matrix.shape[1] != num_image_elements:
            raise ValueError(
                f"Sensing matrix columns ({self.sensing_matrix.shape[1]}) "
                f"must match total number of hyperspectral image elements ({num_image_elements})."
            )
        self.num_measurements = self.sensing_matrix.shape[0]

        print(f"HyperspectralImagingOperator initialized for image cube shape {self.image_shape}.")
        print(f"  Sensing Matrix H: shape {self.sensing_matrix.shape}, type {self.sensing_matrix.dtype}, device {self.sensing_matrix.device}.")
        if self.sensing_matrix.is_sparse:
             # Calculate density for sparse COO tensors
             density = self.sensing_matrix._nnz() / float(np.prod(self.sensing_matrix.shape))
             print(f"  Sensing Matrix is Sparse (density: {density:.4f})")


    def op(self, hyperspectral_cube: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Hyperspectral data cube to sensor measurements. y = Hx
        """
        if hyperspectral_cube.shape != self.image_shape:
            raise ValueError(f"Input cube shape {hyperspectral_cube.shape} must match {self.image_shape}.")
        hyperspectral_cube = hyperspectral_cube.to(self.device)

        image_vector = hyperspectral_cube.reshape(-1) # Flatten the cube

        # Type handling (HSI data usually real, H can be real or complex in some abstract CS cases)
        if self.sensing_matrix.is_complex() and not image_vector.is_complex():
            image_vector = image_vector.to(torch.complex64)
        elif not self.sensing_matrix.is_complex() and image_vector.is_complex():
            print("Warning: Complex HSI cube with real sensing matrix. Taking real part of cube.")
            image_vector = image_vector.real

        # Ensure dtypes match for matmul if one is float and other is double etc.
        # Cast image_vector to sensing_matrix's dtype for matmul
        if image_vector.dtype != self.sensing_matrix.dtype:
            image_vector = image_vector.to(self.sensing_matrix.dtype)


        measurement_data = torch.matmul(self.sensing_matrix, image_vector)
        return measurement_data

    def op_adj(self, measurement_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Sensor measurements to HSI cube domain. x_adj = H^T y or H^H y
        """
        if measurement_data.ndim != 1 or measurement_data.shape[0] != self.num_measurements:
            raise ValueError(f"Input data has invalid shape {measurement_data.shape}. Expected ({self.num_measurements},).")
        measurement_data = measurement_data.to(self.device)

        # Determine adjoint matrix
        if self.sensing_matrix.is_complex():
            sensing_matrix_adj = self.sensing_matrix.conj().T
            if not measurement_data.is_complex(): # Ensure data matches matrix complexity
                 measurement_data = measurement_data.to(torch.complex64)
        else: # Sensing matrix is real
            sensing_matrix_adj = self.sensing_matrix.T
            if measurement_data.is_complex(): # If data is complex but matrix real, result of H^T y should be real
                 measurement_data = measurement_data.real

        # Ensure dtypes match for matmul
        if measurement_data.dtype != sensing_matrix_adj.dtype:
             measurement_data = measurement_data.to(sensing_matrix_adj.dtype)


        reconstructed_vector = torch.matmul(sensing_matrix_adj, measurement_data)
        reconstructed_cube = reconstructed_vector.reshape(self.image_shape)

        return reconstructed_cube

def create_sparse_sensing_matrix(num_measurements: int, num_image_elements: int,
                                 sparsity_factor: float = 0.1, device='cpu',
                                 dtype=torch.float32) -> torch.Tensor:
    """
    Creates a sparse random sensing matrix H (torch.sparse_coo_tensor).
    Each row (measurement) will have approximately sparsity_factor * num_image_elements non-zero entries.
    This is a generic way to make H sparse, not specific to a CASSI architecture.
    """
    if not (0 < sparsity_factor <= 1):
        raise ValueError("Sparsity factor must be between 0 and 1 (exclusive of 0).")

    num_non_zero_per_row = int(np.ceil(num_image_elements * sparsity_factor))
    if num_non_zero_per_row == 0: num_non_zero_per_row = 1

    total_non_zero_elements = num_measurements * num_non_zero_per_row

    # Ensure we don't request more non-zero elements than possible if num_image_elements is small
    if num_non_zero_per_row > num_image_elements :
        num_non_zero_per_row = num_image_elements
        total_non_zero_elements = num_measurements * num_image_elements


    rows = torch.arange(num_measurements, device=device).repeat_interleave(num_non_zero_per_row)

    # For columns, ensure unique selections per row if replace=False, or allow repeats
    # Using replace=True is simpler and often fine for random matrices.
    # If num_non_zero_per_row is small relative to num_image_elements, duplicates are less likely.
    cols = torch.randint(0, num_image_elements, (total_non_zero_elements,), device=device)

    # For truly unique columns per row (more complex to generate efficiently for large sparse matrices):
    # cols_list = []
    # for _ in range(num_measurements):
    #    cols_list.append(torch.randperm(num_image_elements, device=device)[:num_non_zero_per_row])
    # cols = torch.cat(cols_list)

    values = torch.randn(total_non_zero_elements, device=device, dtype=dtype)

    indices = torch.stack([rows, cols], dim=0)

    sparse_matrix = torch.sparse_coo_tensor(indices, values,
                                            (num_measurements, num_image_elements),
                                            device=device)
    return sparse_matrix.coalesce() # Sums duplicate indices if any, and sorts them


if __name__ == '__main__':
    print("\nRunning basic HyperspectralImagingOperator (Sparse H) checks...")
    device_hsi_op = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_s_hsi = (16, 16, 8) # Ny, Nx, N_bands
    num_elements_hsi = np.prod(img_s_hsi)
    num_measurements_hsi = num_elements_hsi // 3

    h_matrix_sparse = create_sparse_sensing_matrix(
        num_measurements_hsi, num_elements_hsi,
        sparsity_factor=0.05,
        device=device_hsi_op,
        dtype=torch.float32
    )

    # Test with a dense matrix as well
    # h_matrix_dense = torch.randn(num_measurements_hsi, num_elements_hsi, dtype=torch.float32, device=device_hsi_op)

    for h_matrix_type, h_matrix in [("Sparse", h_matrix_sparse)]: #, ("Dense", h_matrix_dense)
        print(f"--- Testing with {h_matrix_type} Sensing Matrix ---")
        try:
            hsi_op = HyperspectralImagingOperator(
                image_shape=img_s_hsi,
                sensing_matrix=h_matrix,
                device=device_hsi_op
            )
            print(f"HyperspectralImagingOperator ({h_matrix_type} H) instantiated.")

            phantom_hsi = torch.randn(img_s_hsi, device=device_hsi_op, dtype=torch.float32)

            simulated_measurements = hsi_op.op(phantom_hsi)
            print(f"Forward op output shape: {simulated_measurements.shape}")
            assert simulated_measurements.shape == (num_measurements_hsi,)

            reconstructed_adj = hsi_op.op_adj(simulated_measurements)
            print(f"Adjoint op output shape: {reconstructed_adj.shape}")
            assert reconstructed_adj.shape == img_s_hsi

            # Dot product test
            x_dp = torch.randn_like(phantom_hsi) # Real input
            y_dp_rand = torch.randn_like(simulated_measurements) # Real measurements (since H is real)

            Ax = hsi_op.op(x_dp)
            Aty = hsi_op.op_adj(y_dp_rand)

            lhs = torch.dot(Ax.flatten(), y_dp_rand.flatten())
            rhs = torch.dot(x_dp.flatten(), Aty.flatten())

            print(f"HSI ({h_matrix_type} H) Dot product test: LHS={lhs.item():.6f}, RHS={rhs.item():.6f}")
            assert np.isclose(lhs.item(), rhs.item(), rtol=1e-3, atol=1e-5), f"Dot product test failed for {h_matrix_type} H."

            print(f"HyperspectralImagingOperator ({h_matrix_type} H) __main__ checks completed.")

        except Exception as e:
            print(f"Error in HyperspectralImagingOperator ({h_matrix_type} H) __main__ checks: {e}")
            import traceback
            traceback.print_exc()
