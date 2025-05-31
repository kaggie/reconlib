import torch
from reconlib.operators import Operator
import numpy as np

class EITOperator(Operator):
    """
    Placeholder Forward and Adjoint Operator for Electrical Impedance Tomography (EIT).

    Assumes a **linearized model** where changes in boundary voltage measurements (delta_v)
    are related to changes in internal conductivity (delta_sigma) by a
    sensitivity matrix J (Jacobian):
        delta_v = J @ delta_sigma_flattened

    In this placeholder, J is a randomly generated matrix as calculating a true
    sensitivity matrix requires solving the forward EIT problem (e.g., using FEM).
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx) - for the conductivity map delta_sigma
                 num_measurements: int,        # Number of boundary voltage difference measurements
                 sensitivity_matrix_J: torch.Tensor | None = None, # The Jacobian matrix
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.Ny, self.Nx = self.image_shape
        self.num_image_pixels = self.Ny * self.Nx
        self.num_measurements = num_measurements
        self.device = torch.device(device)

        if sensitivity_matrix_J is not None:
            if sensitivity_matrix_J.shape != (self.num_measurements, self.num_image_pixels):
                raise ValueError(f"Provided sensitivity_matrix_J shape {sensitivity_matrix_J.shape} "
                                 f"is not ({self.num_measurements}, {self.num_image_pixels}).")
            self.J = sensitivity_matrix_J.to(self.device)
        else:
            print("Warning: No sensitivity_matrix_J provided to EITOperator. Using a random placeholder matrix.")
            # Generate a random placeholder J
            self.J = torch.randn(self.num_measurements, self.num_image_pixels, device=self.device) * 0.01
            # Small values to keep outputs in a reasonable range for a placeholder

        self.measurement_shape = (self.num_measurements,)

        print(f"EITOperator (Linearized Placeholder) initialized.")
        print(f"  Image (delta_sigma) Shape: {self.image_shape}, Num Pixels: {self.num_image_pixels}")
        print(f"  Num Measurements (delta_v): {self.num_measurements}")
        print(f"  Sensitivity Matrix J shape: {self.J.shape}")

    def op(self, delta_sigma_map: torch.Tensor) -> torch.Tensor:
        """
        Forward: Conductivity change map (delta_sigma) to boundary voltage change measurements (delta_v).
        delta_v = J @ delta_sigma_flattened
        """
        if delta_sigma_map.shape != self.image_shape:
            raise ValueError(f"Input delta_sigma_map shape {delta_sigma_map.shape} must match {self.image_shape}.")
        delta_sigma_map = delta_sigma_map.to(self.device)

        delta_sigma_flat = delta_sigma_map.reshape(-1) # Flatten

        # Ensure dtypes match for matmul
        if delta_sigma_flat.dtype != self.J.dtype:
            delta_sigma_flat = delta_sigma_flat.to(self.J.dtype)

        delta_v = torch.matmul(self.J, delta_sigma_flat)
        return delta_v

    def op_adj(self, delta_v_measurements: torch.Tensor) -> torch.Tensor:
        """
        Adjoint: Boundary voltage change measurements (delta_v) to conductivity change map (delta_sigma) domain.
        delta_sigma_adj = J.T @ delta_v
        """
        if delta_v_measurements.shape != self.measurement_shape:
            raise ValueError(f"Input delta_v_measurements shape {delta_v_measurements.shape} must match {self.measurement_shape}.")
        delta_v_measurements = delta_v_measurements.to(self.device)

        # Ensure dtypes match for matmul
        if delta_v_measurements.dtype != self.J.T.dtype: # J.T might change dtype if J is special
             delta_v_measurements = delta_v_measurements.to(self.J.T.dtype)

        delta_sigma_adj_flat = torch.matmul(self.J.T, delta_v_measurements)
        return delta_sigma_adj_flat.reshape(self.image_shape)

if __name__ == '__main__':
    print("\nRunning basic EITOperator (Placeholder) checks...")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s = (32, 32)
    n_meas = 64 # Example: 16 electrodes, some number of unique current injection/voltage measurement pairs

    try:
        eit_op = EITOperator(image_shape=img_s, num_measurements=n_meas, device=dev)
        print("EITOperator instantiated with random J.")

        delta_sigma_phantom = torch.zeros(img_s, device=dev)
        delta_sigma_phantom[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 0.1 # Conductivity change
        delta_sigma_phantom[10:15,10:15] = -0.05


        delta_v_sim = eit_op.op(delta_sigma_phantom)
        print(f"Forward op output shape (delta_v): {delta_v_sim.shape}")
        assert delta_v_sim.shape == (n_meas,)

        adj_recon_sigma = eit_op.op_adj(delta_v_sim)
        print(f"Adjoint op output shape (delta_sigma_adj): {adj_recon_sigma.shape}")
        assert adj_recon_sigma.shape == img_s

        # Dot product test
        x_dp = torch.randn_like(delta_sigma_phantom)
        y_dp_rand = torch.randn_like(delta_v_sim)
        Ax = eit_op.op(x_dp)
        Aty = eit_op.op_adj(y_dp_rand)
        lhs = torch.dot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.dot(x_dp.flatten(), Aty.flatten())
        print(f"EIT Dot product test: LHS={lhs.item():.6f}, RHS={rhs.item():.6f}")
        # With random J, precision might be lower for rtol
        assert np.isclose(lhs.item(), rhs.item(), rtol=1e-3, atol=1e-5), "Dot product test failed."

        print("EITOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in EITOperator __main__ checks: {e}")
        import traceback; traceback.print_exc()
