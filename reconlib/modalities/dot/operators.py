import torch
from reconlib.operators import Operator
import numpy as np

class DOTOperator(Operator):
    """
    Placeholder Forward and Adjoint Operator for Diffuse Optical Tomography (DOT).

    Assumes a **linearized model** where changes in boundary measurements (delta_y)
    (e.g., log-amplitude change, phase change) are related to changes in internal
    optical properties (delta_mu_a, delta_mu_s_prime) by a sensitivity matrix J:
        delta_y = J @ delta_mu_flattened

    In this placeholder, J is a randomly generated matrix. Calculating a true J
    requires solving the Diffusion Equation (or other light transport models) using FEM/FDM.
    The image (delta_mu) can be for a single property (e.g., delta_mu_a) or
    for multiple properties stacked (e.g., [delta_mu_a, delta_mu_s_prime]).
    This placeholder assumes a single property map for simplicity.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx) - for the optical property change map (delta_mu)
                 num_measurements: int,        # Number of boundary measurements
                 sensitivity_matrix_J: torch.Tensor | None = None,
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
            print("Warning: No sensitivity_matrix_J provided to DOTOperator. Using a random placeholder matrix.")
            self.J = torch.randn(self.num_measurements, self.num_image_pixels, device=self.device) * 0.001
            # Small values for placeholder J

        self.measurement_shape = (self.num_measurements,)

        print(f"DOTOperator (Linearized Placeholder) initialized.")
        print(f"  Image (delta_mu) Shape: {self.image_shape}, Num Pixels: {self.num_image_pixels}")
        print(f"  Num Measurements (delta_y): {self.num_measurements}")
        print(f"  Sensitivity Matrix J shape: {self.J.shape}")

    def op(self, delta_mu_map: torch.Tensor) -> torch.Tensor:
        """
        Forward: Optical property change map (delta_mu) to boundary measurement changes (delta_y).
        delta_y = J @ delta_mu_flattened
        """
        if delta_mu_map.shape != self.image_shape:
            raise ValueError(f"Input delta_mu_map shape {delta_mu_map.shape} must match {self.image_shape}.")
        delta_mu_map = delta_mu_map.to(self.device)

        delta_mu_flat = delta_mu_map.reshape(-1)

        if delta_mu_flat.dtype != self.J.dtype:
            delta_mu_flat = delta_mu_flat.to(self.J.dtype)

        delta_y = torch.matmul(self.J, delta_mu_flat)
        return delta_y

    def op_adj(self, delta_y_measurements: torch.Tensor) -> torch.Tensor:
        """
        Adjoint: Boundary measurement changes (delta_y) to optical property change map (delta_mu) domain.
        delta_mu_adj = J.T @ delta_y
        """
        if delta_y_measurements.shape != self.measurement_shape:
            raise ValueError(f"Input delta_y_measurements shape {delta_y_measurements.shape} must match {self.measurement_shape}.")
        delta_y_measurements = delta_y_measurements.to(self.device)

        if delta_y_measurements.dtype != self.J.T.dtype:
             delta_y_measurements = delta_y_measurements.to(self.J.T.dtype)

        delta_mu_adj_flat = torch.matmul(self.J.T, delta_y_measurements)
        return delta_mu_adj_flat.reshape(self.image_shape)

if __name__ == '__main__':
    print("\nRunning basic DOTOperator (Placeholder) checks...")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s = (24, 24)
    n_meas = 48

    try:
        dot_op = DOTOperator(image_shape=img_s, num_measurements=n_meas, device=dev)
        print("DOTOperator instantiated with random J.")

        delta_mu_phantom = torch.zeros(img_s, device=dev)
        delta_mu_phantom[img_s[0]//3:img_s[0]*2//3, img_s[1]//3:img_s[1]*2//3] = 0.01 # Absorption change
        delta_mu_phantom[5:10,5:10] = -0.005


        delta_y_sim = dot_op.op(delta_mu_phantom)
        print(f"Forward op output shape (delta_y): {delta_y_sim.shape}")
        assert delta_y_sim.shape == (n_meas,)

        adj_recon_mu = dot_op.op_adj(delta_y_sim)
        print(f"Adjoint op output shape (delta_mu_adj): {adj_recon_mu.shape}")
        assert adj_recon_mu.shape == img_s

        # Dot product test
        x_dp = torch.randn_like(delta_mu_phantom)
        y_dp_rand = torch.randn_like(delta_y_sim)
        Ax = dot_op.op(x_dp)
        Aty = dot_op.op_adj(y_dp_rand)
        lhs = torch.dot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.dot(x_dp.flatten(), Aty.flatten())
        print(f"DOT Dot product test: LHS={lhs.item():.6f}, RHS={rhs.item():.6f}")
        assert np.isclose(lhs.item(), rhs.item(), rtol=1e-3, atol=1e-5), "Dot product test failed."

        print("DOTOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in DOTOperator __main__ checks: {e}")
        import traceback; traceback.print_exc()
