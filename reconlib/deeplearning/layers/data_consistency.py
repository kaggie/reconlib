import torch
import torch.nn as nn
# Adjust path if NUFFTOperator is not directly importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Go up three levels for reconlib
from reconlib.operators import NUFFTOperator # Assuming NUFFTOperator is in reconlib.operators

class DataConsistencyMoDL(nn.Module):
    """
    MoDL Data Consistency block.
    Solves: (A^H A + lambda I) x = A^H y + lambda z
    where A is the NUFFTOperator, y is observed k-space, z is denoiser output.
    The solution is found using Conjugate Gradient.
    """
    def __init__(self, 
                 nufft_op: NUFFTOperator, 
                 lambda_dc: float | torch.Tensor, 
                 num_cg_iterations: int = 5):
        super().__init__()
        self.nufft_op = nufft_op
        if isinstance(lambda_dc, float):
            # If float, make it a non-trainable buffer.
            # If it needs to be learnable, it should be passed as a Parameter.
            self.lambda_dc = torch.tensor(lambda_dc, device=nufft_op.device, dtype=torch.float32) 
        elif isinstance(lambda_dc, torch.Tensor):
            if lambda_dc.numel() == 1:
                self.lambda_dc = lambda_dc.to(device=nufft_op.device, dtype=torch.float32)
            else:
                raise ValueError("lambda_dc tensor must be a scalar.")
        else:
            raise TypeError("lambda_dc must be float or torch.Tensor scalar.")

        self.num_cg_iterations = num_cg_iterations
        self.device = nufft_op.device
        self.image_shape = nufft_op.image_shape

    def _cg_solve(self, b: torch.Tensor, max_iter: int, tol: float = 1e-5) -> torch.Tensor:
        """
        Solves (A^H A + lambda I) x = b using Conjugate Gradient.
        A is self.nufft_op.
        lambda is self.lambda_dc.
        b is the right hand side.
        """
        x = torch.zeros_like(b) # Initial guess for image domain solution
        r = b - self.operator_AHA_plus_lambda_I(x) # Initial residual: b - (A^H A + lambda I)x
        p = r.clone()
        rs_old = torch.sum(torch.conj(r) * r).real

        for i in range(max_iter):
            Ap = self.operator_AHA_plus_lambda_I(p)
            alpha_num = rs_old
            alpha_den = torch.sum(torch.conj(p) * Ap).real
            
            if torch.abs(alpha_den) < 1e-12: # Avoid division by zero or very small denominator
                # print(f"CG iter {i+1}, Denominator too small, stopping.")
                break
            alpha = alpha_num / alpha_den
            
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(torch.conj(r) * r).real
            
            if torch.sqrt(rs_new) < tol:
                # print(f"CG iter {i+1}, Residual below tolerance {tol}, stopping.")
                break
            
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
            
            # if i == max_iter -1:
            #     print(f"CG warning: Max iterations ({max_iter}) reached. Residual: {torch.sqrt(rs_new):.2e}")

        return x

    def operator_AHA_plus_lambda_I(self, image_estimate: torch.Tensor) -> torch.Tensor:
        """ Computes (A^H A + lambda I)x """
        # A x
        kspace_estimate = self.nufft_op.op(image_estimate)
        # A^H (A x)
        aha_x = self.nufft_op.op_adj(kspace_estimate)
        # (A^H A + lambda I) x
        return aha_x + self.lambda_dc * image_estimate

    def forward(self, 
                current_image_estimate_zk: torch.Tensor, # Output of denoiser (z_k)
                observed_k_space_y: torch.Tensor         # Original undersampled k-space (y)
               ) -> torch.Tensor:                       # Returns x_{k+1}
        """
        current_image_estimate_zk: Current estimate from denoiser (z_k in MoDL paper). Shape: self.image_shape
        observed_k_space_y: Acquired k-space data. Shape: (num_k_points,)
        """
        if current_image_estimate_zk.shape != self.image_shape:
            raise ValueError(f"Shape of current_image_estimate_zk {current_image_estimate_zk.shape} must match {self.image_shape}")
        if observed_k_space_y.device != self.device:
            observed_k_space_y = observed_k_space_y.to(self.device)
        if current_image_estimate_zk.device != self.device:
            current_image_estimate_zk = current_image_estimate_zk.to(self.device)

        # Compute right hand side: b = A^H y + lambda * z_k
        Aty = self.nufft_op.op_adj(observed_k_space_y)
        rhs_b = Aty + self.lambda_dc * current_image_estimate_zk
        
        # Solve (A^H A + lambda I) x = b using CG
        # Initial guess for x can be current_image_estimate_zk or zeros
        # Using zeros as initial guess for CG solver for system Ax=b
        solved_image_xk_plus_1 = self._cg_solve(rhs_b, max_iter=self.num_cg_iterations)
        
        return solved_image_xk_plus_1

if __name__ == '__main__':
    # This is a placeholder for a proper test, which should be in tests/
    print("Running DataConsistencyMoDL basic example...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup a dummy NUFFTOperator (requires reconlib.nufft to be available)
    # For this standalone test, we might need to mock NUFFTOperator if reconlib.nufft is complex to setup
    # Or use a simplified version if possible.
    # Let's assume NUFFTOperator can be instantiated simply for shape purposes here.
    
    # Mocking NUFFTOperator for the sake of this example if reconlib.nufft is not easily accessible
    class MockNUFFTOperator:
        def __init__(self, image_shape, k_traj_len, device_):
            self.image_shape = image_shape
            self.k_trajectory = torch.zeros(k_traj_len, len(image_shape)) # Dummy
            self.device = device_
        def op(self, x): # A
            # Simulate forward: image -> k-space
            # For test, just return sum of pixels for each k-point (wrong but gives shape)
            return torch.ones(self.k_trajectory.shape[0], dtype=x.dtype, device=x.device) * torch.sum(x)
        def op_adj(self, y): # A_H
            # Simulate adjoint: k-space -> image
            # For test, just broadcast sum of k-space to image shape (wrong but gives shape)
            return torch.ones(self.image_shape, dtype=y.dtype, device=y.device) * torch.sum(y)

    img_s = 32
    k_pts = 100
    dims = 2
    if dims == 2:
        ishape = (img_s, img_s)
    else:
        ishape = (img_s//2, img_s, img_s)

    try:
        # Attempt to use the real NUFFTOperator if reconlib is structured for it
        from reconlib.operators import NUFFTOperator # Ensure this path is correct
        
        # Minimal params for NUFFTOperator
        k_traj_dummy = torch.rand(k_pts, dims, device=device) - 0.5
        oversamp_factor_dummy = tuple([2.0]*dims)
        kb_J_dummy = tuple([4]*dims)
        kb_alpha_dummy = tuple([2.34 * J for J in kb_J_dummy])
        Ld_dummy = tuple([2**8]*dims)

        nufft_op_mock = NUFFTOperator(
            image_shape=ishape,
            k_trajectory=k_traj_dummy,
            oversamp_factor=oversamp_factor_dummy,
            kb_J=kb_J_dummy,
            kb_alpha=kb_alpha_dummy,
            Ld=Ld_dummy,
            device=device
        )
        print("Using actual NUFFTOperator for test.")
    except Exception as e:
        print(f"Could not init actual NUFFTOperator ({e}), using MockNUFFTOperator for DataConsistencyMoDL example.")
        nufft_op_mock = MockNUFFTOperator(image_shape=ishape, k_traj_len=k_pts, device_=device)


    lambda_val = 0.05
    dc_layer = DataConsistencyMoDL(nufft_op=nufft_op_mock, lambda_dc=lambda_val, num_cg_iterations=3).to(device)

    # Dummy data
    z_k = torch.randn(ishape, dtype=torch.complex64, device=device)
    y_k_space = torch.randn(k_pts, dtype=torch.complex64, device=device)

    print(f"Input z_k shape: {z_k.shape}")
    print(f"Input y_k_space shape: {y_k_space.shape}")

    # Run forward pass
    x_k_plus_1 = dc_layer(z_k, y_k_space)
    print(f"Output x_k_plus_1 shape: {x_k_plus_1.shape}")
    assert x_k_plus_1.shape == ishape, "Output shape mismatch!"
    print("DataConsistencyMoDL basic example run completed.")
