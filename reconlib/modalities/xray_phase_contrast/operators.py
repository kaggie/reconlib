import torch
import numpy as np # For np.pi, if needed for k
from reconlib.operators import Operator
# Import gradient/divergence helpers if they were made generic,
# otherwise, implement simple versions here.
# For now, implementing simple finite differences directly.

class XRayPhaseContrastOperator(Operator):
    """
    Forward and Adjoint Operator for X-ray Phase-Contrast Imaging (XPCI).

    Models a simplified differential phase contrast scenario where the signal
    is proportional to the sum of the gradients of the phase image.
    Forward model: y = k_wave_number * (grad_x(x_phase) + grad_y(x_phase))
    Adjoint model: x_adj = -k_wave_number * divergence(y_gx, y_gy) assuming y was split.
                   If y is sum of grads, adj is k * (div_x(y) + div_y(y)) (approx).

    Args:
        image_shape (tuple[int, int]): Shape of the input 2D phase image (Height, Width).
        k_wave_number (float): X-ray wave number (2 * pi / wavelength).
        pixel_size_m (float or tuple[float,float], optional): Size of pixels in meters.
                                     If float, assumes square pixels.
                                     If tuple (ph, pw), specifies (pixel_height, pixel_width).
                                     Used for scaling gradients if true quantitative values needed,
                                     but for this operator, it's more about relative changes
                                     unless k_wave_number is precisely scaled.
                                     Defaults to 1.0 (unitless gradients).
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (H, W)
                 k_wave_number: float,
                 pixel_size_m: float | tuple[float,float] = 1.0,
                 device: str | torch.device = 'cpu'):
        super().__init__()
        if len(image_shape) != 2:
            raise ValueError("image_shape must be a 2-tuple (H, W).")
        self.image_shape = image_shape
        self.k_wave_number = k_wave_number

        if isinstance(pixel_size_m, (float, int)):
            self.pixel_h_m = float(pixel_size_m)
            self.pixel_w_m = float(pixel_size_m)
        elif isinstance(pixel_size_m, tuple) and len(pixel_size_m) == 2:
            self.pixel_h_m = float(pixel_size_m[0])
            self.pixel_w_m = float(pixel_size_m[1])
        else:
            raise ValueError("pixel_size_m must be a float or a 2-tuple (ph, pw).")

        self.device = torch.device(device)

    def _gradient_sum(self, x_phase_image: torch.Tensor) -> torch.Tensor:
        """ Calculates sum of gradients (Gx + Gy) using simple finite differences. """
        # Assumes x_phase_image is (H, W) or (B, C, H, W), grad over last 2 dims.
        # Forward differences with circular padding (implicit via roll for first element)

        # grad_h = x[h] - x[h-1] (rolled) = x - roll(x, shifts=1, dims=-2)
        # grad_w = x[w] - x[w-1] (rolled) = x - roll(x, shifts=1, dims=-1)

        # User pseudocode: np.diff(x, axis=1, prepend=x[:, -1:])
        # This is x_i - x_{i-1} with circular boundary.
        # For torch: x - torch.roll(x, shifts=1, dims=-1) for grad_w
        #            x - torch.roll(x, shifts=1, dims=-2) for grad_h

        grad_w = x_phase_image - torch.roll(x_phase_image, shifts=1, dims=-1) # diff along width
        grad_h = x_phase_image - torch.roll(x_phase_image, shifts=1, dims=-2) # diff along height

        # Scale by 1/pixel_size if true gradient is needed, but k can absorb this.
        # For simplicity, using unit pixel size in gradient calc here, k_wave_number handles overall scaling.
        # grad_w = grad_w / self.pixel_w_m
        # grad_h = grad_h / self.pixel_h_m

        return grad_w + grad_h

    def _divergence_sum_adj(self, y_grad_sum_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of sum of gradients. If G = Gx + Gy, then G_adj = Gx_adj + Gy_adj.
        If Gx(x) = x - roll(x,1,-1), then Gx_adj(y) = y - roll(y,-1,-1) = -div_x_forward_diff(y).
        So, this is like - (div_x(y) + div_y(y)) where div is forward difference.
        """
        # Adjoint of (x - roll(x,1,dim)) is (y - roll(y,-1,dim))
        div_w_component = y_grad_sum_data - torch.roll(y_grad_sum_data, shifts=-1, dims=-1)
        div_h_component = y_grad_sum_data - torch.roll(y_grad_sum_data, shifts=-1, dims=-2)

        # The pseudocode for adjoint_phase_contrast was:
        # x += np.diff(y, axis=1, append=y[:,:1])
        # x += np.diff(y, axis=0, append=y[:1,:])
        # x *= -params['k']
        # np.diff(y, append=y[:,:1]) is [y1-y0, y2-y1, ..., y0-yN-1] (forward diff with circular append for last element)
        # This is effectively y_i+1 - y_i (forward difference operator)
        # The adjoint of forward difference Gx(x_i) = x_{i+1} - x_i is Gx_adj(y_i) = y_{i-1} - y_i = -(y_i - y_{i-1}) = -div_backward(y_i)
        # The user's gradient was x_i - x_{i-1}. Its adjoint is y_i - y_{i+1}.

        return div_w_component + div_h_component


    def op(self, x_phase_image: torch.Tensor) -> torch.Tensor:
        """
        Forward XPCI operation: y = k * (grad_x(x) + grad_y(x)).
        x_phase_image: (H, W) phase image.
        Returns: (H, W) differential phase contrast image.
        """
        if x_phase_image.shape[-2:] != self.image_shape: # Allow batch/channel dims
            raise ValueError(f"Input x_phase_image spatial shape {x_phase_image.shape[-2:]} must match {self.image_shape}.")
        if x_phase_image.device != self.device:
            x_phase_image = x_phase_image.to(self.device)
        # Phase is usually real, but allow complex for operator generality
        if not torch.is_complex(x_phase_image) and x_phase_image.dtype != torch.float32 and x_phase_image.dtype != torch.float64:
             x_phase_image = x_phase_image.to(torch.float32) # Default to float32 if not complex/float

        sum_of_gradients = self._gradient_sum(x_phase_image)
        y_differential_phase = self.k_wave_number * sum_of_gradients

        return y_differential_phase

    def op_adj(self, y_differential_phase_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint XPCI operation.
        y_differential_phase_data: (H, W) differential phase contrast image.
        Returns: (H, W) reconstructed phase image (approximation).
        """
        if y_differential_phase_data.shape[-2:] != self.image_shape:
            raise ValueError(f"Input y_differential_phase_data spatial shape {y_differential_phase_data.shape[-2:]} must match {self.image_shape}.")
        if y_differential_phase_data.device != self.device:
            y_differential_phase_data = y_differential_phase_data.to(self.device)
        if not torch.is_complex(y_differential_phase_data) and y_differential_phase_data.dtype != torch.float32 and y_differential_phase_data.dtype != torch.float64:
             y_differential_phase_data = y_differential_phase_data.to(torch.float32)

        # Adjoint of y = k * Sum(Grad(x)) is x_adj = k * Sum(Div(y_sum_grad_comp))
        # where Div is adjoint of Grad.
        # If Grad_i(x) = x - roll(x,1,i), then Div_i(y) = y - roll(y,-1,i)
        # So this is -k * divergence_sum(y) based on the Div defined earlier
        # The pseudocode was x *= -params['k'] after summing two diffs.
        # If forward is k * (Gx + Gy)x, adjoint is k * (Gx^T + Gy^T)y.
        # If Gx(x) = x - roll(x,1,-1), then Gx^T(y) = y - roll(y,-1,-1).

        sum_of_adjoint_gradients = self._divergence_sum_adj(y_differential_phase_data)
        x_phase_adj = self.k_wave_number * sum_of_adjoint_gradients # k is real, so k*adj(Op) = adj(k*Op)

        return x_phase_adj

if __name__ == '__main__':
    print("Running basic XRayPhaseContrastOperator checks...")
    device = torch.device('cpu')
    img_shape_xrpc = (32, 32) # H, W

    # Example parameters from user pseudocode
    k_val = 2 * np.pi / 1e-10 # X-ray wave number (for 10 keV)
    dx_val = 1e-6             # Pixel size (1 micron)

    try:
        xrpc_op_test = XRayPhaseContrastOperator(
            image_shape=img_shape_xrpc,
            k_wave_number=k_val,
            pixel_size_m=dx_val,
            device=device
        )
        print("XRayPhaseContrastOperator instantiated.")

        phantom_phase = torch.randn(img_shape_xrpc, dtype=torch.float32, device=device)
        # Add a ramp to ensure gradients are non-zero
        ramp_h = torch.linspace(0,1,img_shape_xrpc[0], device=device).unsqueeze(1).repeat(1,img_shape_xrpc[1])
        ramp_w = torch.linspace(0,1,img_shape_xrpc[1], device=device).unsqueeze(0).repeat(img_shape_xrpc[0],1)
        phantom_phase += ramp_h + ramp_w

        diff_phase_data = xrpc_op_test.op(phantom_phase)
        print(f"Forward op output shape: {diff_phase_data.shape}")
        assert diff_phase_data.shape == img_shape_xrpc

        recon_phase_adj = xrpc_op_test.op_adj(diff_phase_data)
        print(f"Adjoint op output shape: {recon_phase_adj.shape}")
        assert recon_phase_adj.shape == img_shape_xrpc

        # Basic dot product test
        x_dp_xrpc = torch.randn_like(phantom_phase)
        y_dp_rand_xrpc = torch.randn_like(diff_phase_data)

        Ax_xrpc = xrpc_op_test.op(x_dp_xrpc)
        Aty_xrpc = xrpc_op_test.op_adj(y_dp_rand_xrpc)

        # Ensure complex for vdot if inputs were real
        Ax_xrpc_c = Ax_xrpc.to(torch.complex64) if not Ax_xrpc.is_complex() else Ax_xrpc
        y_dp_rand_xrpc_c = y_dp_rand_xrpc.to(torch.complex64) if not y_dp_rand_xrpc.is_complex() else y_dp_rand_xrpc
        x_dp_xrpc_c = x_dp_xrpc.to(torch.complex64) if not x_dp_xrpc.is_complex() else x_dp_xrpc
        Aty_xrpc_c = Aty_xrpc.to(torch.complex64) if not Aty_xrpc.is_complex() else Aty_xrpc

        lhs_xrpc = torch.vdot(Ax_xrpc_c.flatten(), y_dp_rand_xrpc_c.flatten())
        rhs_xrpc = torch.vdot(x_dp_xrpc_c.flatten(), Aty_xrpc_c.flatten())

        print(f"XRPC Dot product test: LHS={lhs_xrpc.item():.4e}, RHS={rhs_xrpc.item():.4e}")
        # Finite difference operators should satisfy this well.
        if not np.isclose(lhs_xrpc.real.item(), rhs_xrpc.real.item(), rtol=1e-4, atol=1e-5) or \
           not np.isclose(lhs_xrpc.imag.item(), rhs_xrpc.imag.item(), rtol=1e-4, atol=1e-5):
           print(f"Warning: XRPC Dot product components differ more than expected. LHS: {lhs_xrpc}, RHS: {rhs_xrpc}")

        print("XRayPhaseContrastOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in XRayPhaseContrastOperator __main__ checks: {e}")
