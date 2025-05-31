import torch
from reconlib.operators import Operator
import numpy as np
from reconlib.modalities.fluorescence_microscopy.operators import generate_gaussian_psf # Re-use for PSF

class SIMOperator(Operator):
    """
    Placeholder Forward and Adjoint Operator for Structured Illumination Microscopy (SIM).

    SIM works by illuminating the sample with patterned light (e.g., stripes),
    acquiring multiple images with different pattern phases/orientations.
    This effectively encodes high-frequency information into observable moiré fringes.
    Reconstruction decodes this to achieve super-resolution.

    A simplified forward model:
    1. True high-res image X_hr.
    2. For each illumination pattern P_i:
       - Modulated image: M_i = X_hr * P_i
       - Observed raw image: Y_i = PSF_det * M_i (convolution with detection PSF)
    The operator outputs a stack of Y_i.

    The adjoint is complex. This placeholder will be very basic.
    """
    def __init__(self,
                 hr_image_shape: tuple[int, int], # (Ny_hr, Nx_hr) - high-res ground truth
                 num_patterns: int, # Total number of raw SIM images (e.g., 3 phases * 3 angles = 9)
                 psf_detection: torch.Tensor, # Detection PSF (low-pass effect)
                 patterns: torch.Tensor | None = None, # Pre-generated patterns (num_patterns, Ny_hr, Nx_hr)
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.hr_image_shape = hr_image_shape
        self.num_patterns = num_patterns
        self.psf_detection = psf_detection.to(torch.device(device))
        self.device = torch.device(device)

        if self.psf_detection.ndim != 2:
            raise ValueError("Detection PSF must be 2D for this basic SIM operator.")

        # Padding for 'same' convolution with detection PSF
        self.psf_padding = [(s - 1) // 2 for s in self.psf_detection.shape]
        # PyTorch conv2d padding is (pad_H, pad_W)
        # self.psf_padding.reverse() # Not needed if (H,W) order is kept

        if patterns is not None:
            if patterns.shape != (num_patterns, *hr_image_shape):
                raise ValueError(f"Provided patterns shape {patterns.shape} is not ({num_patterns}, {hr_image_shape[0]}, {hr_image_shape[1]}).")
            self.patterns = patterns.to(self.device)
        else:
            print("Warning: No patterns provided to SIMOperator. Using dummy constant patterns.")
            self.patterns = torch.ones((num_patterns, *hr_image_shape), device=self.device) / num_patterns
            # In a real scenario, these would be e.g. sin(k*x + phase) for different k, phase

        # Output shape: stack of raw SIM images
        self.measurement_shape = (num_patterns, *hr_image_shape) # Each raw image has same shape as HR for simplicity here

        print(f"SIMOperator (Placeholder) initialized.")
        print(f"  HR Image Shape: {self.hr_image_shape}, Num Patterns: {self.num_patterns}")
        print(f"  Detection PSF shape: {self.psf_detection.shape}")

    def op(self, hr_image: torch.Tensor) -> torch.Tensor:
        """
        Forward: High-resolution image to a stack of raw SIM images.
        """
        if hr_image.shape != self.hr_image_shape:
            raise ValueError(f"Input hr_image shape {hr_image.shape} must match {self.hr_image_shape}.")
        hr_image = hr_image.to(self.device)

        raw_sim_images = torch.zeros(self.measurement_shape, device=self.device)

        # Unsqueeze for conv2d
        psf_exp = self.psf_detection.unsqueeze(0).unsqueeze(0) # (1,1,kH,kW)

        for i in range(self.num_patterns):
            pattern_i = self.patterns[i, ...] # (Ny_hr, Nx_hr)
            modulated_image = hr_image * pattern_i # Element-wise multiplication

            # Apply detection PSF (blurring)
            # Input for conv2d: (batch, channels, H, W)
            modulated_image_exp = modulated_image.unsqueeze(0).unsqueeze(0) # (1,1,H,W)

            blurred_modulated_image = torch.nn.functional.conv2d(
                modulated_image_exp, psf_exp, padding=self.psf_padding
            )
            raw_sim_images[i, ...] = blurred_modulated_image.squeeze()

        return raw_sim_images

    def op_adj(self, raw_sim_images_stack: torch.Tensor) -> torch.Tensor:
        """
        Adjoint: Stack of raw SIM images to a high-resolution image estimate.
        Placeholder: Correlate each raw image with its pattern and sum up, after adjoint PSF.
        """
        if raw_sim_images_stack.shape != self.measurement_shape:
            raise ValueError(f"Input stack shape {raw_sim_images_stack.shape} must match {self.measurement_shape}.")
        raw_sim_images_stack = raw_sim_images_stack.to(self.device)

        hr_image_estimate = torch.zeros(self.hr_image_shape, device=self.device)

        # Flipped PSF for adjoint convolution
        psf_flipped_exp = torch.flip(self.psf_detection, dims=[0,1]).unsqueeze(0).unsqueeze(0)

        for i in range(self.num_patterns):
            raw_image_i_exp = raw_sim_images_stack[i, ...].unsqueeze(0).unsqueeze(0) # (1,1,H,W)

            # Adjoint of PSF convolution
            adj_conv_raw_image = torch.nn.functional.conv2d(
                raw_image_i_exp, psf_flipped_exp, padding=self.psf_padding
            ).squeeze() # (H,W)

            # Adjoint of pattern multiplication (element-wise with same pattern if real)
            pattern_i = self.patterns[i, ...]
            hr_image_estimate += adj_conv_raw_image * pattern_i # Element-wise

        return hr_image_estimate / self.num_patterns # Average contribution

if __name__ == '__main__':
    print("\nRunning basic SIMOperator (Placeholder) checks...")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hr_shape = (64, 64)
    n_pats = 9
    psf_det_sim = generate_gaussian_psf(shape=(7,7), sigma=2.0, device=dev) # Wider PSF for noticeable blur

    # Example patterns (simple stripes for demo)
    example_patterns = torch.zeros((n_pats, *hr_shape), device=dev)
    x_coords = torch.linspace(-np.pi, np.pi, hr_shape[1], device=dev)
    for i in range(n_pats):
        angle = (i // 3) * (np.pi / 3) # 3 angles
        phase = (i % 3) * (2 * np.pi / 3) # 3 phases
        kx = torch.cos(angle) * 5 # Example spatial frequency
        ky = torch.sin(angle) * 5
        xx, yy = torch.meshgrid(torch.arange(hr_shape[0],device=dev), torch.arange(hr_shape[1],device=dev), indexing='ij')
        example_patterns[i,...] = (torch.cos( (xx*ky + yy*kx) * (2*np.pi/hr_shape[0]) + phase) + 1)/2 # Positive patterns

    try:
        sim_op = SIMOperator(hr_image_shape=hr_shape, num_patterns=n_pats,
                             psf_detection=psf_det_sim, patterns=example_patterns, device=dev)
        print("SIMOperator instantiated.")

        phantom_hr = torch.zeros(hr_shape, device=dev)
        phantom_hr[hr_shape[0]//4:hr_shape[0]*3//4, hr_shape[1]//4:hr_shape[1]*3//4] = 1.0
        phantom_hr[10:20,10:20]=2.0


        raw_images = sim_op.op(phantom_hr)
        print(f"Forward op output shape: {raw_images.shape}")
        assert raw_images.shape == (n_pats, *hr_shape)

        adj_recon = sim_op.op_adj(raw_images)
        print(f"Adjoint op output shape: {adj_recon.shape}")
        assert adj_recon.shape == hr_shape

        # Dot product test
        x_dp = torch.randn_like(phantom_hr)
        y_dp_rand = torch.randn_like(raw_images)
        Ax = sim_op.op(x_dp)
        Aty = sim_op.op_adj(y_dp_rand)
        lhs = torch.dot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.dot(x_dp.flatten(), Aty.flatten())
        print(f"SIM Dot product test: LHS={lhs.item():.6f}, RHS={rhs.item():.6f}")
        assert np.isclose(lhs.item(), rhs.item(), rtol=1e-3), "Dot product test failed."

        print("SIMOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in SIMOperator __main__ checks: {e}")
        import traceback; traceback.print_exc()
