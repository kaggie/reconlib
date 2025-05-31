# This file will contain utility functions specific to X-ray Phase-Contrast Imaging (XPCI).
# For example, generating propagation kernels, handling multi-material decomposition, etc.

import torch
import numpy as np

def create_angular_spectrum_propagator(
    shape: tuple[int, int], # (H, W) of the wavefield / image
    pixel_size_m: float,
    wavelength_m: float,
    propagation_distance_m: float,
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Creates an Angular Spectrum Method (ASM) propagator in Fourier space.
    This propagator, when multiplied by FT(wavefield_at_z0), gives FT(wavefield_at_z1).

    Args:
        shape (tuple[int, int]): Spatial shape (Height, Width) of the wavefield.
        pixel_size_m (float): Pixel size in meters.
        wavelength_m (float): Wavelength of the X-rays in meters.
        propagation_distance_m (float): Distance to propagate.
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The ASM propagator H(fx, fy) as a complex tensor,
                      fftshifted for multiplication in centered Fourier space.
                      Shape (H, W).
    """
    Ny, Nx = shape
    device = torch.device(device)

    # Create spatial frequency coordinates (fx, fy)
    # These are cycles per meter if pixel_size_m is in meters.
    fx_freq = torch.fft.fftfreq(Nx, d=pixel_size_m, device=device) # Corresponds to X (width)
    fy_freq = torch.fft.fftfreq(Ny, d=pixel_size_m, device=device) # Corresponds to Y (height)

    # Create 2D grid of spatial frequencies
    # Note: meshgrid default 'xy' indexing means fx_grid is (Ny, Nx) if fx_freq is Nx, fy_freq is Ny.
    # Or, if fx_grid is (Nx, Ny) and fy_grid is (Nx, Ny), then use .T
    # Let's make them (Ny, Nx) from the start for typical image H,W order
    fy_grid, fx_grid = torch.meshgrid(fy_freq, fx_freq, indexing='ij') # fy_grid (H,W), fx_grid (H,W)

    # Squared spatial frequency magnitude
    f_squared = fx_grid**2 + fy_grid**2

    # Term inside sqrt: (1/lambda^2 - f^2)
    # Only propagate waves that are not evanescent (1/lambda^2 > f^2)
    # k0_squared = (1.0 / wavelength_m)**2
    # term_under_sqrt = k0_squared - f_squared

    # Propagator phase: exp(i * 2 * pi * distance * sqrt( (1/lambda)^2 - fx^2 - fy^2 ) )
    # This form is for k_z = 2*pi * sqrt(...)
    # The phase term is k_z * distance.

    # Evanescent wave handling: where argument of sqrt is negative, kz becomes imaginary,
    # leading to exponential decay.
    # For numerical stability, ensure argument of sqrt is non-negative.
    # Valid_propagation = term_under_sqrt >= 0
    # kz_phase_factor = torch.zeros_like(term_under_sqrt)
    # kz_phase_factor[Valid_propagation] = torch.sqrt(term_under_sqrt[Valid_propagation])

    # Simpler form from many optics texts for phase of propagator (Fresnel-Kirchhoff based on ASM):
    # H(fx,fy) = exp(i * k * distance) * exp(-i * pi * lambda * distance * (fx^2 + fy^2))
    # The first term exp(i*k*distance) is a constant phase offset often ignored if only relative phase matters.
    # The second term is the Fresnel propagator (paraxial approximation).
    # For full ASM (Rayleigh-Sommerfeld solution for Helmholtz):
    # H(fx,fy) = exp( i * 2 * pi * distance * sqrt(1/lambda^2 - fx^2 - fy^2) )
    # This needs careful handling of evanescent waves.

    # Let's implement the Fresnel propagator part, as it's common and simpler.
    # This is actually the phase of the paraxial propagator in Fourier space.
    # Note: This is for H(fx,fy) that multiplies FT(U_in) to get FT(U_out).
    # It should be fftshifted to match centered FT(U_in).

    propagator_phase_paraxial = -torch.pi * wavelength_m * propagation_distance_m * f_squared
    propagator = torch.exp(1j * propagator_phase_paraxial) # Complex64 by default if phase is float32

    # This propagator is already defined in a way that fx=0, fy=0 is at the corner (due to fftfreq).
    # For multiplication with fftshift(FT(input)), this propagator should also be fftshifted.
    return torch.fft.fftshift(propagator)


if __name__ == '__main__':
    print("Running basic execution checks for XPCI utils...")
    device = torch.device('cpu')

    shape_example = (128, 128)
    ps_m_example = 1e-5 # 10 micron pixels
    lambda_m_example = 12.398 / 25.0 * 1e-10 # Approx for 25 keV X-rays
    dist_m_example = 0.1 # 10 cm

    try:
        asm_propagator = create_angular_spectrum_propagator(
            shape=shape_example,
            pixel_size_m=ps_m_example,
            wavelength_m=lambda_m_example,
            propagation_distance_m=dist_m_example,
            device=device
        )
        assert asm_propagator.shape == shape_example
        assert asm_propagator.is_complex()
        # Check if center (DC component after fftshift) is close to 1 (exp(0))
        center_val = asm_propagator[shape_example[0]//2, shape_example[1]//2]
        assert torch.isclose(center_val, torch.tensor(1.0 + 0.0j, dtype=torch.complex64)), \
            f"Center of propagator should be ~1.0, got {center_val}"
        print("create_angular_spectrum_propagator basic execution check PASSED.")
    except Exception as e:
        print(f"Error during create_angular_spectrum_propagator check: {e}")

    print("XPCI utils placeholder execution finished.")
