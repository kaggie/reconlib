# This file will contain utility functions specific to OCT.
# For example, dispersion compensation, k-space resampling, etc.

import torch
import numpy as np # Often used for OCT processing steps

def compensate_dispersion_placeholder(
    oct_data: torch.Tensor,
    dispersion_coeffs: list[float] | None = None
) -> torch.Tensor:
    """
    Placeholder for a dispersion compensation routine.
    Actual implementation would involve phase correction in k-space.

    Args:
        oct_data (torch.Tensor): Raw OCT data (interferogram), typically complex.
                                 Assumed shape (..., num_depth_samples).
        dispersion_coeffs (list[float], optional): Coefficients for dispersion
                                                   (e.g., a0, a1, a2 for polynomial phase).
                                                   Defaults to None (no compensation).
    Returns:
        torch.Tensor: Dispersion compensated OCT data.
    """
    if dispersion_coeffs is None:
        print("Dispersion compensation skipped (no coefficients provided).")
        return oct_data

    print(f"Applying dispersion compensation (placeholder) with coeffs: {dispersion_coeffs}")

    # Ensure data is complex
    if not oct_data.is_complex():
        oct_data_complex = oct_data.to(torch.complex64)
    else:
        oct_data_complex = oct_data

    # 1. Go to k-space (frequency domain for depth)
    k_space_data = torch.fft.fft(oct_data_complex, dim=-1)

    # 2. Create phase correction term
    num_depth_samples = oct_data_complex.shape[-1]
    # Normalized k-axis (frequency axis for depth) from -0.5 to 0.5
    k_norm = torch.fft.fftfreq(num_depth_samples, device=oct_data.device, dtype=oct_data.dtype)
                                # if oct_data is real, then dtype will be real. ensure complex for phase.
    if k_norm.is_complex(): k_norm = k_norm.real # fftfreq can return complex if input N is complex, ensure real for phase calc

    phase_correction = torch.zeros_like(k_norm)
    # Example polynomial phase: phi(k) = c0 + c1*k + c2*k^2 + ...
    for i, coeff in enumerate(dispersion_coeffs):
        phase_correction += coeff * (k_norm ** i) # i=0 for c0 (const phase), i=1 for c1*k (linear phase/delay), etc.

    # Apply phase correction (exp(1j * phi(k)))
    # Reshape phase_correction to be broadcastable with k_space_data if needed
    # Current k_space_data shape (..., num_depth_samples), phase_correction shape (num_depth_samples)
    # Broadcasting should work fine.
    k_space_compensated = k_space_data * torch.exp(1j * phase_correction)

    # 3. Transform back to interferogram space
    compensated_data = torch.fft.ifft(k_space_compensated, dim=-1)

    return compensated_data


if __name__ == '__main__':
    # Example usage/test
    num_a_scans = 10
    depth_samples = 512
    dummy_oct_interferogram = torch.randn(num_a_scans, depth_samples, dtype=torch.complex64) + \
                              1j * torch.randn(num_a_scans, depth_samples, dtype=torch.complex64)

    print(f"Dummy OCT interferogram shape: {dummy_oct_interferogram.shape}")

    # Example dispersion coefficients (e.g., for quadratic and cubic phase terms)
    # These would typically be determined experimentally or through calibration.
    # Coeffs for k^0 (const), k^1 (linear), k^2 (GDD), k^3 (TOD)
    coeffs = [0.0, 0.0, 5e-4, 1e-5]  # Example quadratic and cubic dispersion

    compensated_signal = compensate_dispersion_placeholder(dummy_oct_interferogram, coeffs)
    print(f"Dispersion compensated signal shape: {compensated_signal.shape}")
    assert compensated_signal.shape == dummy_oct_interferogram.shape, "Shape mismatch after dispersion compensation."

    compensated_signal_no_coeffs = compensate_dispersion_placeholder(dummy_oct_interferogram, None)
    assert torch.allclose(compensated_signal_no_coeffs, dummy_oct_interferogram), "Data changed with no coeffs."

    print("OCT utils placeholder execution finished.")
