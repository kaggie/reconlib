# This file will contain utility functions specific to Seismic Imaging.
# For example, Ricker wavelet generation, travel time calculation, velocity model manipulation.

import torch
import numpy as np

def ricker_wavelet(
    duration_s: float, # Total duration of the wavelet in seconds
    dt_s: float,       # Time sampling interval in seconds
    peak_freq_hz: float, # Peak frequency of the wavelet in Hz
    delay_s: float = 0.0 # Time delay for the peak (relative to start of wavelet vector)
                         # If 0, peak might be cut off if duration is short.
                         # Often set to duration_s / 2 or 1/peak_freq_hz for symmetry.
) -> torch.Tensor:
    """
    Generates a Ricker wavelet (Mexican hat wavelet).

    Args:
        duration_s (float): Total time duration of the wavelet array.
        dt_s (float): Time sampling interval (1 / sampling_rate).
        peak_freq_hz (float): Peak frequency of the wavelet.
        delay_s (float, optional): Time shift for the wavelet peak.
                                   If not set, defaults to center of duration if possible.
                                   A common choice is 1.0 / peak_freq_hz or duration_s / 2.0.
                                   Defaults to 0.0 (peak at start, likely not ideal).

    Returns:
        torch.Tensor: A 1D tensor representing the Ricker wavelet.
    """
    if peak_freq_hz <= 0:
        raise ValueError("Peak frequency must be positive.")
    if dt_s <= 0:
        raise ValueError("Time sampling interval dt_s must be positive.")
    if duration_s <= 0:
        raise ValueError("Duration must be positive.")

    num_samples = int(round(duration_s / dt_s))
    if num_samples < 1:
        raise ValueError("Resulting number of samples is less than 1. Check duration and dt_s.")

    t = torch.arange(num_samples, dtype=torch.float32) * dt_s - delay_s

    # Ricker wavelet formula: (1 - 2 * (pi * f_peak * t)^2) * exp(-(pi * f_peak * t)^2)
    pi_fp_t = torch.pi * peak_freq_hz * t
    pi_fp_t_sq = pi_fp_t**2

    wavelet = (1.0 - 2.0 * pi_fp_t_sq) * torch.exp(-pi_fp_t_sq)

    return wavelet


if __name__ == '__main__':
    print("Running basic execution checks for Seismic utils...")
    device = torch.device('cpu') # Ricker wavelet is typically small, CPU is fine.

    duration = 0.1 # seconds
    dt = 0.001   # 1 ms sampling
    peak_freq = 25 # Hz

    # Test with delay to center the wavelet
    delay1 = duration / 2.0
    try:
        wavelet1 = ricker_wavelet(duration, dt, peak_freq, delay_s=delay1)
        assert wavelet1.ndim == 1
        assert wavelet1.shape[0] == int(round(duration/dt))
        # Check if peak is roughly at center (due to discrete sampling)
        # Max value should be close to t=delay1 (index = delay1/dt)
        # For Ricker, peak is at t=0 relative to its defining formula, so t-delay1 = 0
        # print(f"Wavelet 1 (centered): peak at index {torch.argmax(torch.abs(wavelet1)).item()} vs expected {int(delay1/dt)}")
        print(f"Generated Ricker wavelet 1 (centered) of shape {wavelet1.shape}.")
        print("ricker_wavelet (centered) basic execution check PASSED.")
    except Exception as e:
        print(f"Error during ricker_wavelet (centered) test: {e}")

    # Test with a common delay related to peak frequency
    delay2 = 1.2 / peak_freq # Common factor to ensure most of wavelet is captured
    duration2 = delay2 * 2 # Ensure duration covers the wavelet
    try:
        wavelet2 = ricker_wavelet(duration2, dt, peak_freq, delay_s=delay2)
        assert wavelet2.ndim == 1
        # print(f"Wavelet 2 (peak_freq delay): peak at index {torch.argmax(torch.abs(wavelet2)).item()} vs expected {int(delay2/dt)}")
        print(f"Generated Ricker wavelet 2 (peak_freq delay) of shape {wavelet2.shape}.")
        print("ricker_wavelet (peak_freq delay) basic execution check PASSED.")
    except Exception as e:
        print(f"Error during ricker_wavelet (peak_freq delay) test: {e}")

    print("Seismic utils placeholder execution finished.")
