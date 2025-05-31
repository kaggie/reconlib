from .base import EPRImaging

class ContinuousWaveEPR(EPRImaging):
  """
  Class for Continuous Wave (CW) EPR imaging techniques.

  CW-EPR data is typically acquired by sweeping an external magnetic field
  while applying a continuous, fixed-frequency microwave field to the sample.
  The absorption of microwave energy by paramagnetic species is detected,
  often using a lock-in amplifier synchronized with a small modulation of the
  magnetic field, yielding a derivative spectrum.

  Key Parameters:
  - Microwave Frequency: The constant frequency of the microwave irradiation.
  - Magnetic Field Sweep Range: The start and end points of the magnetic field sweep.
  - Sweep Rate: Speed at which the magnetic field is swept.
  - Modulation Amplitude: The peak-to-peak amplitude of the magnetic field modulation.
  - Modulation Frequency: The frequency of the magnetic field modulation.
  - Microwave Power: The power level of the microwave irradiation.
  - Time Constant: The time constant of the lock-in amplifier or detection system.

  Common Artifacts or Distortions:
  - Baseline Drift: Slow variations in the signal baseline.
  - Noise: Random fluctuations from various sources (thermal, detector, etc.).
  - Saturation Effects: Distortion of the signal shape and intensity at high
    microwave powers, where the spin system cannot relax quickly enough.
  - Modulation Broadening: Broadening of spectral lines if the modulation
    amplitude is too large compared to the intrinsic linewidth.
  - Passage Effects: Distortions that can occur if the sweep rate is too fast
    relative to the relaxation times of the spin system.

  Reconstruction Considerations:
  The fundamental physics of CW-EPR dictates the reconstruction approach. The
  resonance condition (h * nu = g * mu_B * B, where nu is microwave frequency,
  B is magnetic field, g is the g-factor, mu_B is Bohr magneton) links the
  swept magnetic field to the g-factor of the resonant spins. The intensity of
  absorption reflects the concentration of these spins.
  - Spectral-Spatial Imaging: If no field gradients are used, the spectrum itself
    can provide spatial information if different parts of the sample have distinct
    spectral signatures (e.g., different g-factors or linewidths) or if the sample
    is moved/rotated. Reconstruction may involve deconvolving instrumental
    broadening (e.g., from modulation) and then mapping spectral features to
    spatial distributions.
  - Gradient-Based Imaging: If magnetic field gradients are applied, the resonance
    condition becomes position-dependent (B = B0 + G*r). The EPR spectrum then
    represents a projection of the spin distribution along the gradient direction.
    Reconstruction in this case uses techniques similar to X-ray Computed
    Tomography (CT), such as filtered backprojection, applied to a series of
    projections obtained at different gradient orientations.
  """

  def __init__(self, metadata: dict, data: dict, sweep_parameters: dict):
    """Initializes the ContinuousWaveEPR class.

    Args:
      metadata: A dictionary containing metadata for the EPR experiment.
                This might include microwave frequency, power, temperature, etc.
      data: A dictionary containing the EPR data. This typically includes the
            detected signal (e.g., absorption derivative) as a function of
            the magnetic field.
      sweep_parameters: A dictionary containing parameters specific to the
                        magnetic field sweep. Examples:
                        - 'center_field_mT': The center of the magnetic field sweep in mT.
                        - 'sweep_width_mT': The range of the magnetic field sweep in mT.
                        - 'n_points': Number of data points acquired during the sweep.
                        - 'modulation_amplitude_mT': Peak-to-peak modulation amplitude.
                        - 'modulation_frequency_kHz': Modulation frequency.
    """
    super().__init__(metadata, data)
    self.sweep_parameters = sweep_parameters

  def get_physics_model(self) -> dict:
    """
    Returns a dictionary describing the physics model for CW EPR.

    This model includes key parameters, acquisition details, and common
    artifacts relevant to the technique.
    """
    return {
        "technique": "Continuous Wave EPR",
        "description": "Detects microwave absorption during a magnetic field sweep "
                       "with continuous microwave irradiation, often using field "
                       "modulation and lock-in detection.",
        "acquisition_mode": "Magnetic Field Sweep (most common)",
        "alternative_acquisition": "Frequency Sweep (less common)",
        "key_parameters": [
            "microwave_frequency_GHz",       # e.g., 9.5 (X-band), 35 (Q-band)
            "magnetic_field_center_mT",
            "magnetic_field_sweep_width_mT",
            "sweep_rate_mT_per_s",
            "modulation_amplitude_mT",       # Peak-to-peak
            "modulation_frequency_kHz",      # e.g., 100 kHz
            "microwave_power_mW",
            "time_constant_ms",              # Lock-in amplifier time constant
            "temperature_K"
        ],
        "signal_type": "Typically derivative of absorption spectrum",
        "common_artifacts": [
            "baseline_drift",
            "noise (1/f, thermal, detector)",
            "saturation_effects (power broadening)",
            "modulation_broadening",
            "passage_effects (rapid sweep distortions)",
            "microphonics (vibrational noise)"
        ],
        "imaging_method": "Can be spectral-spatial or gradient-based (projection imaging)"
    }

  def reconstruct(self, *args, **kwargs) -> str:
    """
    Performs image reconstruction from CW-EPR data.

    The specific algorithm depends on the data acquisition method:
    - For spectral-spatial CW-EPR (no field gradients), this might involve
      deconvolution and analysis of spectral features.
    - For gradient-based CW-EPR, this typically involves projection
      reconstruction algorithms (e.g., filtered backprojection).

    Args:
      *args: Additional arguments depending on the reconstruction type.
      **kwargs: Additional keyword arguments.

    Returns:
      A string indicating that the reconstruction is not yet implemented or
      the result of the reconstruction (e.g., an image).
    """
    # Placeholder for actual reconstruction logic
    return "Reconstruction for Continuous Wave EPR is not yet implemented (details depend on acquisition)."
