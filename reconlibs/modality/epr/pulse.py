from .base import EPRImaging

class PulseEPR(EPRImaging):
  """
  Class for Pulse EPR imaging techniques.

  Pulse EPR methods involve applying one or more short, intense microwave pulses
  to a sample and detecting the transient response (e.g., free induction decay (FID)
  or spin echoes). These techniques offer advantages in sensitivity, resolution,
  and the ability to probe specific spin interactions or relaxation pathways.

  Common Pulse Sequences:
  - Hahn Echo (90 - tau - 180 - tau - echo): Basic sequence for measuring T2.
  - Inversion Recovery (180 - tau - 90 - FID): Basic sequence for measuring T1.
  - Two-Pulse ESEEM (Electron Spin Echo Envelope Modulation): Detects hyperfine
    interactions with nearby nuclei.
  - ELDOR (Electron-Electron Double Resonance): Measures interactions between
    different electron spins, often by observing one spin while pulsing another.
  - DEER/PELDOR (Double Electron-Electron Resonance / Pulsed ELDOR): Measures
    distances between spin labels by observing dipolar interactions.

  Spatial Encoding (Imaging):
  In pulse EPR imaging, spatial information is typically encoded by applying
  magnetic field gradients. Similar to MRI, these gradients impart a
  position-dependent Larmor frequency to the spins.
  - For direct imaging, gradients can be applied during signal acquisition (readout gradient)
    or between pulses (phase encoding gradients) to sample k-space.
  - The detected signal (e.g., echo amplitude or FID) is then Fourier transformed
    to reconstruct the image.

  Critical Factors and Challenges:
  - Dead Time: The period after a pulse during which the detector is saturated,
    preventing signal acquisition. This can lead to loss of fast-decaying signals.
  - Pulse Imperfections:
    - Finite Pulse Width: Real pulses have finite duration, leading to imperfect
      excitation profiles across the spectrum.
    - B1 (Microwave Field) Inhomogeneity: Spatial variations in the microwave
      field strength can lead to non-uniform spin excitation and signal loss.
  - Relaxation Times (T1, T2, Tm):
    - T1 (Spin-Lattice Relaxation): Affects signal recovery between shots and overall
      sensitivity.
    - T2 (Spin-Spin Relaxation) / Tm (Phase Memory Time): Determines the decay rate
      of transient signals and limits the duration of pulse sequences and echo
      acquisition windows. Short T2/Tm values are a major challenge in EPR imaging.
  - Bandwidth Limitations: The spectrometer and resonator bandwidth can limit the
    excitation/detection range and the ability to rapidly switch gradients.

  Reconstruction Considerations:
  Pulse EPR image reconstruction heavily relies on k-space formalism, analogous
  to MRI. The strength (G), duration (t_grad), and timing of magnetic field
  gradients within the pulse sequence determine the trajectory covered in k-space.
  The k-space coordinate is proportional to the integral of G(t)dt.
  - k-Space Trajectories: Common trajectories include Cartesian (stepped gradients),
    radial (rotating gradients), and spiral. The choice impacts acquisition time,
    motion sensitivity, and artifact behavior.
  - Common Corrections:
    - Dead-time effects: Extrapolation or modeling of the initial part of the FID/echo.
    - Echo distortions: Phase correction, echo centering.
    - Field Inhomogeneities (B0 and B1): Can cause image distortion or signal loss;
      may require field mapping and correction.
    - T2/T2* decay: Signal decay during acquisition can blur images or reduce
      resolution. Compensation techniques (e.g., weighting k-space data) may be applied.
    - Gradient non-linearity and eddy currents: May require pre-calibration and correction.
  - Reconstruction Algorithms:
    - Gridding and FFT: For non-Cartesian k-space data (e.g., radial, spiral),
      data is first interpolated onto a Cartesian grid (gridding) followed by
      a Fast Fourier Transform (FFT). Density compensation is crucial here.
    - Filtered Backprojection (FBP): Can be used for specific trajectories like
      radial acquisitions, similar to CT.
    - Iterative Methods: Can incorporate complex physical models and priors,
      useful for undersampled data or strong artifacts, but computationally intensive.
  """

  def __init__(self, metadata: dict, data: dict, pulse_sequence_details: dict):
    """Initializes the PulseEPR class.

    Args:
      metadata: A dictionary containing metadata for the EPR experiment.
                This might include microwave frequency, temperature, static
                magnetic field strength, gradient strengths, etc.
      data: A dictionary containing the EPR data. This could be raw time-domain
            signals (FIDs, echoes), k-space data, or partially processed data.
            The structure of 'data' should be well-defined, e.g.,
            {'kspace_data': k_data_array, 'gradient_waveforms': grad_forms}.
      pulse_sequence_details: A dictionary containing parameters specific to
                               the pulse sequence used. Examples:
                               - 'sequence_name': e.g., 'HahnEcho', 'ESEEM', 'DEER'.
                               - 'pulse_lengths_ns': Durations of pi/2, pi pulses.
                               - 'inter_pulse_delays_ns': Delays like 'tau' in Hahn echo.
                               - 'gradient_amplitudes_mT_m': Strength of imaging gradients.
                               - 'gradient_durations_us': Duration for which gradients are on.
                               - 'n_averages': Number of signal averages.
                               - 'kspace_trajectory': e.g., 'radial', 'cartesian'.
    """
    super().__init__(metadata, data)
    self.pulse_sequence_details = pulse_sequence_details

  def get_physics_model(self) -> dict:
    """
    Returns a dictionary describing the physics model for Pulse EPR.

    This model includes key parameters, common sequences, spatial encoding
    details, critical factors, and k-space sampling methods.
    """
    return {
        "technique": "Pulse EPR",
        "description": "Utilizes sequences of microwave pulses to manipulate electron "
                       "spins and detect transient responses (FID, echoes).",
        "common_sequences": [
            "Free Induction Decay (FID)",
            "Hahn Echo (two-pulse)",
            "Spin Echo Electron Modulation (ESEEM - two or three pulse)",
            "Electron-Electron Double Resonance (ELDOR - e.g., DEER/PELDOR)",
            "Inversion Recovery (for T1 measurement)",
            "Specialized imaging sequences (e.g., gradient echo, spin echo imaging)"
        ],
        "spatial_encoding": {
            "method": "Magnetic Field Gradients",
            "principle": "Gradients encode spatial information by making the Larmor "
                         "frequency position-dependent, sampling k-space.",
            "k_space_relation": "k is proportional to integral(Gradient(t) dt)"
        },
        "key_experimental_parameters": [ # Parameters often found in `pulse_sequence_details` or `metadata`
            "microwave_frequency_GHz",
            "pulse_lengths_ns (for pi/2, pi, etc.)",
            "inter_pulse_delays_ns (e.g., tau)",
            "gradient_strength_mT_per_m",
            "gradient_duration_us",
            "gradient_shape (e.g., sinusoidal, trapezoidal, rectangular)",
            "repetition_time_ms_or_s",
            "number_of_averages",
            "temperature_K"
        ],
        "critical_factors_and_challenges": [
            "dead_time_ns (receiver recovery time after pulse)",
            "pulse_imperfections (finite width, B1 inhomogeneity, off-resonance effects)",
            "relaxation_times_T1_us_ms (spin-lattice)",
            "relaxation_times_T2_ns_us (spin-spin/transverse)",
            "phase_memory_time_Tm_ns_us (often similar to T2)",
            "spectrometer_bandwidth_MHz",
            "resonator_Q_factor_and_bandwidth",
            "gradient_fidelity_and_eddy_currents"
        ],
        "typical_kspace_sampling_strategies": [
            "Cartesian (projection-reconstruction with stepped phase encodes)",
            "Radial (projections rotated, often with constant gradient strength)",
            "Spiral (gradients varied sinusoidally to trace spiral in k-space)",
            "Single Point Imaging (SPI) / Constant Time Imaging (CTI) variants"
        ],
        "signal_type": "Time-domain signal (FID or echo)"
    }

  def reconstruct(self, *args, **kwargs) -> str:
    """
    Performs image reconstruction from Pulse EPR data.

    This typically involves processing k-space data. Common methods include:
    - Gridding data to a Cartesian grid followed by Fast Fourier Transform (FFT),
      especially for non-Cartesian k-space trajectories (e.g., radial, spiral).
    - Filtered Backprojection (FBP) if a suitable k-space trajectory (e.g., radial)
      and data format are provided.
    - Iterative reconstruction methods for advanced artifact correction or
      incorporation of prior information.

    Pre-processing steps like dead-time correction, echo centering, and T2*
    compensation might be applied before or during reconstruction.

    Args:
      *args: Additional arguments depending on the reconstruction algorithm.
      **kwargs: Additional keyword arguments (e.g., 'algorithm': 'gridding_fft').

    Returns:
      A string indicating that the reconstruction is not yet implemented or
      the result of the reconstruction (e.g., an image as a NumPy array).
    """
    # Placeholder for actual reconstruction logic
    recon_algorithm = kwargs.get("algorithm", "gridding_fft")
    return f"Reconstruction for Pulse EPR using {recon_algorithm} is not yet implemented."
