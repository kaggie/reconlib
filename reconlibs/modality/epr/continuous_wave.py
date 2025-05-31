from .base import EPRImaging
import numpy as np
# Assuming reconstruction module is in the same directory or package
from .reconstruction import ARTReconstructor, preprocess_cw_epr_data


class ContinuousWaveEPR(EPRImaging):
  """
  Class for Continuous Wave (CW) EPR imaging techniques.

  CW-EPR data is typically acquired by sweeping an external magnetic field
  while applying a continuous, fixed-frequency microwave field to the sample.
  The absorption of microwave energy by paramagnetic species is detected,
  often using a lock-in amplifier synchronized with a small modulation of the
  magnetic field, yielding a derivative spectrum. For imaging, magnetic field
  gradients are typically applied to encode spatial information.

  Key Parameters:
  - Microwave Frequency: The constant frequency of the microwave irradiation.
  - Magnetic Field Sweep Range: The start and end points of the magnetic field sweep.
  - Sweep Rate: Speed at which the magnetic field is swept.
  - Modulation Amplitude: The peak-to-peak amplitude of the magnetic field modulation.
  - Modulation Frequency: The frequency of the magnetic field modulation.
  - Microwave Power: The power level of the microwave irradiation.
  - Time Constant: The time constant of the lock-in amplifier or detection system.
  - Gradient Angles: List of angles for which projections were acquired.

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
  For gradient-based CW-EPR imaging, the EPR spectrum recorded at each gradient
  orientation represents a projection of the spin distribution. Reconstruction
  uses tomographic techniques like Filtered Backprojection or iterative methods
  like ART (Algebraic Reconstruction Technique).
  """

  def __init__(self, metadata: dict, data: np.ndarray, sweep_parameters: dict):
    """Initializes the ContinuousWaveEPR class.

    Args:
      metadata: A dictionary containing metadata for the EPR experiment.
                This might include microwave frequency, power, temperature,
                and potentially 'gradient_angles' if not passed to reconstruct.
      data: A 2D NumPy array representing the projection data (num_angles x num_bins).
            It's assumed that self.data['projections'] holds this if data is a dict.
            For simplicity, this init now expects data to be the np.ndarray directly.
      sweep_parameters: A dictionary containing parameters specific to the
                        magnetic field sweep.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input 'data' for ContinuousWaveEPR must be a 2D NumPy array (projections x bins).")
    super().__init__(metadata, data) # self.data is now the np.ndarray
    self.sweep_parameters = sweep_parameters
    # self.gradient_angles could be initialized here if always present in metadata
    # e.g., self.gradient_angles = metadata.get('gradient_angles')

  def get_physics_model(self) -> dict:
    """
    Returns a dictionary describing the physics model for CW EPR.
    """
    return {
        "technique": "Continuous Wave EPR",
        "description": "Detects microwave absorption during a magnetic field sweep "
                       "with continuous microwave irradiation, often using field "
                       "modulation and lock-in detection.",
        "acquisition_mode": "Magnetic Field Sweep (most common)",
        "alternative_acquisition": "Frequency Sweep (less common)",
        "key_parameters": [
            "microwave_frequency_GHz",
            "magnetic_field_center_mT",
            "magnetic_field_sweep_width_mT",
            "sweep_rate_mT_per_s",
            "modulation_amplitude_mT",
            "modulation_frequency_kHz",
            "microwave_power_mW",
            "time_constant_ms",
            "temperature_K",
            "gradient_angles_degrees" # Added
        ],
        "signal_type": "Typically derivative of absorption spectrum (projections)",
        "common_artifacts": [
            "baseline_drift",
            "noise (1/f, thermal, detector)",
            "saturation_effects (power broadening)",
            "modulation_broadening",
            "passage_effects (rapid sweep distortions)",
            "microphonics (vibrational noise)"
        ],
        "imaging_method": "Gradient-based (projection imaging) commonly reconstructed with FBP or ART."
    }

  def reconstruct(self,
                  gradient_angles: list, # List of angles in degrees
                  grid_size: tuple,      # (num_pixels_y, num_pixels_x)
                  num_iterations: int,
                  relaxation_param: float,
                  regularization_weight: float = 0.0,
                  non_negativity: bool = True,
                  art_preprocessing_params: dict = None,
                  **kwargs) -> np.ndarray:
    """
    Performs image reconstruction from CW-EPR projection data using ART.

    Args:
      gradient_angles: List or 1D NumPy array of gradient angles (in degrees)
                       corresponding to each row in self.data.
      grid_size: Tuple (height, width) defining the dimensions of the
                 reconstruction grid.
      num_iterations: The number of iterations for ART.
      relaxation_param: The relaxation parameter (lambda) for ART.
      regularization_weight: Weight for Tikhonov regularization in ART.
                             Default is 0.0.
      non_negativity: If True, enforces non-negativity in ART. Default is True.
      art_preprocessing_params: Optional dictionary of parameters for
                                `preprocess_cw_epr_data`.
      **kwargs: Additional keyword arguments (currently unused).

    Returns:
      A 2D NumPy array representing the reconstructed image.
    """
    if not isinstance(self.data, np.ndarray):
        raise TypeError("self.data is not a NumPy array. Ensure it's loaded correctly.")
    if self.data.ndim != 2:
        raise ValueError("self.data must be a 2D array of projection data (angles x bins).")
    if len(gradient_angles) != self.data.shape[0]:
        raise ValueError("Number of gradient_angles must match the number of projections in self.data.")

    # Preprocessing (optional)
    # self.data is assumed to be the raw projection data (angles x bins)
    processed_data = preprocess_cw_epr_data(self.data, params=art_preprocessing_params)

    # Instantiate ARTReconstructor
    art_solver = ARTReconstructor(
        projection_data=processed_data,
        gradient_angles=gradient_angles,
        grid_size=grid_size,
        num_iterations=num_iterations,
        relaxation_param=relaxation_param,
        regularization_weight=regularization_weight,
        non_negativity=non_negativity
    )

    # Perform reconstruction
    reconstructed_image = art_solver.reconstruct()

    return reconstructed_image
