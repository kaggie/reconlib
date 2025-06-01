"""
EPR reconstruction algorithms.

This module will contain algorithms for both Continuous Wave (CW) and Pulse EPR.
"""
import numpy as np
import pywt
from scipy.signal import correlate, find_peaks
from scipy.sparse import csc_matrix, eye, diags # eye is not used in current _baseline_als, but kept from previous commit
from scipy.sparse.linalg import spsolve
from scipy.special import wofz

__all__ = [
    'gaussian_lineshape',
    'lorentzian_lineshape',
    'voigt_lineshape',
    'deconvolve_cw_spectrum',
    'apply_kspace_corrections',
    'preprocess_cw_epr_data',
    'radial_recon_2d',
    'radial_recon_3d',
    'ARTReconstructor'
]

# --- Lineshape Functions ---

def gaussian_lineshape(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """
    Generates a Gaussian lineshape normalized to peak height 1.

    Args:
        x (np.ndarray): Array of x-values (e.g., field or frequency).
        center (float): Center of the peak.
        fwhm (float): Full Width at Half Maximum of the Gaussian peak.
                      Must be positive.

    Returns:
        np.ndarray: The Gaussian lineshape values at each x.
    """
    if fwhm <= 0:
        raise ValueError("FWHM must be positive for Gaussian lineshape.")
    # sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma)**2) # Area normalized
    # For peak height 1:
    return np.exp(-4 * np.log(2) * ((x - center) / fwhm)**2)

def lorentzian_lineshape(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """
    Generates a Lorentzian lineshape normalized to peak height 1.

    Args:
        x (np.ndarray): Array of x-values (e.g., field or frequency).
        center (float): Center of the peak.
        fwhm (float): Full Width at Half Maximum of the Lorentzian peak.
                      Must be positive.

    Returns:
        np.ndarray: The Lorentzian lineshape values at each x.
    """
    if fwhm <= 0:
        raise ValueError("FWHM must be positive for Lorentzian lineshape.")
    gamma = fwhm / 2.0  # HWHM
    # For peak height 1 (A=1):
    return (gamma**2) / ((x - center)**2 + gamma**2)

def voigt_lineshape(x: np.ndarray, center: float, fwhm_gaussian: float, fwhm_lorentzian: float) -> np.ndarray:
    """
    Generates a Voigt lineshape normalized to peak height 1.

    The Voigt profile is a convolution of Gaussian and Lorentzian profiles.

    Args:
        x (np.ndarray): Array of x-values (e.g., field or frequency).
        center (float): Center of the peak.
        fwhm_gaussian (float): Full Width at Half Maximum of the Gaussian component.
                               Must be non-negative.
        fwhm_lorentzian (float): Full Width at Half Maximum of the Lorentzian component.
                                 Must be non-negative.

    Returns:
        np.ndarray: The Voigt lineshape values at each x, normalized to peak height 1.
                    Returns Gaussian if fwhm_lorentzian is zero or negligible.
                    Returns Lorentzian if fwhm_gaussian is zero or negligible.
    """
    if fwhm_gaussian < 0 or fwhm_lorentzian < 0:
        raise ValueError("FWHM values must be non-negative for Voigt lineshape.")

    if fwhm_gaussian < 1e-8 and fwhm_lorentzian < 1e-8: # Both effectively zero
        # Return a delta-like function or raise error, here returning array of zeros for safety
        # or perhaps better, a narrow Gaussian centered at 'center' if x covers it.
        # For now, if x contains center, put 1 there. This is an edge case.
        # A true delta function is harder to represent on a discrete grid.
        # Let's assume FWHM are practically zero, so it's a spike if x=center.
        # However, a function that is zero everywhere except one point is not typical.
        # More practically, if both are zero, it should be a very narrow Gaussian or Lorentzian.
        # Fallback to a very narrow Gaussian if both are zero.
        # This case should ideally be handled by the caller or by defining a minimum width.
        # For now, let's treat it as a very narrow Gaussian if both are zero.
        if fwhm_gaussian < 1e-8: fwhm_gaussian = 1e-5 # Avoid division by zero, make it a very narrow Gaussian

    if fwhm_gaussian < 1e-8: # Effectively Lorentzian
        return lorentzian_lineshape(x, center, fwhm_lorentzian)
    if fwhm_lorentzian < 1e-8: # Effectively Gaussian
        return gaussian_lineshape(x, center, fwhm_gaussian)

    sigma_g = fwhm_gaussian / (2 * np.sqrt(2 * np.log(2)))
    gamma_l = fwhm_lorentzian / 2.0

    # Complex argument for wofz
    z = (x - center + 1j * gamma_l) / (sigma_g * np.sqrt(2))

    voigt_profile = np.real(wofz(z))

    # Normalize to peak height 1 by dividing by the max value of the profile
    # This is an empirical normalization as the analytical max of wofz(z) can be complex.
    # We calculate the profile over the given x range and normalize to its max in that range.
    # For a well-behaved peak centered in x, this should be close to the true max.
    # If the peak is off-center or x range is too small, this might not be perfect.

    # To get a more robust peak normalization, we can evaluate wofz at the center
    # for the imaginary part gamma_l / (sigma_g * sqrt(2))
    # z_center = (1j * gamma_l) / (sigma_g * np.sqrt(2))
    # peak_value = np.real(wofz(z_center))
    # if peak_value > 0:
    #    return voigt_profile / peak_value
    # else: # Fallback if peak_value is zero or negative (should not happen for valid params)
    #    current_max = np.max(voigt_profile)
    #    if current_max > 0:
    #        return voigt_profile / current_max
    #    return voigt_profile # Should not happen if input is valid

    # Simpler: calculate profile, then divide by its max.
    # This assumes the x-range is sufficient to capture the peak.
    profile_max = np.max(voigt_profile)
    if profile_max > 1e-9: # Avoid division by zero or near-zero if profile is flat
        return voigt_profile / profile_max

    # If profile_max is very small (e.g. x-range far from center),
    # avoid division by zero and return the (small) profile as is.
    return voigt_profile


def deconvolve_cw_spectrum(spectrum, modulation_amplitude_mT: float, other_params: dict = None):
  """
  Deconvolves a CW-EPR spectrum to remove instrumental broadening.

  In CW-EPR, magnetic field modulation is often used for sensitivity enhancement,
  leading to a derivative spectrum that is also broadened by the modulation
  amplitude. Deconvolution aims to recover the underlying absorption spectrum
  and improve spectral resolution. This is often a crucial preprocessing
  step for spectral-spatial CW-EPR imaging or for accurate lineshape analysis.

  Args:
    spectrum: The recorded CW-EPR spectrum (often a derivative).
    modulation_amplitude_mT: The peak-to-peak modulation amplitude in mT.
    other_params: Dictionary of other parameters relevant to deconvolution,
                  e.g., linewidth, lineshape model (Gaussian/Lorentzian).

  Returns:
    A placeholder string for the deconvolved CW spectrum.
    This will be replaced with an actual spectrum (e.g., NumPy array).
  """
  # Placeholder for actual deconvolution logic
  print(f"Spectrum received: {type(spectrum)}, Mod Amp: {modulation_amplitude_mT}")
  if other_params:
    print(f"Other params: {other_params}")
  return "Deconvolved CW spectrum placeholder"

def apply_kspace_corrections(raw_kspace_data, correction_parameters: dict):
  """
  Applies common corrections to raw k-space data from Pulse EPR experiments.

  Before Fourier Transformation, k-space data often requires corrections for:
  - Dead-time effects: Missing data points at the beginning of FID/echo.
  - Echo centering: Ensuring the peak of the echo is at the correct k-space origin.
  - Phase errors: Correcting phase shifts due to hardware or sequence timing.
  - Baseline distortions: Removing offsets or drifts in the signal.
  - T2*/T2 decay compensation: Optionally weighting data to counteract signal decay.

  Args:
    raw_kspace_data: The acquired k-space data (e.g., NumPy array).
    correction_parameters: A dictionary containing parameters for each
                           correction to be applied, e.g.,
                           {'dead_time_points': 5, 'apply_phase_correction': True}.

  Returns:
    A placeholder string for the corrected k-space data.
    This will be replaced with the actual corrected k-space data (e.g., NumPy array).
  """
  print(f"Raw k-space data received: {type(raw_kspace_data)}")
  print(f"Correction parameters: {correction_parameters}")
  return "Corrected k-space data placeholder"

def preprocess_cw_epr_data(projection_data: np.ndarray, params: dict = None) -> np.ndarray:
  """
  Preprocesses projection data from CW-EPR imaging experiments.

  CW-EPR projection data, where each row typically corresponds to a projection
  angle and columns are spectral bins, often requires several preprocessing
  steps before reconstruction. These steps aim to correct artifacts, reduce
  noise, and normalize the data.

  Typical Preprocessing Steps:
  1.  Baseline Correction:
      - 'polynomial': Fits and subtracts a low-order polynomial.
        Requires `baseline_correct_order`.
      - 'als': Asymmetric Least Squares baseline correction.
        Requires `als_lambda`, `als_p_asymmetry`, `als_niter`.
  2.  Noise Reduction:
      - 'savitzky-golay': Savitzky-Golay filter. (Preserved if existing)
        Requires `savgol_window_length`, `savgol_polyorder`.
      - 'wavelet': Wavelet denoising.
        Requires `wavelet_type`, `wavelet_level`, `wavelet_threshold_sigma_multiplier`.
  3.  Spectral Alignment (Field Shift Correction):
      - If `align_spectra` is True.
      - Uses cross-correlation with a reference projection.
      - `reference_projection_index` specifies the reference.
      - Optional: `align_peak_prominence` for peak-based alignment. (Future)
  4.  Normalization:
      - Controlled by `normalize_method`.
      - 'max': Normalizes each projection by its maximum value. (Previously `normalize_per_projection`)
      - 'area': Normalizes each projection by its total area (sum of absolute values).
      - 'intensity_sum': Normalizes by the sum of values.
      - None: No normalization.

  Order of Operations:
  1. Baseline Correction
  2. Denoising
  3. Alignment
  4. Normalization

  Args:
    projection_data: A 2D NumPy array where rows are projection angles and
                     columns are spectral data points.
    params: An optional dictionary of parameters to control preprocessing.
            Example:
            ```python
            params = {
                'baseline_correct_method': 'als', # 'polynomial', 'als', None
                'baseline_correct_order': 2,      # For polynomial
                'als_lambda': 1e6,
                'als_p_asymmetry': 0.01,
                'als_niter': 10,
                'denoise_method': 'wavelet',      # 'savitzky-golay', 'wavelet', None
                # 'savgol_window_length': 11,     # For Savitzky-Golay
                # 'savgol_polyorder': 2,          # For Savitzky-Golay
                'wavelet_type': 'db4',
                'wavelet_level': 4,
                'wavelet_threshold_sigma_multiplier': 3,
                'align_spectra': True,
                'reference_projection_index': 0,
                # 'align_peak_prominence': 0.1, # For future peak-based alignment
                'normalize_method': 'area',       # 'max', 'area', 'intensity_sum', None
            }
            ```

  Returns:
    A 2D NumPy array of preprocessed projection data.
  """
  if not isinstance(projection_data, np.ndarray):
    raise TypeError("projection_data must be a NumPy array.")
  if projection_data.ndim != 2:
    raise ValueError("projection_data must be a 2D array (angles x spectral_bins).")

  processed_data = projection_data.copy()
  num_projections, num_points = processed_data.shape

  if not params:
    params = {} # Ensure params is a dict

  # 1. Baseline Correction
  baseline_method = params.get('baseline_correct_method')
  if baseline_method == 'polynomial':
    order = params.get('baseline_correct_order', 2)
    for i in range(num_projections):
      x = np.arange(num_points)
      if np.all(processed_data[i,:] == 0) : continue # Skip if row is all zeros
      poly_coeffs = np.polyfit(x, processed_data[i, :], order)
      baseline = np.polyval(poly_coeffs, x)
      processed_data[i, :] -= baseline
  elif baseline_method == 'als':
    lam = params.get('als_lambda', 1e6)
    p = params.get('als_p_asymmetry', 0.01)
    niter = params.get('als_niter', 10)
    for i in range(num_projections):
      if np.all(processed_data[i,:] == 0) : continue
      baseline = _baseline_als(processed_data[i, :], lam, p, niter)
      processed_data[i, :] -= baseline

  # 2. Denoising
  denoise_method = params.get('denoise_method')
  if denoise_method == 'wavelet':
    wavelet_type = params.get('wavelet_type', 'db4')
    wavelet_level = params.get('wavelet_level', 4)
    sigma_multiplier = params.get('wavelet_threshold_sigma_multiplier', 3) # VisuShrink-like
    for i in range(num_projections):
      if np.all(processed_data[i,:] == 0) : continue
      coeffs = pywt.wavedec(processed_data[i, :], wavelet=wavelet_type, level=wavelet_level)
      # Estimate noise variance from the median absolute deviation (MAD) of the first level detail coefficients
      sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
      threshold = sigma_multiplier * sigma

      # Soft thresholding
      thresholded_coeffs = [coeffs[0]] # Keep approximation coefficients
      for j in range(1, len(coeffs)):
          thresholded_coeffs.append(pywt.threshold(coeffs[j], value=threshold, mode='soft'))
      processed_data[i, :] = pywt.waverec(thresholded_coeffs, wavelet=wavelet_type)
      # Ensure output length matches input length after wavelet reconstruction
      if len(processed_data[i, :]) != num_points:
          processed_data[i, :] = processed_data[i, :][:num_points]

  elif denoise_method == 'savitzky-golay':
    window_length = params.get('savgol_window_length', 11)
    polyorder = params.get('savgol_polyorder', 2)
    if window_length % 2 == 0: window_length+=1 # must be odd
    if polyorder >= window_length: polyorder = window_length -1 # must be less than window_length

    from scipy.signal import savgol_filter # Local import if only used here
    for i in range(num_projections):
        if np.all(processed_data[i,:] == 0) : continue
        if len(processed_data[i,:]) > window_length: # Filter only if data is long enough
            processed_data[i, :] = savgol_filter(processed_data[i, :], window_length, polyorder)


  # 3. Spectral Alignment
  if params.get('align_spectra', False):
    ref_idx = params.get('reference_projection_index', 0)
    if not (0 <= ref_idx < num_projections):
        print(f"Warning: Invalid reference_projection_index {ref_idx}. Using 0 as default.")
        ref_idx = 0

    reference_projection = processed_data[ref_idx, :].copy() # Use a copy

    # Optional: Use peak finding for alignment reference if prominence is given
    align_peak_prominence = params.get('align_peak_prominence')
    if align_peak_prominence:
        ref_peaks, _ = find_peaks(reference_projection, prominence=align_peak_prominence)
        if not ref_peaks.size:
            print(f"Warning: No peak found in reference spectrum {ref_idx} with prominence {align_peak_prominence}. Using full spectrum correlation.")
            align_peak_prominence = None # Fallback to full correlation
        else:
            # Use the most prominent peak for alignment
            # This logic might need refinement based on specific peak characteristics desired
            prominences = _['prominences']
            main_ref_peak_idx = ref_peaks[np.argmax(prominences)]


    for i in range(num_projections):
      if i == ref_idx:
        continue # Skip reference projection

      current_projection = processed_data[i, :]

      if align_peak_prominence and ref_peaks.size: # Peak-based alignment
          target_peaks, _ = find_peaks(current_projection, prominence=align_peak_prominence)
          if not target_peaks.size:
              print(f"Warning: No peak found in target spectrum {i} with prominence {align_peak_prominence}. Skipping alignment for this spectrum.")
              continue

          # Find the target peak that is closest in index to the main_ref_peak_idx
          # This assumes peaks don't shift *too* dramatically
          # More robust would be to match based on shape or other features if multiple peaks
          main_target_peak_idx = target_peaks[np.argmin(np.abs(target_peaks - main_ref_peak_idx))]
          shift = main_ref_peak_idx - main_target_peak_idx

      else: # Cross-correlation based alignment
          correlation = correlate(current_projection, reference_projection, mode='same')
          # The 'lag' is the offset from the center of the correlation array
          lag_array = np.arange(-num_points // 2, num_points // 2 + (num_points % 2))
          if correlation.size == 0: # Should not happen with mode='same'
              shift = 0
          else:
              shift = lag_array[np.argmax(correlation)]


      processed_data[i, :] = np.roll(current_projection, shift)
      # Handle edge effects by zeroing out the introduced elements if desired,
      # or use other padding methods (np.pad) before rolling.
      # For simplicity, np.roll wraps around. If shifting introduces invalid data,
      # those parts might need to be set to 0 or a fill value.
      if shift > 0:
          processed_data[i, :shift] = 0 # Or some fill value
      elif shift < 0:
          processed_data[i, shift:] = 0 # Or some fill value


  # 4. Normalization
  normalize_method = params.get('normalize_method')
  if normalize_method == 'max': # This was the previous 'normalize_per_projection'
    for i in range(num_projections):
      row_max = processed_data[i, :].max()
      if row_max != 0:
        processed_data[i, :] = processed_data[i, :] / row_max
  elif normalize_method == 'area':
    for i in range(num_projections):
      row_area = np.sum(np.abs(processed_data[i, :]))
      if row_area != 0:
        processed_data[i, :] = processed_data[i, :] / row_area
  elif normalize_method == 'intensity_sum':
    for i in range(num_projections):
      row_sum = np.sum(processed_data[i, :])
      if row_sum != 0:
        processed_data[i, :] = processed_data[i, :] / row_sum
  # elif normalize_method == 'reference_peak':
      # This would require identifying the peak after alignment and then normalizing.
      # Needs robust peak identification. If align_peak_prominence was used,
      # the peak height at main_target_peak_idx (after shift) could be used.
      # This is more complex and might be better as a separate step or refined.
      # print("Normalization by 'reference_peak' is not fully implemented yet.")
      # pass


  return processed_data

# Helper function for Asymmetric Least Squares baseline correction
def _baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2), format="csc")
    D = D.dot(D.transpose()) # D = D^T * D in CSR format
    w = np.ones(L)
    for i in range(niter):
        W = csc_matrix(np.diag(w)) # Use csc_matrix for sparse diagonal matrix
        Z = W + lam * D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def radial_recon_2d(data, angles):
  """
  Performs 2D radial reconstruction (e.g., filtered backprojection).

  This is directly applicable to Pulse EPR when a radial k-space acquisition
  scheme (projections acquired at different angles) is used. Each 'data' entry
  would be a 1D k-space line (e.g., an echo profile acquired under a readout
  gradient, with the gradient rotated for each new angle).
  It's also applicable to CW-EPR if projection data is acquired, for example,
  by rotating a sample in a fixed gradient or by rotating the magnetic field
  gradients around the sample.

  Args:
    data: A list or NumPy array of 1D projections (k-space lines for Pulse EPR).
    angles: A list or NumPy array of projection angles in radians.

  Returns:
    A placeholder string for the 2D reconstructed image.
    This will be replaced with an actual image (e.g., NumPy array)
    once numerical libraries are integrated.
  """
  # In a real implementation, this would involve steps like:
  # 1. Applying a Ram-Lak filter (or similar) to each projection.
  # 2. Backprojecting the filtered projections onto an image grid.
  # For Pulse EPR, data might need gridding before FBP if not perfectly on radial lines.
  print(f"Data received: {type(data)}, Angles received: {type(angles)}")
  return "2D Radial Reconstruction Placeholder"

def radial_recon_3d(data, angles_phi, angles_theta):
  """
  Performs 3D radial reconstruction.

  Similar to 2D, this is directly applicable to Pulse EPR with 3D radial
  (or "kooshball") k-space sampling. Each 'data' entry is a 1D k-space line.
  It's also applicable to CW-EPR with 3D field gradients, where spectra under
  different gradient orientations (defined by phi and theta) serve as projections.

  Args:
    data: A list or NumPy array of 1D projections (k-space lines for Pulse EPR).
    angles_phi: A list or NumPy array of projection angles (phi) in radians.
    angles_theta: A list or NumPy array of projection angles (theta) in radians.

  Returns:
    A placeholder string for the 3D reconstructed image.
    This will be replaced with an actual image (e.g., NumPy array)
    once numerical libraries are integrated.
  """
  # Similar to 2D, but with 3D backprojection.
  # For Pulse EPR, data might need gridding.
  print(f"Data received: {type(data)}, Angles (phi) received: {type(angles_phi)}, Angles (theta) received: {type(angles_theta)}")
  return "3D Radial Reconstruction Placeholder"


class ARTReconstructor:
  """
  Algebraic Reconstruction Technique (ART) for tomographic reconstruction.

  ART is an iterative algorithm used to reconstruct an image from a set of
  projections. It is particularly useful when the projection data is limited
  or noisy.
  """
  def __init__(self,
               projection_data: np.ndarray,
               gradient_angles: list, # List of angles in degrees or radians
               grid_size: tuple,      # (num_pixels_x, num_pixels_y)
               num_iterations: int,
               relaxation_param: float, # Often denoted as lambda
               regularization_weight: float = 0.0, # For optional regularization
               non_negativity: bool = True,
               lineshape_model: str = None,
               lineshape_params: dict = None,
               projector_type: str = 'nearest'):
    """
    Initializes the ARTReconstructor.

    Args:
      projection_data: 2D NumPy array of projection data (num_angles x num_bins).
      gradient_angles: List or 1D NumPy array of gradient angles corresponding
                       to each row in projection_data.
      grid_size: Tuple (height, width) defining the dimensions of the
                 reconstruction grid.
      num_iterations: The number of iterations to perform.
      relaxation_param: The relaxation parameter (lambda) controlling the
                        update step size. Typically 0 < lambda < 2.
      regularization_weight: Weight for an optional regularization term (e.g., Tikhonov).
                             Default is 0.0 (no regularization).
      non_negativity: If True, enforces non-negativity in the reconstructed image
                      at each iteration. Default is True.
      lineshape_model: Optional string specifying the EPR lineshape model.
                       E.g., 'gaussian', 'lorentzian', 'voigt', or None.
                       If None, a delta function like projection is assumed (original behavior).
                       Default is None.
      lineshape_params: Optional dictionary of parameters for the lineshape model.
                        E.g., {'fwhm': 0.5} for Gaussian/Lorentzian (in bin units),
                        or {'fwhm_g': 0.3, 'fwhm_l': 0.2} for Voigt (in bin units).
                        Default is None.
      projector_type: Type of projector to use for the system matrix.
                      'nearest': Pixel center projected to nearest bin (original behavior).
                      'siddon_like': Ray-pixel intersection length (simplified).
                      Default is 'nearest'.
    """
    # Validate projection_data
    if not isinstance(projection_data, np.ndarray):
      raise TypeError("projection_data must be a NumPy array.")
    if projection_data.ndim != 2:
      raise ValueError("projection_data must be a 2D array (num_angles x num_bins).")
    self.projection_data = projection_data

    # Validate gradient_angles
    if not isinstance(gradient_angles, (list, np.ndarray)):
      raise TypeError("gradient_angles must be a list or NumPy array.")
    if len(gradient_angles) != self.projection_data.shape[0]:
      raise ValueError("Length of gradient_angles must match the number of rows in projection_data.")
    self.gradient_angles = np.array(gradient_angles) # Ensure it's a numpy array

    # Validate grid_size
    if not (isinstance(grid_size, tuple) and len(grid_size) == 2 and
            isinstance(grid_size[0], int) and grid_size[0] > 0 and
            isinstance(grid_size[1], int) and grid_size[1] > 0):
      raise ValueError("grid_size must be a tuple of two positive integers (height, width).")
    self.grid_size = grid_size

    # Validate num_iterations
    if not (isinstance(num_iterations, int) and num_iterations > 0):
      raise ValueError("num_iterations must be a positive integer.")
    self.num_iterations = num_iterations

    # Validate relaxation_param
    if not isinstance(relaxation_param, (float, int)): # allow int for convenience
        raise TypeError("relaxation_param must be a float.")
    if not (0 < relaxation_param <= 2): # Common range, can be strict or warning
        print(f"Warning: relaxation_param ({relaxation_param}) is typically between 0 and 2.")
    self.relaxation_param = float(relaxation_param)


    # Validate regularization_weight
    if not isinstance(regularization_weight, (float, int)):
        raise TypeError("regularization_weight must be a float.")
    if regularization_weight < 0:
      raise ValueError("regularization_weight must be non-negative.")
    self.regularization_weight = float(regularization_weight)

    # Validate non_negativity
    if not isinstance(non_negativity, bool):
      raise TypeError("non_negativity must be a boolean value.")
    self.non_negativity = non_negativity
    self.lineshape_model = lineshape_model
    self.lineshape_params = lineshape_params if lineshape_params is not None else {}
    self.projector_type = projector_type

    # Validate lineshape and projector params
    if self.lineshape_model and not isinstance(self.lineshape_model, str):
        raise TypeError("lineshape_model must be a string or None.")
    if self.lineshape_params and not isinstance(self.lineshape_params, dict):
        raise TypeError("lineshape_params must be a dictionary or None.")
    if not isinstance(self.projector_type, str) or self.projector_type not in ['nearest', 'siddon_like']:
        raise ValueError("projector_type must be 'nearest' or 'siddon_like'.")

    if self.lineshape_model:
        if self.lineshape_model == 'gaussian' and 'fwhm' not in self.lineshape_params:
            raise ValueError("Missing 'fwhm' in lineshape_params for 'gaussian' model.")
        if self.lineshape_model == 'lorentzian' and 'fwhm' not in self.lineshape_params:
            raise ValueError("Missing 'fwhm' in lineshape_params for 'lorentzian' model.")
        if self.lineshape_model == 'voigt' and ('fwhm_g' not in self.lineshape_params or 'fwhm_l' not in self.lineshape_params):
            raise ValueError("Missing 'fwhm_g' or 'fwhm_l' in lineshape_params for 'voigt' model.")


    # Derived attributes
    self.num_angles = self.projection_data.shape[0]
    self.num_bins_per_projection = self.projection_data.shape[1]
    self.num_pixels_x = self.grid_size[1] # width
    self.num_pixels_y = self.grid_size[0] # height
    self.num_pixels = self.num_pixels_x * self.num_pixels_y

    # Total number of individual data points in all projections
    self.total_projection_bins = self.num_angles * self.num_bins_per_projection

    # Initialize image estimate (flattened array)
    self.image_estimate = np.zeros(self.num_pixels, dtype=np.float64)

    # Placeholder for system matrix (optional, can be computed on-the-fly)
    self.system_matrix = None # Or initialize if pre-computed

    # Note: The self.image_estimate was already initialized as a flattened array.
    # The _initialize_image method will provide a 2D version if needed,
    # or could be used to reset the self.image_estimate if it were 2D.
    # For ART, operating on a flattened image_estimate is often convenient.

    print(f"ARTReconstructor initialized: {self.grid_size[0]}x{self.grid_size[1]} grid, "
          f"{self.num_angles} angles, {self.num_bins_per_projection} bins/projection.")
    print(f"Iterations: {self.num_iterations}, Relaxation: {self.relaxation_param}, "
          f"Regularization: {self.regularization_weight}, Non-negativity: {self.non_negativity}")

    # Initialize system matrix (can be large, consider on-the-fly for memory efficiency)
    # For now, we pre-calculate it.
    # self.system_matrix, self.row_norms_sq = self._initialize_system_matrix() # Call this in reconstruct

  def _calculate_ray_pixel_intersection_length(self, px_min_x, px_max_x, px_min_y, px_max_y, ray_orig_x, ray_orig_y, ray_dir_x, ray_dir_y):
    """
    Calculates the length of a ray segment intersecting a rectangular pixel.
    Simplified version of Siddon's algorithm for a single ray and pixel.

    Args:
        px_min_x, px_max_x: X boundaries of the pixel.
        px_min_y, px_max_y: Y boundaries of the pixel.
        ray_orig_x, ray_orig_y: Origin point of the ray.
        ray_dir_x, ray_dir_y: Direction vector of the ray (dx, dy).

    Returns:
        float: The intersection length. Returns 0.0 if no intersection.
    """
    # Normalize ray direction vector (optional, but helps if t is actual length)
    # norm_ray_dir = np.sqrt(ray_dir_x**2 + ray_dir_y**2)
    # if norm_ray_dir < 1e-9: return 0.0 # Avoid division by zero for zero direction vector
    # ray_dir_x /= norm_ray_dir
    # ray_dir_y /= norm_ray_dir

    t_min = -np.inf
    t_max = np.inf

    # Intersection with X planes
    if abs(ray_dir_x) > 1e-9: # Ray is not parallel to Y axis
        t1 = (px_min_x - ray_orig_x) / ray_dir_x
        t2 = (px_max_x - ray_orig_x) / ray_dir_x
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
    elif not (px_min_x <= ray_orig_x < px_max_x): # Ray is parallel and outside X-slab
        return 0.0

    # Intersection with Y planes
    if abs(ray_dir_y) > 1e-9: # Ray is not parallel to X axis
        t1 = (px_min_y - ray_orig_y) / ray_dir_y
        t2 = (px_max_y - ray_orig_y) / ray_dir_y
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
    elif not (px_min_y <= ray_orig_y < px_max_y): # Ray is parallel and outside Y-slab
        return 0.0

    if t_max > t_min and t_max > 0: # Intersection segment exists and is at least partially in front of ray origin
        # We need to ensure the segment is "physical" if t can be negative.
        # If ray origin can be anywhere, t_min and t_max define the segment.
        # The length is (t_max - t_min) * norm_of_direction_vector.
        # If direction vector was normalized, length is t_max - t_min.
        # Let's assume t represents distance along the normalized ray.
        # The current implementation of t_min/t_max assumes ray can extend infinitely.
        # For ART, we only care about positive intersection lengths.
        intersection_length = t_max - t_min

        # If ray_dir was not normalized, need to multiply by its norm
        # intersection_length *= norm_ray_dir
        # However, the t values are distances along the (potentially unnormalized) ray vector.
        # So, the length of the segment in terms of parameter t is (t_max - t_min).
        # The actual length is this difference times the magnitude of (ray_dir_x, ray_dir_y)
        # Let's assume the ray_dir passed is already scaled or handled by `w_geom` context.
        # The FWHM for lineshapes are in "bin units". `w_geom` should be proportional to actual length.
        # So, this function should return a value proportional to length.
        # Using (t_max - t_min) as the weight is fine if consistent.
        # Let's assume ray_dir_x, ray_dir_y are cos(theta_perp), sin(theta_perp) (normalized)

        return max(0.0, intersection_length) # ensure non-negative length
    return 0.0


  def _initialize_image(self) -> np.ndarray:
    """
    Initializes (or re-initializes) the reconstruction image grid.

    Returns:
      A NumPy array of zeros with the shape specified by self.grid_size
      (num_pixels_y, num_pixels_x) and dtype np.float64.
    """
    # self.grid_size is (height, width) which is (num_pixels_y, num_pixels_x)
    return np.zeros(self.grid_size, dtype=np.float64)

  def _initialize_system_matrix(self) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes the system matrix A for ART.

    The method of construction depends on `self.projector_type` and `self.lineshape_model`.

    If `self.projector_type == 'nearest'`:
      A pixel's contribution is assigned to the single nearest projection bin
      based on the projection of the pixel's center.
    If `self.projector_type == 'siddon_like'`:
      A simplified ray-driven model is used. For each projection ray (center of a bin),
      the intersection length with each pixel is calculated. This length is the
      geometric weight `w_geom`.

    If `self.lineshape_model` is specified (e.g., 'gaussian', 'lorentzian', 'voigt'):
      The geometric contribution `w_geom` (either 1.0 for 'nearest' or intersection
      length for 'siddon_like') is convolved with the specified lineshape.
      The lineshape's FWHM is provided in `self.lineshape_params` in units of
      projection bins. A single pixel will then contribute to multiple bins.
    If `self.lineshape_model` is None:
      The contribution `w_geom` is assigned directly to the central bin(s)
      determined by the projector.

    The projection axis spans from `p_min` to `p_max`, calculated based on
    image dimensions, and is divided into `self.num_bins_per_projection`.

    Rotation center is assumed to be the center of the image grid.

    Returns:
      A tuple containing:
        - A (np.ndarray): The system matrix with shape
          (total_projection_bins, num_pixels).
        - row_norms_sq (np.ndarray): 1D array of squared L2 norms of rows of A.
    """
    A = np.zeros((self.total_projection_bins, self.num_pixels), dtype=np.float64)

    # Pixel dimensions (assuming square pixels of size 1x1 in image space)
    pixel_width = 1.0
    pixel_height = 1.0

    image_center_x = self.num_pixels_x / 2.0
    image_center_y = self.num_pixels_y / 2.0

    # Determine the span of the projection axis
    # This should ideally match the physical extent of the projections
    max_img_dim = max(self.num_pixels_x, self.num_pixels_y)
    p_min = -max_img_dim / 2.0
    p_max = max_img_dim / 2.0
    projection_axis_length = p_max - p_min
    bin_width = projection_axis_length / self.num_bins_per_projection

    if bin_width <= 0:
        raise ValueError("Calculated bin_width is not positive. Check grid_size and num_bins_per_projection.")

    # Prepare lineshape function and parameters if a model is chosen
    lineshape_func = None
    lineshape_bin_axis = None
    lineshape_values_centered = None
    max_lineshape_spread_bins = 0 # How many bins away from center the lineshape spreads

    if self.lineshape_model:
        if self.lineshape_model == 'gaussian':
            lineshape_func = gaussian_lineshape
            fwhm_bins = self.lineshape_params['fwhm']
        elif self.lineshape_model == 'lorentzian':
            lineshape_func = lorentzian_lineshape
            fwhm_bins = self.lineshape_params['fwhm']
        elif self.lineshape_model == 'voigt':
            lineshape_func = voigt_lineshape
            fwhm_g_bins = self.lineshape_params['fwhm_g']
            fwhm_l_bins = self.lineshape_params['fwhm_l']
        else:
            raise ValueError(f"Unknown lineshape model: {self.lineshape_model}")

        # Determine spread of lineshape (e.g., +/- 3*FWHM for Gaussian-like, or adapt)
        # For simplicity, let's use a fixed multiple of FWHM or a reasonable number of bins.
        # This determines the range of 'offset' in the loops later.
        # A more adaptive way would be to find where lineshape drops below a threshold.
        effective_fwhm_for_spread = fwhm_bins if self.lineshape_model != 'voigt' else max(fwhm_g_bins, fwhm_l_bins)
        max_lineshape_spread_bins = int(np.ceil(effective_fwhm_for_spread * 2.5)) # e.g., cover out to where it's small
        if max_lineshape_spread_bins == 0 : max_lineshape_spread_bins = 1 # ensure it covers at least the center bin

        # Create a generic x-axis for the lineshape, centered at 0
        # It should span from -max_lineshape_spread_bins to +max_lineshape_spread_bins
        lineshape_bin_axis = np.arange(-max_lineshape_spread_bins, max_lineshape_spread_bins + 1)

        if self.lineshape_model == 'voigt':
            lineshape_values_centered = lineshape_func(lineshape_bin_axis, 0, fwhm_g_bins, fwhm_l_bins)
        else: # gaussian or lorentzian
            lineshape_values_centered = lineshape_func(lineshape_bin_axis, 0, fwhm_bins)


    for angle_idx, theta_deg in enumerate(self.gradient_angles):
        theta_rad = np.deg2rad(theta_deg)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        for r_idx in range(self.num_pixels_y):  # y-coordinate of pixel
            for c_idx in range(self.num_pixels_x):  # x-coordinate of pixel
                pixel_flat_idx = r_idx * self.num_pixels_x + c_idx

                # Pixel boundaries in image space (origin at top-left)
                px_x_min = c_idx
                px_x_max = c_idx + pixel_width
                px_y_min = r_idx
                px_y_max = r_idx + pixel_height

                # Pixel center relative to image center (for projection)
                # Image center is (self.num_pixels_x / 2.0, self.num_pixels_y / 2.0)
                center_x_img_coord = c_idx + pixel_width / 2.0
                center_y_img_coord = r_idx + pixel_height / 2.0

                center_x_rel = center_x_img_coord - image_center_x
                center_y_rel = center_y_img_coord - image_center_y

                if self.projector_type == 'siddon_like':
                    # Loop through each projection bin's central ray for this angle
                    for k_bin_idx in range(self.num_bins_per_projection):
                        # Determine the ray for this bin k_bin_idx at angle theta_rad
                        # Ray origin (p_val) on the projection axis (from p_min to p_max)
                        p_val = p_min + (k_bin_idx + 0.5) * bin_width

                        # Convert ray origin (p_val, angle) back to Cartesian for intersection calculation
                        # A point on the ray: (p_val*cos(theta), p_val*sin(theta)) in rotated system
                        # Ray direction vector: (-sin(theta), cos(theta))
                        # This needs careful setup of coordinate systems.
                        # For a simplified Siddon-like:
                        # We project pixel corners to the projection axis, and if the bin is between
                        # the min/max projected corners, we approximate intersection.
                        # A true Siddon requires finding t_min, t_max for intersections with pixel boundaries.

                        # Simplified: Use pixel center projection to find central bin for this pixel
                        # then use a fixed geometric factor (e.g. 1.0 if ray passes close to center)
                        # This is not true Siddon, but a step towards it.
                        # For now, let's stick to the planned logic:
                        # calculate_intersection_length(pixel_boundaries, ray_origin_k, ray_direction_k)
                        # This helper is non-trivial.
                        # As a placeholder for calculate_intersection_length:
                        # Project pixel center. If it falls into k_bin_idx, w_geom = 1, else 0.
                        # This makes 'siddon_like' currently identical to 'nearest' for w_geom source.
                        # The true Siddon would calculate actual intersection length.
                        # --- Start of Siddon-like w_geom calculation ---
                        # Ray definition (direction perpendicular to gradient)
                        ray_dir_x = -sin_theta # Direction of the ray
                        ray_dir_y = cos_theta

                        # A point on the ray (center of the current projection bin k_bin_idx on the projection axis,
                        # then mapped back to image_center relative coordinates)
                        # p_val is distance along projection axis (which is along gradient direction)
                        # So, a point on the line x*cos+y*sin = p_val
                        # Let this point be (xo, yo) in image_center_rel coords.
                        # A point on the ray: (p_val * cos_theta, p_val * sin_theta) is NOT on the ray,
                        # it's on the line *defining* the projection value p_val.
                        # The ray is perpendicular to the gradient vector (cos_theta, sin_theta).
                        # The ray equation: (x_rel - Xc)*(-sin_theta) + (y_rel - Yc)*cos_theta = 0
                        # where (Xc, Yc) is a point on the ray.
                        # Let's choose the point where the ray crosses the axis that is perpendicular
                        # to the gradient and passes through the origin of the relative coord system.
                        # This point is (p_val * -sin_theta, p_val * cos_theta) if p_val were distance along that axis.
                        # It's simpler: the ray is defined by:
                        # (x_rel * cos_theta + y_rel * sin_theta) = p_val  -- This is projection equation.
                        # The ray itself is a line. Let's use its definition in global coords for intersection.
                        # Ray origin for intersection function (a point on the line):
                        # If cos_theta is not too small: ry_orig_y = image_center_y, ry_orig_x = (p_val - (ry_orig_y - image_center_y)*sin_theta)/cos_theta + image_center_x
                        # else: ry_orig_x = image_center_x, ry_orig_y = (p_val - (ry_orig_x - image_center_x)*cos_theta)/sin_theta + image_center_y
                        # This is getting complicated. Let's use the ray's normal form:
                        # n_x*x + n_y*y - d = 0, where n=(cos_theta, sin_theta), d = p_val (if x,y relative to image center)
                        # For intersection, we need parametric form of the ray.
                        # Point on ray: (p_val * cos_theta + image_center_x, p_val * sin_theta + image_center_y) is a point on the *gradient line*.
                        # The ray is perpendicular to this.
                        # Ray origin (a point on the ray, in global image coords [0,Ngrid]):
                        # Let's use the center of the projection bin on the projection axis, mapped to global image coords.
                        # The projection axis passes through image_center_x, image_center_y.
                        # A point on the projection axis at distance p_val from center:
                        proj_axis_pt_x = image_center_x + p_val * cos_theta
                        proj_axis_pt_y = image_center_y + p_val * sin_theta
                        # The ray is the line passing through (proj_axis_pt_x, proj_axis_pt_y)
                        # with direction (-sin_theta, cos_theta).
                        ray_origin_x_global = proj_axis_pt_x
                        ray_origin_y_global = proj_axis_pt_y

                        w_geom = self._calculate_ray_pixel_intersection_length(
                            c_idx, c_idx + pixel_width, # pixel x_min, x_max
                            r_idx, r_idx + pixel_height, # pixel y_min, y_max
                            ray_origin_x_global, ray_origin_y_global,
                            ray_dir_x, ray_dir_y
                        )
                        # --- End of Siddon-like w_geom calculation ---

                        if w_geom > 0:
                            if self.lineshape_model and lineshape_func:
                                for offset_idx, offset in enumerate(lineshape_bin_axis):
                                    target_bin_idx = k_bin_idx + offset
                                    if 0 <= target_bin_idx < self.num_bins_per_projection:
                                        lineshape_val = lineshape_values_centered[offset_idx]
                                        global_row_idx = angle_idx * self.num_bins_per_projection + target_bin_idx
                                        A[global_row_idx, pixel_flat_idx] += w_geom * lineshape_val
                            else: # No lineshape model
                                global_row_idx = angle_idx * self.num_bins_per_projection + k_bin_idx
                                A[global_row_idx, pixel_flat_idx] += w_geom

                elif self.projector_type == 'nearest':
                    # Project pixel center onto the projection axis
                    p_coord_of_pixel_center = center_x_rel * cos_theta + center_y_rel * sin_theta

                    # Map projected coordinate 'p' to a central bin index
                    p_shifted = p_coord_of_pixel_center - p_min
                    center_bin_float = p_shifted / bin_width
                    center_bin_idx = int(np.floor(center_bin_float))

                    if 0 <= center_bin_idx < self.num_bins_per_projection:
                        w_geom = 1.0 # Standard weight for nearest neighbor
                        if self.lineshape_model and lineshape_func:
                            for offset_idx, offset in enumerate(lineshape_bin_axis):
                                target_bin_idx = center_bin_idx + offset
                                if 0 <= target_bin_idx < self.num_bins_per_projection:
                                    lineshape_val = lineshape_values_centered[offset_idx]
                                    global_row_idx = angle_idx * self.num_bins_per_projection + target_bin_idx
                                    A[global_row_idx, pixel_flat_idx] += w_geom * lineshape_val
                        else: # No lineshape model
                            global_row_idx = angle_idx * self.num_bins_per_projection + center_bin_idx
                            A[global_row_idx, pixel_flat_idx] += w_geom

    row_norms_sq = np.sum(A**2, axis=1)
    # Handle cases where a row norm is zero to avoid division by zero later in ART update
    # This can happen if no pixel center projects into a particular bin for a given angle.
    if np.any(row_norms_sq == 0):
        print("Warning: Some rows in the system matrix have zero norm. "
              "This might indicate issues with projection geometry or binning.")

    return A, row_norms_sq

  def reconstruct(self) -> np.ndarray:
    """
    Performs ART reconstruction using the Kaczmarz row-action method.
    """
    image_2d = self._initialize_image() # (height, width) or (num_pixels_y, num_pixels_x)
    image_flat = image_2d.flatten().astype(np.float64)

    A, A_row_norms_sq = self._initialize_system_matrix()

    epsilon = 1e-9 # Small number to compare against for zero norm

    for iteration in range(self.num_iterations):
      print(f"Iteration {iteration + 1}/{self.num_iterations}")
      for global_row_idx in range(A.shape[0]):
        A_row = A[global_row_idx]
        norm_sq = A_row_norms_sq[global_row_idx]

        if norm_sq < epsilon:
          continue # Skip this projection if norm is effectively zero

        angle_idx = global_row_idx // self.num_bins_per_projection
        bin_idx = global_row_idx % self.num_bins_per_projection

        measured_val = self.projection_data[angle_idx, bin_idx]
        predicted_val = np.dot(A_row, image_flat)
        residual = measured_val - predicted_val

        image_flat += self.relaxation_param * (residual / norm_sq) * A_row

      # --- Regularization and Constraints (to be implemented) ---
      # if self.regularization_weight > 0.0:
      #   image_flat = self._apply_tikhonov_regularization(image_flat) # Placeholder
      # if self.non_negativity:
      #   image_flat = self._enforce_non_negativity(image_flat) # Placeholder
      # --- Regularization and Constraints ---
      if self.regularization_weight > 0.0:
        image_flat = self._apply_tikhonov_regularization(image_flat)
      if self.non_negativity:
        image_flat = self._enforce_non_negativity(image_flat)

    reconstructed_image = image_flat.reshape(self.grid_size)
    return reconstructed_image

  def _apply_tikhonov_regularization(self, image_flat: np.ndarray) -> np.ndarray:
    """
    Applies simplified Tikhonov regularization to the image.
    image_new = image_old / (1 + lambda)
    where lambda is self.regularization_weight.
    This is a form of damping, assuming the regularization term is ||Lx||^2
    and L is the identity matrix, and the problem is Ax=b, the update comes from
    (A^T A + lambda*I)x = A^T b. If applied iteratively, it simplifies.
    Here, we use a very direct damping factor on the image itself.
    A more common iterative Tikhonov might be:
    x_k+1 = x_k - alpha * (grad_data_fidelity + lambda * grad_regularizer)
    The chosen form is simpler and acts as a post-iteration damping.
    """
    if self.regularization_weight <= 0: # Should not be called if weight is not positive
        return image_flat
    return image_flat / (1.0 + self.regularization_weight)

  def _enforce_non_negativity(self, image_flat: np.ndarray) -> np.ndarray:
    """
    Enforces non-negativity constraint on the image.
    Sets all negative pixel values to zero.
    """
    image_flat[image_flat < 0] = 0
    return image_flat
