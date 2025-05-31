"""
EPR reconstruction algorithms.

This module will contain algorithms for both Continuous Wave (CW) and Pulse EPR.
"""
import numpy as np

__all__ = [
    'deconvolve_cw_spectrum',
    'apply_kspace_corrections',
    'preprocess_cw_epr_data',
    'radial_recon_2d',
    'radial_recon_3d',
    'ARTReconstructor'  # Added ARTReconstructor
]

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
      - Polynomial Fitting: Fits and subtracts a low-order polynomial from each
        projection to remove baseline drifts or offsets.
      - Wavelet Methods: Utilizes wavelet decomposition to identify and remove
        baseline components.
      - Other methods: e.g., iterative morphological filtering.
  2.  Noise Reduction:
      - Savitzky-Golay Filter: A polynomial smoothing filter that can preserve
        signal features better than simple moving averages.
      - Gaussian Filter: Convolves the data with a Gaussian kernel to smooth noise.
      - Median Filter: Effective for removing salt-and-pepper noise.
  3.  Spectral Alignment (Field Shift Correction):
      - If there are drifts in the magnetic field or microwave frequency between
        projections, spectra might need to be aligned. This can be done by
        cross-correlation with a reference spectrum or by identifying and
        aligning a common peak.
  4.  Normalization:
      - To Acquisition Parameters: Normalize by number of scans, receiver gain,
        Q-factor of the resonator (if monitored), or other instrumental factors
        that might vary between projections or experiments.
      - To Total Signal Intensity: Normalize each projection by its total integral
        (sum of absolute values or sum of squares) if overall intensity variations
        are not of interest for the image contrast.
      - To a Reference Peak: If a stable reference peak exists in the spectrum
        (e.g., from a standard sample), its intensity can be used for normalization.
      - Per-Projection Normalization (as implemented below for demonstration):
        Normalizing each projection by its own maximum can be useful for
        visualizing or if relative variations within each projection are key.

  Args:
    projection_data: A 2D NumPy array where rows are projection angles and
                     columns are spectral data points.
    params: An optional dictionary of parameters to control preprocessing.
            Example: {'normalize_per_projection': True,
                      'baseline_correct_method': 'polynomial',
                      'baseline_correct_order': 2}

  Returns:
    A 2D NumPy array of preprocessed projection data.
  """
  if not isinstance(projection_data, np.ndarray):
    raise TypeError("projection_data must be a NumPy array.")
  if projection_data.ndim != 2:
    raise ValueError("projection_data must be a 2D array (angles x spectral_bins).")

  processed_data = projection_data.copy()

  if params:
    if params.get('normalize_per_projection', False):
      for i in range(processed_data.shape[0]):
        row_max = processed_data[i, :].max()
        if row_max != 0:
          processed_data[i, :] = processed_data[i, :] / row_max
        # else: row remains as is (all zeros or contains non-positives)
    # Add other preprocessing steps here based on params
    # e.g., baseline correction:
    # if params.get('baseline_correct_method') == 'polynomial':
    #   order = params.get('baseline_correct_order', 2)
    #   # ... implement polynomial baseline correction for each row ...
    #   pass

  return processed_data

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
               non_negativity: bool = True):
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

    This implementation uses a simplified geometric projection model where a
    pixel contributes to a projection bin if the projection of the pixel's
    center falls within that bin.

    **Important Simplifications and Assumptions:**
    1.  **Binary Contribution**: A pixel either contributes fully (1.0) or not at all (0.0)
        to a projection bin. This is a nearest-neighbor type assignment based on the
        projected pixel center.
    2.  **No Lineshape Convolution**: This model does NOT account for EPR lineshapes.
        In reality, each "point" in the object is an EPR spectrum, and the projection
        is a sum of these spectra. For CW-EPR with field gradients, this means each
        pixel would contribute its spectral value to multiple bins if the lineshape
        is broader than a single bin, or the projection of the pixel's spectral
        response would be convolved with the gradient field.
    3.  **Pixel Footprint**: It does not accurately model the footprint of the pixel
        (e.g., using Siddon's algorithm for exact ray-pixel intersection lengths).
    4.  **Rotation Center**: Assumed to be the center of the image grid (num_pixels_x / 2, num_pixels_y / 2).
    5.  **Projection Bin Mapping**:
        - The projection axis (onto which pixel centers are projected) is assumed
          to span from `p_min` to `p_max`.
        - `p_min` is set to `-max(self.num_pixels_x, self.num_pixels_y) / 2.0`.
        - `p_max` is set to `+max(self.num_pixels_x, self.num_pixels_y) / 2.0`.
        - This range is then divided into `self.num_bins_per_projection`.
        This choice ensures that pixels from corners of a square image can project
        into the bins. The actual field-of-view of the projection needs to match
        this assumption.

    For accurate EPR image reconstruction, especially CW-EPR with gradients, a more
    sophisticated forward projector that incorporates lineshapes and potentially
    gradient non-idealities would be required.

    Returns:
      A tuple containing:
        - A (np.ndarray): The system matrix with shape
          (total_projection_bins, num_pixels).
        - row_norms_sq (np.ndarray): 1D array of squared L2 norms of rows of A.
    """
    A = np.zeros((self.total_projection_bins, self.num_pixels), dtype=np.float64)

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

    for angle_idx, theta_deg in enumerate(self.gradient_angles):
      theta_rad = np.deg2rad(theta_deg)
      cos_theta = np.cos(theta_rad)
      sin_theta = np.sin(theta_rad)

      for r_idx in range(self.num_pixels_y):  # y-coordinate, rows
        for c_idx in range(self.num_pixels_x):  # x-coordinate, columns
          pixel_flat_idx = r_idx * self.num_pixels_x + c_idx

          # Pixel center coordinates (relative to image_center for projection)
          center_x_rel = (c_idx + 0.5) - image_center_x
          center_y_rel = (r_idx + 0.5) - image_center_y

          # Project pixel center onto the axis perpendicular to the gradient
          # Standard Radon transform projection coordinate for angle theta
          p = center_x_rel * cos_theta + center_y_rel * sin_theta

          # Map projected coordinate 'p' to a bin index
          # Shift p to be relative to p_min for bin calculation
          p_shifted = p - p_min
          bin_idx_float = p_shifted / bin_width

          # Assign to nearest bin (can be refined with interpolation for better models)
          bin_idx = int(np.floor(bin_idx_float)) # Using floor for consistency

          if 0 <= bin_idx < self.num_bins_per_projection:
            global_row_idx = angle_idx * self.num_bins_per_projection + bin_idx
            A[global_row_idx, pixel_flat_idx] = 1.0
          # else: pixel projects outside the defined projection range for this angle

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
