## Photoacoustic Tomography

*   **Enhancement Idea 1 (k-Wave Integration):** Provide an option to use the k-Wave toolbox (if installed by the user) for highly accurate forward and adjoint operators. This would require a wrapper around k-Wave's Python bindings or calling its command-line tools.
*   **Enhancement Idea 2 (Analytical Operators):** Implement analytical forward/adjoint operators for specific geometries like circular or linear arrays, which can be faster than full numerical solutions for those cases.
*   **Enhancement Idea 3 (Time Domain vs. Frequency Domain Data):** The current placeholder assumes time-domain sensor data. Consider if frequency-domain data processing is also relevant for some PAT reconstruction approaches.
*   **Enhancement Idea 4 (Multi-wavelength PAT):** For spectral photoacoustic imaging (sPAT), the operator might need to handle data from multiple illumination wavelengths to estimate chromophore concentrations. This could be a more advanced feature.

## Terahertz Imaging

*   **Enhancement Idea 1 (Specific THz Modes):** Implement operators for specific THz imaging modes, such as:
    *   `THzRadonTransformOperator` for CT based on transmission/absorption.
    *   `THzPulsedOperator` for time-domain reflection/transmission imaging, possibly including deconvolution.
    *   `THzNearFieldOperator` for near-field THz imaging.
*   **Enhancement Idea 2 (Frequency-Domain Processing):** Many THz systems (especially THz-TDS) acquire data that is naturally processed in the frequency domain. Operators could directly work with spectral data (amplitude and phase).
*   **Enhancement Idea 3 (Material Parameter Estimation):** Extend operators/reconstructors to estimate material parameters (e.g., complex refractive index, permittivity) from THz signals, not just an 'image'.
*   **Enhancement Idea 4 (Polarimetric THz Imaging):** Support for THz systems that measure polarization states, which can provide additional contrast mechanisms.

## Infrared Thermography

*   **Enhancement Idea 1 (Heat Equation Solvers):** Implement more realistic forward operators based on numerical solutions (Finite Difference, Finite Element) of the Bioheat or Pennes' bioheat equation for biological tissues, or standard heat equation for materials. This would require defining thermal properties (conductivity, capacity, density).
*   **Enhancement Idea 2 (Specific Active Thermography Modes):**
    *   `PulsedThermographyOperator`: Model the response to a Dirac-like heat pulse.
    *   `LockInThermographyOperator`: Model the response to a sinusoidal heat input, possibly operating in the frequency domain.
*   **Enhancement Idea 3 (Analytical Models):** For simple geometries (e.g., 1D heat flow, layered materials), implement analytical solutions (e.g., Green's functions) as operators.
*   **Enhancement Idea 4 (Defect Parameterization):** Instead of reconstructing a full subsurface map, the 'image' could be a set of parameters describing discrete defects (e.g., location, size, depth, thermal resistance). The operator would then map these parameters to surface temperature.
*   **Enhancement Idea 5 (Multi-frequency Lock-in):** Support for lock-in thermography data acquired at multiple modulation frequencies.

## Microwave Imaging

*   **Enhancement Idea 1 (Non-Linear Solvers):** Implement iterative non-linear solvers like BIM, DBIM, or CSI. This would require a forward solver for the full wave equation (e.g., using Finite Difference Time Domain - FDTD, or Method of Moments - MoM) and its corresponding adjoint.
*   **Enhancement Idea 2 (Specific Antenna Array Geometries):** Provide tools or operators tailored for common antenna array configurations (e.g., circular, planar) and data acquisition schemes (multi-static, multi-view).
*   **Enhancement Idea 3 (Frequency Hopping/Multi-Frequency MWI):** Support for MWI systems that use multiple frequencies to improve resolution and stability. The operator would need to handle data from different frequencies.
*   **Enhancement Idea 4 (Quantitative Imaging):** Focus on reconstructing quantitative dielectric property values (complex permittivity) rather than just qualitative contrast maps.
*   **Enhancement Idea 5 (Integration with EM Solvers):** Allow integration with external electromagnetic simulation tools (e.g., via Python APIs if available) to act as the forward/adjoint operator for advanced users.

## Hyperspectral Imaging

*   **Enhancement Idea 1 (Specific HSI System Operators):**
    *   `CASSISimulatorOperator`: Model a CASSI system, including coded aperture effects, prism/grating dispersion, and detector integration. This would be more complex than a generic sensing matrix.
    *   `PushbroomHSIOperator`: Model line-scanning HSI systems, potentially including spatial and spectral PSFs.
*   **Enhancement Idea 2 (Advanced Sparsity Regularizers):** Implement regularizers tailored for HSI, such as:
    *   *Total Variation 3D (TV3D)* or *Anisotropic Spatial-Spectral TV*.
    *   *Sparsity in learned dictionaries* (Dictionary Learning).
    *   *Low-rank tensor decomposition* regularizers (e.g., Tucker, PARAFAC).
*   **Enhancement Idea 3 (Spectral Unmixing Module):** Develop a separate module or set of tools for spectral unmixing, including endmember extraction and abundance estimation algorithms. This is a distinct problem from typical inverse problem reconstruction but highly relevant to HSI.
*   **Enhancement Idea 4 (Data Preprocessing Utilities):** Add utilities for common HSI preprocessing steps like atmospheric correction (for remote sensing), noise estimation, or bad band removal.
*   **Enhancement Idea 5 (Integration with Spectral Libraries):** Allow use of standard spectral libraries (e.g., USGS, SPECCHIO) for tasks like material identification or generating realistic phantoms with known endmembers.

## Fluorescence Microscopy

*   **Enhancement Idea 1 (Realistic PSF Generation):** Implement more sophisticated PSF models (e.g., Gibson-Lanni, Born & Wolf) that can account for optical parameters like numerical aperture (NA), wavelength, refractive index mismatch.
*   **Enhancement Idea 2 (Specific Super-Resolution Operators/Reconstructors):**
    *   `SparseLocalizationOperator`: For PALM/STORM, an operator that maps a sparse grid of emitter locations and intensities to a series of blurred image frames (simulating blinking). The adjoint would project back.
    *   Reconstructors specifically for fitting PSFs or using L1-norm minimization for sparse localization.
*   **Enhancement Idea 3 (Blind Deconvolution):** Implement algorithms that can estimate both the true image and the PSF when the PSF is unknown or partially known.
*   **Enhancement Idea 4 (GPU-Accelerated Convolution):** Ensure convolution operations are efficiently implemented on GPU, as deconvolution can be computationally intensive. (PyTorch's convNd is already good).
*   **Enhancement Idea 5 (Handling Different Noise Models):** Extend reconstructors to handle Poisson noise (common in low-light microscopy) or mixed Poisson-Gaussian noise, beyond simple L2 data fidelity.

# Future Modalities TODO (User Request)

## Structured Illumination Microscopy (SIM)
*   **Description:** A super-resolution microscopy technique using patterned illumination to enhance resolution beyond the diffraction limit.
*   **Regularization:**
    *   L1: Used in super-resolution SIM (SR-SIM) to exploit sparsity in high-frequency components during reconstruction.
    *   L2: Applied in Fourier domain reconstruction (e.g., generalized Wiener filtering) to stabilize solutions against noise.
    *   TV: Employed in regularization-based iterative optimization to preserve edges and reduce noise in SR-SIM.
*   **Rationale:** SIM’s reconstruction involves solving inverse problems with structured illumination patterns, where L1/TV and L2 improve resolution and robustness.
*   **Example:** SR-SIM uses regularization-based iterative optimization for robust reconstruction under Gaussian or Poisson noise.

## X-ray Diffraction Imaging
*   **Description:** Uses X-ray diffraction patterns to image crystalline structures, often in materials science or structural biology.
*   **Regularization:**
    *   L1: Applied in phase retrieval problems (e.g., ptychography) to exploit sparsity in electron density or wavelet domains.
    *   L2: Used in iterative phase retrieval (e.g., Gerchberg-Saxton algorithms) to stabilize solutions.
    *   TV: Enhances edge preservation in reconstructed diffraction images.
*   **Rationale:** Phase retrieval in diffraction imaging is an ill-posed inverse problem, where L1/TV promotes sparsity, and L2 reduces noise.
*   **Example:** Fourier ptychography in X-ray diffraction uses L1 regularization for phase recovery.

## Electrical Impedance Tomography (EIT)
*   **Description:** Reconstructs internal conductivity distributions from surface electrical measurements, used in medical and industrial imaging.
*   **Regularization:**
    *   L1: Promotes sparse conductivity changes, useful in dynamic EIT or sparse imaging scenarios.
    *   L2: Common in Tikhonov regularization to smooth conductivity maps and stabilize the ill-posed inverse problem.
    *   TV: Widely used to preserve sharp boundaries in conductivity distributions.
*   **Rationale:** EIT’s nonlinear inverse problem benefits from L1/TV for sparsity and edge preservation and L2 for smoothness.
*   **Example:** TV regularization is used in EIT to reconstruct sharp conductivity contrasts in medical imaging.

## Diffuse Optical Tomography (DOT)
*   **Description:** Uses near-infrared light to image optical properties (e.g., absorption, scattering) in tissues, often for brain or breast imaging.
*   **Regularization:**
    *   L1: Applied in sparse reconstruction to model localized absorbers or scatterers.
    *   L2: Used to stabilize reconstructions in the presence of noise and ill-posedness.
    *   TV: Enhances edge-preserving reconstruction of optical property maps.
*   **Rationale:** DOT’s ill-posed inverse problem benefits from L1/TV for sparse features and L2 for noise reduction.
*   **Example:** DOT reconstruction often uses TV to resolve sharp boundaries in tissue optical properties.
