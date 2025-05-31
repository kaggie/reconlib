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
*   **Description:** A super-resolution microscopy technique using patterned illumination to enhance resolution beyond the diffraction limit. It typically involves acquiring multiple images with different phases and orientations of the illumination pattern. These raw images are then processed to reconstruct a super-resolved image.
*   **Key Papers:**
    *   Gustafsson, M. G. (2000). Surpassing the lateral resolution limit by a factor of two using structured illumination microscopy. Journal of microscopy, 198(2), 82-87. (Foundational paper on SIM)
    *   Heintzmann, R., & Cremer, C. G. (1999). Laterally modulated excitation microscopy: improvement of resolution by using a diffraction grating. Optical Preamplifiers and their Applications. (Early concepts related to SIM)
*   **Important Algorithms:**
    *   **SIM Reconstruction Algorithms:** Often involve processing in Fourier space. The core idea is that the patterned illumination shifts high-frequency information (normally inaccessible) into the observable region of the optical transfer function (OTF). By acquiring multiple images with different patterns, one can separate these components and extend the effective OTF.
    *   **Generalized Wiener Filter / Iterative Methods:** Used for separating and recombining the frequency components, often with regularization to handle noise and ill-conditioning.
    *   **Parameter Estimation:** Precise knowledge of illumination pattern parameters (frequency, phase, orientation) is critical and often needs to be estimated from the data.
*   **Key Features:**
    *   Doubles resolution compared to widefield microscopy (in 2D SIM, can be more in 3D SIM).
    *   Relatively fast and compatible with live-cell imaging compared to some other super-resolution methods.
    *   Requires multiple raw images per final super-resolved image (e.g., 9-15 for 2D/3D SIM).
    *   Sensitive to artifacts if parameters are incorrect or noise is high.
*   **Regularization in SIM:**
    *   **L1:** Can be used in super-resolution SIM (SR-SIM) or non-linear SIM variants to exploit sparsity in high-frequency components or object features during reconstruction.
    *   **L2 (Wiener-like):** Applied in Fourier domain reconstruction (e.g., generalized Wiener filtering) to stabilize solutions against noise, especially when separating frequency components.
    *   **TV:** Employed in regularization-based iterative optimization (e.g., for non-linear SIM or robust reconstruction) to preserve edges and reduce noise in the final super-resolved image.
*   **Enhancement Idea 1 (Basic SIM Operator):** Model the forward process of SIM. This would involve:
    *   Defining an input high-resolution image (ground truth).
    *   Generating illumination patterns (e.g., sinusoidal stripes) with varying phase/orientation.
    *   Simulating the moiré effect: multiplying the image by the pattern.
    *   Applying a low-pass filter (representing the microscope's OTF) to get the raw SIM images.
    The operator would take the true image and output a stack of these raw images.
*   **Enhancement Idea 2 (Basic SIM Reconstructor):** Implement a simplified reconstruction algorithm. This could initially be based on the core idea of shifting and recombining components in Fourier space, perhaps with a simple Wiener filter.
*   **Enhancement Idea 3 (Pattern Generation Utilities):** Functions to generate various illumination patterns (stripes, grids) with specified frequency, phase, and orientation.
*   **Enhancement Idea 4 (Parameter Estimation Simulation):** Placeholder for simulating the effect of misestimated illumination parameters, which is a common challenge in real SIM.
*   **Rationale for Regularization:** SIM’s reconstruction involves solving inverse problems to separate and recombine frequency components that are mixed due to the patterned illumination and OTF limitations. L1/TV can promote sparsity or sharpness in the reconstructed super-resolved image, while L2 (often in Wiener filtering) helps stabilize the solution against noise, especially when dealing with division by small OTF values.
*   **Example Application:** SR-SIM often uses regularization-based iterative optimization or sophisticated Wiener filtering for robust reconstruction, particularly when dealing with low signal-to-noise ratios or non-ideal patterns.

## X-ray Diffraction Imaging
*   **Description:** Uses X-ray diffraction patterns to image crystalline structures and electron density distributions, crucial in materials science, structural biology (e.g., protein crystallography), and non-destructive testing. Phase retrieval is often a key challenge as detectors typically only measure intensity (magnitude squared of the diffraction pattern).
*   **Key Papers:**
    *   Sayre, D. (1952). Some implications of a theorem due to Shannon. Acta Crystallographica, 5(6), 843-843. (Early work on phasing)
    *   Fienup, J. R. (1982). Phase retrieval algorithms: a comparison. Applied optics, 21(15), 2758-2769. (Classic paper on phase retrieval algorithms)
    *   Rodenburg, J. M., & Faulkner, H. M. L. (2004). A phase retrieval algorithm for shifting illumination. Applied Physics Letters, 85(20), 4795-4797. (Ptychography concept)
*   **Important Algorithms:**
    *   **Phase Retrieval Algorithms:**
        *   *Gerchberg-Saxton (GS):* Iterates between real and Fourier space, applying known constraints in each domain.
        *   *Hybrid Input-Output (HIO):* An improvement over GS, often more robust.
        *   *Difference Map:* Another iterative phase retrieval algorithm.
    *   **Ptychography:** A scanning coherent diffraction imaging technique that uses multiple overlapping diffraction patterns to reconstruct both the object and the illumination probe, often providing robust phase retrieval.
    *   **Direct Methods:** Used in crystallography, leveraging statistical properties of diffraction data.
*   **Key Features:**
    *   Provides atomic or near-atomic resolution information about material structure.
    *   Sensitive to crystal lattice, orientation, strain, and electron density.
    *   **Phase Problem:** A central challenge; detectors measure intensities, so the phase of the diffracted waves is lost and must be recovered.
    *   Coherent Diffraction Imaging (CDI) techniques use coherent X-ray beams.
*   **Regularization in X-ray Diffraction Imaging:**
    *   **L1:** Applied in phase retrieval problems (e.g., ptychography, CDI) to exploit sparsity in the object's electron density, its gradient (TV-like), or in a wavelet domain. This helps in regularizing the ill-posed phase problem.
    *   **L2:** Used in iterative phase retrieval algorithms (e.g., variants of Gerchberg-Saxton or gradient descent methods for minimizing error metrics) to stabilize solutions and ensure consistency with measured magnitudes.
    *   **TV (Total Variation):** Enhances edge preservation and reduces noise in reconstructed electron density maps or object images, especially when combined with iterative phase retrieval.
*   **Enhancement Idea 1 (Basic Diffraction Operator):**
    *   Model forward process: Object (e.g., electron density) -> Fourier Transform (diffraction pattern, complex) -> Magnitude (intensity).
    *   The `op` would take a real-space object representation and output the magnitude of its Fourier transform.
    *   The `op_adj` is non-trivial due to the loss of phase. A simple adjoint might involve taking the measured magnitudes, possibly embedding them with zero or random phase, and inverse Fourier transforming.
*   **Enhancement Idea 2 (Placeholder Phase Retrieval Reconstructor):**
    *   A very basic reconstructor might iterate between applying Fourier domain magnitude constraints (from measured diffraction intensities) and real-space constraints (e.g., support, non-negativity) using the operator's `op` (for magnitude) and a modified `op_adj` (for inverse transform with estimated phase). This would be a highly simplified Gerchberg-Saxton like approach.
*   **Enhancement Idea 3 (Ptychography Operator - Advanced):** Model the ptychographic forward process, involving a scanning probe and multiple diffraction patterns. This is significantly more complex.
*   **Enhancement Idea 4 (Phantom Generation):** Utilities to create phantoms with crystalline-like structures or defined electron density maps.
*   **Rationale for Regularization:** Phase retrieval in diffraction imaging is inherently ill-posed due to the loss of Fourier phase. L1/TV regularization helps by promoting sparsity or structural priors (like sharp edges) in the object domain, guiding the algorithm towards a more plausible solution among the many that might fit the measured magnitudes. L2 regularization is often used in the error metric (e.g., difference between measured and estimated magnitudes) during iterative optimization.
*   **Example Application:** Fourier ptychography often incorporates L1 or TV regularization within its iterative updates to improve the quality and robustness of both phase and amplitude reconstruction of the object.

## Electrical Impedance Tomography (EIT)
*   **Description:** A non-invasive imaging technique that reconstructs the internal conductivity (or impedance) distribution of a body from electrical measurements made at its surface. Electrodes are attached to the surface, small currents are injected, and resulting voltages are measured.
*   **Key Papers:**
    *   Barber, D. C., & Brown, B. H. (1984). Applied potential tomography. Journal of Physics E: Scientific Instruments, 17(9), 723. (Early work and review)
    *   Cheney, M., Isaacson, D., & Newell, J. C. (1999). Electrical impedance tomography. SIAM review, 41(1), 85-101. (Mathematical overview)
    *   Adler, A., & Guardo, R. (1996). Electrical impedance tomography: regularized imaging and contrast detection. IEEE Transactions on medical imaging, 15(2), 170-179.
*   **Important Algorithms:**
    *   **Linearized Reconstruction:** Based on a sensitivity matrix (Jacobian) derived from a forward model (e.g., Complete Electrode Model). Common algorithms include Filtered Back-Projection (FBP-like), Tikhonov regularization.
    *   **Iterative Non-linear Reconstruction:** Algorithms like Gauss-Newton, Levenberg-Marquardt, or iteratively reweighted least squares that solve the full non-linear inverse problem. These require repeated solutions of the forward problem.
    *   **Regularization Methods:** Tikhonov (L2), Total Variation (TV), Laplace regularization are crucial due to the ill-posed nature of EIT.
    *   **Difference Imaging:** Reconstructing changes in conductivity over time, often more robust than absolute imaging.
*   **Key Features:**
    *   Non-invasive, portable, relatively low-cost.
    *   No ionizing radiation.
    *   Used for monitoring physiological functions (e.g., lung ventilation, gastric emptying, brain activity) and industrial process tomography.
    *   Highly ill-posed inverse problem: small changes in internal conductivity can lead to very small changes in surface measurements, making it sensitive to noise and modeling errors.
    *   Spatial resolution is generally low compared to other tomographic modalities.
*   **Regularization in EIT:**
    *   **L1:** Promotes sparse conductivity changes, useful in dynamic EIT (imaging changes over time) or when expecting localized conductivity anomalies.
    *   **L2 (Tikhonov):** Most common for stabilizing the ill-posed inverse problem by penalizing large variations in the conductivity map, leading to smoother solutions.
    *   **TV (Total Variation):** Widely used to preserve sharp boundaries between regions of different conductivity, which is often desired in medical images. It can help counteract the smoothing effect of L2 regularization.
*   **Enhancement Idea 1 (Basic EIT Operator - Linearized):**
    *   The core of many EIT reconstructions is a sensitivity matrix `J` (Jacobian), such that `delta_v = J * delta_sigma`, where `delta_v` are changes in boundary voltages and `delta_sigma` are changes in conductivity.
    *   The `op` could take a `delta_sigma` map (flattened) and multiply by a placeholder `J` to get `delta_v`.
    *   The `op_adj` would multiply by `J.T`.
    *   Generating a realistic `J` is complex (requires FEM solver for the forward EIT problem). The placeholder `J` would be random or very simplified.
*   **Enhancement Idea 2 (Placeholder EIT Reconstructor):**
    *   A simple reconstructor could implement a regularized least-squares solution, e.g., `delta_sigma_recon = (J.T J + lambda I)^-1 J.T delta_v_measured`.
    *   Alternatively, use `ProximalGradientReconstructor` with the above operator and a TV or L2 regularizer.
*   **Enhancement Idea 3 (EIDORS Integration - Advanced):** EIDORS is an open-source software suite for EIT. Consider ways to interface with it for more realistic forward modeling or reconstruction if feasible.
*   **Enhancement Idea 4 (Phantom Generation):** Utilities to create phantoms with defined conductivity regions (e.g., lungs, heart within a torso-like shape).
*   **Rationale for Regularization:** EIT is severely ill-posed. L2 regularization is almost always needed to get a stable, smooth solution. L1/TV are used to introduce sparsity or preserve edges, which is important because L2 tends to blur boundaries. The choice depends on the expected nature of conductivity changes.
*   **Example Application:** TV regularization is often favored in medical EIT for reconstructing images of lung aeration or detecting organ boundaries, as it can better preserve edges than pure L2 regularization.

## Diffuse Optical Tomography (DOT)
*   **Description:** Uses near-infrared light to image optical properties (e.g., absorption, scattering) in tissues, often for brain or breast imaging.
*   **Key Papers:**
    *   Arridge, S. R. (1999). Optical tomography in medical imaging. Inverse problems, 15(2), R41. (Comprehensive review of DOT physics and maths)
    *   Boas, D. A., Brooks, D. H., Miller, E. L., DiMarzio, C. A., Kilmer, M., Gaudette, R. J., & Zhang, Q. (2001). Imaging the body with diffuse optical tomography. IEEE signal processing magazine, 18(6), 57-75.
    *   Gibson, A. P., Hebden, J. C., & Arridge, S. R. (2005). Recent advances in diffuse optical imaging. Physics in medicine & biology, 50(4), R1.
*   **Important Algorithms:**
    *   **Forward Model Solvers:** Based on approximations to the Radiative Transfer Equation (RTE), most commonly the Diffusion Equation (DE). Finite Element Method (FEM) is often used to solve the DE for complex geometries.
    *   **Inverse Problem Solvers:**
        *   *Linearized Methods:* Using a sensitivity matrix (Jacobian) relating changes in optical properties to changes in measurements. Iterative methods like ART, SART, or regularized least-squares are used.
        *   *Non-linear Iterative Methods:* Gauss-Newton, Levenberg-Marquardt, gradient descent, that iteratively update optical property estimates by minimizing the difference between measured and predicted data.
    *   **Regularization:** Tikhonov (L2), Total Variation (TV), L1-norm (for sparsity) are essential due to the highly ill-posed and diffuse nature of DOT.
*   **Key Features:**
    *   Non-invasive, uses non-ionizing NIR light.
    *   Can measure functional information (e.g., hemoglobin concentration changes related to brain activity or tumor angiogenesis).
    *   Relatively low cost and potential for portable systems.
    *   Highly scattering nature of light in tissue leads to a very ill-posed inverse problem and low spatial resolution compared to X-ray CT or MRI.
    *   Different data types: Continuous Wave (CW), Frequency Domain (FD), Time Domain (TD), each providing different information content and complexity.
*   **Regularization in DOT:**
    *   **L1:** Applied in sparse reconstruction to model localized absorbers or scatterers, or to find sparse changes in optical properties (e.g., in functional brain imaging).
    *   **L2 (Tikhonov):** Commonly used to stabilize reconstructions in the presence of noise and the inherent ill-posedness of the inverse problem, typically promoting smoother solutions.
    *   **TV (Total Variation):** Enhances edge-preserving reconstruction of optical property maps, useful for delineating regions with different optical characteristics.
*   **Enhancement Idea 1 (Basic DOT Operator - Linearized):**
    *   Similar to EIT, a linearized DOT model uses a sensitivity matrix (Jacobian `J`) which relates changes in optical properties (`delta_mu_a`, `delta_mu_s'`) to changes in measured boundary data (e.g., log-amplitude attenuation, phase shift).
    *   `delta_y = J @ delta_mu_flat`.
    *   The `op` would multiply a flattened `delta_mu` map by a placeholder `J`.
    *   The `op_adj` would be `J.T`.
    *   A realistic `J` requires solving the Diffusion Equation (e.g., via FEM). Placeholder `J` would be random.
*   **Enhancement Idea 2 (Placeholder DOT Reconstructor):**
    *   Implement regularized least-squares or use `ProximalGradientReconstructor` with the linearized operator and TV/L2 regularization to reconstruct `delta_mu_a` and/or `delta_mu_s'`.
*   **Enhancement Idea 3 (NIRFAST Integration - Advanced):** NIRFAST is an open-source package for DOT forward modeling and reconstruction. Explore possibilities for interfacing.
*   **Enhancement Idea 4 (Phantom Generation):** Utilities for creating 2D/3D phantoms with regions of varying absorption (`mu_a`) and scattering (`mu_s'`) coefficients.
*   **Enhancement Idea 5 (Multi-wavelength DOT):** Extend to handle data from multiple wavelengths to allow for spectroscopic DOT (estimating concentrations of chromophores like oxy- and deoxy-hemoglobin).
*   **Rationale for Regularization:** DOT is severely ill-posed due to strong light scattering. L2 regularization is essential for stability. L1/TV are valuable for reconstructing sparse or piece-wise constant changes in optical properties, improving spatial localization and reducing artifacts.
*   **Example Application:** DOT reconstruction for functional brain imaging often uses TV or L1 regularization to better localize activated regions, which appear as changes in absorption due to neurovascular coupling.
