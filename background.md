## Photoacoustic Tomography

*   **Key Papers:**
    *   Wang, L. V., & Yao, J. (2016). A practical guide to photoacoustic tomography in the life sciences. Nature methods, 13(8), 627-638. (General overview and practical aspects)
    *   Xu, M., & Wang, L. V. (2006). Photoacoustic imaging in biomedicine. Review of scientific instruments, 77(4). (Fundamental principles and review)
    *   Treeby, B. E., & Cox, B. T. (2010). k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields. Journal of biomedical optics, 15(2), 021314. (k-Wave simulation toolbox)
*   **Important Algorithms:**
    *   **Universal Back-Projection (UBP):** A common analytical reconstruction algorithm for various geometries.
    *   **Time Reversal:** Uses the time-reversed recorded signals as sources to computationally backpropagate the wave.
    *   **Model-Based Reconstruction:** Iterative methods that incorporate an accurate forward model and often regularization (e.g., using k-Wave adjoint).
    *   **Fourier Domain Reconstruction:** Algorithms that operate in the k-space/frequency domain, particularly for specific sensor geometries.
*   **Key Features:**
    *   Combines optical contrast with ultrasound resolution.
    *   Scalable imaging depth and resolution.
    *   Can be implemented with various sensor geometries (linear, circular, spherical).

## Terahertz Imaging

*   **Key Papers:**
    *   Tonouchi, M. (2007). Cutting-edge terahertz technology. Nature photonics, 1(2), 97-105. (Overview of THz technology)
    *   Jepsen, P. U., Cooke, D. G., & Koch, M. (2011). Terahertz spectroscopy and imaging-Modern techniques and applications. Laser & Photonics Reviews, 5(1), 124-166. (Review of THz spectroscopy and imaging)
    *   Chan, W. L., Deibel, J., & Mittleman, D. M. (2007). Imaging with terahertz radiation. Reports on Progress in Physics, 70(8), 1325. (Focus on imaging aspects)
*   **Important Algorithms:**
    *   **THz-TDS (Time-Domain Spectroscopy) Deconvolution:** Extracting material parameters from time-resolved THz pulses.
    *   **Filtered Back-Projection (FBP):** For THz Computed Tomography (CT) if using Radon-like projections.
    *   **Iterative Reconstruction (e.g., ART, SART):** For THz CT, especially with limited data.
    *   **Compressed Sensing Algorithms:** Leveraging sparsity for undersampled data (e.g., in spatial or frequency domains).
    *   **Migration Algorithms:** Adapted from seismic imaging for THz reflection tomography.
*   **Key Features:**
    *   Non-ionizing radiation, safe for biological tissues.
    *   Sensitivity to water content, useful for moisture analysis.
    *   Penetrates many dielectric materials (plastics, ceramics, paper, textiles).
    *   Unique spectral fingerprints for many chemicals and materials.

## Infrared Thermography

*   **Key Papers:**
    *   Vavilov, V., & Maldague, X. (2001). Infrared thermography: Non-destructive testing methods. Encyclopedia of Materials: Science and Technology, 1-9. (General NDT overview)
    *   Maldague, X. P. (2001). Theory and practice of infrared technology for non-destructive testing. John Wiley & Sons. (Comprehensive book)
    *   Shepard, S. M. (2001). Advances in pulsed thermography. Thermosense XXIII, 4360, 511-515. (Pulsed thermography focus)
*   **Important Algorithms:**
    *   **Pulsed Phase Thermography (PPT):** Analysis of thermal data in the frequency domain (phase and amplitude images) to reduce noise and enhance defect detection.
    *   **Thermographic Signal Reconstruction (TSR):** Fits thermal decay curves to a polynomial model to obtain derivative images, improving defect contrast.
    *   **Principal Component Thermography (PCT):** Applies PCA to thermal image sequences to separate defect signals from background variations.
    *   **Quantitative Inverse Heat Conduction Problems (IHCP):** Model-based approaches to estimate defect properties (size, depth, thermal resistance) by solving the heat equation inversely.
*   **Key Features:**
    *   Non-contact, full-field imaging.
    *   Sensitive to subsurface defects, delaminations, voids, cracks, and material variations that affect heat flow.
    *   **Active Thermography:** Uses external heat stimulation (e.g., flash lamps, lasers, hot air) to induce thermal contrast.
        *   *Pulsed Thermography:* A short heat pulse is applied.
        *   *Lock-in (Modulated) Thermography:* Periodic heating is applied, and the phase/amplitude of the temperature response is analyzed.
        *   *Step Heating:* A prolonged heating period is applied.
    *   **Passive Thermography:** Observes temperature differences due to existing thermal gradients or self-heating components (e.g., in electronics).

## Microwave Imaging

*   **Key Papers:**
    *   Fear, E. C. (2017). Microwave imaging of the breast: A clinical reality?. IEEE Antennas and Wireless Propagation Letters, 16, 2366-2369. (Focus on breast imaging)
    *   Semnov, S. Y. (2009). Microwave tomography: review of the progress and perspectives. IEEE Antennas and Propagation Magazine, 51(1), 13-26. (Review of tomography aspects)
    *   Pastorino, M. (2010). Microwave imaging. John Wiley & Sons. (Comprehensive book)
*   **Important Algorithms:**
    *   **Born Iterative Method (BIM) / Distorted Born Iterative Method (DBIM):** Iterative methods to solve the non-linear inverse scattering problem by linearizing at each step.
    *   **Contrast Source Inversion (CSI):** Formulates the problem in terms of contrast sources and iteratively solves for them and the dielectric contrast.
    *   **Gauss-Newton Inversion (GNI):** A gradient-based optimization method applied to the non-linear problem.
    *   **Linear Sampling Method (LSM) / Factorization Method:** Qualitative methods that can determine the shape of scatterers without solving the full non-linear problem.
    *   **Adjoint-Based Optimization:** Iterative methods using the adjoint of the forward solver to compute gradients for updating dielectric properties.
*   **Key Features:**
    *   Non-ionizing radiation.
    *   Sensitive to dielectric property contrast (permittivity and conductivity), which can differ significantly between healthy and malignant tissues (e.g., in breast cancer).
    *   Can penetrate deeper into biological tissues than optical methods, but with lower resolution.
    *   Applications in medical imaging (breast, brain, bone), industrial NDT, security screening.
    *   Data acquisition typically involves an array of antennas transmitting and receiving microwave signals.
    *   Ill-posed and often non-linear inverse problem.

## Hyperspectral Imaging

*   **Key Papers:**
    *   Goetz, A. F. H., Vane, G., Solomon, J. E., & Rock, B. N. (1985). Imaging spectrometry for earth remote sensing. Science, 228(4704), 1147-1153. (Pioneering paper on HSI for remote sensing)
    *   Bioucas-Dias, J. M., Plaza, A., Camps-Valls, G., Scheunders, P., Nasrabadi, N., & Chanussot, J. (2013). Hyperspectral remote sensing data analysis and future challenges. IEEE Geoscience and Remote Sensing Magazine, 1(2), 6-36. (Comprehensive review)
    *   Wagadarikar, A., John, R., Willett, R., & Brady, D. (2009). Single disperser design for coded aperture snapshot spectral imaging. Applied optics, 48(10), B33-B40. (Example of compressed HSI - CASSI)
*   **Important Algorithms:**
    *   **Compressed Sensing HSI Reconstruction:** Algorithms like TwIST, FISTA, ADMM applied with sparsity-promoting regularizers (e.g., spatial TV, spectral TV, wavelet sparsity, dictionary learning) to recover the HSI cube from undersampled measurements.
    *   **Spectral Unmixing:**
        *   *Endmember Extraction Algorithms (EEA):* Pixel Purity Index (PPI), N-FINDR, Vertex Component Analysis (VCA).
        *   *Abundance Estimation:* Fully Constrained Least Squares (FCLS), Non-negative Matrix Factorization (NMF).
    *   **Pansharpening:** Fusing a low-resolution HSI cube with a high-resolution panchromatic (single band) image to get a high spatial and spectral resolution HSI cube.
    *   **Denoising Algorithms:** PCA-based denoising, BM3D/BM4D variants for HSI (e.g., HSI-BM3D).
*   **Key Features:**
    *   Acquires images in hundreds of contiguous narrow spectral bands, providing detailed spectral information for each pixel.
    *   Enables material identification, classification, and quantification based on unique spectral signatures.
    *   Applications: Remote sensing (agriculture, geology, environmental monitoring), medical imaging (disease detection, surgical guidance), food quality control, art analysis.
    *   Data is a 3D cube (2 spatial dimensions, 1 spectral dimension).
    *   Challenges: High data dimensionality ('curse of dimensionality'), data redundancy, noise, atmospheric correction (for remote sensing).
    *   Compressed HSI (e.g., CASSI - Coded Aperture Snapshot Spectral Imaging) aims to reduce acquisition time/data volume by acquiring encoded projections.

## Fluorescence Microscopy

*   **Key Papers:**
    *   Betzig, E., Patterson, G. H., Sougrat, R., Lindwasser, O. W., Olenych, S., Bonifacino, J. S., ... & Hess, H. F. (2006). Imaging intracellular fluorescent proteins at nanometer resolution. Science, 313(5793), 1642-1645. (PALM - a super-resolution technique)
    *   Rust, M. J., Bates, M., & Zhuang, X. (2006). Sub-diffraction-limit imaging by stochastic optical reconstruction microscopy (STORM). Nature methods, 3(10), 793-796. (STORM - another super-resolution technique)
    *   Sibarita, J. B. (2005). Deconvolution microscopy. Microscopy techniques, 95-109. (Overview of deconvolution)
*   **Important Algorithms:**
    *   **Deconvolution Algorithms:**
        *   *Richardson-Lucy (RL):* Iterative algorithm based on Bayes' theorem, non-negative.
        *   *Iterative Constrained Tikhonov-Miller (ICTM):* Linear iterative method with Tikhonov regularization.
        *   *Wiener Filter:* Linear filter, optimal in mean square error sense if signal and noise statistics are known.
    *   **Super-Resolution Localization Algorithms (for PALM, STORM, etc.):**
        *   *Gaussian Fitting:* Fit PSF of isolated emitters to a 2D/3D Gaussian to find their precise location.
        *   *Centroid Calculation:* Simpler localization by finding the center of mass of emitter spots.
        *   *Iterative Sparsity-Based Reconstruction (e.g., L1):* Reconstruct an image of sparse emitter locations from many frames.
    *   **Image Restoration/Denoising:** Non-local means, BM3D, deep learning based denoisers.
*   **Key Features:**
    *   High specificity due to fluorescent labeling of specific molecules or structures.
    *   Widely used in cell biology, neuroscience, and materials science.
    *   **Diffraction Limit:** Conventional fluorescence microscopy is limited in resolution by the diffraction of light (~200-250 nm laterally).
    *   **Deconvolution Microscopy:** Computationally improves resolution and contrast by reassigning blurred light to its original location, based on knowledge of the PSF.
    *   **Super-Resolution Microscopy (Nanoscopy):** Techniques like PALM, STORM, STED, SIM that overcome the diffraction limit to achieve resolutions down to tens of nanometers. These often involve specialized optics, fluorophores, and complex image processing.
    *   Data is typically 2D or 3D images (stacks of 2D images). Time-lapse (4D) imaging is also common.

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

## Diffuse Optical Tomography (DOT)

*   **Description:** A non-invasive imaging technique that uses near-infrared (NIR) light to image optical properties (primarily absorption `mu_a` and reduced scattering `mu_s'`) of biological tissues. Light is delivered to the tissue surface via source optodes, and detected at other locations by detector optodes after it has propagated through the tissue.
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

## Photon Counting CT (PCCT) - User Requested TODO

*   **Description:** An advanced X-ray computed tomography technique that uses energy-resolving photon-counting detectors. These detectors count individual X-ray photons and measure their energy, allowing for material decomposition, noise reduction, and potentially lower radiation doses.
*   **Key Features & Challenges:**
    *   Energy discrimination capabilities (multiple energy bins).
    *   Improved contrast-to-noise ratio (CNR).
    *   Material decomposition (e.g., separating bone, soft tissue, contrast agents).
    *   Potential for K-edge imaging.
    *   Higher spatial resolution than conventional CT.
    *   Challenges: Pulse pile-up, charge sharing, detector calibration, high data rates, spectral distortion, complex image reconstruction algorithms (statistical, model-based).
*   **Regularization in PCCT:**
    *   **L1/Sparsity:** Useful for material-specific images (e.g., if a contrast agent is sparse) or for regularizing sinograms in sparse-view CT.
    *   **TV (Total Variation):** Widely used for preserving edges and reducing noise in reconstructed attenuation maps for each energy bin or in material-decomposed images.
    *   **L2 (Tikhonov):** Can be used for general smoothing and stabilizing iterative reconstructions.
    *   **Spectral Regularization:** Priors that enforce smoothness or correlations across energy bins or material basis images (e.g., spectral TV).
