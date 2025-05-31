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
