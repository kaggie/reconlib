## Microwave Imaging

*   **Key Papers:**
    *   [Example Paper 1 URL] - Briefly describe paper
*   **Important Algorithms:**
    *   Algorithm A: Description
*   **Key Features:**
    *   Feature X: Description
## Hyperspectral Imaging

*   **Key Papers:**
    *   [Example Paper 1 URL] - Briefly describe paper
*   **Important Algorithms:**
    *   Algorithm A: Description
*   **Key Features:**
    *   Feature X: Description
## Fluorescence Microscopy

*   **Key Papers:**
    *   [Example Paper 1 URL] - Briefly describe paper
*   **Important Algorithms:**
    *   Algorithm A: Description
*   **Key Features:**
    *   Feature X: Description

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
