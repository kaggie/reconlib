# X-ray Diffraction Imaging

## Description
Uses X-ray diffraction patterns to image crystalline structures and electron density distributions, crucial in materials science, structural biology (e.g., protein crystallography), and non-destructive testing. Phase retrieval is often a key challenge as detectors typically only measure intensity (magnitude squared of the diffraction pattern).

## Key Papers
*   Sayre, D. (1952). Some implications of a theorem due to Shannon. Acta Crystallographica, 5(6), 843-843. (Early work on phasing)
*   Fienup, J. R. (1982). Phase retrieval algorithms: a comparison. Applied optics, 21(15), 2758-2769. (Classic paper on phase retrieval algorithms)
*   Rodenburg, J. M., & Faulkner, H. M. L. (2004). A phase retrieval algorithm for shifting illumination. Applied Physics Letters, 85(20), 4795-4797. (Ptychography concept)

## Important Algorithms
*   **Phase Retrieval Algorithms:**
    *   *Gerchberg-Saxton (GS):* Iterates between real and Fourier space, applying known constraints in each domain.
    *   *Hybrid Input-Output (HIO):* An improvement over GS, often more robust.
    *   *Difference Map:* Another iterative phase retrieval algorithm.
*   **Ptychography:** A scanning coherent diffraction imaging technique that uses multiple overlapping diffraction patterns to reconstruct both the object and the illumination probe, often providing robust phase retrieval.
*   **Direct Methods:** Used in crystallography, leveraging statistical properties of diffraction data.

## Key Features
*   Provides atomic or near-atomic resolution information about material structure.
*   Sensitive to crystal lattice, orientation, strain, and electron density.
*   **Phase Problem:** A central challenge; detectors measure intensities, so the phase of the diffracted waves is lost and must be recovered.
*   Coherent Diffraction Imaging (CDI) techniques use coherent X-ray beams.

## Regularization in X-ray Diffraction Imaging
*   **L1:** Applied in phase retrieval problems (e.g., ptychography, CDI) to exploit sparsity in the object's electron density, its gradient (TV-like), or in a wavelet domain. This helps in regularizing the ill-posed phase problem.
*   **L2:** Used in iterative phase retrieval algorithms (e.g., variants of Gerchberg-Saxton or gradient descent methods for minimizing error metrics) to stabilize solutions and ensure consistency with measured magnitudes.
*   **TV (Total Variation):** Enhances edge preservation and reduces noise in reconstructed electron density maps or object images, especially when combined with iterative phase retrieval.

## Enhancement Ideas
*   **Enhancement Idea 1 (Basic Diffraction Operator):**
    *   Model forward process: Object (e.g., electron density) -> Fourier Transform (diffraction pattern, complex) -> Magnitude (intensity).
    *   The `op` would take a real-space object representation and output the magnitude of its Fourier transform.
    *   The `op_adj` is non-trivial due to the loss of phase. A simple adjoint might involve taking the measured magnitudes, possibly embedding them with zero or random phase, and inverse Fourier transforming.
*   **Enhancement Idea 2 (Placeholder Phase Retrieval Reconstructor):**
    *   A very basic reconstructor might iterate between applying Fourier domain magnitude constraints (from measured diffraction intensities) and real-space constraints (e.g., support, non-negativity) using the operator's `op` (for magnitude) and a modified `op_adj` (for inverse transform with estimated phase). This would be a highly simplified Gerchberg-Saxton like approach.
*   **Enhancement Idea 3 (Ptychography Operator - Advanced):** Model the ptychographic forward process, involving a scanning probe and multiple diffraction patterns. This is significantly more complex.
*   **Enhancement Idea 4 (Phantom Generation):** Utilities to create phantoms with crystalline-like structures or defined electron density maps.

## Rationale for Regularization
Phase retrieval in diffraction imaging is inherently ill-posed due to the loss of Fourier phase. L1/TV regularization helps by promoting sparsity or structural priors (like sharp edges) in the object domain, guiding the algorithm towards a more plausible solution among the many that might fit the measured magnitudes. L2 regularization is often used in the error metric (e.g., difference between measured and estimated magnitudes) during iterative optimization.

## Example Application
Fourier ptychography often incorporates L1 or TV regularization within its iterative updates to improve the quality and robustness of both phase and amplitude reconstruction of the object.
