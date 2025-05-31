# Hyperspectral Imaging Enhancements

### Enhancement Idea 1 (Specific HSI System Operators)
*   `CASSISimulatorOperator`: Model a CASSI system, including coded aperture effects, prism/grating dispersion, and detector integration. This would be more complex than a generic sensing matrix.
*   `PushbroomHSIOperator`: Model line-scanning HSI systems, potentially including spatial and spectral PSFs.

### Enhancement Idea 2 (Advanced Sparsity Regularizers)
Implement regularizers tailored for HSI, such as:
*   *Total Variation 3D (TV3D)* or *Anisotropic Spatial-Spectral TV*.
*   *Sparsity in learned dictionaries* (Dictionary Learning).
*   *Low-rank tensor decomposition* regularizers (e.g., Tucker, PARAFAC).

### Enhancement Idea 3 (Spectral Unmixing Module)
Develop a separate module or set of tools for spectral unmixing, including endmember extraction and abundance estimation algorithms. This is a distinct problem from typical inverse problem reconstruction but highly relevant to HSI.

### Enhancement Idea 4 (Data Preprocessing Utilities)
Add utilities for common HSI preprocessing steps like atmospheric correction (for remote sensing), noise estimation, or bad band removal.

### Enhancement Idea 5 (Integration with Spectral Libraries)
Allow use of standard spectral libraries (e.g., USGS, SPECCHIO) for tasks like material identification or generating realistic phantoms with known endmembers.
