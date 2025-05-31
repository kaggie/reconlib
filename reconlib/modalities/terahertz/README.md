# Terahertz Imaging Enhancements

### Enhancement Idea 1 (Specific THz Modes)
Implement operators for specific THz imaging modes, such as:
*   `THzRadonTransformOperator` for CT based on transmission/absorption.
*   `THzPulsedOperator` for time-domain reflection/transmission imaging, possibly including deconvolution.
*   `THzNearFieldOperator` for near-field THz imaging.

### Enhancement Idea 2 (Frequency-Domain Processing)
Many THz systems (especially THz-TDS) acquire data that is naturally processed in the frequency domain. Operators could directly work with spectral data (amplitude and phase).

### Enhancement Idea 3 (Material Parameter Estimation)
Extend operators/reconstructors to estimate material parameters (e.g., complex refractive index, permittivity) from THz signals, not just an 'image'.

### Enhancement Idea 4 (Polarimetric THz Imaging)
Support for THz systems that measure polarization states, which can provide additional contrast mechanisms.
