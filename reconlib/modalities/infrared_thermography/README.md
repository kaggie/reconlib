# Infrared Thermography Enhancements

### Enhancement Idea 1 (Heat Equation Solvers)
Implement more realistic forward operators based on numerical solutions (Finite Difference, Finite Element) of the Bioheat or Pennes' bioheat equation for biological tissues, or standard heat equation for materials. This would require defining thermal properties (conductivity, capacity, density).

### Enhancement Idea 2 (Specific Active Thermography Modes)
*   `PulsedThermographyOperator`: Model the response to a Dirac-like heat pulse.
*   `LockInThermographyOperator`: Model the response to a sinusoidal heat input, possibly operating in the frequency domain.

### Enhancement Idea 3 (Analytical Models)
For simple geometries (e.g., 1D heat flow, layered materials), implement analytical solutions (e.g., Green's functions) as operators.

### Enhancement Idea 4 (Defect Parameterization)
Instead of reconstructing a full subsurface map, the 'image' could be a set of parameters describing discrete defects (e.g., location, size, depth, thermal resistance). The operator would then map these parameters to surface temperature.

### Enhancement Idea 5 (Multi-frequency Lock-in)
Support for lock-in thermography data acquired at multiple modulation frequencies.
