# Notes on System Matrix Calculation for Voronoi-Based PET Reconstruction

## Introduction

The system matrix `A` in PET reconstruction models the probability that a positron annihilation event occurring within a specific region of the image space (here, a Voronoi cell) is detected along a particular Line of Response (LOR). For Voronoi-based PET, where image regions are irregular polygons (2D) or polyhedra (3D), calculating the elements `A_ij` (probability of LOR `i` detecting an event from Voronoi cell `j`) is a key challenge.

## Core Component: `compute_lor_cell_intersection`

The most computationally intensive part of building the system matrix is the function `compute_lor_cell_intersection(lor, cell_geometry)`. This function must determine the geometric intersection (e.g., length of overlap in 2D, or volume/area of overlap if more complex factors are considered) between a given LOR and a specific Voronoi cell.

### Challenges:

1.  **Irregular Cell Geometry**: Unlike voxel grids where cells are uniform squares or cubes, Voronoi cells are arbitrary convex polygons/polyhedra. Standard ray-tracing algorithms for regular grids are not directly applicable.
2.  **Geometric Intersection Algorithms**:
    *   **2D**: Requires robust algorithms for line segment (LOR) - polygon (Voronoi cell) intersection. This involves checking intersections with all edges of the polygon and determining the segment of the LOR that lies within the polygon.
    *   **3D**: This becomes significantly more complex, requiring line segment - polyhedron intersection. This can involve ray-casting against all faces of the polyhedron or more sophisticated geometric intersection tests.
3.  **Computational Cost**: For `M` LORs and `N` Voronoi cells, `M*N` intersection calculations are needed. Given that `M` can be very large (millions of LORs) and `N` can also be substantial (thousands of cells), efficiency is paramount.
4.  **Accuracy and Numerical Stability**: Geometric calculations with floating-point numbers can suffer from precision issues. Robust algorithms are needed to handle edge cases and degenerate geometries correctly.
5.  **System Matrix Size**: The full system matrix can be enormous (`M*N`) and often cannot be stored explicitly in memory. "Matrix-free" methods (where elements are computed on-the-fly during forward/back projection) or sparse/িকালেd representations are typically necessary, especially for 3D.

### Initial Implementation Focus:

*   The initial development will focus on a **2D implementation**.
*   `compute_lor_cell_intersection_2d` will calculate the **length of the LOR segment contained within the 2D Voronoi polygon**. This length is directly proportional to the detection probability, assuming uniform activity within the cell and uniform detector sensitivity along the LOR.
*   Future extension to 3D will require developing or integrating robust line-polyhedron intersection algorithms.

## Considerations for System Matrix Elements:

Beyond geometric intersection, a complete system matrix element `A_ij` would also incorporate:
*   **Attenuation**: Correction for photon attenuation along the LOR, both from the source cell `j` to the edge of the phantom, and potentially from within the cell itself if activity distribution is non-uniform (though Voronoi assumes uniform activity per cell).
*   **Detector Efficiency/Sensitivity**: Variation in detector sensitivity across different LORs.
*   **Positron Range and Acollinearity**: Physical effects that blur the LOR.
*   **Scatter and Randoms**: While often handled as corrections to the projection data `p` or as additive terms in the model, some advanced models might try to incorporate aspects into `A`.

For the initial Voronoi-PET implementation, these additional physical factors might be simplified or addressed as pre-corrections to the sinogram data, with the system matrix primarily handling the geometric component.

## Conclusion

The development of an accurate and efficient `compute_lor_cell_intersection` routine is critical for the success of the Voronoi-based PET reconstruction method. This will be a primary focus area, starting with 2D and aiming for robustness and reasonable performance.
