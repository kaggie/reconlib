# TODO: Extend Voronoi-Based PET Reconstruction to 3D

This document outlines the necessary steps and considerations for extending the current 2D Voronoi-based PET reconstruction framework (`VoronoiPETReconstructor2D`) to support 3D PET data.

## Core Requirements for 3D Extension:

1.  **3D Voronoi Tessellation Integration**:
    *   Utilize `reconlib.voronoi.delaunay_3d.delaunay_triangulation_3d` to get the 3D Delaunay tetrahedralization of the generator points.
    *   Employ `reconlib.voronoi.voronoi_from_delaunay.construct_voronoi_polyhedra_3d` to obtain the geometric definition of 3D Voronoi cells (polyhedra, typically defined by their faces, which are lists of vertex indices).

2.  **3D Geometric Helper Functions**:
    *   **`_compute_lor_cell_intersection_3d(lor_p1_3d, lor_p2_3d, cell_polyhedron_faces, unique_voronoi_vertices_3d)`**: This is the most critical new geometric component. It needs to calculate the intersection length of a 3D LOR segment with a 3D convex polyhedron (the Voronoi cell).
        *   This will likely involve robust algorithms for line segment-polyhedron intersection, potentially by checking intersections with each face of the polyhedron.
        *   Careful handling of geometric predicates, edge cases, and numerical stability in 3D will be essential.
    *   **Cell Validation**: Adapt `_validate_voronoi_cells_3d` to use `ConvexHull(cell_vertices).volume` (from `reconlib.voronoi.geometry_core`) to check for degenerate 3D cells (near-zero volume).
    *   **Generator Point Validation**: Adapt `_validate_generator_points_3d` to check for coplanarity of generator points using `ConvexHull(points).volume`.

3.  **3D System Matrix Calculation**:
    *   The `_compute_system_matrix_3d` method will need to iterate through 3D LORs and 3D Voronoi cells, using the new `_compute_lor_cell_intersection_3d`.
    *   LOR definition in 3D will require appropriate parameterization (e.g., from sinogram indices if using cylindrical PET geometry, or direct LOR endpoints).

4.  **Adapt MLEM Algorithm**:
    *   The core MLEM update loop structure in `VoronoiPETReconstructor3D` will remain similar to the 2D version, but will operate on 3D data structures (e.g., 3D activity map represented by activities per 3D Voronoi cell).
    *   Forward and back-projection helpers (`_forward_project_3d`, `_back_project_3d`) will use the new 3D system matrix.

5.  **Unit Testing for 3D**:
    *   Develop new test cases specifically for 3D:
        *   Test 3D generator point validation (e.g., with coplanar points).
        *   Test `_compute_lor_cell_intersection_3d` with simple 3D geometries (e.g., LOR through a cube).
        *   Test the full 3D MLEM reconstruction with a simple 3D phantom and a small number of 3D Voronoi cells.

## Implementation Strategy:

*   Create a new class `VoronoiPETReconstructor3D`.
*   Share common logic with `VoronoiPETReconstructor2D` if possible, perhaps through a base class or helper functions, but the geometric aspects will be dimension-specific.
*   Focus heavily on robust 3D geometric intersection routines. Consider existing geometry libraries or implement carefully.

## Potential Challenges:

*   **Computational Complexity**: 3D LOR-polyhedron intersections are significantly more complex and computationally intensive than their 2D counterparts. Efficient algorithms will be crucial.
*   **Numerical Stability**: 3D geometric predicates and calculations are more prone to floating-point precision issues.
*   **Visualization**: Debugging and visualizing 3D Voronoi cells and LOR intersections will require appropriate tools.

## Future Considerations (Post-3D Implementation):

*   Optimization of system matrix calculation (e.g., using symmetries, on-the-fly calculation, GPU acceleration).
*   Support for more complex LOR definitions (e.g., list-mode data).
*   Integration of advanced PET corrections (attenuation, scatter, randoms) more deeply into the Voronoi system model if not handled by pre-correcting input sinograms.
