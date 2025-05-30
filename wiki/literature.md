# Literature and References for ReconLib

This document lists key literature and references that have inspired or are relevant to the algorithms and techniques implemented within the `reconlib` library.

## From the Voronoi Submodule Integration

The following points were originally listed in the `LITERATURE.md` file of the Voronoi submodule, which was integrated into `reconlib`.

*Literature and References*
*(Content from reconlib/voronoi/LITERATURE.md was: "# Literature and References" - which is minimal. This section can be expanded based on the actual content if it were more detailed, or specific items from the Voronoi library's cited works.)*

Given the minimal content, key literature relevant to the Voronoi submodule would typically include:

*   **Voronoi Diagrams & Delaunay Triangulations**:
    *   Aurenhammer, F. (1991). "Voronoi diagrams—a survey of a fundamental geometric data structure." ACM Computing Surveys (CSUR), 23(3), 345-405.
    *   De Berg, M., Van Kreveld, M., Overmars, M., & Schwarzkopf, O. (2000). "Computational Geometry: Algorithms and Applications." Springer. (Chapters on Voronoi diagrams and Delaunay triangulations).
    *   Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000). "Spatial Tessellations: Concepts and Applications of Voronoi Diagrams." John Wiley & Sons.
    *   Guibas, L., & Stolfi, J. (1985). "Primitives for the manipulation of general subdivisions and the computation of Voronoi diagrams." ACM Transactions on Graphics (TOG), 4(2), 74-123. (For Delaunay triangulation algorithms).
    *   Shewchuk, J. R. (1996). "Triangle: Engineering a 2D quality mesh generator and Delaunay triangulator." Applied Computational Geometry. (Referenced for robust triangulation).

*   **Voronoi Density Estimation / NUFFT Density Compensation**:
    *   Pipe, J. G., & Menon, P. (1999). "Sampling density compensation in MRI: Rationale and an iterative numerical solution." Magnetic Resonance in Medicine, 41(1), 179-186.
    *   Lawson, C. L. (1977). "Software for C1 surface interpolation." In Mathematical Software III (J. R. Rice, Ed.), pp. 161–194. Academic Press. (Natural Neighbor Interpolation, related to Voronoi cells).
    *   Amidror, I. (2002). "Scattered data interpolation methods for electronic imaging systems: a survey." Journal of Electronic Imaging, 11(2), 157-176. (Section on Voronoi/Delaunay based methods).

*   **Convex Hulls**:
    *   Barber, C. B., Dobkin, D. P., & Huhdanpaa, H. (1996). "The Quickhull algorithm for convex hulls." ACM Transactions on Mathematical Software (TOMS), 22(4), 469-483. (Qhull algorithm, often used by SciPy).
    *   Preparata, F. P., & Hong, S. J. (1977). "Convex hulls of finite sets of points in two and three dimensions." Communications of the ACM, 20(2), 87-93.

*   **Phase Unwrapping using Voronoi Diagrams or Region Growing**:
    *   Ghiglia, D. C., & Pritt, M. D. (1998). "Two-dimensional phase unwrapping: theory, algorithms, and software." John Wiley & Sons. (General reference for phase unwrapping, including region growing).
    *   Abdul-Rahman, H. S., Gdeisat, M. A., Burton, D. R., & Lalor, M. J. (2007). "Fast three-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path." Optical Engineering, 46(6), 065601. (Example of quality-guided region growing).
    *   References specific to PUROR or ROMEO if those were the inspiration for the Voronoi-based unwrapper's advanced steps.

## General Reconlib Literature

This section is a placeholder for future literature and references relevant to the broader `reconlib` library.

*   **MRI Physics and Reconstruction**:
    *   Haacke, E. M., Brown, R. W., Thompson, M. R., & Venkatesan, R. (1999). "Magnetic resonance imaging: Physical principles and sequence design." John Wiley & Sons.
    *   Bernstein, M. A., King, K. F., & Zhou, X. J. (2004). "Handbook of MRI pulse sequences." Elsevier.
    *   Fessler, J. A. (2009). "Optimization methods for image reconstruction." IEEE Signal Processing Magazine, 26(5), 134-143. (Covers iterative methods).

*   **Specific Algorithms (Examples)**:
    *   Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P. (1999). "SENSE: sensitivity encoding for fast MRI." Magnetic Resonance in Medicine, 42(5), 952-962.
    *   Griswold, M. A., Jakob, P. M., Heidemann, R. M., Nittka, M., Jellus, V., Wang, J., ... & Haase, A. (2002). "Generalized autocalibrating partially parallel acquisitions (GRAPPA)." Magnetic Resonance in Medicine, 47(6), 1202-1210.
    *   Lustig, M., Donoho, D., & Pauly, J. M. (2007). "Sparse MRI: The application of compressed sensing for rapid MR imaging." Magnetic Resonance in Medicine, 58(6), 1182-1195.
    *   Beck, A., & Teboulle, M. (2009). "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM Journal on Imaging Sciences, 2(1), 183-202. (FISTA algorithm).
    *   Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning, 3(1), 1-122. (ADMM algorithm).

*(This list should be expanded with specific citations as algorithms are implemented or refined.)*
