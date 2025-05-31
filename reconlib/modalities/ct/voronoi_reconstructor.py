import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import traceback

try:
    from reconlib.voronoi.delaunay_2d import delaunay_triangulation_2d
    from reconlib.voronoi.voronoi_from_delaunay import construct_voronoi_polygons_2d
    from reconlib.voronoi.geometry_core import ConvexHull, EPSILON as GEOMETRY_EPSILON
except ImportError as e:
    print(f"Warning: Voronoi utilities not fully imported for VoronoiCTReconstructor2D: {e}. Using placeholders.")
    if 'GEOMETRY_EPSILON' not in globals(): GEOMETRY_EPSILON = 1e-7 # type: ignore
    def delaunay_triangulation_2d(points: torch.Tensor) -> torch.Tensor: # type: ignore
        print("Warning: Using placeholder delaunay_triangulation_2d.")
        return torch.empty((0,3), dtype=torch.long, device=points.device if hasattr(points, 'device') else 'cpu')
    def construct_voronoi_polygons_2d(points: torch.Tensor, simplices: torch.Tensor) -> Tuple[List[List[torch.Tensor]], torch.Tensor]: # type: ignore
        print("Warning: Using placeholder construct_voronoi_polygons_2d.")
        return ([[] for _ in range(points.shape[0])], torch.empty((0,2), device=points.device if hasattr(points, 'device') else 'cpu'))
    class ConvexHull: # type: ignore
        def __init__(self, points: torch.Tensor, tol: float = 1e-7):
            self.points = points; self.device = points.device if hasattr(points, 'device') else 'cpu'
            self.area = torch.tensor(0.0, device=self.device); self.volume = torch.tensor(0.0, device=self.device)
            self.vertices = torch.empty((0,), dtype=torch.long, device=self.device)
            self.simplices = torch.empty((0,2), dtype=torch.long, device=self.device)
            print("Warning: Using dummy ConvexHull placeholder.")
        def get_vertices_tensor(self) -> torch.Tensor: return self.points
        def get_edges(self) -> List[Tuple[torch.Tensor, torch.Tensor]]: return []

class VoronoiCTReconstructor2D:
    """
    Performs 2D Computed Tomography (CT) reconstruction using a Voronoi diagram
    to represent the image space. The attenuation coefficient is assumed to be
    uniform within each Voronoi cell. Reconstruction is performed using an
    iterative Simultaneous Algebraic Reconstruction Technique (SART) algorithm.

    Key Steps:
    1. Validation of input generator points for Voronoi cells.
    2. Computation of the Voronoi tessellation from these generator points.
    3. Validation of the geometric properties of the resulting Voronoi cells.
    4. Computation of a system matrix (A_ij), where each element represents the
       path length of X-ray 'j' through Voronoi cell 'i'.
    5. Iterative update of attenuation coefficients within each cell using the SART
       algorithm until convergence or a specified number of iterations.

    The class returns a dictionary containing the reconstructed attenuation map,
    the vertices of the Voronoi cells, and status information.
    """
    def __init__(self,
                 num_iterations: int = 10,
                 relaxation_factor: float = 0.15,
                 epsilon: float = GEOMETRY_EPSILON,
                 sart_epsilon: float = 1e-9,
                 verbose: bool = False,
                 device: str = 'cpu',
                 positivity_constraint: bool = True):
        """
        Initializes the VoronoiCTReconstructor2D.

        Args:
            num_iterations (int): Number of iterations for the SART algorithm.
                Defaults to 10.
            relaxation_factor (float): Relaxation factor (lambda) for the SART update.
                Controls the step size of the update. Defaults to 0.15.
            epsilon (float): Small float value used for various geometric
                comparisons and numerical stability (e.g., checking for
                duplicate points, collinearity, zero areas, intersection
                tolerances). Defaults to `GEOMETRY_EPSILON` from
                `reconlib.voronoi.geometry_core`.
            sart_epsilon (float): Small epsilon value added to denominators in the
                SART update rule to prevent division by zero and enhance numerical stability.
                Defaults to 1e-9.
            verbose (bool): If True, prints progress information, warnings, and intermediate
                values during reconstruction. Defaults to False.
            device (str): Computational device ('cpu' or 'cuda') on which to perform
                tensor operations. Defaults to 'cpu'.
            positivity_constraint (bool): If True, enforces non-negativity on the
                attenuation estimates after each SART iteration update.
                Defaults to True.
        """
        self.num_iterations = num_iterations
        self.relaxation_factor = relaxation_factor
        self.epsilon = epsilon # Geometric epsilon
        self.sart_epsilon = sart_epsilon # SART algorithm stability epsilon
        self.verbose = verbose
        self.device = torch.device(device)
        self.positivity_constraint = positivity_constraint

        if self.verbose:
            print(f"VoronoiCTReconstructor2D initialized on device {self.device}")
            print(f"  Iterations: {self.num_iterations}, Relaxation: {self.relaxation_factor}")
            print(f"  Geometric Epsilon: {self.epsilon}, SART Epsilon: {self.sart_epsilon}")
            print(f"  Positivity Constraint: {self.positivity_constraint}")

    def _validate_generator_points_2d(self, generator_points: torch.Tensor) -> Tuple[bool, str]:
        """
        Validates the input 2D generator points for Voronoi tessellation.

        Checks for:
        - Correct tensor type and device.
        - Correct shape (M, 2), where M is the number of points.
        - Minimum number of points (at least 3 for 2D Delaunay/Voronoi).
        - Duplicate or near-duplicate points within `self.epsilon` tolerance.
        - Degeneracy: collinearity of all points, checked by ensuring the area of their
          convex hull is significantly greater than `self.epsilon**2`.

        Args:
            generator_points (torch.Tensor): Tensor of generator points, shape (M, 2).

        Returns:
            Tuple[bool, str]: `(is_invalid, status_message)`.
                              `is_invalid` is True if validation fails.
        """
        if not isinstance(generator_points, torch.Tensor): return True, "Generator points must be a PyTorch Tensor."
        if generator_points.device != self.device: return True, f"Generator points device ({generator_points.device}) does not match reconstructor device ({self.device})."
        if generator_points.ndim != 2: return True, f"Generator points must be a 2D tensor (M, 2). Got shape {generator_points.shape}."
        M, N_dim = generator_points.shape
        if N_dim != 2: return True, f"Generator points must have 2 columns for 2D coordinates. Got {N_dim} columns."
        if M < 3: return True, f"Insufficient generator points: need at least 3 for 2D. Got {M}."
        if M > 1000 and self.verbose: print("Warning: Pairwise duplicate check on many points.")
        for i in range(M):
            for j in range(i + 1, M):
                if torch.norm(generator_points[i] - generator_points[j]) < self.epsilon:
                    return True, f"Duplicate or near-duplicate generator points (indices {i}, {j})."
        try:
            hull = ConvexHull(generator_points.float(), tol=self.epsilon)
            if hull.area < self.epsilon**2 : return True, f"Generator points degenerate (convex hull area ~ {hull.area.item():.2e} < {self.epsilon**2:.2e})."
        except RuntimeError as qe: return True, f"Degeneracy check (ConvexHull) failed: {qe}."
        except Exception as e: return True, f"Unexpected error in degeneracy check: {e}."
        return False, "Generator points validated successfully."

    def _compute_voronoi_diagram_2d(self, generator_points: torch.Tensor) -> Tuple[Optional[List[List[torch.Tensor]]], Optional[torch.Tensor], str]:
        """
        Computes the Voronoi diagram from generator points via Delaunay triangulation.

        Relies on `delaunay_triangulation_2d` and `construct_voronoi_polygons_2d` from
        the `reconlib.voronoi` submodule. Assumes these utilities handle CPU/GPU transfer
        appropriately or expect CPU tensors (current implementation passes CPU tensors).

        Args:
            generator_points (torch.Tensor): Validated 2D generator points, shape (M, 2),
                                             assumed to be on `self.device`.

        Returns:
            Tuple[Optional[List[List[torch.Tensor]]], Optional[torch.Tensor], str]:
                - `voronoi_cells_vertices`: A list where each element is another list of
                  1D PyTorch Tensors (each shape (2,)), representing the vertices of a
                  Voronoi cell. Vertices are on `self.device`. None if computation failed.
                - `unique_voronoi_vertices`: A 2D PyTorch Tensor (V, 2) of unique Voronoi
                  vertices on `self.device`. None if computation failed.
                - `status_message`: String indicating success or failure reason.
        """
        try:
            generator_points_cpu = generator_points.cpu() # Voronoi libs might expect CPU
            delaunay_simplices = delaunay_triangulation_2d(generator_points_cpu)
            if delaunay_simplices is None: return None, None, "Delaunay triangulation failed (returned None)."
            if delaunay_simplices.shape[0] == 0 and generator_points_cpu.shape[0] >= 3:
                return None, None, "Delaunay triangulation resulted in zero simplices (input may be degenerate)."

            voronoi_cells_verts_list_cpu, unique_voronoi_vertices_cpu = construct_voronoi_polygons_2d(generator_points_cpu, delaunay_simplices)

            unique_v_verts = unique_voronoi_vertices_cpu.to(self.device) if unique_voronoi_vertices_cpu is not None else None
            cells_verts = [[v.to(self.device) for v in cell_cpu] for cell_cpu in voronoi_cells_verts_list_cpu]

            return cells_verts, unique_v_verts, "Voronoi diagram computed successfully."
        except RuntimeError as qe: return None, None, f"Error during Voronoi computation (Qhull): {qe}."
        except Exception as e: return None, None, f"Unexpected error during Voronoi computation: {e}"

    def _validate_voronoi_cells_2d(self, voronoi_cells_vertices_list: List[List[torch.Tensor]]) -> Tuple[bool, str]:
        """
        Validates the computed Voronoi cells.

        Checks for:
        - Each cell being non-empty list of vertices.
        - Each cell having enough vertices to form a polygon with area (at least 3 distinct points).
        - Each cell's polygon having a non-negligible area (using ConvexHull area > epsilon**2).

        Args:
            voronoi_cells_vertices_list (List[List[torch.Tensor]]): List of Voronoi cell vertices.
                Each inner list contains 2D Tensors (on `self.device`) for a cell's vertices.

        Returns:
            Tuple[bool, str]: `(is_invalid, status_message)`.
                              `is_invalid` is True if any cell fails validation.
        """
        if not voronoi_cells_vertices_list: return True, "Voronoi cell list is empty."
        for i, cell_v_list in enumerate(voronoi_cells_vertices_list):
            if not cell_v_list: # Cell has no vertices at all
                 if self.verbose: print(f"Warning: Cell {i} has no vertices.")
                 return True, f"Voronoi cell {i} has no vertices."

            # Check if vertices can form a tensor and are sufficient
            try:
                cell_v_tensor = torch.stack(cell_v_list).float() # Stack and ensure float
            except Exception as e_stack:
                 if self.verbose: print(f"Warning: Could not stack vertices for cell {i}: {e_stack}")
                 return True, f"Could not stack vertices for Voronoi cell {i}: {e_stack}"

            if cell_v_tensor.shape[0] < 3:
                if self.verbose: print(f"Warning: Cell {i} has {cell_v_tensor.shape[0]} effective vertices, less than 3. Considered degenerate.")
                return True, f"Degenerate Voronoi cell {i} (vertices form a line or point - {cell_v_tensor.shape[0]} effective vertices)."

            # Check area using ConvexHull
            try:
                cell_hull = ConvexHull(cell_v_tensor, tol=self.epsilon)
                if cell_hull.area < self.epsilon**2:
                    return True, f"Degenerate Voronoi cell {i} (area {cell_hull.area:.2e} < threshold {self.epsilon**2:.2e})."
            except RuntimeError as qe: return True, f"ConvexHull for cell {i} failed (Qhull error: {qe}). Cell vertices might be too degenerate."
            except Exception as e: return True, f"Validation of cell {i} (area check) failed: {e}."
        return False, "Voronoi cells validated successfully."

    @staticmethod
    def _line_segment_intersection_2d(p1: torch.Tensor, p2: torch.Tensor,
                                      p3: torch.Tensor, p4: torch.Tensor,
                                      epsilon: float) -> Optional[torch.Tensor]:
        """
        Calculates the intersection point of two 2D line segments [p1,p2] and [p3,p4].
        Returns the intersection point if it exists and lies on both segments, else None.
        """
        d1 = p2 - p1; d2 = p4 - p3
        d1_cross_d2 = d1[0] * d2[1] - d1[1] * d2[0]
        if torch.abs(d1_cross_d2) < epsilon: return None # Parallel or collinear

        p3_minus_p1 = p3 - p1
        t_numerator = p3_minus_p1[0] * d2[1] - p3_minus_p1[1] * d2[0] # (p3-p1) x d2
        u_numerator = p3_minus_p1[0] * d1[1] - p3_minus_p1[1] * d1[0] # (p3-p1) x d1

        t = t_numerator / d1_cross_d2
        u = u_numerator / d1_cross_d2 # Solves for u in p3-p1 = t*d1 - u*d2 (by crossing with d1)

        # Parameters t and u must be in [0,1] for intersection to be on both segments
        if (0.0 - epsilon) <= t <= (1.0 + epsilon) and (0.0 - epsilon) <= u <= (1.0 + epsilon):
            return p1 + t * d1
        return None

    @staticmethod
    def _is_point_inside_polygon_2d(point: torch.Tensor, polygon_vertices: torch.Tensor,
                                    epsilon: float) -> bool:
        """
        Checks if a point is inside a 2D polygon using the ray casting algorithm.
        Points on the edge are considered inside.
        `polygon_vertices` must be an ordered (N,2) tensor.
        """
        n = polygon_vertices.shape[0]
        if n < 3: return False # Not a polygon

        inside = False
        px, py = point[0], point[1]

        # Check if point is one of the vertices (on boundary)
        for i in range(n):
            if torch.allclose(point, polygon_vertices[i], atol=epsilon): return True

        p1x, p1y = polygon_vertices[0,0], polygon_vertices[0,1]
        for i in range(n + 1): # Iterate over edges (V_i, V_{i+1})
            p2x, p2y = polygon_vertices[i % n, 0], polygon_vertices[i % n, 1]

            # Check if point is on the current edge p1-p2
            # Check collinearity: cross_product((p2-p1), (point-p1)) should be near zero
            edge_vec_x, edge_vec_y = p2x - p1x, p2y - p1y
            point_vec_x, point_vec_y = px - p1x, py - p1y
            cross_product = edge_vec_x * point_vec_y - edge_vec_y * point_vec_x
            if torch.abs(cross_product) < epsilon**2: # Collinear (epsilon might need to be scaled by edge length)
                # Check if point is within bounding box of segment
                if (min(p1x, p2x) - epsilon <= px <= max(p1x, p2x) + epsilon and
                    min(p1y, p2y) - epsilon <= py <= max(p1y, p2y) + epsilon):
                    return True # Point on edge

            # Ray casting: (py is between p1y and p2y) and (px is to the left of intersection)
            if (p1y <= py + epsilon < p2y - epsilon or p2y <= py + epsilon < p1y - epsilon): # Point y is between edge y-endpoints (exclusive for one endpoint for robustness)
                # Compute x-intersection of horizontal ray from point with the edge line
                # Avoid division by zero for vertical edges (p2y - p1y == 0)
                if abs(p2y - p1y) > epsilon: # Edge is not horizontal
                    x_intersection = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if x_intersection > px - epsilon: # Intersection is to the right of or at the point
                        inside = not inside

            p1x, p1y = p2x, p2y # Move to next edge
        return inside

    def _get_lor_endpoints_2d(self, angles_rad_flat: torch.Tensor, radial_offsets_flat: torch.Tensor, fov_width: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates X-ray path start and end points based on projection geometry.

        This method assumes a parallel beam geometry where X-ray paths are defined by
        their angle (`angles_rad_flat`) and radial offset (`radial_offsets_flat`) from
        the origin. The Field of View width (`fov_width`) determines the length of
        these paths across the imaging area.

        The equation for a line (X-ray path) is `x*cos(angle) + y*sin(angle) = offset`.
        Endpoints are determined by extending `fov_width / 2` in both directions along
        the X-ray path from its point of closest approach to the origin.

        Args:
            angles_rad_flat (torch.Tensor): Flattened 1D tensor of X-ray projection angles in radians.
            radial_offsets_flat (torch.Tensor): Flattened 1D tensor of X-ray radial offsets.
            fov_width (float): The width of the field of view, used to define the length
                               of the X-ray paths.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - p1s (torch.Tensor): Tensor of shape (2, num_paths) with (x,y) coordinates of X-ray start points.
                - p2s (torch.Tensor): Tensor of shape (2, num_paths) with (x,y) coordinates of X-ray end points.
        """
        nx = torch.cos(angles_rad_flat); ny = torch.sin(angles_rad_flat) # Normal vector to the X-ray path
        ray_dx = -ny; ray_dy = nx # Direction vector of the X-ray path
        mid_x = radial_offsets_flat * nx; mid_y = radial_offsets_flat * ny # Point on X-ray path closest to origin
        half_ray_length = fov_width / 2.0
        p1_x = mid_x - ray_dx * half_ray_length; p1_y = mid_y - ray_dy * half_ray_length
        p2_x = mid_x + ray_dx * half_ray_length; p2_y = mid_y + ray_dy * half_ray_length
        return torch.stack([p1_x, p1_y], dim=0), torch.stack([p2_x, p2_y], dim=0)

    def _compute_lor_cell_intersection_2d(self, xray_p1: torch.Tensor, xray_p2: torch.Tensor,
                                          cell_polygon_vertices_list: List[torch.Tensor]) -> float:
        """
        Computes the length of intersection of a single X-ray path segment with a 2D Voronoi cell polygon.

        The X-ray path is defined by its start point `xray_p1` and end point `xray_p2`.
        The Voronoi cell is defined by `cell_polygon_vertices_list`, a list of 2D tensors
        representing its ordered vertices.

        The method calculates all intersection points between the X-ray path segment and
        the cell's edges. It also considers cases where the X-ray path's endpoints might
        be inside the cell. The final intersection length is the distance between the
        two extreme valid intersection points along the X-ray path that lie within both
        the X-ray segment and the cell.

        Args:
            xray_p1 (torch.Tensor): Start point of the X-ray path (2D tensor).
            xray_p2 (torch.Tensor): End point of the X-ray path (2D tensor).
            cell_polygon_vertices_list (List[torch.Tensor]): A list of 2D tensors,
                where each tensor represents a vertex of the Voronoi cell polygon.
                Vertices are assumed to be ordered.

        Returns:
            float: The calculated intersection length of the X-ray path segment with the
                   Voronoi cell. Returns 0.0 if there is no intersection or if the
                   cell is degenerate.
        """
        if not cell_polygon_vertices_list or len(cell_polygon_vertices_list) < 3: return 0.0
        try:
            cell_v_tensor = torch.stack(cell_polygon_vertices_list).to(dtype=xray_p1.dtype, device=self.device)
            if cell_v_tensor.shape[0] < 3: return 0.0
        except Exception:
            if self.verbose: print(f"Warning: Could not stack vertices for a cell in intersection test for X-ray {xray_p1}-{xray_p2}.")
            return 0.0

        collected_physical_points = []
        num_cell_verts = cell_v_tensor.shape[0]
        for i in range(num_cell_verts):
            poly_edge_p1 = cell_v_tensor[i]
            poly_edge_p2 = cell_v_tensor[(i + 1) % num_cell_verts]
            intersect_pt = self._line_segment_intersection_2d(xray_p1, xray_p2, poly_edge_p1, poly_edge_p2, self.epsilon)
            if intersect_pt is not None: collected_physical_points.append(intersect_pt)

        if self._is_point_inside_polygon_2d(xray_p1, cell_v_tensor, self.epsilon): collected_physical_points.append(xray_p1)
        if self._is_point_inside_polygon_2d(xray_p2, cell_v_tensor, self.epsilon): collected_physical_points.append(xray_p2)

        if not collected_physical_points: return 0.0

        unique_points = []
        if collected_physical_points:
            temp_stacked_points = torch.stack(collected_physical_points)
            if temp_stacked_points.shape[0] > 0:
                unique_points.append(temp_stacked_points[0])
                for k_pt in range(1, temp_stacked_points.shape[0]):
                    is_duplicate = any(torch.allclose(temp_stacked_points[k_pt], up, atol=self.epsilon) for up in unique_points)
                    if not is_duplicate: unique_points.append(temp_stacked_points[k_pt])

        if len(unique_points) < 2: return 0.0

        path_vec = xray_p2 - xray_p1; path_len_sq = torch.dot(path_vec, path_vec)
        if path_len_sq < self.epsilon**2: return 0.0

        t_values = []
        for pt in unique_points:
            t_val = torch.dot(pt - xray_p1, path_vec) / path_len_sq
            path_actual_len = torch.sqrt(path_len_sq)
            t_tolerance = self.epsilon / (path_actual_len + 1e-9) # Tolerance for t values being on segment
            if (-t_tolerance) <= t_val <= (1.0 + t_tolerance):
                 t_values.append(torch.clamp(t_val, 0.0, 1.0)) # Clamp to ensure points are strictly on segment

        if len(t_values) < 2: return 0.0 # Need two distinct points on the segment for a length

        t_tensor = torch.stack(t_values); min_t = torch.min(t_tensor); max_t = torch.max(t_tensor)
        intersection_length = (max_t - min_t) * torch.sqrt(path_len_sq) # Length is delta_t * segment_length
        return intersection_length.item() if intersection_length > self.epsilon else 0.0 # Return 0 if length is negligible

    def _compute_system_matrix_2d(self, xray_descriptor: Dict, voronoi_cells_vertices_list: List[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        """
        Computes the 2D system matrix (A_ij) where A_ij is the path length of X-ray 'i' through Voronoi cell 'j'.

        Args:
            xray_descriptor (Dict): Dictionary describing the X-ray projection geometry.
                Expected keys:
                - 'angles_rad' (torch.Tensor): 1D tensor of unique projection angles in radians.
                - 'radial_offsets' (torch.Tensor): 1D tensor of unique radial offsets from origin.
                - 'fov_width' (float): Width of the Field of View, determining X-ray path lengths.
            voronoi_cells_vertices_list (List[List[torch.Tensor]]): A list where each
                inner list contains 2D PyTorch Tensors representing the vertices of a Voronoi cell.

        Returns:
            Optional[torch.Tensor]: The computed system matrix of shape (num_total_paths, num_cells),
                                    or None if an error occurs (e.g., invalid descriptor).
        """
        if not all(k in xray_descriptor for k in ['angles_rad', 'radial_offsets', 'fov_width']):
            if self.verbose: print("Error: X-ray descriptor missing required keys: 'angles_rad', 'radial_offsets', 'fov_width'.")
            return None
        angles_rad = xray_descriptor['angles_rad'].to(self.device); radial_offsets = xray_descriptor['radial_offsets'].to(self.device)
        fov_width = lor_descriptor['fov_width']
        num_angles = angles_rad.shape[0]; num_radial_bins = radial_offsets.shape[0]
        num_total_paths = num_angles * num_radial_bins; num_cells = len(voronoi_cells_vertices_list) # num_total_xrays
        if self.verbose: print(f"Computing system matrix for {num_total_paths} X-ray paths and {num_cells} cells.")

        angles_grid, radials_grid = torch.meshgrid(angles_rad, radial_offsets, indexing='ij')
        path_p1s, path_p2s = self._get_lor_endpoints_2d(angles_grid.flatten(), radials_grid.flatten(), fov_width) # xray_p1s, xray_p2s

        system_matrix = torch.zeros((num_total_paths, num_cells), device=self.device, dtype=torch.float32) # num_total_xrays
        for i_cell, cell_v_list_for_intersect in enumerate(voronoi_cells_vertices_list):
            if not cell_v_list_for_intersect:
                if self.verbose: print(f"Warning: Cell {i_cell} has no vertices, skipping in system matrix.")
                continue
            for j_path in range(num_total_paths): # j_xray
                system_matrix[j_path, i_cell] = self._compute_lor_cell_intersection_2d(path_p1s[:, j_path], path_p2s[:, j_path], cell_v_list_for_intersect) # xray_p1s, xray_p2s

        if self.verbose: print(f"System matrix computed. Shape: {system_matrix.shape}, Sum of elements: {torch.sum(system_matrix).item():.2e}")
        return system_matrix

    def _forward_project_2d(self, attenuation_per_cell: torch.Tensor, system_matrix: torch.Tensor) -> torch.Tensor:
        """ Forward projects cell attenuation coefficients using the system matrix. """
        # attenuation_per_cell is mu_j (num_cells,)
        # system_matrix is L_ij (num_paths, num_cells)
        # result is p_i = sum_j (L_ij * mu_j) (num_paths,)
        mu = attenuation_per_cell.unsqueeze(1) if attenuation_per_cell.ndim == 1 else attenuation_per_cell
        projection_data_flat = torch.matmul(system_matrix, mu).squeeze(-1)
        return projection_data_flat

    def _back_project_2d(self, projection_data_flat: torch.Tensor, system_matrix: torch.Tensor) -> torch.Tensor:
        """ Back projects flat projection data using the system matrix transpose. """
        # projection_data_flat is w_i (num_paths,)
        # system_matrix.T is L_ji (num_cells, num_paths)
        # result is b_j = sum_i (L_ji * w_i) (num_cells,)
        proj = projection_data_flat.unsqueeze(1) if projection_data_flat.ndim == 1 else projection_data_flat
        back_projected_values = torch.matmul(system_matrix.T, proj).squeeze(-1)
        return back_projected_values

    def reconstruct(self, sinogram_2d: torch.Tensor, generator_points_2d: torch.Tensor,
                    lor_descriptor: Dict, initial_estimate: Optional[torch.Tensor] = None) -> Dict[str, Any]: # xray_descriptor
        """
        Performs Voronoi-based CT reconstruction using a SART-like algorithm.

        Args:
            sinogram_2d (torch.Tensor): The measured 2D CT sinogram data, representing line integrals
                of attenuation. Expected shape is (num_angles, num_radial_bins), which will be
                flattened internally to match the system matrix's first dimension.
            generator_points_2d (torch.Tensor): A 2D tensor of shape (M, 2) containing the (x,y)
                coordinates of the M generator points for the Voronoi cells.
            lor_descriptor (Dict): Dictionary describing the X-ray path geometry (scanner setup).
                This will be passed to `_compute_system_matrix_2d` as `xray_descriptor`.
                Expected keys:
                - 'angles_rad' (torch.Tensor): 1D tensor of unique projection angles in radians.
                - 'radial_offsets' (torch.Tensor): 1D tensor of unique radial offsets.
                - 'fov_width' (float): Width of the Field of View.
            initial_estimate (Optional[torch.Tensor]): An optional initial estimate for the
                attenuation coefficients within each Voronoi cell. Shape should be (M,).
                If None, reconstruction starts from an image of zeros. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the reconstruction results:
                - "attenuation" (torch.Tensor): 1D tensor of shape (M,) with the estimated
                  attenuation coefficient for each Voronoi cell.
                - "voronoi_cells_vertices" (List[List[torch.Tensor]]): A list where each
                  element is another list of 1D PyTorch Tensors (each shape (2,)),
                  representing the vertices of a successfully computed Voronoi cell.
                - "status" (str): A string indicating the final status of the reconstruction
                  (e.g., "completed", "failed: ...").
                - "degenerate_input" (bool): True if the input `generator_points_2d` were
                  found to be degenerate (e.g., collinear, too few points), preventing
                  valid Voronoi tessellation.
                - "error_log" (List[str]): A list of error or warning messages encountered
                  during the reconstruction process.
        """
        result: Dict[str, Any] = {"attenuation": torch.empty(0, device=self.device), "voronoi_cells_vertices": [], "status": "Not started", "degenerate_input": False, "error_log": []}
        log_entry = lambda msg: (result["error_log"].append(msg), print(msg)) if self.verbose else result["error_log"].append(msg)

        log_entry(f"Starting Voronoi CT (SART) reconstruction on device {self.device}...");
        sinogram_2d = sinogram_2d.to(self.device); generator_points_2d = generator_points_2d.to(self.device)

        is_gen_invalid, gen_status = self._validate_generator_points_2d(generator_points_2d)
        log_entry(f"Generator Point Validation: {gen_status}"); result["status"] = gen_status
        if is_gen_invalid: result["degenerate_input"] = True; return result

        cells_verts, _, vor_status = self._compute_voronoi_diagram_2d(generator_points_2d)
        log_entry(f"Voronoi Diagram Computation: {vor_status}"); result["status"] = vor_status
        if cells_verts is None: return result
        result["voronoi_cells_vertices"] = cells_verts

        is_cell_invalid, cell_status = self._validate_voronoi_cells_2d(cells_verts)
        log_entry(f"Voronoi Cell Validation: {cell_status}"); result["status"] = cell_status
        if is_cell_invalid: return result

        log_entry("Validation and Tessellation complete. Computing system matrix...")
        # Pass 'lor_descriptor' as 'xray_descriptor' to _compute_system_matrix_2d
        system_matrix = self._compute_system_matrix_2d(lor_descriptor, cells_verts)
        if system_matrix is None:
            result["status"] = "Failed: System matrix computation error."; log_entry(result["status"]); return result
        log_entry(f"System matrix computed. Shape: {system_matrix.shape}")

        num_cells = generator_points_2d.shape[0]
        attenuation_estimates: torch.Tensor # Renamed from activity_estimates
        if initial_estimate is not None:
            if initial_estimate.shape != (num_cells,):
                err_msg = f"Failed: Initial estimate shape {initial_estimate.shape} does not match num_cells ({num_cells})."; log_entry(err_msg); result["status"] = err_msg; return result
            attenuation_estimates = initial_estimate.clone().to(self.device)
        else:
            attenuation_estimates = torch.zeros(num_cells, device=self.device, dtype=torch.float32) # Initialize with zeros for CT

        if self.positivity_constraint: attenuation_estimates = torch.clamp(attenuation_estimates, min=0.0)

        sinogram_flat = sinogram_2d.reshape(-1)
        if sinogram_flat.shape[0] != system_matrix.shape[0]:
            err_msg = f"Failed: Flattened sinogram paths ({sinogram_flat.shape[0]}) != System Matrix paths ({system_matrix.shape[0]}). Check X-ray descriptor consistency."; log_entry(err_msg); result["status"] = err_msg; return result

        # Pre-calculate row sums (for projection error normalization) and column sums (for backprojection weighting)
        row_sums_system_matrix = torch.sum(system_matrix, dim=1) + self.sart_epsilon
        col_sums_system_matrix = torch.sum(system_matrix, dim=0) + self.sart_epsilon

        log_entry("Starting SART iterations...")
        for iteration in range(self.num_iterations):
            attenuation_prev_iter = attenuation_estimates.clone()

            expected_projections = self._forward_project_2d(attenuation_estimates, system_matrix)
            projection_error = sinogram_flat - expected_projections

            # Normalizing the projection error before backprojection
            weighted_error = projection_error / row_sums_system_matrix

            # Backproject the weighted error
            correction_term_numerator = self._back_project_2d(weighted_error, system_matrix)

            # Weighted backprojection (dividing by column sums)
            update_step = correction_term_numerator / col_sums_system_matrix

            attenuation_estimates += self.relaxation_factor * update_step

            if self.positivity_constraint:
                attenuation_estimates = torch.clamp(attenuation_estimates, min=0.0)

            change = torch.norm(attenuation_estimates - attenuation_prev_iter) / (torch.norm(attenuation_prev_iter) + self.sart_epsilon)
            if self.verbose:
                log_entry(f"SART Iter {iteration + 1}/{self.num_iterations}, Change: {change.item():.4e}, Attenuation Sum: {attenuation_estimates.sum().item():.2e}")

        result['attenuation'] = attenuation_estimates # Renamed from 'activity'
        final_status = "Voronoi-based CT reconstruction completed."
        log_entry(final_status); result['status'] = final_status
        return result

__all__ = ['VoronoiCTReconstructor2D']
