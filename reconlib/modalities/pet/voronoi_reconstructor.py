import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import traceback

try:
    from reconlib.voronoi.delaunay_2d import delaunay_triangulation_2d
    from reconlib.voronoi.voronoi_from_delaunay import construct_voronoi_polygons_2d
    from reconlib.voronoi.geometry_core import ConvexHull, EPSILON as GEOMETRY_EPSILON
except ImportError as e:
    print(f"Warning: Voronoi utilities not fully imported for VoronoiPETReconstructor2D: {e}. Using placeholders.")
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

class VoronoiPETReconstructor2D:
    """
    Performs 2D PET reconstruction using a Voronoi diagram to represent the image space.
    The activity is assumed to be uniform within each Voronoi cell.
    Reconstruction is performed using an MLEM-like iterative algorithm.

    The process involves:
    1. Validating input generator points.
    2. Computing the Voronoi tessellation from these points.
    3. Validating the resulting Voronoi cells (e.g., ensuring they have area).
    4. Computing a system matrix where each element (A_ij) represents the intersection
       length of Line of Response (LOR) j with Voronoi cell i.
    5. Iteratively updating activity estimates within each cell using an MLEM algorithm.
    """
    def __init__(self,
                 num_iterations: int = 10,
                 epsilon: float = GEOMETRY_EPSILON,
                 verbose: bool = False,
                 device: str = 'cpu',
                 positivity_constraint: bool = True):
        """
        Initializes the VoronoiPETReconstructor2D.

        Args:
            num_iterations (int, optional): Number of iterations for the MLEM algorithm.
                                          Defaults to 10.
            epsilon (float, optional): Small float value used for various geometric
                                       comparisons and numerical stability (e.g., checking for
                                       duplicate points, collinearity, zero areas, intersection
                                       tolerances). Defaults to `reconlib.voronoi.geometry_core.EPSILON`.
            verbose (bool, optional): If True, prints progress information and warnings during
                                      reconstruction. Defaults to False.
            device (str, optional): Computational device ('cpu' or 'cuda'). Defaults to 'cpu'.
            positivity_constraint (bool, optional): If True, enforces non-negativity on the
                                                  activity estimates at each MLEM iteration.
                                                  Defaults to True.
        """
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.mlem_epsilon = 1e-12 # Specific epsilon for MLEM division stability
        self.verbose = verbose
        self.device = torch.device(device)
        self.positivity_constraint = positivity_constraint

        if self.verbose:
            print(f"VoronoiPETReconstructor2D initialized on device {self.device}")
            print(f"  Iterations: {self.num_iterations}, Geometric Epsilon: {self.epsilon}, MLEM Epsilon: {self.mlem_epsilon}")
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
        Calculates LOR start/end points from angles, radial offsets, and FOV width.
        Assumes LORs are lines defined by `x*cos(angle) + y*sin(angle) = offset`,
        and endpoints are `fov_width / 2` distance from the LOR's closest point to origin.
        """
        nx = torch.cos(angles_rad_flat); ny = torch.sin(angles_rad_flat) # Normal to LOR
        lor_dx = -ny; lor_dy = nx # Direction of LOR
        mid_x = radial_offsets_flat * nx; mid_y = radial_offsets_flat * ny # Point on LOR closest to origin
        half_lor_length = fov_width / 2.0
        p1_x = mid_x - lor_dx * half_lor_length; p1_y = mid_y - lor_dy * half_lor_length
        p2_x = mid_x + lor_dx * half_lor_length; p2_y = mid_y + lor_dy * half_lor_length
        return torch.stack([p1_x, p1_y], dim=0), torch.stack([p2_x, p2_y], dim=0)

    def _compute_lor_cell_intersection_2d(self, lor_p1: torch.Tensor, lor_p2: torch.Tensor,
                                          cell_polygon_vertices_list: List[torch.Tensor]) -> float:
        """
        Computes the length of intersection of a Line of Response (LOR) with a 2D Voronoi cell.
        This is the accurate geometric version.
        """
        if not cell_polygon_vertices_list or len(cell_polygon_vertices_list) < 3: return 0.0
        try:
            cell_v_tensor = torch.stack(cell_polygon_vertices_list).to(dtype=lor_p1.dtype, device=self.device)
            if cell_v_tensor.shape[0] < 3: return 0.0
        except Exception:
            if self.verbose: print(f"Warning: Could not stack vertices for a cell in intersection test for LOR {lor_p1}-{lor_p2}.")
            return 0.0

        collected_physical_points = []
        num_cell_verts = cell_v_tensor.shape[0]
        for i in range(num_cell_verts):
            poly_edge_p1 = cell_v_tensor[i]
            poly_edge_p2 = cell_v_tensor[(i + 1) % num_cell_verts]
            intersect_pt = self._line_segment_intersection_2d(lor_p1, lor_p2, poly_edge_p1, poly_edge_p2, self.epsilon)
            if intersect_pt is not None: collected_physical_points.append(intersect_pt)

        if self._is_point_inside_polygon_2d(lor_p1, cell_v_tensor, self.epsilon): collected_physical_points.append(lor_p1)
        if self._is_point_inside_polygon_2d(lor_p2, cell_v_tensor, self.epsilon): collected_physical_points.append(lor_p2)

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

        lor_vec = lor_p2 - lor_p1; lor_len_sq = torch.dot(lor_vec, lor_vec)
        if lor_len_sq < self.epsilon**2: return 0.0

        t_values = []
        for pt in unique_points:
            t_val = torch.dot(pt - lor_p1, lor_vec) / lor_len_sq
            lor_actual_len = torch.sqrt(lor_len_sq)
            t_tolerance = self.epsilon / (lor_actual_len + 1e-9)
            if (-t_tolerance) <= t_val <= (1.0 + t_tolerance):
                 t_values.append(torch.clamp(t_val, 0.0, 1.0))

        if len(t_values) < 2: return 0.0

        t_tensor = torch.stack(t_values); min_t = torch.min(t_tensor); max_t = torch.max(t_tensor)
        intersection_length = (max_t - min_t) * torch.sqrt(lor_len_sq) # Use sqrt for actual length
        return intersection_length.item() if intersection_length > self.epsilon else 0.0

    def _compute_system_matrix_2d(self, lor_descriptor: Dict, voronoi_cells_vertices_list: List[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        """
        Computes the 2D system matrix where A_ij is the intersection length of LOR i with cell j.

        Args:
            lor_descriptor: Dict with 'angles_rad', 'radial_offsets', 'fov_width'.
            voronoi_cells_vertices_list: List of lists of vertex tensors for each cell.
        """
        if not all(k in lor_descriptor for k in ['angles_rad', 'radial_offsets', 'fov_width']):
            if self.verbose: print("Error: lor_descriptor missing required keys: 'angles_rad', 'radial_offsets', 'fov_width'.")
            return None
        angles_rad = lor_descriptor['angles_rad'].to(self.device); radial_offsets = lor_descriptor['radial_offsets'].to(self.device)
        fov_width = lor_descriptor['fov_width']
        num_angles = angles_rad.shape[0]; num_radial_bins = radial_offsets.shape[0]
        num_total_lors = num_angles * num_radial_bins; num_cells = len(voronoi_cells_vertices_list)
        if self.verbose: print(f"Computing system matrix for {num_total_lors} LORs and {num_cells} cells.")

        angles_grid, radials_grid = torch.meshgrid(angles_rad, radial_offsets, indexing='ij')
        lor_p1s, lor_p2s = self._get_lor_endpoints_2d(angles_grid.flatten(), radials_grid.flatten(), fov_width)

        system_matrix = torch.zeros((num_total_lors, num_cells), device=self.device, dtype=torch.float32)
        for i_cell, cell_v_list_for_lor_intersect in enumerate(voronoi_cells_vertices_list):
            if not cell_v_list_for_lor_intersect:
                if self.verbose: print(f"Warning: Cell {i_cell} has no vertices, skipping in system matrix.")
                continue
            for j_lor in range(num_total_lors):
                system_matrix[j_lor, i_cell] = self._compute_lor_cell_intersection_2d(lor_p1s[:, j_lor], lor_p2s[:, j_lor], cell_v_list_for_lor_intersect)

        if self.verbose: print(f"System matrix computed. Shape: {system_matrix.shape}, Sum of elements: {torch.sum(system_matrix).item():.2e}")
        return system_matrix

    def _forward_project_2d(self, activity_per_cell: torch.Tensor, system_matrix: torch.Tensor) -> torch.Tensor:
        """ Forward projects cell activities using the system matrix. """
        act = activity_per_cell.unsqueeze(1) if activity_per_cell.ndim == 1 else activity_per_cell # (num_cells, 1)
        # system_matrix: (num_lors, num_cells)
        projection_data_flat = torch.matmul(system_matrix, act).squeeze(-1) # (num_lors,)
        return projection_data_flat

    def _back_project_2d(self, projection_data_flat: torch.Tensor, system_matrix: torch.Tensor) -> torch.Tensor:
        """ Back projects flat projection data using the system matrix transpose. """
        proj = projection_data_flat.unsqueeze(1) if projection_data_flat.ndim == 1 else projection_data_flat # (num_lors, 1)
        # system_matrix.T: (num_cells, num_lors)
        back_projected_activity = torch.matmul(system_matrix.T, proj).squeeze(-1) # (num_cells,)
        return back_projected_activity

    def reconstruct(self, sinogram_2d: torch.Tensor, generator_points_2d: torch.Tensor,
                    lor_descriptor: Dict, initial_estimate: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Performs Voronoi-based PET reconstruction using an MLEM-like algorithm.

        Args:
            sinogram_2d (torch.Tensor): 2D PET sinogram data.
                                        Expected shape (num_angles, num_radial_bins), which will be flattened.
            generator_points_2d (torch.Tensor): Initial generator points for Voronoi cells.
                                                Shape (M, 2), where M is the number of cells.
            lor_descriptor (Dict): Dictionary describing LOR geometry. Expected keys:
                'angles_rad' (torch.Tensor): 1D tensor of unique projection angles.
                'radial_offsets' (torch.Tensor): 1D tensor of unique radial offsets.
                'fov_width' (float): Width of the FOV to define LOR segment length.
            initial_estimate (Optional[torch.Tensor], optional): Initial activity estimate for each cell.
                                                               Shape (M,). If None, defaults to ones.

        Returns:
            Dict[str, Any]: A dictionary containing reconstruction results:
                - "activity" (torch.Tensor): Estimated activity per Voronoi cell (Shape: (M,)).
                - "voronoi_cells_vertices" (List[List[torch.Tensor]]): Vertices of each Voronoi cell.
                - "status" (str): Final status message.
                - "degenerate_input" (bool): True if input generator points were degenerate.
                - "error_log" (List[str]): Log of messages and errors during reconstruction.
        """
        result: Dict[str, Any] = {"activity": torch.empty(0, device=self.device), "voronoi_cells_vertices": [], "status": "Not started", "degenerate_input": False, "error_log": []}
        log_entry = lambda msg: (result["error_log"].append(msg), print(msg)) if self.verbose else result["error_log"].append(msg)

        log_entry(f"Starting Voronoi PET reconstruction on device {self.device}...");
        sinogram_2d = sinogram_2d.to(self.device); generator_points_2d = generator_points_2d.to(self.device)

        is_gen_invalid, gen_status = self._validate_generator_points_2d(generator_points_2d)
        log_entry(f"Generator Point Validation: {gen_status}"); result["status"] = gen_status
        if is_gen_invalid: result["degenerate_input"] = True; return result

        cells_verts, _, vor_status = self._compute_voronoi_diagram_2d(generator_points_2d) # unique_v_verts not used further for now
        log_entry(f"Voronoi Diagram Computation: {vor_status}"); result["status"] = vor_status
        if cells_verts is None: return result
        result["voronoi_cells_vertices"] = cells_verts

        is_cell_invalid, cell_status = self._validate_voronoi_cells_2d(cells_verts)
        log_entry(f"Voronoi Cell Validation: {cell_status}"); result["status"] = cell_status
        if is_cell_invalid: return result

        log_entry("Validation and Tessellation complete. Computing system matrix...")
        system_matrix = self._compute_system_matrix_2d(lor_descriptor, cells_verts)
        if system_matrix is None:
            result["status"] = "Failed: System matrix computation error."; log_entry(result["status"]); return result
        log_entry(f"System matrix computed. Shape: {system_matrix.shape}")

        num_cells = generator_points_2d.shape[0]
        activity_estimates: torch.Tensor
        if initial_estimate is not None:
            if initial_estimate.shape != (num_cells,):
                err_msg = f"Failed: Initial estimate shape {initial_estimate.shape} does not match num_cells ({num_cells})."; log_entry(err_msg); result["status"] = err_msg; return result
            activity_estimates = initial_estimate.clone().to(self.device)
        else: activity_estimates = torch.ones(num_cells, device=self.device, dtype=torch.float32)

        if self.positivity_constraint: activity_estimates = torch.clamp(activity_estimates, min=0.0)

        sinogram_flat = sinogram_2d.reshape(-1)
        if sinogram_flat.shape[0] != system_matrix.shape[0]:
            err_msg = f"Failed: Flattened sinogram LORs ({sinogram_flat.shape[0]}) != System Matrix LORs ({system_matrix.shape[0]}). Check lor_descriptor consistency with sinogram shape."; log_entry(err_msg); result["status"] = err_msg; return result

        log_entry("Calculating sensitivity image for MLEM...");
        sensitivity_image = self._back_project_2d(torch.ones(system_matrix.shape[0], device=self.device, dtype=torch.float32), system_matrix)
        sensitivity_image_clamped = torch.clamp(sensitivity_image, min=self.mlem_epsilon)
        log_entry("Sensitivity image calculated.")

        log_entry("Starting MLEM iterations...")
        for iteration in range(self.num_iterations):
            activity_prev_iter = activity_estimates.clone()
            expected_counts_flat = self._forward_project_2d(activity_estimates, system_matrix)
            ratio = sinogram_flat / (expected_counts_flat + self.mlem_epsilon)
            correction_term = self._back_project_2d(ratio, system_matrix)
            activity_estimates = activity_estimates * correction_term / sensitivity_image_clamped
            if self.positivity_constraint: activity_estimates = torch.clamp(activity_estimates, min=0.0)

            change = torch.norm(activity_estimates - activity_prev_iter) / (torch.norm(activity_prev_iter) + self.mlem_epsilon)
            if self.verbose:
                log_entry(f"MLEM Iter {iteration + 1}/{self.num_iterations}, Change: {change.item():.4e}, Activity Sum: {activity_estimates.sum().item():.2e}")

        result['activity'] = activity_estimates
        final_status = "Voronoi-based PET reconstruction completed."
        log_entry(final_status); result['status'] = final_status
        return result

__all__ = ['VoronoiPETReconstructor2D']
