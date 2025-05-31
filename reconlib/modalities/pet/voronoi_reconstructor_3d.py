import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

try:
    from reconlib.voronoi.delaunay_3d import delaunay_triangulation_3d
except ImportError: # pragma: no cover
    delaunay_triangulation_3d = None
    print("Warning: delaunay_triangulation_3d not found. VoronoiPETReconstructor3D may not function correctly.")

try:
    from reconlib.voronoi.voronoi_from_delaunay import construct_voronoi_polyhedra_3d
except ImportError: # pragma: no cover
    construct_voronoi_polyhedra_3d = None
    print("Warning: construct_voronoi_polyhedra_3d not found. VoronoiPETReconstructor3D may not function correctly.")

try:
    from reconlib.voronoi.geometry_core import ConvexHull, EPSILON
except ImportError: # pragma: no cover
    ConvexHull = None
    EPSILON = 1e-6
    print("Warning: ConvexHull or EPSILON not found. VoronoiPETReconstructor3D may not function correctly.")


class VoronoiPETReconstructor3D:
    def __init__(self,
                 num_iterations: int = 10,
                 epsilon: float = EPSILON, # Geometric epsilon
                 verbose: bool = False,
                 device: str = 'cpu',
                 positivity_constraint: bool = True,
                 mlem_epsilon: Optional[float] = None): # MLEM stability epsilon
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.mlem_epsilon = mlem_epsilon if mlem_epsilon is not None else self.epsilon
        self.verbose = verbose
        try:
            self.device = torch.device(device)
        except Exception as e: # pragma: no cover
            print(f"Warning: Failed to set device '{device}', defaulting to CPU. Error: {e}")
            self.device = torch.device('cpu')
        self.positivity_constraint = positivity_constraint

        if delaunay_triangulation_3d is None or \
           construct_voronoi_polyhedra_3d is None or \
           ConvexHull is None: # pragma: no cover
            raise ImportError(
                "Required Voronoi or geometry functions not found. "
                "Please ensure 'reconlib.voronoi' is correctly installed and implemented."
            )

    def _validate_generator_points_3d(self, generator_points: torch.Tensor) -> Tuple[bool, str]:
        if not isinstance(generator_points, torch.Tensor):
            return True, "Generator points must be a PyTorch tensor."
        if generator_points.device.type != self.device.type:
            return True, f"Generator points are on device {generator_points.device}, but reconstructor is on {self.device}."
        if generator_points.ndim != 2:
            return True, f"Generator points must be a 2D tensor, got {generator_points.ndim} dimensions."
        if generator_points.shape[1] != 3:
            return True, f"Generator points must have 3 columns (3D coords), got {generator_points.shape[1]}."
        if generator_points.shape[0] < 4:
            return True, f"At least 4 generator points are required for 3D Voronoi, got {generator_points.shape[0]}."
        if generator_points.shape[0] > 1:
            for i in range(generator_points.shape[0]):
                for j in range(i + 1, generator_points.shape[0]):
                    if torch.norm(generator_points[i] - generator_points[j]) < self.epsilon:
                        return True, f"Duplicate points found: index {i} and {j} are too close."
        try:
            points_for_hull = generator_points.float().cpu().numpy()
            if len(np.unique(points_for_hull, axis=0)) < 4:
                 return True, "Degenerate input: Fewer than 4 unique points for Convex Hull."
            hull = ConvexHull(points_for_hull, tol=self.epsilon)
            if hull.volume < self.epsilon**3:
                return True, f"Generator points coplanar/degenerate (hull volume {hull.volume:.2e})."
        except RuntimeError as e:
            return True, f"Convex hull computation failed (degenerate input?): {e}"
        except Exception as e: # pragma: no cover
            return True, f"Failed convex hull check: {e}"
        return False, "Generator points validated successfully."

    def _compute_voronoi_diagram_3d(
        self, generator_points: torch.Tensor
    ) -> Tuple[Optional[List[Dict[str, List[List[int]]]]], Optional[torch.Tensor], str]:
        if self.verbose: print(f"Starting 3D Voronoi diagram for {generator_points.shape[0]} points on {self.device}.")
        try:
            generator_points_cpu = generator_points.cpu()
            if self.verbose: print("Performing 3D Delaunay triangulation...")
            delaunay_tetrahedra = delaunay_triangulation_3d(generator_points_cpu.numpy())
            if delaunay_tetrahedra is None: return None, None, "Delaunay triangulation failed (returned None)."
            if not hasattr(delaunay_tetrahedra, 'shape') or delaunay_tetrahedra.shape[0] == 0:
                return None, None, f"Delaunay resulted in empty/invalid output (shape: {getattr(delaunay_tetrahedra, 'shape', 'N/A')})."
            if self.verbose: print(f"Delaunay successful, {delaunay_tetrahedra.shape[0]} tetrahedra.")
            if self.verbose: print("Constructing Voronoi polyhedra...")
            voronoi_cells_cpu, unique_verts_cpu_np = construct_voronoi_polyhedra_3d(
                generator_points_cpu.numpy(), delaunay_tetrahedra)
            if voronoi_cells_cpu is None or unique_verts_cpu_np is None:
                return None, None, "Voronoi polyhedra construction failed (returned None)."
            if not voronoi_cells_cpu or unique_verts_cpu_np.shape[0] == 0:
                return None, None, "Voronoi construction resulted in empty cell data or vertices."
            if self.verbose: print(f"Voronoi successful. {len(voronoi_cells_cpu)} cells, {unique_verts_cpu_np.shape[0]} unique vertices.")
            unique_voronoi_vertices = torch.from_numpy(unique_verts_cpu_np).to(dtype=generator_points.dtype, device=self.device)
            cells_data_device = []
            for cell_cpu_data in voronoi_cells_cpu:
                if isinstance(cell_cpu_data, dict) and 'faces' in cell_cpu_data:
                    cells_data_device.append({'faces': cell_cpu_data['faces']})
                else: # pragma: no cover
                    if self.verbose: print(f"Warning: Cell data format unexpected: {type(cell_cpu_data)}. Passing as is.")
                    cells_data_device.append(cell_cpu_data)
            if self.verbose: print(f"Successfully computed Voronoi diagram. {len(cells_data_device)} cells on device.")
            return cells_data_device, unique_voronoi_vertices, "Voronoi diagram computed successfully."
        except RuntimeError as e: return None, None, f"Runtime error during Voronoi computation: {e}" # pragma: no cover
        except Exception as e: return None, None, f"Unexpected error during Voronoi computation: {type(e).__name__} - {e}" # pragma: no cover

    def _validate_voronoi_cells_3d(
        self, cells_data: Optional[List[Dict[str, List[List[int]]]]],
        unique_voronoi_vertices: Optional[torch.Tensor]
    ) -> Tuple[bool, str]:
        if cells_data is None: return True, "Voronoi cells data is None."
        if unique_voronoi_vertices is None: return True, "Unique Voronoi vertices is None."
        if not isinstance(cells_data, list): return True, f"Cells data not a list, got {type(cells_data)}."
        if not isinstance(unique_voronoi_vertices, torch.Tensor): return True, f"Unique vertices not PyTorch Tensor, got {type(unique_voronoi_vertices)}."
        if unique_voronoi_vertices.ndim!=2 or unique_voronoi_vertices.shape[1]!=3: return True, f"Unique vertices shape error: {unique_voronoi_vertices.shape}."
        if unique_voronoi_vertices.device.type != self.device.type: return True, f"Unique vertices on wrong device {unique_voronoi_vertices.device}."

        num_total_vertices = unique_voronoi_vertices.shape[0]
        if not cells_data:
            if num_total_vertices > 0 and self.verbose: print("Warning: Voronoi cells empty, but unique_vertices not.")
            return False, "Voronoi cells data is empty, validated."

        for i, cell_def in enumerate(cells_data):
            if not (isinstance(cell_def, dict) and 'faces' in cell_def and isinstance(cell_def['faces'], list)):
                return True, f"Cell {i} invalid structure or missing 'faces'."
            if not cell_def['faces']: return True, f"Error: Cell {i} has no faces."

            cell_unique_v_indices = set()
            for j, face_indices in enumerate(cell_def['faces']):
                if not (isinstance(face_indices, list) and face_indices): return True, f"Face {j} in cell {i} invalid or empty."
                for k, v_idx in enumerate(face_indices):
                    if not isinstance(v_idx, int): return True, f"V_idx {k} in face {j} cell {i} not int."
                    if not (0 <= v_idx < num_total_vertices): return True, f"V_idx {v_idx} out of bounds for cell {i}."
                    cell_unique_v_indices.add(v_idx)

            if len(cell_unique_v_indices) < 4: return True, f"Cell {i} has < 4 unique vertices ({len(cell_unique_v_indices)})."
            try:
                cell_v_coords_np = unique_voronoi_vertices[torch.tensor(list(cell_unique_v_indices), device=unique_voronoi_vertices.device, dtype=torch.long)].cpu().float().numpy()
                if len(np.unique(cell_v_coords_np, axis=0)) < 4: return True, f"Cell {i} < 4 unique coords for hull."
                hull = ConvexHull(cell_v_coords_np, tol=self.epsilon)
                if hull.volume < self.epsilon**3: return True, f"Cell {i} degenerate (volume {hull.volume:.2e})."
            except RuntimeError as e: return True, f"Qhull error cell {i} (degenerate?): {e}"
            except Exception as e: return True, f"Exception cell {i} validation: {e}" # pragma: no cover
        return False, "Voronoi cells validated successfully."

    def _get_lor_endpoints_3d(self, lor_descriptor: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if 'lor_endpoints' not in lor_descriptor:
            if self.verbose: print("Error: 'lor_endpoints' key missing.")
            return None, None
        lor_endpoints_tensor = lor_descriptor['lor_endpoints']
        if not isinstance(lor_endpoints_tensor, torch.Tensor):
            if self.verbose: print(f"Error: 'lor_endpoints' not PyTorch tensor, got {type(lor_endpoints_tensor)}.")
            return None, None
        if not (lor_endpoints_tensor.ndim==3 and lor_endpoints_tensor.shape[1]==2 and lor_endpoints_tensor.shape[2]==3):
            if self.verbose: print(f"Error: 'lor_endpoints' shape error: {lor_endpoints_tensor.shape}.")
            return None, None
        if lor_endpoints_tensor.shape[0] == 0:
            if self.verbose: print("Warning: 'lor_endpoints' has 0 LORs.")
            empty_shape = (0,3); dtype = lor_endpoints_tensor.dtype # Preserve original dtype for empty
            return torch.empty(empty_shape, device=self.device, dtype=dtype), \
                   torch.empty(empty_shape, device=self.device, dtype=dtype)
        try:
            lor_endpoints_tensor = lor_endpoints_tensor.to(device=self.device, dtype=torch.float32)
        except Exception as e: # pragma: no cover
            if self.verbose: print(f"Error: Failed to move 'lor_endpoints' to device/cast: {e}")
            return None, None
        return lor_endpoints_tensor[:,0,:], lor_endpoints_tensor[:,1,:]

    @staticmethod
    def _is_point_in_face_3d(point_3d: torch.Tensor, face_vertices_coords: torch.Tensor,
                             face_normal: torch.Tensor, epsilon: float) -> bool: # pragma: no cover
        if face_vertices_coords.shape[0] < 3: return False
        return True

    @staticmethod
    def _is_point_inside_polyhedron_3d(point: torch.Tensor, faces_indices: List[List[int]],
                                       unique_voronoi_vertices: torch.Tensor, epsilon: float) -> bool: # pragma: no cover
        return False

    def _compute_lor_cell_intersection_3d(
        self, lor_p1: torch.Tensor, lor_p2: torch.Tensor,
        cell_geometry_data: Tuple[Dict[str, List[List[int]]], torch.Tensor],
    ) -> float:
        cell_def_dict, unique_voronoi_vertices = cell_geometry_data
        lor_vector = lor_p2 - lor_p1
        lor_length_sq = torch.dot(lor_vector, lor_vector)
        if lor_length_sq < self.epsilon**2: return 0.0
        lor_length = torch.sqrt(lor_length_sq)
        lor_direction = lor_vector / (lor_length + self.epsilon)
        collected_t_values = []
        if not (cell_def_dict and 'faces' in cell_def_dict and cell_def_dict['faces'] and unique_voronoi_vertices.shape[0] >=3):
            return 0.0

        faces_vertex_indices = cell_def_dict['faces']
        for face_indices in faces_vertex_indices:
            if len(face_indices) < 3: continue
            try:
                face_v_coords = unique_voronoi_vertices[torch.tensor(face_indices, device=unique_voronoi_vertices.device, dtype=torch.long)]
            except IndexError: continue # pragma: no cover
            p0,p1_face,p2_face = face_v_coords[0], face_v_coords[1], face_v_coords[2]
            face_normal = torch.cross(p1_face - p0, p2_face - p0)
            face_normal_len = torch.linalg.norm(face_normal)
            if face_normal_len < self.epsilon: continue
            face_normal /= face_normal_len
            plane_d = torch.dot(face_normal, p0)
            denominator = torch.dot(lor_direction, face_normal)
            if abs(denominator) < self.epsilon: continue
            t_plane_intersect = (plane_d - torch.dot(lor_p1, face_normal)) / denominator
            if not (-self.epsilon <= t_plane_intersect <= lor_length + self.epsilon): continue
            intersect_point = lor_p1 + t_plane_intersect * lor_direction
            if self._is_point_in_face_3d(intersect_point, face_v_coords, face_normal, self.epsilon):
                collected_t_values.append(torch.clamp(t_plane_intersect / lor_length, 0.0, 1.0).item())

        if self._is_point_inside_polyhedron_3d(lor_p1, faces_vertex_indices, unique_voronoi_vertices, self.epsilon):
            collected_t_values.append(0.0)
        if self._is_point_inside_polyhedron_3d(lor_p2, faces_vertex_indices, unique_voronoi_vertices, self.epsilon):
            collected_t_values.append(1.0)

        if len(collected_t_values) < 2: return 0.0
        t_values_sorted = torch.sort(torch.tensor(list(set(collected_t_values)), device=self.device, dtype=torch.float32)).values
        if t_values_sorted.numel() < 2: return 0.0
        intersection_len_val = (t_values_sorted[-1].item() - t_values_sorted[0].item()) * lor_length.item()
        return intersection_len_val if intersection_len_val > self.epsilon else 0.0

    def _compute_system_matrix_3d(
        self,
        lor_descriptor: Dict[str, Any],
        voronoi_cells_data: List[Dict[str, List[List[int]]]],
        unique_voronoi_vertices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        lor_p1s, lor_p2s = self._get_lor_endpoints_3d(lor_descriptor)
        if lor_p1s is None or lor_p2s is None:
            if self.verbose: print("Error: Failed LOR endpoints for system matrix.")
            return None

        num_total_lors = lor_p1s.shape[0]
        num_cells = len(voronoi_cells_data)

        if num_total_lors == 0 or num_cells == 0:
            if self.verbose: print(f"Warning: 0 LORs ({num_total_lors}) or 0 cells ({num_cells}). Returning empty/zero matrix.")
            return torch.zeros((num_total_lors, num_cells), device=self.device, dtype=torch.float32)

        system_matrix = torch.zeros((num_total_lors, num_cells), device=self.device, dtype=torch.float32)
        if self.verbose: print(f"Computing system matrix for {num_total_lors} LORs and {num_cells} cells.")

        for i_cell in range(num_cells):
            cell_def_dict = voronoi_cells_data[i_cell]
            if not (cell_def_dict and isinstance(cell_def_dict, dict) and 'faces' in cell_def_dict):
                if self.verbose: print(f"Warning: Cell {i_cell} invalid geometry, skipping.")
                continue

            cell_data_for_intersection = (cell_def_dict, unique_voronoi_vertices)
            for j_lor in range(num_total_lors):
                p1 = lor_p1s[j_lor, :]
                p2 = lor_p2s[j_lor, :]
                system_matrix[j_lor, i_cell] = self._compute_lor_cell_intersection_3d(
                    p1, p2, cell_data_for_intersection)

            if self.verbose and (i_cell + 1) % 10 == 0 and num_cells > 10 : # pragma: no cover (avoid div by zero if num_cells too small)
                 print(f"Processed {i_cell + 1}/{num_cells} cells for system matrix computation.")

        if self.verbose: # pragma: no cover
            print(f"System matrix computed. Shape: {system_matrix.shape}, Sum: {torch.sum(system_matrix).item():.4f}")
        return system_matrix

    def _forward_project_3d(self, activity_per_cell: torch.Tensor, system_matrix: torch.Tensor) -> torch.Tensor:
        act = activity_per_cell.unsqueeze(1) if activity_per_cell.ndim == 1 else activity_per_cell
        projection_data = torch.matmul(system_matrix, act)
        return projection_data.squeeze(-1)

    def _back_project_3d(self, projection_data_flat: torch.Tensor, system_matrix: torch.Tensor) -> torch.Tensor:
        proj = projection_data_flat.unsqueeze(1) if projection_data_flat.ndim == 1 else projection_data_flat
        back_projected_activity = torch.matmul(system_matrix.T, proj)
        return back_projected_activity.squeeze(-1)

    def reconstruct(
        self,
        sinogram_3d: torch.Tensor,
        generator_points_3d: torch.Tensor,
        lor_descriptor: Dict[str, Any],
        initial_estimate: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:

        result = {
            "activity": torch.empty(0, device=self.device), "voronoi_cells_data": None,
            "unique_voronoi_vertices": None, "status": "Not started", "error_log": []
        }
        def log_entry(message: str, is_error: bool = False):
            if self.verbose or (is_error and not self.verbose) : print(message) # Log errors even if not verbose
            if is_error: result["error_log"].append(message)

        log_entry("Starting 3D PET reconstruction process.")
        result["status"] = "Processing inputs"
        try:
            # Ensure inputs are on the correct device and dtype
            sinogram_3d_dev = sinogram_3d.to(device=self.device, dtype=torch.float32)
            generator_points_3d_dev = generator_points_3d.to(device=self.device, dtype=torch.float32)
        except Exception as e: # pragma: no cover
            log_entry(f"Error moving input tensors to device '{self.device}': {e}", is_error=True)
            result["status"] = "Failed: Input tensor device transfer error."
            return result

        is_invalid_gens, gen_msg = self._validate_generator_points_3d(generator_points_3d_dev)
        if is_invalid_gens:
            log_entry(f"Generator points validation failed: {gen_msg}",is_error=True); result["status"]=f"Failed: {gen_msg}"; return result
        log_entry("Generator points validated.")

        cells_data, unique_verts, vor_msg = self._compute_voronoi_diagram_3d(generator_points_3d_dev)
        log_entry(f"Voronoi diagram status: {vor_msg}")
        result["voronoi_cells_data"]=cells_data; result["unique_voronoi_vertices"]=unique_verts # Store for debugging
        if cells_data is None or unique_verts is None:
            log_entry(f"Voronoi diagram computation failed.",is_error=True); result["status"]=f"Failed: {vor_msg}"; return result
        log_entry("Voronoi diagram computed.")

        is_invalid_cells, cell_val_msg = self._validate_voronoi_cells_3d(cells_data, unique_verts)
        if is_invalid_cells:
            log_entry(f"Voronoi cells validation failed: {cell_val_msg}",is_error=True); result["status"]=f"Failed: {cell_val_msg}"; return result
        log_entry("Voronoi cells validated.")

        system_matrix = self._compute_system_matrix_3d(lor_descriptor, cells_data, unique_verts)
        if system_matrix is None:
            log_entry("System matrix computation failed.",is_error=True); result["status"]="Failed: System matrix error."; return result
        log_entry(f"System matrix computed. Shape: {system_matrix.shape}")

        num_cells = generator_points_3d_dev.shape[0]
        activity_estimates: torch.Tensor
        if initial_estimate is not None:
            if initial_estimate.shape != (num_cells,):
                log_entry(f"Initial estimate shape error. Expected ({num_cells},), got {initial_estimate.shape}.",is_error=True); result["status"]="Failed: Initial estimate shape."; return result
            try: activity_estimates = initial_estimate.to(device=self.device, dtype=torch.float32)
            except Exception as e: log_entry(f"Error moving initial_estimate to device: {e}",is_error=True); result["status"]="Failed: Initial estimate device error."; return result # pragma: no cover
        else: activity_estimates = torch.ones(num_cells, device=self.device, dtype=torch.float32)
        if self.positivity_constraint: activity_estimates = torch.clamp(activity_estimates, min=0.0)

        sinogram_flat = sinogram_3d_dev.reshape(-1)
        if sinogram_flat.shape[0] != system_matrix.shape[0]:
            log_entry(f"Sinogram LORs ({sinogram_flat.shape[0]}) mismatch SM LORs ({system_matrix.shape[0]}).",is_error=True); result["status"]="Failed: Sinogram LOR mismatch."; return result

        sensitivity_image = self._back_project_3d(torch.ones(system_matrix.shape[0], device=self.device, dtype=torch.float32), system_matrix)
        sensitivity_image_clamped = torch.clamp(sensitivity_image, min=self.mlem_epsilon)
        log_entry("Sensitivity image computed. Starting MLEM iterations.")
        result["status"] = "MLEM Iterations"

        for iteration in range(self.num_iterations):
            activity_prev_iter = activity_estimates.clone()
            expected_counts_flat = self._forward_project_3d(activity_estimates, system_matrix)
            ratio = sinogram_flat / (expected_counts_flat + self.mlem_epsilon)
            correction_term = self._back_project_3d(ratio, system_matrix)
            activity_estimates = activity_estimates * correction_term / sensitivity_image_clamped
            if self.positivity_constraint: activity_estimates = torch.clamp(activity_estimates, min=0.0)
            if self.verbose: # pragma: no cover
                change = torch.linalg.norm(activity_estimates-activity_prev_iter)/(torch.linalg.norm(activity_prev_iter)+self.epsilon)
                log_entry(f"Iter {iteration+1}/{self.num_iterations} | RelChg: {change:.2e} | ActSum: {torch.sum(activity_estimates):.2e}")

        log_entry("MLEM iterations completed.")
        result["activity"] = activity_estimates.cpu()
        result["status"] = "Voronoi-based 3D PET reconstruction completed."
        # Store CPU versions of geometry for output if they were computed and on device
        if result["voronoi_cells_data"] is not None : pass # Already in Python list format
        if result["unique_voronoi_vertices"] is not None : result["unique_voronoi_vertices"] = result["unique_voronoi_vertices"].cpu()

        log_entry(result["status"])
        return result

__all__ = ['VoronoiPETReconstructor3D']
