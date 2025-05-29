import torch 
import numpy as np # Only for type hints in compute_voronoi_density_weights for vor.vertices
from scipy.spatial import Voronoi # For Voronoi diagram computation

EPSILON = 1e-9

# Helper functions provided in the issue description
def monotone_chain_2d(points: torch.Tensor, tol: float = 1e-6):
    """
    Computes the convex hull of 2D points using the Monotone Chain algorithm.
    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D.
        tol (float): Tolerance for floating point comparisons.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - hull_vertices_indices (torch.Tensor): Indices of points forming the convex hull, ordered.
            - hull_simplices (torch.Tensor): Pairs of indices forming the hull edges.
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points tensor must be 2-dimensional with shape (N, 2).")
    if points.shape[0] < 3:
        indices = torch.arange(points.shape[0], device=points.device)
        if points.shape[0] == 2:
            simplices = torch.tensor([[0, 1]], device=points.device, dtype=torch.long)
        else: 
            simplices = torch.empty((0, 2), device=points.device, dtype=torch.long)
        return indices, simplices

    sorted_indices = torch.lexsort((points[:, 1], points[:, 0]))
    # sorted_points = points[sorted_indices] # Not directly used after this line in current logic

    if points[sorted_indices].shape[0] <= 2: # Check based on sorted unique points logic
        final_indices = torch.unique(sorted_indices) 
        if final_indices.shape[0] == 2:
            simplices = torch.tensor([[final_indices[0], final_indices[1]]], device=points.device, dtype=torch.long)
            # Fallback for original points < 3 (already handled by initial check, but for logical flow)
            if points.shape[0] < 3: 
                original_indices = torch.arange(points.shape[0], device=points.device)
                if points.shape[0] == 2:
                    simplices = torch.tensor([[0,1]], dtype=torch.long, device=points.device)
                else:
                    simplices = torch.empty((0,2), dtype=torch.long, device=points.device)
                return original_indices, simplices
            return final_indices, simplices # Return unique indices of the two points forming the line
        else: # Less than 2 unique points
             original_indices = torch.arange(points.shape[0], device=points.device) # Should be caught by points.shape[0] < 3
             simplices = torch.empty((0,2), dtype=torch.long, device=points.device)
             return original_indices, simplices


    upper_hull = []
    lower_hull = []

    def cross_product_orientation(p1_idx, p2_idx, p3_idx, pts_tensor):
        p1 = pts_tensor[p1_idx]
        p2 = pts_tensor[p2_idx]
        p3 = pts_tensor[p3_idx]
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    for i in range(points.shape[0]): # Iterate through original indices after sorting
        current_original_idx = sorted_indices[i]
        while len(upper_hull) >= 2:
            p1_orig_idx = upper_hull[-2]
            p2_orig_idx = upper_hull[-1]
            orientation = cross_product_orientation(p1_orig_idx, p2_orig_idx, current_original_idx, points)
            if orientation >= -tol: 
                upper_hull.pop()
            else:
                break
        upper_hull.append(current_original_idx.item())

    for i in range(points.shape[0] - 1, -1, -1): # Iterate through original indices after sorting (in reverse)
        current_original_idx = sorted_indices[i]
        while len(lower_hull) >= 2:
            p1_orig_idx = lower_hull[-2]
            p2_orig_idx = lower_hull[-1]
            orientation = cross_product_orientation(p1_orig_idx, p2_orig_idx, current_original_idx, points)
            if orientation >= -tol: 
                lower_hull.pop()
            else:
                break
        lower_hull.append(current_original_idx.item())

    hull_vertices_indices_list = upper_hull[:-1] + lower_hull[:-1]
    hull_vertices_indices_list = list(dict.fromkeys(hull_vertices_indices_list)) # Preserve order while making unique
    hull_vertices_indices = torch.tensor(hull_vertices_indices_list, dtype=torch.long, device=points.device)

    num_hull_vertices = hull_vertices_indices.shape[0]
    if num_hull_vertices < 2: 
        simplices = torch.empty((0, 2), dtype=torch.long, device=points.device)
    else:
        simplices_list = []
        for i in range(num_hull_vertices):
            simplices_list.append([hull_vertices_indices[i].item(), hull_vertices_indices[(i + 1) % num_hull_vertices].item()])
        simplices = torch.tensor(simplices_list, dtype=torch.long, device=points.device)

    return hull_vertices_indices, simplices


def monotone_chain_convex_hull_3d(points: torch.Tensor, tol: float = 1e-7):
    n, dim = points.shape
    device = points.device
    if dim != 3: raise ValueError("Points must be 3D.")
    if n < 4:
        unique_indices = torch.unique(torch.arange(n, device=device))
        return unique_indices, torch.empty((0, 3), dtype=torch.long, device=device)

    p0_idx = torch.argmin(points[:, 0])
    p0 = points[p0_idx]
    dists_from_p0 = torch.sum((points - p0)**2, dim=1)
    dists_from_p0[p0_idx] = -1 
    p1_idx = torch.argmax(dists_from_p0)
    p1 = points[p1_idx]
    line_vec = p1 - p0
    if torch.norm(line_vec) < tol:
        for i in range(n):
            if i != p0_idx.item() and torch.norm(points[i] - p0) > tol: # check i != p0_idx.item()
                p1_idx = torch.tensor(i, device=device); p1 = points[p1_idx]; line_vec = p1 - p0; break
        if torch.norm(line_vec) < tol: return torch.unique(torch.tensor([p0_idx.item(), p1_idx.item()], device=device)), torch.empty((0,3),dtype=torch.long,device=device)
    
    ap = points - p0
    t = torch.matmul(ap, line_vec) / (torch.dot(line_vec, line_vec) + EPSILON)
    projections = p0.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
    dists_from_line = torch.sum((points - projections)**2, dim=1)
    dists_from_line[p0_idx] = -1; dists_from_line[p1_idx] = -1
    p2_idx = torch.argmax(dists_from_line)
    p2 = points[p2_idx]

    def compute_plane_normal(pt0, pt1, pt2): return torch.cross(pt1 - pt0, pt2 - pt0)
    normal_p0p1p2 = compute_plane_normal(p0, p1, p2)
    if torch.norm(normal_p0p1p2) < tol:
        found_non_collinear = False
        for i in range(n):
            if i != p0_idx.item() and i != p1_idx.item():
                temp_normal = compute_plane_normal(p0, p1, points[i])
                if torch.norm(temp_normal) > tol:
                    p2_idx = torch.tensor(i, device=device); p2 = points[p2_idx]; normal_p0p1p2 = temp_normal; found_non_collinear = True; break
        if not found_non_collinear: return torch.unique(torch.tensor([p0_idx.item(), p1_idx.item(), p2_idx.item()], device=device)), torch.empty((0,3), dtype=torch.long, device=device)

    dists_from_plane = torch.matmul(points - p0.unsqueeze(0), normal_p0p1p2)
    dists_from_plane[p0_idx] = 0; dists_from_plane[p1_idx] = 0; dists_from_plane[p2_idx] = 0
    p3_idx = torch.argmax(torch.abs(dists_from_plane))
    p3 = points[p3_idx]

    if torch.abs(dists_from_plane[p3_idx]) < tol:
        all_indices = torch.tensor([p0_idx.item(), p1_idx.item(), p2_idx.item(), p3_idx.item()], device=device)
        return torch.unique(all_indices), torch.empty((0, 3), dtype=torch.long, device=device)

    initial_simplex_indices = [p0_idx.item(), p1_idx.item(), p2_idx.item(), p3_idx.item()]
    if torch.dot(p3 - p0, normal_p0p1p2) < 0:
        initial_simplex_indices = [p0_idx.item(), p2_idx.item(), p1_idx.item(), p3_idx.item()]
    
    s = initial_simplex_indices
    faces = [[s[0],s[1],s[2]], [s[0],s[3],s[1]], [s[1],s[3],s[2]], [s[0],s[2],s[3]]]
    current_faces = torch.tensor(faces, dtype=torch.long, device=device)
    hull_vertex_indices = list(set(initial_simplex_indices))
    is_in_simplex = torch.zeros(n, dtype=torch.bool, device=device); 
    for idx_val in initial_simplex_indices: is_in_simplex[idx_val] = True 
    candidate_points_indices = torch.arange(n, device=device)[~is_in_simplex]

    for pt_idx_tensor in candidate_points_indices:
        pt_idx = pt_idx_tensor.item()
        current_point = points[pt_idx]
        visible_faces_indices = []
        for i_face, face_idxs in enumerate(current_faces): 
            p_f0,p_f1,p_f2 = points[face_idxs[0]], points[face_idxs[1]], points[face_idxs[2]] 
            face_normal = compute_plane_normal(p_f0, p_f1, p_f2)
            if torch.dot(current_point - p_f0, face_normal) > tol:
                visible_faces_indices.append(i_face)
        if not visible_faces_indices: continue
        hull_vertex_indices.append(pt_idx)
        edge_count = {}
        for i_face in visible_faces_indices: 
            face = current_faces[i_face]
            edges_on_face = [tuple(sorted((face[0].item(), face[1].item()))), tuple(sorted((face[1].item(), face[2].item()))), tuple(sorted((face[2].item(), face[0].item())))]
            for edge in edges_on_face: edge_count[edge] = edge_count.get(edge, 0) + 1
        horizon_edges = [edge for edge, count in edge_count.items() if count == 1]
        faces_to_keep_mask = torch.ones(current_faces.shape[0], dtype=torch.bool, device=device)
        for i_face in visible_faces_indices: faces_to_keep_mask[i_face] = False 
        new_faces_list = [f.tolist() for f in current_faces[faces_to_keep_mask]]
        for edge_tuple in horizon_edges: new_faces_list.append([pt_idx, edge_tuple[1], edge_tuple[0]])
        if not new_faces_list and pt_idx in hull_vertex_indices and not horizon_edges : hull_vertex_indices.pop()
        current_faces = torch.tensor(new_faces_list, dtype=torch.long, device=device) if new_faces_list else torch.empty((0,3),dtype=torch.long,device=device)
        if current_faces.numel() == 0 and n >=4 : break
    
    if current_faces.numel() > 0: final_hull_vertex_indices = torch.unique(current_faces.flatten())
    elif n > 0 : 
        if len(hull_vertex_indices) > 0: final_hull_vertex_indices = torch.tensor(list(set(hull_vertex_indices)), dtype=torch.long, device=device)
        else: unique_pts, _ = torch.sort(torch.unique(torch.arange(n, device=device))); final_hull_vertex_indices = unique_pts
    else: final_hull_vertex_indices = torch.empty((0,), dtype=torch.long, device=device)
    valid_faces = []
    if current_faces.numel() > 0:
        for face_idx_list_item in current_faces: 
            if len(torch.unique(face_idx_list_item)) == 3: valid_faces.append(face_idx_list_item.tolist()) 
    final_faces = torch.tensor(valid_faces, dtype=torch.long, device=device)
    return final_hull_vertex_indices, final_faces


class ConvexHull:
    def __init__(self, points: torch.Tensor, tol: float = 1e-6):
        if not isinstance(points, torch.Tensor): raise ValueError("Input points must be a PyTorch tensor.")
        if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")
        self.points, self.device, self.dtype, self.dim, self.tol = points, points.device, points.dtype, points.shape[1], tol
        if self.dim not in [2, 3]: raise ValueError("Only 2D and 3D points are supported.")
        self.vertices, self.simplices, self._area, self._volume = None, None, None, None
        self._compute_hull()

    def _compute_hull(self):
        if self.dim == 2: self._convex_hull_2d()
        else: self._convex_hull_3d()

    def _convex_hull_2d(self):
        self.vertices, self.simplices = monotone_chain_2d(self.points, self.tol)
        if self.vertices.shape[0] < 3: self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            hull_pts = self.points[self.vertices]; x,y = hull_pts[:,0], hull_pts[:,1]
            self._area = (0.5 * torch.abs(torch.sum(x * torch.roll(y,-1) - torch.roll(x,-1) * y))).to(self.dtype)
    
    def _compute_face_normal(self,v0,v1,v2): return torch.cross(v1-v0, v2-v0)

    def _convex_hull_3d(self):
        self.vertices, self.simplices = monotone_chain_convex_hull_3d(self.points, self.tol)
        if self.simplices is None or self.simplices.shape[0]==0:
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            if self.vertices is None or self.vertices.numel()==0: self.vertices = torch.empty((0,),dtype=torch.long,device=self.device)
            return
        if self.simplices.dtype != torch.long: self.simplices = self.simplices.to(dtype=torch.long)
        if self.vertices.numel() < 1: self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype); return # Use numel for 0-dim or empty
        ref_pt = self.points[self.vertices[0]] 
        total_vol = torch.tensor(0.0,device=self.device,dtype=self.points.dtype) # Match dtype with points for precision
        for face_idxs in self.simplices: 
            p0,p1,p2 = self.points[face_idxs[0]],self.points[face_idxs[1]],self.points[face_idxs[2]] 
            total_vol += torch.dot(p0-ref_pt, torch.cross(p1-ref_pt, p2-ref_pt))/6.0
        self._volume = torch.abs(total_vol)

    @property
    def area(self) -> torch.Tensor:
        if self.dim == 2:
            if self._area is None: self._convex_hull_2d()
            return self._area if self._area is not None else torch.tensor(0.0,device=self.device,dtype=self.dtype)
        else: 
            surface_area = torch.tensor(0.0, device=self.device, dtype=self.points.dtype) # Match dtype
            if self.simplices is not None and self.simplices.shape[0] > 0:
                for face_idxs in self.simplices: 
                    idx = face_idxs.long() 
                    p0,p1,p2 = self.points[idx[0]],self.points[idx[1]],self.points[idx[2]]
                    surface_area += 0.5 * torch.norm(self._compute_face_normal(p0,p1,p2))
            return surface_area

    @property
    def volume(self) -> torch.Tensor:
        if self.dim == 3:
            if self._volume is None: self._convex_hull_3d()
            return self._volume if self._volume is not None else torch.tensor(0.0,device=self.device,dtype=self.dtype)
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

# --- Sutherland-Hodgman Polygon Clipping Helpers ---
def _sutherland_hodgman_is_inside(point: torch.Tensor, edge_type: str, clip_value: float) -> bool:
    if edge_type == 'left': return point[0] >= clip_value
    elif edge_type == 'top': return point[1] <= clip_value
    elif edge_type == 'right': return point[0] <= clip_value
    elif edge_type == 'bottom': return point[1] >= clip_value
    return False

def _sutherland_hodgman_intersect(p1: torch.Tensor, p2: torch.Tensor, 
                                  clip_edge_p1: torch.Tensor, clip_edge_p2: torch.Tensor) -> torch.Tensor:
    x1, y1 = p1[0].to(torch.float64), p1[1].to(torch.float64)
    x2, y2 = p2[0].to(torch.float64), p2[1].to(torch.float64)
    x3, y3 = clip_edge_p1[0].to(torch.float64), clip_edge_p1[1].to(torch.float64)
    x4, y4 = clip_edge_p2[0].to(torch.float64), clip_edge_p2[1].to(torch.float64)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if torch.abs(denominator) < EPSILON: return p2 
    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_numerator / denominator
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    return torch.tensor([intersect_x, intersect_y], dtype=p1.dtype, device=p1.device)

# --- Delaunay Triangulation 3D: PyTorch Implementation ---
def _orientation3d_pytorch(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, tol: float) -> int:
    mat = torch.stack((p2 - p1, p3 - p1, p4 - p1), dim=0)
    det_val = torch.det(mat.to(dtype=torch.float64))
    if torch.abs(det_val) < tol: return 0
    return 1 if det_val > 0 else -1

def _in_circumsphere3d_pytorch(p: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, tol: float) -> bool:
    points_for_mat = [t1, t2, t3, t4, p]
    mat_rows = []
    for pt_i in points_for_mat:
        pt_i_64 = pt_i.to(dtype=torch.float64)
        sum_sq = torch.sum(pt_i_64**2)
        mat_rows.append(torch.cat((pt_i_64, sum_sq.unsqueeze(0), torch.tensor([1.0], dtype=torch.float64, device=p.device))))
    mat_5x5 = torch.stack(mat_rows, dim=0)
    orient_mat = torch.stack((t2 - t1, t3 - t1, t4 - t1), dim=0).to(dtype=torch.float64)
    orient_det_val = torch.det(orient_mat)
    if torch.abs(orient_det_val) < tol: return False 
    circumsphere_det_val = torch.det(mat_5x5)
    return (orient_det_val * circumsphere_det_val) > tol

def delaunay_triangulation_3d(points: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    n_input_points, dim = points.shape
    device = points.device; original_dtype = points.dtype
    if dim != 3: raise ValueError("Input points must be 3-dimensional.")
    if n_input_points < 4: return torch.empty((0, 4), dtype=torch.long, device=device)
    points_calc = points
    min_coords, max_coords = torch.min(points_calc,0).values, torch.max(points_calc,0).values
    center = (min_coords + max_coords) / 2.0
    diag_len = torch.norm(max_coords - min_coords); 
    if diag_len < tol : diag_len = 1.0 
    scale_factor_super = max(5.0 * diag_len, 10.0) 
    sp_idx_start = n_input_points
    sp_v0 = center + torch.tensor([-scale_factor_super,-scale_factor_super/2,-scale_factor_super/2], device=device,dtype=original_dtype)
    sp_v1 = center + torch.tensor([scale_factor_super,-scale_factor_super/2,-scale_factor_super/2], device=device,dtype=original_dtype)
    sp_v2 = center + torch.tensor([0.0,scale_factor_super,-scale_factor_super/2], device=device,dtype=original_dtype)
    sp_v3 = center + torch.tensor([0.0,0.0,scale_factor_super], device=device,dtype=original_dtype)
    super_tetra_vertices = torch.stack([sp_v0,sp_v1,sp_v2,sp_v3],dim=0)
    all_points = torch.cat([points_calc, super_tetra_vertices], dim=0)
    st_indices = torch.tensor([sp_idx_start,sp_idx_start+1,sp_idx_start+2,sp_idx_start+3],dtype=torch.long,device=device)
    p0_st,p1_st,p2_st,p3_st = all_points[st_indices[0]],all_points[st_indices[1]],all_points[st_indices[2]],all_points[st_indices[3]]
    if _orientation3d_pytorch(p0_st,p1_st,p2_st,p3_st,tol) < 0: st_indices[[1,2]] = st_indices[[2,1]]
    triangulation = [st_indices.tolist()]
    permutation = torch.arange(n_input_points, device=device)
    for i_point_orig_idx in range(n_input_points):
        current_point_idx = permutation[i_point_orig_idx].item()
        current_point_coords = all_points[current_point_idx]
        bad_tetrahedra_indices = []
        for tet_idx, tet_v_indices_list in enumerate(triangulation): 
            tet_v_indices = torch.tensor(tet_v_indices_list,dtype=torch.long,device=device) 
            v1,v2,v3,v4 = all_points[tet_v_indices[0]],all_points[tet_v_indices[1]],all_points[tet_v_indices[2]],all_points[tet_v_indices[3]]
            if _in_circumsphere3d_pytorch(current_point_coords,v1,v2,v3,v4,tol): bad_tetrahedra_indices.append(tet_idx)
        if not bad_tetrahedra_indices: continue
        boundary_faces, face_counts = [], {}
        for tet_idx in bad_tetrahedra_indices:
            tet = triangulation[tet_idx]
            faces_of_tet = [sorted([tet[0],tet[1],tet[2]]), sorted([tet[0],tet[1],tet[3]]), sorted([tet[0],tet[2],tet[3]]), sorted([tet[1],tet[2],tet[3]])]
            for face_nodes_s_list in faces_of_tet: face_counts[tuple(face_nodes_s_list)] = face_counts.get(tuple(face_nodes_s_list),0)+1 
        for face_tuple, count in face_counts.items(): 
            if count == 1: boundary_faces.append(list(face_tuple))
        for tet_idx in sorted(bad_tetrahedra_indices,reverse=True): triangulation.pop(tet_idx)
        for face_nodes_list_item in boundary_faces: 
            new_tet_nodes = face_nodes_list_item + [current_point_idx] 
            p1_f,p2_f,p3_f = all_points[face_nodes_list_item[0]],all_points[face_nodes_list_item[1]],all_points[face_nodes_list_item[2]] 
            current_orientation = _orientation3d_pytorch(p1_f,p2_f,p3_f,current_point_coords,tol)
            if current_orientation == 0: continue
            new_tet_final_nodes = [face_nodes_list_item[0],face_nodes_list_item[2],face_nodes_list_item[1],current_point_idx] if current_orientation < 0 else new_tet_nodes 
            triangulation.append(new_tet_final_nodes)
    final_triangulation = [tet_nodes_list_item for tet_nodes_list_item in triangulation if not any(node_idx >= n_input_points for node_idx in tet_nodes_list_item)] 
    if not final_triangulation: return torch.empty((0,4),dtype=torch.long,device=device)
    return torch.tensor(final_triangulation,dtype=torch.long,device=device)

# --- Sutherland-Hodgman Polygon Clipping Main Function ---
def clip_polygon_2d(polygon_vertices: torch.Tensor, clip_bounds: torch.Tensor) -> torch.Tensor:
    """
    Clips a 2D polygon against a rectangular bounding box using the Sutherland-Hodgman algorithm.

    Args:
        polygon_vertices (torch.Tensor): A tensor of shape (N, 2) representing the
                                         ordered vertices of the input polygon.
        clip_bounds (torch.Tensor): A tensor of shape (2, 2) defining the rectangular
                                    clipping window: [[min_x, min_y], [max_x, max_y]].

    Returns:
        torch.Tensor: A tensor of shape (M, 2) representing the ordered vertices
                      of the clipped polygon. Returns an empty tensor (0, 2)
                      if the polygon is entirely outside the clip_bounds.
    """
    if not isinstance(polygon_vertices, torch.Tensor) or polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
        raise ValueError("polygon_vertices must be a tensor of shape (N, 2).")
    if not isinstance(clip_bounds, torch.Tensor) or clip_bounds.shape != (2,2):
        raise ValueError("clip_bounds must be a tensor of shape (2, 2).")

    min_x, min_y = clip_bounds[0, 0], clip_bounds[0, 1]
    max_x, max_y = clip_bounds[1, 0], clip_bounds[1, 1]

    if not (min_x <= max_x and min_y <= max_y):
        raise ValueError("Clip bounds min must be less than or equal to max for each dimension.")

    # Device and dtype for new tensors should match polygon_vertices
    device = polygon_vertices.device
    dtype = polygon_vertices.dtype

    # Define clip edges: (edge_type, clip_value, clip_edge_p1, clip_edge_p2)
    # Points defining clip edges must be on the same device and dtype for intersection function
    clip_edges = [
        ('left',   min_x, torch.tensor([min_x, min_y], device=device, dtype=dtype), torch.tensor([min_x, max_y], device=device, dtype=dtype)), # Left edge x = min_x
        ('top',    max_y, torch.tensor([min_x, max_y], device=device, dtype=dtype), torch.tensor([max_x, max_y], device=device, dtype=dtype)), # Top edge y = max_y
        ('right',  max_x, torch.tensor([max_x, max_y], device=device, dtype=dtype), torch.tensor([max_x, min_y], device=device, dtype=dtype)), # Right edge x = max_x
        ('bottom', min_y, torch.tensor([max_x, min_y], device=device, dtype=dtype), torch.tensor([min_x, min_y], device=device, dtype=dtype))  # Bottom edge y = min_y
    ]

    output_vertices = polygon_vertices.clone()

    for edge_type, clip_value, clip_edge_p1, clip_edge_p2 in clip_edges:
        input_vertices = output_vertices.clone() # Vertices from previous clipping stage
        
        if input_vertices.shape[0] == 0: # Polygon fully clipped already
            output_vertices = torch.empty((0, 2), dtype=dtype, device=device)
            break 
        
        clipped_polygon_against_this_edge = [] # Python list to store vertices for this stage
        num_input_verts = input_vertices.shape[0]
        
        # Start with the last vertex to form edge with the first
        p1 = input_vertices[num_input_verts - 1] 

        for k_idx in range(num_input_verts):
            p2 = input_vertices[k_idx]
            
            p1_is_inside = _sutherland_hodgman_is_inside(p1, edge_type, clip_value)
            p2_is_inside = _sutherland_hodgman_is_inside(p2, edge_type, clip_value)

            if p1_is_inside and p2_is_inside: # Case 1: Both points inside
                clipped_polygon_against_this_edge.append(p2)
            elif p1_is_inside and not p2_is_inside: # Case 2: P1 inside, P2 outside (outgoing edge)
                intersection_pt = _sutherland_hodgman_intersect(p1, p2, clip_edge_p1, clip_edge_p2)
                clipped_polygon_against_this_edge.append(intersection_pt)
            elif not p1_is_inside and not p2_is_inside: # Case 3: Both points outside
                pass # Do nothing
            elif not p1_is_inside and p2_is_inside: # Case 4: P1 outside, P2 inside (incoming edge)
                intersection_pt = _sutherland_hodgman_intersect(p1, p2, clip_edge_p1, clip_edge_p2)
                clipped_polygon_against_this_edge.append(intersection_pt)
                clipped_polygon_against_this_edge.append(p2)
            
            p1 = p2 # Move to next edge

        if len(clipped_polygon_against_this_edge) > 0:
            output_vertices = torch.stack(clipped_polygon_against_this_edge)
        else: # Polygon was entirely clipped by this edge
            output_vertices = torch.empty((0, 2), dtype=dtype, device=device)
            break # No need to clip against further edges

    return output_vertices


def compute_polygon_area(points: torch.Tensor) -> float:
    """
    Computes the area of a 2D polygon (convex hull) using the PyTorch ConvexHull class.

    Args:
        points: A PyTorch tensor of shape (N, 2) representing the polygon's vertices,
                where N is the number of points.

    Returns:
        The area of the polygon's convex hull.

    Raises:
        ValueError: If the input points are invalid (not a PyTorch tensor,
                      not 2D, less than 3 points, or not (N, 2) shape).
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must be a 2D tensor with shape (N, 2).")
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to compute polygon area.")
    
    # Uses the new PyTorch ConvexHull class from the same file
    # The ConvexHull class's 'area' property is designed for 2D.
    # Pass the module-level EPSILON as the tolerance for the ConvexHull class.
    hull = ConvexHull(points, tol=EPSILON) 
    
    # hull.area returns a torch.Tensor. Use .item() to get the float value.
    calculated_area = hull.area.item()

    # The ConvexHull class itself does not check against EPSILON for the final area.
    # The internal monotone_chain_2d uses its tol for geometric comparisons.
    # The Shoelace formula computes the area directly.
    # It's good practice to check for very small areas here.
    # If the computed area is extremely small, treat as zero.
    if abs(calculated_area) < EPSILON:
        return 0.0
    return calculated_area

def compute_convex_hull_volume(points: torch.Tensor) -> float:
    """
    Computes the volume of the convex hull of a set of 3D points using the PyTorch ConvexHull class.

    Args:
        points: A PyTorch tensor of shape (N, 3) representing the 3D points,
                where N is the number of points.

    Returns:
        The volume of the convex hull.

    Raises:
        ValueError: If the input points are invalid (not a PyTorch tensor,
                      not 2D, not (N, 3) shape, or less than 4 points).
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be a 3D tensor with shape (N, 3).")
    if points.shape[0] < 4:
        raise ValueError("At least 4 points are required to compute 3D convex hull volume.")

    # Uses the new PyTorch ConvexHull class from the same file
    hull = ConvexHull(points, tol=EPSILON) # Pass module EPSILON as tol
    
    # The ConvexHull class's 'volume' property is designed for 3D.
    # hull.volume returns a torch.Tensor. Use .item() to get the float value.
    calculated_volume = hull.volume.item()

    # The ConvexHull class itself does not check against EPSILON for the final volume.
    # The internal monotone_chain_convex_hull_3d uses its tol for geometric comparisons.
    # The volume calculation sums tetrahedra volumes.
    # It's good practice to check for very small volumes here.
    # If the computed volume is extremely small, treat as zero.
    if abs(calculated_volume) < EPSILON:
        return 0.0
    return calculated_volume

def normalize_weights(weights: torch.Tensor, target_sum: float = 1.0, tol: float = 1e-6) -> torch.Tensor:
    """ Normalizes a tensor of weights to sum to a target value (default 1.0).

    Args:
        weights (torch.Tensor): Input tensor of weights (1D, shape (N,)).
        target_sum (float): Desired sum of the normalized weights (default: 1.0).
        tol (float): Tolerance for handling zero or negative weights and sum checks.

    Returns:
        torch.Tensor: Normalized weights with the same shape as input, summing to target_sum.

    Raises:
        AssertionError: If weights is not a 1D tensor or contains values less than -tol.
        ValueError: If sum of weights after clamping is less than tol.
    """
    if not isinstance(weights, torch.Tensor):
        raise TypeError("Input weights must be a PyTorch tensor.")
    assert weights.dim() == 1, "Weights must be a 1D tensor"
    assert torch.all(weights >= -tol), f"Weights must be non-negative (or within -{tol} tolerance)."
    weights = torch.clamp(weights, min=0.0)
    weight_sum = torch.sum(weights)
    if weight_sum < tol:
        raise ValueError(f"Sum of weights ({weight_sum.item()}) is less than tolerance ({tol}); cannot normalize.")
    return weights * (target_sum / weight_sum)

from scipy.spatial import Voronoi 
# ConvexHull is already available from this file (defined above). EPSILON is also global.

def compute_voronoi_density_weights(points: torch.Tensor, 
                                    bounds: torch.Tensor = None, 
                                    space_dim: int = None) -> torch.Tensor:
    """
    Computes Voronoi-based density compensation weights for a set of k-space points.
    Uses Sutherland-Hodgman for 2D clipping if bounds are provided.
    Uses a heuristic (convex hull of Voronoi vertices + bound corners) for 3D open regions with bounds.
    """
    if not isinstance(points, torch.Tensor): raise TypeError("Input points must be a PyTorch tensor.")
    if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")

    original_device = points.device
    points_cpu_numpy = points.cpu().numpy()

    if space_dim is None: space_dim = points.shape[1]
    if space_dim not in [2, 3]: raise ValueError(f"space_dim must be 2 or 3, got {space_dim}.")
    if points.shape[1] != space_dim: raise ValueError(f"Points dim {points.shape[1]} != space_dim {space_dim}.")

    n_points = points.shape[0]
    if n_points == 0: return torch.empty(0, dtype=points.dtype, device=original_device)
    if n_points <= space_dim:
        return torch.full((n_points,), 1.0/(n_points if n_points>0 else 1.0), dtype=points.dtype, device=original_device)

    bounds_cpu = None
    if bounds is not None:
        if not isinstance(bounds, torch.Tensor): raise TypeError("Bounds must be a PyTorch tensor.")
        bounds_cpu = bounds.cpu()
        expected_bounds_shape = (2, space_dim)
        if bounds_cpu.shape != expected_bounds_shape: raise ValueError(f"Bounds shape must be {expected_bounds_shape}, got {bounds_cpu.shape}.")
    
    try:
        qhull_opt = 'Qbb Qc Qz' if space_dim == 2 else 'Qbb Qc'
        vor = Voronoi(points_cpu_numpy, qhull_options=qhull_opt)
    except Exception:
        return torch.full((n_points,), EPSILON, dtype=points.dtype, device=original_device)

    weights_list = []
    min_measure_floor = EPSILON

    for i in range(n_points):
        region_idx = vor.point_region[i]
        region_vertex_indices = vor.regions[region_idx]

        if not region_vertex_indices:
            weights_list.append(1.0 / min_measure_floor); continue
        
        is_open_region = -1 in region_vertex_indices
        finite_region_v_indices = [v_idx for v_idx in region_vertex_indices if v_idx != -1]

        if not finite_region_v_indices and is_open_region:
            weights_list.append(1.0 / min_measure_floor); continue
        
        current_region_vor_vertices_np = vor.vertices[finite_region_v_indices] if finite_region_v_indices else np.empty((0, space_dim))
        current_region_vor_vertices_torch_cpu = torch.tensor(current_region_vor_vertices_np, dtype=points.dtype, device='cpu')

        vertices_for_final_hull_calc = None

        if space_dim == 2:
            if bounds_cpu is not None:
                if not is_open_region: # Closed 2D region, with bounds
                    if current_region_vor_vertices_torch_cpu.shape[0] >= 3:
                        vertices_for_final_hull_calc = clip_polygon_2d(current_region_vor_vertices_torch_cpu, bounds_cpu)
                    else: weights_list.append(1.0 / min_measure_floor); continue
                else: # Open 2D region, with bounds
                    temp_vertices_for_heuristic_hull = []
                    if current_region_vor_vertices_torch_cpu.shape[0] > 0:
                         temp_vertices_for_heuristic_hull.append(current_region_vor_vertices_torch_cpu)
                    min_coord, max_coord = bounds_cpu[0], bounds_cpu[1]
                    corners = torch.tensor([[min_coord[0],min_coord[1]], [max_coord[0],min_coord[1]], [min_coord[0],max_coord[1]], [max_coord[0],max_coord[1]]], dtype=points.dtype,device='cpu')
                    temp_vertices_for_heuristic_hull.append(corners)
                    
                    if not temp_vertices_for_heuristic_hull: weights_list.append(1.0 / min_measure_floor); continue
                    combined_heuristic_points = torch.cat(temp_vertices_for_heuristic_hull, dim=0)
                    unique_heuristic_points = torch.unique(combined_heuristic_points, dim=0)

                    if unique_heuristic_points.shape[0] < 3: weights_list.append(1.0 / min_measure_floor); continue
                    try:
                        initial_closed_hull = ConvexHull(unique_heuristic_points, tol=EPSILON)
                        if initial_closed_hull.vertices is None or initial_closed_hull.vertices.numel() < 3:
                            weights_list.append(1.0 / min_measure_floor); continue
                        initial_closed_polygon_vertices = initial_closed_hull.points[initial_closed_hull.vertices]
                        vertices_for_final_hull_calc = clip_polygon_2d(initial_closed_polygon_vertices, bounds_cpu)
                    except (ValueError, RuntimeError): weights_list.append(1.0 / min_measure_floor); continue
            elif not is_open_region: # Closed 2D region, no bounds
                vertices_for_final_hull_calc = current_region_vor_vertices_torch_cpu
            else: # Open 2D region, no bounds
                weights_list.append(1.0 / min_measure_floor); continue
        
        elif space_dim == 3: # 3D logic
            if is_open_region and bounds_cpu is not None: # Open 3D region, with bounds (heuristic)
                temp_vertices_for_heuristic_hull = []
                if current_region_vor_vertices_torch_cpu.shape[0] > 0:
                     temp_vertices_for_heuristic_hull.append(current_region_vor_vertices_torch_cpu)
                min_coord, max_coord = bounds_cpu[0], bounds_cpu[1]
                corners = torch.tensor([[min_c[0],min_c[1],min_c[2]], [max_c[0],min_c[1],min_c[2]], [min_c[0],max_c[1],min_c[2]], [max_c[0],max_c[1],min_c[2]],[min_c[0],min_c[1],max_c[2]], [max_c[0],min_c[1],max_c[2]],[min_c[0],max_c[1],max_c[2]], [max_c[0],max_c[1],max_c[2]]] for min_c,max_c in [(min_coord,max_coord)]][0], dtype=points.dtype, device='cpu') # Simplified corner generation
                vertices_for_hull_list_3d = [current_region_vor_vertices_torch_cpu] if current_region_vor_vertices_torch_cpu.shape[0]>0 else []
                vertices_for_hull_list_3d.append(corners)
                if not vertices_for_hull_list_3d : weights_list.append(1.0/min_measure_floor); continue
                combined_vertices_for_3d_open = torch.cat(vertices_for_hull_list_3d, dim=0)
                vertices_for_final_hull_calc = torch.unique(combined_vertices_for_3d_open, dim=0)
            elif not is_open_region: # Closed 3D region (with or without bounds, clipping not implemented for 3D polyhedra)
                vertices_for_final_hull_calc = current_region_vor_vertices_torch_cpu
            else: # Open 3D region, no bounds
                 weights_list.append(1.0 / min_measure_floor); continue
        else: # Should not be reached if space_dim is 2 or 3
            weights_list.append(1.0 / min_measure_floor); continue


        if vertices_for_final_hull_calc is None or vertices_for_final_hull_calc.shape[0] < space_dim + 1:
            weights_list.append(1.0 / min_measure_floor); continue
        try:
            hull_of_region = ConvexHull(vertices_for_final_hull_calc, tol=EPSILON) 
        except (ValueError, RuntimeError): 
            weights_list.append(1.0 / min_measure_floor); continue

        cell_measure_tensor = hull_of_region.area if space_dim == 2 else hull_of_region.volume
        cell_measure = cell_measure_tensor.item() 
        actual_measure = min_measure_floor if abs(cell_measure) < min_measure_floor else abs(cell_measure)
        weights_list.append(1.0 / actual_measure)

    final_weights = torch.tensor(weights_list, dtype=points.dtype, device='cpu') 
    return final_weights.to(original_device)

# --- 3D Polyhedron Clipping Helpers ---
def _point_plane_signed_distance(points_tensor: torch.Tensor, plane_normal: torch.Tensor, plane_dist_offset: float) -> torch.Tensor:
    # points_tensor: (N, 3), plane_normal: (3,)
    # Returns: (N,) tensor of signed distances. Positive if on normal side.
    return torch.matmul(points_tensor, plane_normal) - plane_dist_offset

def _segment_plane_intersection(p1: torch.Tensor, p2: torch.Tensor, 
                                plane_normal: torch.Tensor, plane_dist_offset: float, 
                                tol: float = 1e-7) -> torch.Tensor | None:
    dp = p2 - p1
    den = torch.dot(dp, plane_normal)

    if torch.abs(den) < tol: # Segment parallel to plane
        # Check if p1 (and thus segment) is on the plane
        # if torch.abs(torch.dot(p1, plane_normal) - plane_dist_offset) < tol:
        #     return p1 # Or some other indicator that segment is on plane. For clipping, this means no unique intersection.
        return None # No unique intersection point

    t = (plane_dist_offset - torch.dot(p1, plane_normal)) / den
    
    # Intersection must be within the segment [p1, p2]
    # Using tol to avoid issues at endpoints that are already classified by is_inside checks
    if tol <= t <= 1.0 - tol: 
        return p1 + t * dp
    return None

def clip_polyhedron_3d(input_poly_vertices: torch.Tensor, 
                       bounding_box: torch.Tensor, 
                       tol: float = 1e-7) -> torch.Tensor:
    """
    Clips a 3D convex polyhedron against an axis-aligned bounding box.
    Collects original vertices inside the box and intersection points of edges with box planes,
    then computes the convex hull of these collected points.

    Args:
        input_poly_vertices (torch.Tensor): Shape (N, 3), vertices of the input convex polyhedron.
        bounding_box (torch.Tensor): Shape (2, 3), [[min_x,min_y,min_z], [max_x,max_y,max_z]].
        tol (float): Tolerance for geometric computations.

    Returns:
        torch.Tensor: Shape (M, 3), vertices of the clipped polyhedron. Empty (0,3) if fully clipped or degenerate.
    """
    if not isinstance(input_poly_vertices, torch.Tensor) or input_poly_vertices.ndim != 2 or input_poly_vertices.shape[1] != 3:
        raise ValueError("input_poly_vertices must be a tensor of shape (N, 3).")
    if not isinstance(bounding_box, torch.Tensor) or bounding_box.shape != (2,3):
        raise ValueError("bounding_box must be a tensor of shape (2, 3).")

    device = input_poly_vertices.device
    dtype = input_poly_vertices.dtype

    min_coords = bounding_box[0]
    max_coords = bounding_box[1]

    if not (torch.all(min_coords <= max_coords)):
        raise ValueError("Bounding box min_coords must be less than or equal to max_coords for each dimension.")

    # Define the 6 planes of the bounding box: (normal, d_offset)
    # Normals point inwards for the "inside" check: dot(v, normal) - d >= -tol
    planes = [
        (torch.tensor([1,0,0], dtype=dtype, device=device),  min_coords[0]), # x >= min_x
        (torch.tensor([-1,0,0], dtype=dtype, device=device), -max_coords[0]),# x <= max_x  (-x >= -max_x)
        (torch.tensor([0,1,0], dtype=dtype, device=device),  min_coords[1]), # y >= min_y
        (torch.tensor([0,-1,0], dtype=dtype, device=device), -max_coords[1]),# y <= max_y
        (torch.tensor([0,0,1], dtype=dtype, device=device),  min_coords[2]), # z >= min_z
        (torch.tensor([0,0,-1], dtype=dtype, device=device), -max_coords[2]) # z <= max_z
    ]

    final_vertex_candidates = []

    # 1. Collect original vertices that are inside all 6 planes
    for v_idx in range(input_poly_vertices.shape[0]):
        v = input_poly_vertices[v_idx]
        is_fully_inside = True
        for plane_normal, plane_d_offset in planes:
            if _point_plane_signed_distance(v.unsqueeze(0), plane_normal, plane_d_offset).item() < -tol:
                is_fully_inside = False
                break
        if is_fully_inside:
            final_vertex_candidates.append(v)

    # 2. Collect intersection points of polyhedron edges with bounding box planes
    if input_poly_vertices.shape[0] >= 4: # Need enough points to form a 3D hull for edges
        try:
            # Ensure input_poly_vertices is on CPU if ConvexHull's PyTorch part isn't fully robust to GPU inputs
            # or if its fallback to SciPy is triggered. The current ConvexHull should handle it.
            initial_hull = ConvexHull(input_poly_vertices, tol=tol)
            
            if initial_hull.simplices is not None and initial_hull.simplices.numel() > 0:
                unique_edges = set()
                # Simplices are faces (triangles), extract edges
                for face in initial_hull.simplices: # face is [idx0, idx1, idx2]
                    face_indices = face.tolist()
                    edges_on_face = [
                        tuple(sorted((face_indices[0], face_indices[1]))),
                        tuple(sorted((face_indices[1], face_indices[2]))),
                        tuple(sorted((face_indices[2], face_indices[0])))
                    ]
                    unique_edges.update(edges_on_face)

                for edge_indices in unique_edges:
                    p1 = input_poly_vertices[edge_indices[0]]
                    p2 = input_poly_vertices[edge_indices[1]]
                    
                    for plane_normal, plane_d_offset in planes:
                        intersect_pt = _segment_plane_intersection(p1, p2, plane_normal, plane_d_offset, tol)
                        if intersect_pt is not None:
                            # Verify this intersection point is within the *entire* bounding box
                            is_intersection_valid_for_box = True
                            for check_normal, check_d_offset in planes:
                                if _point_plane_signed_distance(intersect_pt.unsqueeze(0), check_normal, check_d_offset).item() < -tol:
                                    is_intersection_valid_for_box = False
                                    break
                            if is_intersection_valid_for_box:
                                final_vertex_candidates.append(intersect_pt)
        except (ValueError, RuntimeError): # Catch potential errors from ConvexHull (e.g. degenerate input)
            # If initial hull fails, we might not be able to get edges, proceed with vertices found so far
            pass 


    if not final_vertex_candidates:
        return torch.empty((0, 3), dtype=dtype, device=device)

    unique_final_candidates = torch.unique(torch.stack(final_vertex_candidates), dim=0)

    if unique_final_candidates.shape[0] < 4:
        return torch.empty((0, 3), dtype=dtype, device=device)

    try:
        clipped_hull = ConvexHull(unique_final_candidates, tol=tol)
        if clipped_hull.simplices is None or clipped_hull.simplices.numel() == 0 or clipped_hull.vertices.numel() < 4 :
            return torch.empty((0, 3), dtype=dtype, device=device)
        
        # Return the vertices that form the faces of the new clipped hull
        # These are indices into unique_final_candidates
        final_vertices_indices = torch.unique(clipped_hull.simplices.flatten())
        return unique_final_candidates[final_vertices_indices]

    except (ValueError, RuntimeError): # Catch errors from the final ConvexHull
        return torch.empty((0, 3), dtype=dtype, device=device)


def compute_polygon_area(points: torch.Tensor) -> float:
    """
    Computes the area of a 2D polygon (convex hull) using the PyTorch ConvexHull class.

    Args:
        points: A PyTorch tensor of shape (N, 2) representing the polygon's vertices,
                where N is the number of points.

    Returns:
        The area of the polygon's convex hull.

    Raises:
        ValueError: If the input points are invalid (not a PyTorch tensor,
                      not 2D, less than 3 points, or not (N, 2) shape).
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must be a 2D tensor with shape (N, 2).")
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to compute polygon area.")
    
    # Uses the new PyTorch ConvexHull class from the same file
    # The ConvexHull class's 'area' property is designed for 2D.
    # Pass the module-level EPSILON as the tolerance for the ConvexHull class.
    hull = ConvexHull(points, tol=EPSILON) 
    
    # hull.area returns a torch.Tensor. Use .item() to get the float value.
    calculated_area = hull.area.item()

    # The ConvexHull class itself does not check against EPSILON for the final area.
    # The internal monotone_chain_2d uses its tol for geometric comparisons.
    # The Shoelace formula computes the area directly.
    # It's good practice to check for very small areas here.
    # If the computed area is extremely small, treat as zero.
    if abs(calculated_area) < EPSILON:
        return 0.0
    return calculated_area

def compute_convex_hull_volume(points: torch.Tensor) -> float:
    """
    Computes the volume of the convex hull of a set of 3D points using the PyTorch ConvexHull class.

    Args:
        points: A PyTorch tensor of shape (N, 3) representing the 3D points,
                where N is the number of points.

    Returns:
        The volume of the convex hull.

    Raises:
        ValueError: If the input points are invalid (not a PyTorch tensor,
                      not 2D, not (N, 3) shape, or less than 4 points).
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be a 3D tensor with shape (N, 3).")
    if points.shape[0] < 4:
        raise ValueError("At least 4 points are required to compute 3D convex hull volume.")

    # Uses the new PyTorch ConvexHull class from the same file
    hull = ConvexHull(points, tol=EPSILON) # Pass module EPSILON as tol
    
    # The ConvexHull class's 'volume' property is designed for 3D.
    # hull.volume returns a torch.Tensor. Use .item() to get the float value.
    calculated_volume = hull.volume.item()

    # The ConvexHull class itself does not check against EPSILON for the final volume.
    # The internal monotone_chain_convex_hull_3d uses its tol for geometric comparisons.
    # The volume calculation sums tetrahedra volumes.
    # It's good practice to check for very small volumes here.
    # If the computed volume is extremely small, treat as zero.
    if abs(calculated_volume) < EPSILON:
        return 0.0
    return calculated_volume

def normalize_weights(weights: torch.Tensor, target_sum: float = 1.0, tol: float = 1e-6) -> torch.Tensor:
    """ Normalizes a tensor of weights to sum to a target value (default 1.0).

    Args:
        weights (torch.Tensor): Input tensor of weights (1D, shape (N,)).
        target_sum (float): Desired sum of the normalized weights (default: 1.0).
        tol (float): Tolerance for handling zero or negative weights and sum checks.

    Returns:
        torch.Tensor: Normalized weights with the same shape as input, summing to target_sum.

    Raises:
        AssertionError: If weights is not a 1D tensor or contains values less than -tol.
        ValueError: If sum of weights after clamping is less than tol.
    """
    if not isinstance(weights, torch.Tensor):
        raise TypeError("Input weights must be a PyTorch tensor.")
    assert weights.dim() == 1, "Weights must be a 1D tensor"
    assert torch.all(weights >= -tol), f"Weights must be non-negative (or within -{tol} tolerance)."
    weights = torch.clamp(weights, min=0.0)
    weight_sum = torch.sum(weights)
    if weight_sum < tol:
        raise ValueError(f"Sum of weights ({weight_sum.item()}) is less than tolerance ({tol}); cannot normalize.")
    return weights * (target_sum / weight_sum)

from scipy.spatial import Voronoi 
# ConvexHull is already available from this file (defined above). EPSILON is also global.

def compute_voronoi_density_weights(points: torch.Tensor, 
                                    bounds: torch.Tensor = None, 
                                    space_dim: int = None) -> torch.Tensor:
    """
    Computes Voronoi-based density compensation weights for a set of k-space points.
    Uses Sutherland-Hodgman for 2D clipping if bounds are provided.
    Uses a heuristic (convex hull of Voronoi vertices + bound corners) for 3D open regions with bounds.
    """
    if not isinstance(points, torch.Tensor): raise TypeError("Input points must be a PyTorch tensor.")
    if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")

    original_device = points.device
    points_cpu_numpy = points.cpu().numpy()

    if space_dim is None: space_dim = points.shape[1]
    if space_dim not in [2, 3]: raise ValueError(f"space_dim must be 2 or 3, got {space_dim}.")
    if points.shape[1] != space_dim: raise ValueError(f"Points dim {points.shape[1]} != space_dim {space_dim}.")

    n_points = points.shape[0]
    if n_points == 0: return torch.empty(0, dtype=points.dtype, device=original_device)
    if n_points <= space_dim:
        return torch.full((n_points,), 1.0/(n_points if n_points>0 else 1.0), dtype=points.dtype, device=original_device)

    bounds_cpu = None
    if bounds is not None:
        if not isinstance(bounds, torch.Tensor): raise TypeError("Bounds must be a PyTorch tensor.")
        bounds_cpu = bounds.cpu()
        expected_bounds_shape = (2, space_dim)
        if bounds_cpu.shape != expected_bounds_shape: raise ValueError(f"Bounds shape must be {expected_bounds_shape}, got {bounds_cpu.shape}.")
    
    try:
        qhull_opt = 'Qbb Qc Qz' if space_dim == 2 else 'Qbb Qc'
        vor = Voronoi(points_cpu_numpy, qhull_options=qhull_opt)
    except Exception:
        return torch.full((n_points,), EPSILON, dtype=points.dtype, device=original_device)

    weights_list = []
    min_measure_floor = EPSILON

    for i in range(n_points):
        region_idx = vor.point_region[i]
        region_vertex_indices = vor.regions[region_idx]

        if not region_vertex_indices:
            weights_list.append(1.0 / min_measure_floor); continue
        
        is_open_region = -1 in region_vertex_indices
        finite_region_v_indices = [v_idx for v_idx in region_vertex_indices if v_idx != -1]

        # Handle cases with too few finite vertices early, especially if open and no bounds
        if not finite_region_v_indices: # Applies to open regions; closed regions must have finite vertices
            if is_open_region and bounds_cpu is None:
                weights_list.append(1.0 / min_measure_floor); continue
            # If it's a closed region with no finite vertices (highly unlikely from Voronoi) or open with no finite but has bounds
            # let it proceed to potentially use bound corners for 2D, or specific handling for 3D.
            # For 2D open with bounds and no finite_region_v_indices, it will use only corners.
        
        current_region_vor_vertices_np = vor.vertices[finite_region_v_indices] if finite_region_v_indices else np.empty((0, space_dim), dtype=points_cpu_numpy.dtype)
        current_region_vor_vertices_torch_cpu = torch.tensor(current_region_vor_vertices_np, dtype=points.dtype, device='cpu')

        vertices_for_final_hull_calc = None

        if space_dim == 2:
            if bounds_cpu is not None: # 2D with bounds
                if not is_open_region: 
                    if current_region_vor_vertices_torch_cpu.shape[0] >= 3:
                        vertices_for_final_hull_calc = clip_polygon_2d(current_region_vor_vertices_torch_cpu, bounds_cpu)
                    else: weights_list.append(1.0 / min_measure_floor); continue
                else: # Open 2D region, with bounds
                    temp_vertices_for_heuristic_hull = []
                    if current_region_vor_vertices_torch_cpu.shape[0] > 0:
                         temp_vertices_for_heuristic_hull.append(current_region_vor_vertices_torch_cpu)
                    min_coord, max_coord = bounds_cpu[0], bounds_cpu[1]
                    corners = torch.tensor([[min_coord[0],min_coord[1]], [max_coord[0],min_coord[1]], [min_coord[0],max_coord[1]], [max_coord[0],max_coord[1]]], dtype=points.dtype,device='cpu')
                    temp_vertices_for_heuristic_hull.append(corners)
                    
                    combined_heuristic_points = torch.cat(temp_vertices_for_heuristic_hull, dim=0)
                    unique_heuristic_points = torch.unique(combined_heuristic_points, dim=0)

                    if unique_heuristic_points.shape[0] < 3: weights_list.append(1.0 / min_measure_floor); continue
                    try:
                        initial_closed_hull = ConvexHull(unique_heuristic_points, tol=EPSILON)
                        if initial_closed_hull.vertices is None or initial_closed_hull.vertices.numel() < 3:
                            weights_list.append(1.0 / min_measure_floor); continue
                        initial_closed_polygon_vertices = initial_closed_hull.points[initial_closed_hull.vertices]
                        vertices_for_final_hull_calc = clip_polygon_2d(initial_closed_polygon_vertices, bounds_cpu)
                    except (ValueError, RuntimeError): weights_list.append(1.0 / min_measure_floor); continue
            elif not is_open_region: # Closed 2D region, no bounds
                vertices_for_final_hull_calc = current_region_vor_vertices_torch_cpu
            else: # Open 2D region, no bounds
                weights_list.append(1.0 / min_measure_floor); continue
        
        elif space_dim == 3: # 3D logic
            if is_open_region and bounds_cpu is not None: # Open 3D region, with bounds
                # Use the polyhedron clipping strategy
                vertices_for_final_hull_calc = clip_polyhedron_3d(current_region_vor_vertices_torch_cpu, bounds_cpu, tol=EPSILON)
            elif not is_open_region: # Closed 3D region (with or without bounds)
                vertices_for_final_hull_calc = current_region_vor_vertices_torch_cpu
            else: # Open 3D region, no bounds
                 weights_list.append(1.0 / min_measure_floor); continue
        else: 
            weights_list.append(1.0 / min_measure_floor); continue


        if vertices_for_final_hull_calc is None or vertices_for_final_hull_calc.shape[0] < space_dim + 1:
            weights_list.append(1.0 / min_measure_floor); continue
        try:
            hull_of_region = ConvexHull(vertices_for_final_hull_calc, tol=EPSILON) 
        except (ValueError, RuntimeError): 
            weights_list.append(1.0 / min_measure_floor); continue

        cell_measure_tensor = hull_of_region.area if space_dim == 2 else hull_of_region.volume
        cell_measure = cell_measure_tensor.item() 
        actual_measure = min_measure_floor if abs(cell_measure) < min_measure_floor else abs(cell_measure)
        weights_list.append(1.0 / actual_measure)

    final_weights = torch.tensor(weights_list, dtype=points.dtype, device='cpu') 
    return final_weights.to(original_device)

# --- 3D Polyhedron Clipping Helpers ---
def _point_plane_signed_distance(points_tensor: torch.Tensor, plane_normal: torch.Tensor, plane_dist_offset: float) -> torch.Tensor:
    # points_tensor: (N, 3), plane_normal: (3,)
    # Returns: (N,) tensor of signed distances. Positive if on normal side.
    return torch.matmul(points_tensor, plane_normal) - plane_dist_offset

def _segment_plane_intersection(p1: torch.Tensor, p2: torch.Tensor, 
                                plane_normal: torch.Tensor, plane_dist_offset: float, 
                                tol: float = 1e-7) -> torch.Tensor | None:
    dp = p2 - p1
    den = torch.dot(dp, plane_normal)

    if torch.abs(den) < tol: # Segment parallel to plane
        return None # No unique intersection point for clipping purposes

    t = (plane_dist_offset - torch.dot(p1, plane_normal)) / den
    
    if tol <= t <= 1.0 - tol: 
        return p1 + t * dp
    return None

def clip_polyhedron_3d(input_poly_vertices: torch.Tensor, 
                       bounding_box: torch.Tensor, 
                       tol: float = 1e-7) -> torch.Tensor:
    """
    Clips a 3D convex polyhedron against an axis-aligned bounding box.
    Collects original vertices inside the box and intersection points of edges with box planes,
    then computes the convex hull of these collected points.

    Args:
        input_poly_vertices (torch.Tensor): Shape (N, 3), vertices of the input convex polyhedron.
                                            Assumed to be on CPU.
        bounding_box (torch.Tensor): Shape (2, 3), [[min_x,min_y,min_z], [max_x,max_y,max_z]].
                                     Assumed to be on CPU.
        tol (float): Tolerance for geometric computations.

    Returns:
        torch.Tensor: Shape (M, 3), vertices of the clipped polyhedron. Empty (0,3) if fully clipped or degenerate.
                      Returns on the same device as input_poly_vertices.
    """
    if not isinstance(input_poly_vertices, torch.Tensor) or input_poly_vertices.ndim != 2 or input_poly_vertices.shape[1] != 3:
        raise ValueError("input_poly_vertices must be a tensor of shape (N, 3).")
    if not isinstance(bounding_box, torch.Tensor) or bounding_box.shape != (2,3):
        raise ValueError("bounding_box must be a tensor of shape (2, 3).")

    device = input_poly_vertices.device # Keep original device for final output
    # Perform computations on CPU for simplicity and compatibility with potential CPU-only parts
    input_poly_vertices_cpu = input_poly_vertices.cpu()
    bounding_box_cpu = bounding_box.cpu()
    dtype = input_poly_vertices.dtype


    min_coords = bounding_box_cpu[0]
    max_coords = bounding_box_cpu[1]

    if not (torch.all(min_coords <= max_coords)):
        raise ValueError("Bounding box min_coords must be less than or equal to max_coords for each dimension.")

    planes = [
        (torch.tensor([1,0,0], dtype=dtype, device='cpu'),  min_coords[0]), 
        (torch.tensor([-1,0,0], dtype=dtype, device='cpu'), -max_coords[0]),
        (torch.tensor([0,1,0], dtype=dtype, device='cpu'),  min_coords[1]), 
        (torch.tensor([0,-1,0], dtype=dtype, device='cpu'), -max_coords[1]),
        (torch.tensor([0,0,1], dtype=dtype, device='cpu'),  min_coords[2]), 
        (torch.tensor([0,0,-1], dtype=dtype, device='cpu'), -max_coords[2]) 
    ]

    final_vertex_candidates = []

    for v_idx in range(input_poly_vertices_cpu.shape[0]):
        v = input_poly_vertices_cpu[v_idx]
        is_fully_inside = True
        for plane_normal, plane_d_offset in planes:
            if _point_plane_signed_distance(v.unsqueeze(0), plane_normal, plane_d_offset).item() < -tol:
                is_fully_inside = False; break
        if is_fully_inside:
            final_vertex_candidates.append(v)

    if input_poly_vertices_cpu.shape[0] >= 4:
        try:
            initial_hull = ConvexHull(input_poly_vertices_cpu, tol=tol) 
            if initial_hull.simplices is not None and initial_hull.simplices.numel() > 0:
                unique_edges = set()
                for face in initial_hull.simplices: 
                    face_indices = face.tolist()
                    edges_on_face = [
                        tuple(sorted((face_indices[0], face_indices[1]))),
                        tuple(sorted((face_indices[1], face_indices[2]))),
                        tuple(sorted((face_indices[2], face_indices[0])))
                    ]
                    unique_edges.update(edges_on_face)

                for edge_indices in unique_edges:
                    # Ensure indices are valid for input_poly_vertices_cpu
                    if edge_indices[0] < input_poly_vertices_cpu.shape[0] and \
                       edge_indices[1] < input_poly_vertices_cpu.shape[0]:
                        p1 = input_poly_vertices_cpu[edge_indices[0]]
                        p2 = input_poly_vertices_cpu[edge_indices[1]]
                    else:
                        # This can happen if initial_hull.simplices refer to points not in the original list
                        # (e.g. if ConvexHull internally adds points or if indices are wrong)
                        # This should not happen if ConvexHull returns indices into the original points.
                        continue 

                    for plane_normal, plane_d_offset in planes:
                        intersect_pt = _segment_plane_intersection(p1, p2, plane_normal, plane_d_offset, tol)
                        if intersect_pt is not None:
                            is_intersection_valid_for_box = True
                            for check_normal, check_d_offset in planes:
                                if _point_plane_signed_distance(intersect_pt.unsqueeze(0), check_normal, check_d_offset).item() < -tol:
                                    is_intersection_valid_for_box = False; break
                            if is_intersection_valid_for_box:
                                final_vertex_candidates.append(intersect_pt)
        except (ValueError, RuntimeError): pass 

    if not final_vertex_candidates:
        return torch.empty((0, 3), dtype=dtype, device=device)

    unique_final_candidates = torch.unique(torch.stack(final_vertex_candidates), dim=0)

    if unique_final_candidates.shape[0] < 4:
        return torch.empty((0, 3), dtype=dtype, device=device)

    try:
        clipped_hull = ConvexHull(unique_final_candidates, tol=tol)
        if clipped_hull.simplices is None or clipped_hull.simplices.numel() == 0 or clipped_hull.vertices.numel() < 4 :
            return torch.empty((0, 3), dtype=dtype, device=device)
        
        final_vertices_indices = torch.unique(clipped_hull.simplices.flatten())
        return unique_final_candidates[final_vertices_indices].to(device) # Move to original device
    except (ValueError, RuntimeError): 
        return torch.empty((0, 3), dtype=dtype, device=device)


# --- compute_polygon_area, compute_convex_hull_volume, normalize_weights,
#     compute_voronoi_density_weights remain unchanged from Turn 35,
#     but are included here as part of the full file content for overwrite.
#     The duplicate import of scipy.spatial.Voronoi from Turn 35 is also removed.
#     The internal helpers for Sutherland-Hodgman polygon clipping are also included from Turn 35.
#     The internal helpers for Delaunay triangulation are also included from Turn 30.

def compute_polygon_area(points: torch.Tensor) -> float:
    """
    Computes the area of a 2D polygon (convex hull) using the PyTorch ConvexHull class.

    Args:
        points: A PyTorch tensor of shape (N, 2) representing the polygon's vertices,
                where N is the number of points.

    Returns:
        The area of the polygon's convex hull.

    Raises:
        ValueError: If the input points are invalid (not a PyTorch tensor,
                      not 2D, less than 3 points, or not (N, 2) shape).
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must be a 2D tensor with shape (N, 2).")
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to compute polygon area.")
    
    hull = ConvexHull(points, tol=EPSILON) 
    calculated_area = hull.area.item()
    if abs(calculated_area) < EPSILON:
        return 0.0
    return calculated_area

def compute_convex_hull_volume(points: torch.Tensor) -> float:
    """
    Computes the volume of the convex hull of a set of 3D points using the PyTorch ConvexHull class.

    Args:
        points: A PyTorch tensor of shape (N, 3) representing the 3D points,
                where N is the number of points.

    Returns:
        The volume of the convex hull.

    Raises:
        ValueError: If the input points are invalid (not a PyTorch tensor,
                      not 2D, not (N, 3) shape, or less than 4 points).
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be a 3D tensor with shape (N, 3).")
    if points.shape[0] < 4:
        raise ValueError("At least 4 points are required to compute 3D convex hull volume.")

    hull = ConvexHull(points, tol=EPSILON) 
    calculated_volume = hull.volume.item()
    if abs(calculated_volume) < EPSILON:
        return 0.0
    return calculated_volume

def normalize_weights(weights: torch.Tensor, target_sum: float = 1.0, tol: float = 1e-6) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor):
        raise TypeError("Input weights must be a PyTorch tensor.")
    assert weights.dim() == 1, "Weights must be a 1D tensor"
    assert torch.all(weights >= -tol), f"Weights must be non-negative (or within -{tol} tolerance)."
    weights = torch.clamp(weights, min=0.0)
    weight_sum = torch.sum(weights)
    if weight_sum < tol:
        raise ValueError(f"Sum of weights ({weight_sum.item()}) is less than tolerance ({tol}); cannot normalize.")
    return weights * (target_sum / weight_sum)

def compute_voronoi_density_weights(points: torch.Tensor, 
                                    bounds: torch.Tensor = None, 
                                    space_dim: int = None) -> torch.Tensor:
    if not isinstance(points, torch.Tensor): raise TypeError("Input points must be a PyTorch tensor.")
    if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")

    original_device = points.device
    points_cpu_numpy = points.cpu().numpy()

    if space_dim is None: space_dim = points.shape[1]
    if space_dim not in [2, 3]: raise ValueError(f"space_dim must be 2 or 3, got {space_dim}.")
    if points.shape[1] != space_dim: raise ValueError(f"Points dim {points.shape[1]} != space_dim {space_dim}.")

    n_points = points.shape[0]
    if n_points == 0: return torch.empty(0, dtype=points.dtype, device=original_device)
    if n_points <= space_dim:
        return torch.full((n_points,), 1.0/(n_points if n_points>0 else 1.0), dtype=points.dtype, device=original_device)

    bounds_cpu = None
    if bounds is not None:
        if not isinstance(bounds, torch.Tensor): raise TypeError("Bounds must be a PyTorch tensor.")
        bounds_cpu = bounds.cpu()
        expected_bounds_shape = (2, space_dim)
        if bounds_cpu.shape != expected_bounds_shape: raise ValueError(f"Bounds shape must be {expected_bounds_shape}, got {bounds_cpu.shape}.")
    
    try:
        qhull_opt = 'Qbb Qc Qz' if space_dim == 2 else 'Qbb Qc'
        vor = Voronoi(points_cpu_numpy, qhull_options=qhull_opt)
    except Exception:
        return torch.full((n_points,), EPSILON, dtype=points.dtype, device=original_device)

    weights_list = []
    min_measure_floor = EPSILON

    for i in range(n_points):
        region_idx = vor.point_region[i]
        region_vertex_indices = vor.regions[region_idx]

        if not region_vertex_indices:
            weights_list.append(1.0 / min_measure_floor); continue
        
        is_open_region = -1 in region_vertex_indices
        finite_region_v_indices = [v_idx for v_idx in region_vertex_indices if v_idx != -1]
        
        current_region_vor_vertices_np = vor.vertices[finite_region_v_indices] if finite_region_v_indices else np.empty((0, space_dim), dtype=points_cpu_numpy.dtype)
        current_region_vor_vertices_torch_cpu = torch.tensor(current_region_vor_vertices_np, dtype=points.dtype, device='cpu')
        
        vertices_for_final_hull_calc = None

        if space_dim == 2:
            if bounds_cpu is not None: 
                if not is_open_region: 
                    if current_region_vor_vertices_torch_cpu.shape[0] >= 3:
                        vertices_for_final_hull_calc = clip_polygon_2d(current_region_vor_vertices_torch_cpu, bounds_cpu)
                    else: weights_list.append(1.0 / min_measure_floor); continue
                else: 
                    temp_vertices_for_heuristic_hull = []
                    if current_region_vor_vertices_torch_cpu.shape[0] > 0:
                         temp_vertices_for_heuristic_hull.append(current_region_vor_vertices_torch_cpu)
                    min_coord, max_coord = bounds_cpu[0], bounds_cpu[1]
                    corners = torch.tensor([[min_coord[0],min_coord[1]], [max_coord[0],min_coord[1]], [min_coord[0],max_coord[1]], [max_coord[0],max_coord[1]]], dtype=points.dtype,device='cpu')
                    temp_vertices_for_heuristic_hull.append(corners)
                    
                    if not temp_vertices_for_heuristic_hull : weights_list.append(1.0 / min_measure_floor); continue # Should not happen if corners are always added
                    combined_heuristic_points = torch.cat(temp_vertices_for_heuristic_hull, dim=0)
                    unique_heuristic_points = torch.unique(combined_heuristic_points, dim=0)

                    if unique_heuristic_points.shape[0] < 3: weights_list.append(1.0 / min_measure_floor); continue
                    try:
                        initial_closed_hull = ConvexHull(unique_heuristic_points, tol=EPSILON)
                        if initial_closed_hull.vertices is None or initial_closed_hull.vertices.numel() < 3:
                            weights_list.append(1.0 / min_measure_floor); continue
                        initial_closed_polygon_vertices = initial_closed_hull.points[initial_closed_hull.vertices]
                        vertices_for_final_hull_calc = clip_polygon_2d(initial_closed_polygon_vertices, bounds_cpu)
                    except (ValueError, RuntimeError): weights_list.append(1.0 / min_measure_floor); continue
            elif not is_open_region: 
                vertices_for_final_hull_calc = current_region_vor_vertices_torch_cpu
            else: 
                weights_list.append(1.0 / min_measure_floor); continue
        
        elif space_dim == 3: 
            if is_open_region and bounds_cpu is not None: 
                # Using the new clip_polyhedron_3d for open 3D regions with bounds
                if current_region_vor_vertices_torch_cpu.shape[0] > 0 : # If there are some finite vertices
                    vertices_for_final_hull_calc = clip_polyhedron_3d(current_region_vor_vertices_torch_cpu, bounds_cpu, tol=EPSILON)
                else: # No finite vertices, try to clip the bounding box itself (conceptually, a large box)
                      # This case is complex. For now, if no finite vertices, it's hard to define the "polyhedron" to clip.
                      # A robust solution might involve creating a large box from the Voronoi point and bounds.
                      # For now, if no finite vertices, we'll assign a small weight.
                    weights_list.append(1.0 / min_measure_floor); continue

            elif not is_open_region: 
                vertices_for_final_hull_calc = current_region_vor_vertices_torch_cpu
            else: # Open 3D region, no bounds
                 weights_list.append(1.0 / min_measure_floor); continue
        else: 
            weights_list.append(1.0 / min_measure_floor); continue

        if vertices_for_final_hull_calc is None or vertices_for_final_hull_calc.shape[0] < space_dim + 1:
            weights_list.append(1.0 / min_measure_floor); continue
        try:
            hull_of_region = ConvexHull(vertices_for_final_hull_calc, tol=EPSILON) 
        except (ValueError, RuntimeError): 
            weights_list.append(1.0 / min_measure_floor); continue

        cell_measure_tensor = hull_of_region.area if space_dim == 2 else hull_of_region.volume
        cell_measure = cell_measure_tensor.item() 
        actual_measure = min_measure_floor if abs(cell_measure) < min_measure_floor else abs(cell_measure)
        weights_list.append(1.0 / actual_measure)

    final_weights = torch.tensor(weights_list, dtype=points.dtype, device='cpu') 
    return final_weights.to(original_device)
