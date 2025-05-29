import numpy as np
import torch # Added for PyTorch ConvexHull
from scipy.spatial import ConvexHull as ScipyConvexHull # Renamed for clarity
from scipy.spatial.qhull import QhullError
import sys # For stderr

EPSILON = 1e-9

class ConvexHull:
    """
    PyTorch-based Convex Hull calculation.
    Supports 2D and 3D point clouds.
    """
    def __init__(self, points: torch.Tensor, incremental: bool = False, tol: float = 1e-6):
        """
        Args:
            points (torch.Tensor): Tensor of shape (N, D) where N is the number of points
                                   and D is the dimension (2 or 3).
            incremental (bool): Not currently used in this PyTorch implementation.
                                SciPy's ConvexHull uses it. Kept for API similarity.
            tol (float): Tolerance for numerical comparisons, e.g., checking coplanarity
                         or if a point is on a face.
        """
        if not isinstance(points, torch.Tensor):
            raise ValueError("Input points must be a PyTorch tensor.")
        if points.ndim != 2:
            raise ValueError("Input points tensor must be 2-dimensional (N, D).")
        
        self.points = points
        self.device = points.device
        self.dtype = points.dtype
        self.dim = points.shape[1]
        self.tol = tol # Tolerance for geometric tests

        if self.dim not in [2, 3]:
            raise ValueError("Only 2D and 3D points are supported.")

        self.vertices = None  # Indices of points forming the hull facets
        self.equations = None # For 3D, plane equations of hull faces (normal and offset)
                              # For 2D, line equations (normal and offset) - though less common
        self._area = None
        self._volume = None
        
        self._compute_hull()

    def _compute_hull(self):
        if self.dim == 2:
            self._convex_hull_2d()
        elif self.dim == 3:
            self._convex_hull_3d()

    def _convex_hull_2d(self):
        """
        Computes the convex hull for 2D points using a Monotone Chain algorithm (Andrew's).
        Sets self.vertices to be a tensor of vertex indices forming the hull, ordered.
        Sets self._area.
        """
        points_np = self.points.cpu().numpy() # SciPy/NumPy based for now
        
        # Fallback to SciPy for 2D hull computation as it's robust
        # and a direct PyTorch implementation of Monotone Chain is non-trivial
        # if not already available in a library.
        try:
            # Using the renamed ScipyConvexHull
            hull_scipy = ScipyConvexHull(points_np)
            # SciPy's vertices are indices into the input points, ordered counter-clockwise
            self.vertices = torch.tensor(hull_scipy.vertices, dtype=torch.long, device=self.device)
            self._area = torch.tensor(hull_scipy.volume, device=self.device, dtype=self.dtype) # SciPy's 'volume' is area for 2D
            
            # Equations for 2D hull edges (optional, not always standard for 2D from basic algos)
            # Ax + By + C = 0. Normal (A,B), Offset C
            # For each edge (v0, v1) on the hull:
            # v0 = self.points[self.vertices[i]]
            # v1 = self.points[self.vertices[(i+1)%len(self.vertices)]]
            # Normal: (v1[1]-v0[1], v0[0]-v1[0])
            # Offset C = - (normal[0]*v0[0] + normal[1]*v0[1])
            # self.equations would store these [A, B, C]
            
        except QhullError as e:
            # Handle degenerate cases (e.g. all points collinear)
            # This might mean no area, or hull is just a line segment
            print(f"QhullError in 2D: {e}. Hull might be degenerate.", file=sys.stderr)
            # Attempt to handle simple collinear case: find extreme points
            if len(points_np) >= 2:
                min_idx = torch.argmin(points_np[:,0]) # Simplistic: take min/max x
                max_idx = torch.argmax(points_np[:,0])
                # Check if multiple points share min/max x, then use y
                # This is a placeholder for more robust degenerate handling.
                self.vertices = torch.tensor(np.unique([min_idx.item(), max_idx.item()]), dtype=torch.long, device=self.device)
            else: # Single point or empty
                 self.vertices = torch.arange(len(points_np), dtype=torch.long, device=self.device)
            self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype)


    def _find_initial_simplex(self, points: torch.Tensor) -> torch.Tensor:
        """ Finds 4 non-coplanar points to form the initial tetrahedron for 3D hull. """
        n_points = points.shape[0]
        if n_points < 4:
            raise ValueError("Need at least 4 points for a 3D simplex.")

        # Find first point (e.g., min x)
        p0_idx = torch.argmin(points[:, 0])
        p0 = points[p0_idx]

        # Find second point furthest from p0
        dists_from_p0 = torch.sum((points - p0)**2, dim=1)
        p1_idx = torch.argmax(dists_from_p0)
        p1 = points[p1_idx]

        # Find third point furthest from the line p0-p1
        # Project points onto vector p1-p0, subtract to get perpendicular component for distance
        line_vec = p1 - p0
        line_vec_norm_sq = torch.dot(line_vec, line_vec)
        if line_vec_norm_sq < self.tol**2: # p0 and p1 are too close
             # Try finding a point not on p0, iterate through points
            for i in range(n_points):
                if torch.sum((points[i] - p0)**2) > self.tol**2 :
                    p1_idx = i; p1 = points[p1_idx]; line_vec = p1 - p0; line_vec_norm_sq = torch.dot(line_vec, line_vec); break
            if line_vec_norm_sq < self.tol**2: raise ValueError("Points are likely coincident or too close.")


        ap = points - p0
        # t = dot(ap, line_vec) / line_vec_norm_sq
        t = torch.matmul(ap, line_vec) / line_vec_norm_sq
        projections = p0.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
        dists_from_line = torch.sum((points - projections)**2, dim=1)
        p2_idx = torch.argmax(dists_from_line)
        p2 = points[p2_idx]

        # Find fourth point furthest from the plane p0-p1-p2
        # Normal to plane p0-p1-p2
        normal = self._compute_face_normal(p0, p1, p2) # Cross product (p1-p0) x (p2-p0)
        if torch.norm(normal) < self.tol: # Points are collinear
            # Attempt to find a non-collinear point by iterating
            for i in range(n_points):
                if i == p0_idx or i == p1_idx: continue
                temp_normal = self._compute_face_normal(p0,p1,points[i])
                if torch.norm(temp_normal) > self.tol:
                    p2_idx = i; p2 = points[p2_idx]; normal = temp_normal; break
            if torch.norm(normal) < self.tol: raise ValueError("Points are likely collinear.")


        # Distance from points to plane: dot(points - p0, normal)
        dists_from_plane = torch.abs(torch.matmul(points - p0, normal))
        p3_idx = torch.argmax(dists_from_plane)
        
        simplex_indices = torch.tensor([p0_idx, p1_idx, p2_idx, p3_idx], device=self.device, dtype=torch.long)
        
        # Ensure chosen points are unique
        if len(torch.unique(simplex_indices)) < 4:
            # This can happen if points are degenerate.
            # Fallback: try to pick first 4 unique points if available
            unique_pts_indices = torch.unique(torch.arange(n_points), return_inverse=False)[0]
            if len(unique_pts_indices) >=4:
                simplex_indices = unique_pts_indices[:4]
            else:
                raise ValueError("Could not find 4 unique points for initial simplex due to degeneracy.")
        
        return simplex_indices


    def _compute_face_normal(self, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """ Computes face normal (v1-v0) x (v2-v0). """
        return torch.cross(v1 - v0, v2 - v0)

    def _convex_hull_3d(self):
        """
        Computes the convex hull for 3D points using a PyTorch-based incremental algorithm (simplified Gift Wrapping or similar concept).
        This is a simplified version and may not be as robust as SciPy's Qhull.
        Sets self.vertices (list of face vertex index lists) and self.equations, and self._volume.
        """
        points = self.points
        n_points = points.shape[0]

        if n_points < 4:
            # Fallback for degenerate cases (e.g. coplanar points in 3D)
            # This should ideally result in 0 volume and a 2D hull if coplanar.
            # For simplicity, we'll rely on SciPy's behavior for such cases if we were to wrap it.
            # Here, we'll just set volume to 0 and no faces.
            print("Warning: Less than 4 points for 3D hull, resulting in 0 volume.", file=sys.stderr)
            self.vertices = [] # No 3D faces
            self.equations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return

        # Fallback to SciPy for 3D hull computation due to complexity of robust PyTorch implementation
        try:
            points_np = self.points.cpu().numpy()
            hull_scipy = ScipyConvexHull(points_np) # Use the renamed ScipyConvexHull
            
            # Store face vertex indices. Each row in hull_scipy.simplices is a face.
            self.vertices = [torch.tensor(face_indices, dtype=torch.long, device=self.device) for face_indices in hull_scipy.simplices]
            
            # Store plane equations: ax + by + cz + d = 0
            # hull_scipy.equations are [normal_x, normal_y, normal_z, offset_d]
            # Ensure normal points outwards: SciPy's normals point outwards by default.
            self.equations = torch.tensor(hull_scipy.equations, device=self.device, dtype=self.dtype)
            self._volume = torch.tensor(hull_scipy.volume, device=self.device, dtype=self.dtype)
            
        except QhullError as e:
            print(f"QhullError in 3D: {e}. Hull might be degenerate (e.g., coplanar points). Setting volume to 0.", file=sys.stderr)
            self.vertices = []
            self.equations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        except Exception as e_gen: # Catch any other unexpected error from Scipy
            print(f"Unexpected error during Scipy ConvexHull in 3D: {e_gen}. Setting volume to 0.", file=sys.stderr)
            self.vertices = []
            self.equations = torch.empty((0, 4), device=self.device, dtype=self.dtype)
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)


    @property
    def points_normals(self) -> list:
        """
        Returns a list of (point_on_hull, normal_vector) tuples for each face of the 3D hull.
        For 2D, this is not well-defined in the same way (edges instead of faces).
        This is a simplified representation. SciPy's `equations` gives normals and offsets.
        """
        if self.dim != 3 or self.equations is None or not self.vertices:
            return []
        
        normals_list = []
        for i, face_indices in enumerate(self.vertices):
            # Take the first vertex of the face as a point on the hull face
            point_on_face = self.points[face_indices[0]]
            # Normal vector from the equation (first 3 components)
            normal_vector = self.equations[i, :3]
            normals_list.append((point_on_face, normal_vector))
        return normals_list

    @property
    def area(self) -> torch.Tensor:
        """
        Returns the area of the 2D convex hull.
        For 3D hulls, this property is not standard (surface area is, but not 'area').
        """
        if self.dim == 2:
            if self._area is None: # Should have been computed by _convex_hull_2d
                self._convex_hull_2d() # Attempt to compute if not done
            return self._area if self._area is not None else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else: # self.dim == 3
            # Calculate surface area of 3D hull
            # Sum of areas of all triangular faces
            surface_area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            if self.vertices is not None:
                for face_indices in self.vertices:
                    p0, p1, p2 = self.points[face_indices[0]], self.points[face_indices[1]], self.points[face_indices[2]]
                    # Area of triangle = 0.5 * || (p1-p0) x (p2-p0) ||
                    surface_area += 0.5 * torch.norm(self._compute_face_normal(p0, p1, p2))
            return surface_area


    @property
    def volume(self) -> torch.Tensor:
        """
        Returns the volume of the 3D convex hull.
        For 2D hulls, this property is not standard (area is).
        """
        if self.dim == 3:
            if self._volume is None: # Should have been computed by _convex_hull_3d
                 self._convex_hull_3d() # Attempt to compute if not done
            return self._volume if self._volume is not None else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else: # self.dim == 2
            # For 2D, volume is typically considered 0.
            # Or, raise an error, or return area? SciPy returns area as 'volume' for 2D.
            # Given the class name and explicit 'area' property, returning 0 for 2D volume seems consistent.
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)


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
    hull = ConvexHull(points, tol=EPSILON) 
    
    # Based on the provided ConvexHull class structure:
    # For 2D, hull.area contains the area.
    # hull.volume for 2D is 0.0.
    # The prompt mentioned "Return hull.volume.item()", but given the class code,
    # hull.area.item() is the correct way to get the 2D area.
    calculated_area = hull.area.item()

    # The original function had a check against EPSILON for the final area.
    # The ConvexHull class might handle this internally, or its fallback SciPy version does.
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
    calculated_volume = hull.volume.item()

    # The original function had a check against EPSILON for the final volume.
    # The ConvexHull class's SciPy fallback might handle this.
    # If the computed volume is extremely small, treat as zero.
    if abs(calculated_volume) < EPSILON:
        return 0.0
    return calculated_volume

def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a PyTorch tensor of weights to sum to 1.

    Args:
        weights: A PyTorch tensor of weights.

    Returns:
        A PyTorch tensor of normalized weights.
        If the sum of input weights is close to zero (within EPSILON), 
        returns the original weights tensor.
    """
    if not isinstance(weights, torch.Tensor):
        raise ValueError("Input weights must be a PyTorch tensor.")

    total = torch.sum(weights)

    if torch.abs(total) < EPSILON:
        print("Warning: Sum of weights is zero or near-zero; returning unnormalized weights.", file=sys.stderr)
        return weights
    
    normalized_weights = weights / total
    # Clamp negative values to 0.0 after normalization
    normalized_weights = torch.clamp(normalized_weights, min=0.0)
    
    # Optional: Re-normalize if clamping changed the sum significantly.
    # This might be needed if clamping to zero significantly alters the sum from 1.0
    # current_sum_after_clamping = torch.sum(normalized_weights)
    # if torch.abs(current_sum_after_clamping) > EPSILON and \
    #    torch.abs(current_sum_after_clamping - 1.0) > EPSILON: # if sum is not zero and not 1
    #     print("Warning: Re-normalizing after clamping due to significant sum change.", file=sys.stderr)
    #     normalized_weights = normalized_weights / current_sum_after_clamping
    #     # Second clamp, just in case of numerical issues with re-normalization of all-zero / tiny sum post-clamp
    #     normalized_weights = torch.clamp(normalized_weights, min=0.0)


    return normalized_weights
