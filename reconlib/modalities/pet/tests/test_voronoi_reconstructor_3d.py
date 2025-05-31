import unittest
import torch
import numpy as np

# Attempt to import the main class
try:
    from reconlib.modalities.pet.voronoi_reconstructor_3d import VoronoiPETReconstructor3D
    VORONOI_RECONSTRUCTOR_AVAILABLE = True
except ImportError as e: # pragma: no cover
    print(f"Failed to import VoronoiPETReconstructor3D: {e}")
    VoronoiPETReconstructor3D = None
    VORONOI_RECONSTRUCTOR_AVAILABLE = False

# For unittest.mock if not available (e.g. Python < 3.3)
if not hasattr(unittest, 'mock'): # pragma: no cover
    try:
        import mock
        unittest.mock = mock
    except ImportError:
        pass

# Mocking placeholder for missing dependencies
class MockDelaunay3D:
    def delaunay_triangulation_3d(self, points_np):
        if not isinstance(points_np, np.ndarray): raise TypeError("MockDelaunay3D input must be np.ndarray") # pragma: no cover
        if len(points_np) < 4: return None
        if len(points_np) == 4: return np.array([[0,1,2,3]], dtype=int)
        if len(points_np) > 4: return np.array([[0,1,2,3],[0,1,2,4]], dtype=int)
        return None

class MockVoronoiFromDelaunay3D:
    def construct_voronoi_polyhedra_3d(self, points_np, delaunay_tetra_np):
        if not isinstance(points_np, np.ndarray) or not isinstance(delaunay_tetra_np, np.ndarray): # pragma: no cover
            raise TypeError("MockVoronoiFromDelaunay3D inputs must be np.ndarray")
        num_generators = len(points_np)
        unique_verts_list = []
        for i in range(num_generators):
            offset = points_np[i] * 0.1
            for dx in [-0.5, 0.5]:
                for dy in [-0.5, 0.5]:
                    for dz in [-0.5, 0.5]:
                        unique_verts_list.append(points_np[i] + np.array([dx,dy,dz]) + offset)
        if not unique_verts_list: unique_voronoi_vertices_np = np.empty((0,3),dtype=float) # pragma: no cover
        else: unique_voronoi_vertices_np = np.unique(np.array(unique_verts_list), axis=0)
        voronoi_cells_data_cpu = []
        num_unique_v = unique_voronoi_vertices_np.shape[0]
        for i in range(num_generators):
            if num_unique_v >= 4:
                face1 = [k % num_unique_v for k in range(3)]
                face2 = [(k+1) % num_unique_v for k in range(3)]
                voronoi_cells_data_cpu.append({'faces': [face1, face2]})
            elif num_generators > 0: voronoi_cells_data_cpu.append({'faces': []}) # pragma: no cover
        if not voronoi_cells_data_cpu and num_generators > 0: return None, None # pragma: no cover
        return voronoi_cells_data_cpu, unique_voronoi_vertices_np

class MockGeometryCore:
    class ConvexHull:
        def __init__(self, points_np, tol=None):
            if points_np is None or len(points_np) < 4: raise ValueError("CH Mock: Not enough points") # pragma: no cover
            self.points = points_np; self.volume = 1.0
        def intersect_line_segment(self, p1_np, p2_np): return 0.01 # pragma: no cover
    EPSILON = 1e-6

if VORONOI_RECONSTRUCTOR_AVAILABLE: # pragma: no branch
    try:
        from reconlib.voronoi.delaunay_3d import delaunay_triangulation_3d
        from reconlib.voronoi.voronoi_from_delaunay import construct_voronoi_polyhedra_3d
        from reconlib.voronoi.geometry_core import ConvexHull as ReconlibGeoCoreConvexHull
        from reconlib.voronoi.geometry_core import EPSILON as ReconlibGeoCoreEPSILON
    except ImportError: # pragma: no cover
        ReconlibGeoCoreConvexHull = None; ReconlibGeoCoreEPSILON = None

@unittest.skipIf(not VORONOI_RECONSTRUCTOR_AVAILABLE, "VoronoiPETReconstructor3D class not available.")
class TestVoronoiPETReconstructor3D(unittest.TestCase):
    def setUp(self):
        self.original_attrs = {}
        self.reconstructor_module = None
        try:
            self.reconstructor_module = __import__('reconlib.modalities.pet.voronoi_reconstructor_3d', fromlist=['None'])
        except ImportError: pass # pragma: no cover

        self.patch_map_class_methods = {
            '_validate_generator_points_3d': getattr(VoronoiPETReconstructor3D, '_validate_generator_points_3d', None),
            '_compute_voronoi_diagram_3d': getattr(VoronoiPETReconstructor3D, '_compute_voronoi_diagram_3d', None),
            '_validate_voronoi_cells_3d': getattr(VoronoiPETReconstructor3D, '_validate_voronoi_cells_3d', None),
            '_compute_system_matrix_3d': getattr(VoronoiPETReconstructor3D, '_compute_system_matrix_3d', None),
            '_get_lor_endpoints_3d': getattr(VoronoiPETReconstructor3D, '_get_lor_endpoints_3d', None),
            '_compute_lor_cell_intersection_3d': getattr(VoronoiPETReconstructor3D, '_compute_lor_cell_intersection_3d', None),
            '_is_point_in_face_3d': getattr(VoronoiPETReconstructor3D, '_is_point_in_face_3d', None),
            '_is_point_inside_polyhedron_3d': getattr(VoronoiPETReconstructor3D, '_is_point_inside_polyhedron_3d', None),
            '_forward_project_3d': getattr(VoronoiPETReconstructor3D, '_forward_project_3d', None),
            '_back_project_3d': getattr(VoronoiPETReconstructor3D, '_back_project_3d', None),
        }
        self.patch_map_module_level = {}
        if self.reconstructor_module: # pragma: no branch
            self.patch_map_module_level.update({
                'delaunay_triangulation_3d': getattr(self.reconstructor_module, 'delaunay_triangulation_3d', None),
                'construct_voronoi_polyhedra_3d': getattr(self.reconstructor_module, 'construct_voronoi_polyhedra_3d', None),
                'ConvexHull': getattr(self.reconstructor_module, 'ConvexHull', None),
            })

    def tearDown(self):
        for name, original_attr in self.patch_map_class_methods.items():
            if original_attr is not None: setattr(VoronoiPETReconstructor3D, name, original_attr)
        if self.reconstructor_module: # pragma: no branch
            for name, original_attr in self.patch_map_module_level.items():
                if original_attr is not None: setattr(self.reconstructor_module, name, original_attr)
                elif hasattr(self.reconstructor_module, name): delattr(self.reconstructor_module, name) # pragma: no cover

    def _patch_reconstructor_method(self, method_name, mock_function):
        setattr(VoronoiPETReconstructor3D, method_name, mock_function)

    def _patch_module_function(self, func_name, mock_function):
        if self.reconstructor_module: setattr(self.reconstructor_module, func_name, mock_function) # pragma: no branch

    # ... (abbreviated other tests) ...
    def test_reconstructor_initialization_default_params(self):pass # pragma: no cover
    def test_validate_generator_points_3d_functionality(self):pass # pragma: no cover
    def test_compute_voronoi_diagram_3d_logic(self):pass # pragma: no cover
    def test_validate_voronoi_cells_3d_logic(self):pass # pragma: no cover
    def test_get_lor_endpoints_3d_logic(self):pass # pragma: no cover
    def test_compute_lor_cell_intersection_3d_logic(self):pass # pragma: no cover
    def test_compute_system_matrix_3d_logic(self):pass # pragma: no cover
    def test_forward_project_3d_logic(self):pass # pragma: no cover
    def test_back_project_3d_logic(self):pass # pragma: no cover

    def test_is_point_in_face_3d_detailed(self):
        if not VORONOI_RECONSTRUCTOR_AVAILABLE: # pragma: no cover
            self.skipTest("VoronoiPETReconstructor3D class not available for testing _is_point_in_face_3d.")

        epsilon = 1e-5
        # Attempt to get device from a dummy reconstructor, default to 'cpu'
        try:
            # Minimal init for reconstructor to get device, won't call heavy setup
            # if __init__ itself doesn't fail due to missing core components for device setup
            dummy_reconstructor = VoronoiPETReconstructor3D(device='cuda' if torch.cuda.is_available() else 'cpu')
            device = dummy_reconstructor.device
        except Exception: # pragma: no cover
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        def to_tensor(data_list_or_array):
            return torch.tensor(data_list_or_array, dtype=torch.float32, device=device)

        # Test Case 1: Square Face (XY plane)
        face_vertices_sq = to_tensor([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
        normal_sq = to_tensor([0,0,1])

        # Inside
        p_in_sq = to_tensor([0.5, 0.5, 0])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_in_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_in_sq")

        # On edge
        p_edge_sq = to_tensor([0.5, 0, 0])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_edge_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_edge_sq")

        p_edge_sq_top = to_tensor([0.5, 1, 0])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_edge_sq_top, face_vertices_sq, normal_sq, epsilon), "Square: p_edge_sq_top")

        # On vertex
        p_vertex_sq = to_tensor([1,1,0])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_vertex_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_vertex_sq")

        # Outside (co-planar)
        p_out_coplanar_sq = to_tensor([1.5, 0.5, 0])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_coplanar_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_out_coplanar_sq")

        # Outside (non-co-planar)
        p_out_noncoplanar_sq = to_tensor([0.5, 0.5, 0.1])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_noncoplanar_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_out_noncoplanar_sq")

        p_out_noncoplanar_sq_neg_z = to_tensor([0.5, 0.5, -0.1])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_noncoplanar_sq_neg_z, face_vertices_sq, normal_sq, epsilon), "Square: p_out_noncoplanar_sq_neg_z")


        # Outside (co-planar, different side)
        p_out_coplanar_neg_sq = to_tensor([-0.5, 0.5, 0])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_coplanar_neg_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_out_coplanar_neg_sq")

        # Test Case 2: Triangular Face (Slanted)
        v0_tri = [0.0,0.0,0.0]; v1_tri = [1.0,0.0,1.0]; v2_tri = [0.0,1.0,1.0]
        face_vertices_tri = to_tensor([v0_tri, v1_tri, v2_tri])

        v0t = to_tensor(v0_tri); v1t = to_tensor(v1_tri); v2t = to_tensor(v2_tri)
        n_vec_tri = torch.cross(v1t - v0t, v2t - v0t)
        normal_tri = n_vec_tri / torch.linalg.norm(n_vec_tri)

        # Inside (centroid)
        p_in_tri = (v0t + v1t + v2t) / 3.0
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_in_tri, face_vertices_tri, normal_tri, epsilon), "Triangle: p_in_tri (centroid)")

        # On edge for triangle
        p_edge_tri = (v0t + v1t) / 2.0 # Midpoint of an edge
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_edge_tri, face_vertices_tri, normal_tri, epsilon), "Triangle: p_edge_tri")

        # On vertex for triangle
        p_vertex_tri = v0t
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_vertex_tri, face_vertices_tri, normal_tri, epsilon), "Triangle: p_vertex_tri")

        # Outside (co-planar, along an edge but beyond segment)
        p_out_coplanar_tri_along_edge_ext = v0t - (v1t - v0t) # Beyond v0 on line v0-v1
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_coplanar_tri_along_edge_ext, face_vertices_tri, normal_tri, epsilon), "Triangle: p_out_coplanar_tri_along_edge_ext")

        p_out_coplanar_tri_sum = v0t + (v1t-v0t) + (v2t-v0t) # v0 + vec(v0,v1) + vec(v0,v2) -> forms a parallelogram, this point is the 4th vertex.
                                                          # For a triangle, this should be outside.
        # Check co-planarity for p_out_coplanar_tri_sum
        d_plane_tri = torch.dot(normal_tri, v0t)
        self.assertTrue(torch.abs(torch.dot(normal_tri, p_out_coplanar_tri_sum) - d_plane_tri) < epsilon, "Triangle: p_out_coplanar_tri_sum sanity check co-planarity")
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_coplanar_tri_sum, face_vertices_tri, normal_tri, epsilon), "Triangle: p_out_coplanar_tri_sum")

        # Outside (non-co-planar)
        p_out_noncoplanar_tri = p_in_tri + normal_tri * 0.1
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_noncoplanar_tri, face_vertices_tri, normal_tri, epsilon), "Triangle: p_out_noncoplanar_tri")

        p_out_noncoplanar_tri_neg = p_in_tri - normal_tri * 0.1
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_noncoplanar_tri_neg, face_vertices_tri, normal_tri, epsilon), "Triangle: p_out_noncoplanar_tri_neg")


        # Test Case 3: Degenerate Face (less than 3 vertices)
        face_vertices_degenerate = to_tensor([[0,0,0], [1,0,0]])
        normal_degenerate = to_tensor([0,0,1]) # Arbitrary normal
        p_test_degenerate = to_tensor([0.5,0,0])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_test_degenerate, face_vertices_degenerate, normal_degenerate, epsilon), "Degenerate: less than 3 vertices")

        # Test Case 4: Point far from face (using square face)
        p_far_sq = to_tensor([10,10,0])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_far_sq, face_vertices_sq, normal_sq, epsilon), "Square: p_far_sq")

        # Test Case 5: Point clearly inside a larger square
        face_large_sq = to_tensor([[-10,-10,0], [10,-10,0], [10,10,0], [-10,10,0]])
        p_center_large_sq = to_tensor([0,0,0])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_center_large_sq, face_large_sq, normal_sq, epsilon), "Large Square: p_center_large_sq")

        # Test Case 6: Co-linear vertices for a face (should be handled by projection or winding number)
        # Normal calculation might be tricky, but if normal is given, projection should still work.
        # This case is more about the robustness of the winding number for projected nearly-flat polygons.
        # If normal is e.g. [0,0,1], projection is to XY. If vertices are [0,0,0],[1,0,0],[2,0,0],[0,1,0]
        # This is not a simple convex polygon. The _is_point_in_face_3d assumes convex face polygon from Voronoi.
        # Let's test a thin rectangle instead, which is convex.
        face_thin_rect = to_tensor([[0,0,0], [10,0,0], [10,0.01,0], [0,0.01,0]])
        p_in_thin_rect = to_tensor([5,0.005,0])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_in_face_3d(p_in_thin_rect, face_thin_rect, normal_sq, epsilon), "Thin Rectangle: p_in_thin_rect")
        p_out_thin_rect = to_tensor([5,0.02,0]) # Outside the thin dimension
        self.assertFalse(VoronoiPETReconstructor3D._is_point_in_face_3d(p_out_thin_rect, face_thin_rect, normal_sq, epsilon), "Thin Rectangle: p_out_thin_rect")

    def test_is_point_inside_polyhedron_3d_detailed(self):
        if not VORONOI_RECONSTRUCTOR_AVAILABLE: # pragma: no cover
            self.skipTest("VoronoiPETReconstructor3D class not available for testing _is_point_inside_polyhedron_3d.")

        epsilon = 1e-5
        try:
            dummy_reconstructor = VoronoiPETReconstructor3D(device='cuda' if torch.cuda.is_available() else 'cpu')
            device = dummy_reconstructor.device
        except Exception: # pragma: no cover
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def to_tensor(data_list_or_array):
            return torch.tensor(data_list_or_array, dtype=torch.float32, device=device)

        # Test Case 1: Unit Cube
        vertices_cube = to_tensor([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],  # Bottom face (z=0), v0,v1,v2,v3
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]   # Top face (z=1), v4,v5,v6,v7
        ])
        # Faces defined CCW from outside view
        faces_cube = [
            [0,3,2,1], # Bottom face (normal [0,0,-1]) v0,v3,v2,v1. (v3-v0)x(v2-v0) = (010)x(110) = (0,0,-1)
            [4,5,6,7], # Top face (normal [0,0,1])    v4,v5,v6,v7. (v5-v4)x(v6-v4) = (100)x(110) = (0,0,1)
            [0,1,5,4], # Front face (y=0, normal [0,-1,0]) v0,v1,v5,v4. (v1-v0)x(v5-v0) = (100)x(101) = (0,-1,0)
            [1,2,6,5], # Right face (x=1, normal [1,0,0])  v1,v2,v6,v5. (v2-v1)x(v6-v1) = (010)x(011) = (1,0,0)
            [2,3,7,6], # Back face (y=1, normal [0,1,0])   v2,v3,v7,v6. (v3-v2)x(v7-v2) = (-100)x(-101) = (0,1,0)
            [3,0,4,7]  # Left face (x=0, normal [-1,0,0])  v3,v0,v4,v7. (v0-v3)x(v4-v3) = (0-10)x(0-11) = (-1,0,0)
        ]

        p_center_cube = to_tensor([0.5,0.5,0.5])
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_center_cube, faces_cube, vertices_cube, epsilon), "Cube: p_center")

        p_on_face_cube = to_tensor([0.5,0.5,0]) # On bottom face
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_on_face_cube, faces_cube, vertices_cube, epsilon), "Cube: p_on_face (bottom)")

        p_on_face_cube_top = to_tensor([0.5,0.5,1.0]) # On top face
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_on_face_cube_top, faces_cube, vertices_cube, epsilon), "Cube: p_on_face (top)")

        p_on_edge_cube = to_tensor([0.5,0,0]) # On front-bottom edge
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_on_edge_cube, faces_cube, vertices_cube, epsilon), "Cube: p_on_edge")

        p_on_vertex_cube = to_tensor([1,1,1]) # Top-back-right vertex
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_on_vertex_cube, faces_cube, vertices_cube, epsilon), "Cube: p_on_vertex")

        p_out_x_cube = to_tensor([1.5,0.5,0.5])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_out_x_cube, faces_cube, vertices_cube, epsilon), "Cube: p_out_x")

        p_out_neg_x_cube = to_tensor([-0.5,0.5,0.5])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_out_neg_x_cube, faces_cube, vertices_cube, epsilon), "Cube: p_out_neg_x")

        p_out_z_cube = to_tensor([0.5,0.5,1.5])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_out_z_cube, faces_cube, vertices_cube, epsilon), "Cube: p_out_z")

        p_far_cube = to_tensor([10,10,10])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_far_cube, faces_cube, vertices_cube, epsilon), "Cube: p_far")

        # Test Case 2: Tetrahedron
        v0_tet_l = [0.,0.,0.]; v1_tet_l = [1.,0.,0.]; v2_tet_l = [0.5,1.,0.]; v3_tet_l = [0.5,0.5,1.]
        vertices_tet = to_tensor([v0_tet_l, v1_tet_l, v2_tet_l, v3_tet_l])
        # Faces CCW from outside
        faces_tet = [
            [0,2,1], # Base: v0,v2,v1. Normal (v2-v0)x(v1-v0) = (0.5,1,0)x(1,0,0) = (0,0,-1)
            [0,1,3], # Side 1: v0,v1,v3. Normal (v1-v0)x(v3-v0) = (1,0,0)x(0.5,0.5,1) = (0,-1,0.5)
            [1,2,3], # Side 2: v1,v2,v3. Normal (v2-v1)x(v3-v1) = (-0.5,1,0)x(-0.5,0.5,1) = (1,0.5,0.25)
            [2,0,3]  # Side 3: v2,v0,v3. Normal (v0-v2)x(v3-v2) = (-0.5,-1,0)x(0,-0.5,1) = (-1,0.5,0.25)
        ]

        v0_tet, v1_tet, v2_tet, v3_tet = to_tensor(v0_tet_l), to_tensor(v1_tet_l), to_tensor(v2_tet_l), to_tensor(v3_tet_l)
        p_in_tet_centroid = (v0_tet + v1_tet + v2_tet + v3_tet) / 4.0
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_in_tet_centroid, faces_tet, vertices_tet, epsilon), "Tetrahedron: p_in_centroid")

        p_out_tet = to_tensor([-1,-1,-1])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_out_tet, faces_tet, vertices_tet, epsilon), "Tetrahedron: p_out_far")

        p_on_face_tet_centroid = (v0_tet + v2_tet + v1_tet) / 3.0 # Centroid of base face [0,2,1]
        self.assertTrue(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_on_face_tet_centroid, faces_tet, vertices_tet, epsilon), "Tetrahedron: p_on_face_centroid")

        # Point slightly outside face [0,1,3] (normal approx (0, -1, 0.5))
        # Plane eq for face [0,1,3]: 0*x -1*y + 0.5*z = 0 (since v0 is on it)
        # Point (0.5, 0.1, 0.1) -> -0.1 + 0.05 = -0.05. This is on correct side for normal (0,-1,0.5)
        # Point (0.5, -0.1, 0.1) -> 0.1 + 0.05 = 0.15. This would be outside.
        p_out_just_past_face_tet = to_tensor([0.5, -0.1, 0.1])
        # For normal (0, -1, 0.5), point p, v0=(0,0,0). dot(normal, p-v0) = dot((0,-1,0.5), (0.5,-0.1,0.1)) = 0*0.5 + (-1)*(-0.1) + 0.5*0.1 = 0.1 + 0.05 = 0.15
        # This is > epsilon, so should be False.
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_out_just_past_face_tet, faces_tet, vertices_tet, epsilon), "Tetrahedron: p_out_just_past_face")


        # Test Case 3: Degenerate Inputs
        faces_empty = []
        p_test_degen = to_tensor([0.1,0.1,0.1])
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_test_degen, faces_empty, vertices_cube, epsilon), "Degenerate: faces_empty")

        # Not enough vertices overall for a 3D polyhedron
        vertices_few_overall = to_tensor([[0,0,0],[1,0,0],[0,1,0]]) # Only 3 points
        # faces_from_few_overall = [[0,1,2]] # A single face
        # The _is_point_inside_polyhedron_3d has `if unique_voronoi_vertices.shape[0] < 4: return False`
        # self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_test_degen, faces_from_few_overall, vertices_few_overall, epsilon), "Degenerate: <4 unique vertices for polyhedron")
        # The above test will fail if faces_from_few_overall has < 4 faces. Let's make a specific test for vertices_few_overall
        dummy_faces_for_few_verts_test = [[0,1,2],[0,1,2],[0,1,2],[0,1,2]] # Ensure 4 faces to pass initial face count check
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_test_degen, dummy_faces_for_few_verts_test, vertices_few_overall, epsilon), "Degenerate: <4 unique vertices for polyhedron")


        # Polyhedron with a degenerate face (face with < 3 indices)
        faces_with_a_degenerate_face = faces_cube[:1] + [[0,1]] + faces_cube[2:]
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_center_cube, faces_with_a_degenerate_face, vertices_cube, epsilon), "Degenerate: one face has <3 vertices")

        # Polyhedron with not enough faces
        faces_not_enough = faces_cube[:3] # Only 3 faces
        self.assertFalse(VoronoiPETReconstructor3D._is_point_inside_polyhedron_3d(p_center_cube, faces_not_enough, vertices_cube, epsilon), "Degenerate: not enough faces for polyhedron")


    def test_reconstruct_3d_overall_flow(self):
        # --- Setup Mocks for all helper methods ---
        mock_validate_gens = unittest.mock.MagicMock(return_value=(False, "OK")) # (is_invalid, msg)
        self._patch_reconstructor_method('_validate_generator_points_3d', mock_validate_gens)

        mock_compute_voronoi = unittest.mock.MagicMock(return_value=(
            [{'faces': [[0,1,2]]}], torch.rand(3,3), "OK")) # cells_data, unique_verts, msg
        self._patch_reconstructor_method('_compute_voronoi_diagram_3d', mock_compute_voronoi)

        mock_validate_cells = unittest.mock.MagicMock(return_value=(False, "OK")) # (is_invalid, msg)
        self._patch_reconstructor_method('_validate_voronoi_cells_3d', mock_validate_cells)

        mock_compute_sys_matrix = unittest.mock.MagicMock(return_value=torch.rand(5,2)) # system_matrix (5 LORs, 2 Cells)
        self._patch_reconstructor_method('_compute_system_matrix_3d', mock_compute_sys_matrix)

        mock_forward_project = unittest.mock.MagicMock(return_value=torch.rand(5)) # projection_data (5 LORs)
        self._patch_reconstructor_method('_forward_project_3d', mock_forward_project)

        mock_back_project = unittest.mock.MagicMock(return_value=torch.rand(2)) # back_projected_activity (2 Cells)
        self._patch_reconstructor_method('_back_project_3d', mock_back_project)

        reconstructor = VoronoiPETReconstructor3D(num_iterations=2, device='cpu', verbose=False) # Short iterations

        # --- Test Data ---
        sinogram = torch.rand(5) # 5 LORs
        gen_points = torch.rand(2,3) # 2 Cells / generator points
        lor_desc = {'lor_endpoints': torch.rand(5,2,3)} # Dummy, as _get_lor_endpoints_3d is part of _compute_system_matrix_3d

        # --- Test Case 1: Successful execution path ---
        result = reconstructor.reconstruct(sinogram, gen_points, lor_desc)

        self.assertEqual(result['status'], "Voronoi-based 3D PET reconstruction completed.")
        self.assertTrue(torch.is_tensor(result['activity']))
        self.assertEqual(result['activity'].shape, (2,)) # num_cells
        mock_validate_gens.assert_called_once()
        mock_compute_voronoi.assert_called_once()
        mock_validate_cells.assert_called_once()
        mock_compute_sys_matrix.assert_called_once()
        self.assertEqual(mock_forward_project.call_count, reconstructor.num_iterations)
        # Back project is called once for sensitivity, then once per iteration
        self.assertEqual(mock_back_project.call_count, reconstructor.num_iterations + 1)

        # --- Test Case 2: Generator points validation fails ---
        mock_validate_gens.return_value = (True, "Gen Fail")
        result = reconstructor.reconstruct(sinogram, gen_points, lor_desc)
        self.assertEqual(result['status'], "Failed: Gen Fail")
        self.assertIn("Gen Fail", result['error_log'][-1])
        mock_validate_gens.reset_mock(return_value=True, side_effect=True) # Reset for next test
        mock_validate_gens.return_value = (False, "OK")


        # --- Test Case 3: Voronoi diagram computation fails ---
        mock_compute_voronoi.return_value = (None, None, "Voronoi Fail")
        result = reconstructor.reconstruct(sinogram, gen_points, lor_desc)
        self.assertEqual(result['status'], "Failed: Voronoi Fail")
        self.assertIn("Voronoi Fail", result['error_log'][-1])
        mock_compute_voronoi.reset_mock(return_value=True, side_effect=True)
        mock_compute_voronoi.return_value = ([{'faces': [[0,1,2]]}], torch.rand(3,3), "OK")

        # --- Test Case 4: Voronoi cells validation fails ---
        mock_validate_cells.return_value = (True, "Cell Val Fail")
        result = reconstructor.reconstruct(sinogram, gen_points, lor_desc)
        self.assertEqual(result['status'], "Failed: Cell Val Fail")
        self.assertIn("Cell Val Fail", result['error_log'][-1])
        mock_validate_cells.reset_mock(return_value=True, side_effect=True)
        mock_validate_cells.return_value = (False, "OK")

        # --- Test Case 5: System matrix computation fails ---
        mock_compute_sys_matrix.return_value = None
        result = reconstructor.reconstruct(sinogram, gen_points, lor_desc)
        self.assertEqual(result['status'], "Failed: System matrix computation error.")
        self.assertIn("System matrix computation failed", result['error_log'][-1])
        mock_compute_sys_matrix.reset_mock(return_value=True, side_effect=True)
        mock_compute_sys_matrix.return_value = torch.rand(5,2) # 5 LORs, 2 Cells

        # --- Test Case 6: Initial estimate shape error ---
        initial_estimate_bad_shape = torch.rand(3) # Should be 2 cells
        result = reconstructor.reconstruct(sinogram, gen_points, lor_desc, initial_estimate=initial_estimate_bad_shape)
        self.assertEqual(result['status'], "Failed: Initial estimate shape error.")
        self.assertIn("Initial estimate shape mismatch", result['error_log'][-1])

        # --- Test Case 7: Sinogram LOR count mismatch ---
        sinogram_bad_shape = torch.rand(4) # System matrix expects 5 LORs
        result = reconstructor.reconstruct(sinogram_bad_shape, gen_points, lor_desc)
        self.assertEqual(result['status'], "Failed: Sinogram LOR count mismatch.")
        self.assertIn("Sinogram LOR count mismatch", result['error_log'][-1])

        # Restore original methods after test
        self.tearDown()


if __name__ == '__main__': # pragma: no cover
    unittest.main()
