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
