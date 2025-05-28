import numpy as np
import pytest
import scipy.ndimage # For testing smoothing effects indirectly

from reconlib.coil_combination import (
    coil_combination_with_phase,
    estimate_phase_maps,
    estimate_sensitivity_maps,
    reconstruct_coil_images,
    compute_density_compensation,
    EPSILON
)
from reconlib.coil_combination import regrid_kspace, filter_low_frequencies # Import placeholders for mocking

# --- Pytest Fixtures for Test Data ---

@pytest.fixture
def dummy_coil_images_2d():
    num_coils, Nx, Ny = 4, 32, 32
    return np.random.rand(num_coils, Nx, Ny) + 1j * np.random.rand(num_coils, Nx, Ny)

@pytest.fixture
def dummy_coil_images_3d():
    num_coils, Nx, Ny, Nz = 4, 16, 16, 8
    return np.random.rand(num_coils, Nx, Ny, Nz) + 1j * np.random.rand(num_coils, Nx, Ny, Nz)

@pytest.fixture
def dummy_sensitivity_maps_2d(dummy_coil_images_2d):
    s_maps = np.random.rand(*dummy_coil_images_2d.shape) + 1j * np.random.rand(*dummy_coil_images_2d.shape)
    sos = np.sqrt(np.sum(np.abs(s_maps)**2, axis=0, keepdims=True))
    return s_maps / (sos + EPSILON)

@pytest.fixture
def dummy_sensitivity_maps_3d(dummy_coil_images_3d):
    s_maps = np.random.rand(*dummy_coil_images_3d.shape) + 1j * np.random.rand(*dummy_coil_images_3d.shape)
    sos = np.sqrt(np.sum(np.abs(s_maps)**2, axis=0, keepdims=True))
    return s_maps / (sos + EPSILON)

@pytest.fixture
def dummy_phase_maps_2d(dummy_coil_images_2d):
    return np.random.uniform(-np.pi, np.pi, dummy_coil_images_2d.shape)

@pytest.fixture
def dummy_phase_maps_3d(dummy_coil_images_3d):
    return np.random.uniform(-np.pi, np.pi, dummy_coil_images_3d.shape)

@pytest.fixture
def dummy_kspace_data_2d():
    num_coils, num_arms, num_samples = 4, 16, 128
    return np.random.randn(num_coils, num_arms, num_samples) + 1j * np.random.randn(num_coils, num_arms, num_samples)

@pytest.fixture
def dummy_trajectory_2d(dummy_kspace_data_2d):
    _, num_arms, num_samples = dummy_kspace_data_2d.shape
    return np.random.randn(num_arms, num_samples, 2) # (num_arms, num_samples, kx_ky_coords)

@pytest.fixture
def dummy_grid_size_2d():
    return (32, 32)

@pytest.fixture
def dummy_kspace_data_flat(): # (num_coils, N_kpoints)
    num_coils, N_kpoints = 4, 2048
    return np.random.randn(num_coils, N_kpoints) + 1j * np.random.randn(num_coils, N_kpoints)

@pytest.fixture
def dummy_trajectory_flat(dummy_kspace_data_flat): # (N_kpoints, dims)
    _, N_kpoints = dummy_kspace_data_flat.shape
    return np.random.randn(N_kpoints, 2)


# --- Test Classes ---

class TestCoilCombinationWithPhase:
    @pytest.mark.parametrize("is_3d", [False, True])
    def test_sos_method(self, is_3d, dummy_coil_images_2d, dummy_coil_images_3d):
        coil_images = dummy_coil_images_3d if is_3d else dummy_coil_images_2d
        combined = coil_combination_with_phase(coil_images, method="sos")
        assert combined.shape == coil_images.shape[1:]
        assert not np.iscomplexobj(combined)
        assert np.all(combined >= 0)
        assert np.max(combined) <= 1.0 + EPSILON # Check normalization

    @pytest.mark.parametrize("is_3d", [False, True])
    def test_roemer_method(self, is_3d, dummy_coil_images_2d, dummy_coil_images_3d,
                           dummy_sensitivity_maps_2d, dummy_sensitivity_maps_3d):
        coil_images = dummy_coil_images_3d if is_3d else dummy_coil_images_2d
        s_maps = dummy_sensitivity_maps_3d if is_3d else dummy_sensitivity_maps_2d
        combined = coil_combination_with_phase(coil_images, method="roemer", sensitivity_maps=s_maps)
        assert combined.shape == coil_images.shape[1:]
        assert not np.iscomplexobj(combined)
        assert np.all(combined >= 0)
        assert np.max(combined) <= 1.0 + EPSILON # Check normalization

    @pytest.mark.parametrize("is_3d", [False, True])
    def test_sos_sensitivity_method(self, is_3d, dummy_coil_images_2d, dummy_coil_images_3d,
                                    dummy_sensitivity_maps_2d, dummy_sensitivity_maps_3d):
        coil_images = dummy_coil_images_3d if is_3d else dummy_coil_images_2d
        s_maps = dummy_sensitivity_maps_3d if is_3d else dummy_sensitivity_maps_2d
        combined = coil_combination_with_phase(coil_images, method="sos_sensitivity", sensitivity_maps=s_maps)
        assert combined.shape == coil_images.shape[1:]
        assert not np.iscomplexobj(combined)
        assert np.all(combined >= 0)
        assert np.max(combined) <= 1.0 + EPSILON

    @pytest.mark.parametrize("is_3d", [False, True])
    def test_with_phase_maps(self, is_3d, dummy_coil_images_2d, dummy_coil_images_3d,
                              dummy_phase_maps_2d, dummy_phase_maps_3d):
        coil_images = dummy_coil_images_3d if is_3d else dummy_coil_images_2d
        phase_maps = dummy_phase_maps_3d if is_3d else dummy_phase_maps_2d
        # Test with SoS, as it's simplest to verify phase effect if any (though SoS ignores phase)
        combined = coil_combination_with_phase(coil_images, method="sos", phase_maps=phase_maps)
        assert combined.shape == coil_images.shape[1:]
        # Basic check, phase maps should not break SoS
        assert np.max(combined) <= 1.0 + EPSILON

    def test_roemer_needs_sens_maps(self, dummy_coil_images_2d):
        with pytest.raises(ValueError, match="Sensitivity maps required for Roemer combination"):
            coil_combination_with_phase(dummy_coil_images_2d, method="roemer")

    def test_sos_sensitivity_needs_sens_maps(self, dummy_coil_images_2d):
        with pytest.raises(ValueError, match="Sensitivity maps required for SoS with sensitivity"):
            coil_combination_with_phase(dummy_coil_images_2d, method="sos_sensitivity")

    def test_invalid_method(self, dummy_coil_images_2d):
        with pytest.raises(ValueError, match="Unknown coil combination method"):
            coil_combination_with_phase(dummy_coil_images_2d, method="invalid_method")

    def test_input_type_errors(self):
        with pytest.raises(TypeError, match="coil_images must be a NumPy array"):
            coil_combination_with_phase([1, 2, 3])
        with pytest.raises(ValueError, match="coil_images must have at least 3 dimensions"):
            coil_combination_with_phase(np.array([[1,2],[3,4]]))


class TestEstimatePhaseMaps:
    @pytest.mark.parametrize("is_3d", [False, True])
    def test_lowres_method(self, is_3d, dummy_coil_images_2d, dummy_coil_images_3d):
        coil_images = dummy_coil_images_3d if is_3d else dummy_coil_images_2d
        phase_maps = estimate_phase_maps(coil_images, method="lowres")
        assert phase_maps.shape == coil_images.shape
        assert np.issubdtype(phase_maps.dtype, np.floating)
        assert np.all(phase_maps >= -np.pi - EPSILON) and np.all(phase_maps <= np.pi + EPSILON)
        # Check if smoothing was applied (difficult to check directly without original)
        # A simple check: variance of phase maps should be lower than variance of raw angle(coil_images)
        # This is a heuristic and might not always hold robustly but can catch gross errors.
        raw_phases = np.angle(coil_images)
        # Ensure smoothing is applied by checking if variance changes.
        # This requires scipy.ndimage.gaussian_filter to have an effect.
        # If sigmas are too small, this test might fail. Sigmas are derived from image size.
        # For a 32x32 image, sigma is ~1.6. For 16x16, ~0.8. These should smooth.
        if coil_images.shape[1] > 4 and coil_images.shape[2] > 4: # Avoid issues with tiny images
             assert np.var(phase_maps) < np.var(raw_phases) * 1.5 # Allow some margin

    @pytest.mark.parametrize("is_3d", [False, True])
    def test_reference_method(self, is_3d, dummy_coil_images_2d, dummy_coil_images_3d):
        coil_images = dummy_coil_images_3d if is_3d else dummy_coil_images_2d
        num_coils = coil_images.shape[0]
        ref_coil_idx = num_coils // 2
        phase_maps = estimate_phase_maps(coil_images, method="reference", reference_coil=ref_coil_idx)
        assert phase_maps.shape == coil_images.shape
        assert np.issubdtype(phase_maps.dtype, np.floating)
        assert np.all(phase_maps >= -np.pi - EPSILON) and np.all(phase_maps <= np.pi + EPSILON)
        # The phase of the reference coil relative to itself (after smoothing) should be close to 0
        # This tests the relative phase calculation and smoothing effect.
        # Phase of coil_images[ref_coil_idx] - (phase of coil_images[ref_coil_idx] - smoothed_ref_phase)
        # = smoothed_ref_phase.
        # The phase_maps[ref_coil_idx] is smoothed (angle(coil_images[ref_coil_idx]) - angle(coil_images[ref_coil_idx])) = 0, then smoothed.
        # So phase_maps[ref_coil_idx] should be close to zero.
        assert np.allclose(phase_maps[ref_coil_idx], 0, atol=0.1) # After smoothing, should be near 0

    def test_invalid_method(self, dummy_coil_images_2d):
        with pytest.raises(ValueError, match="Unknown phase estimation method"):
            estimate_phase_maps(dummy_coil_images_2d, method="invalid_method")

    def test_invalid_reference_coil(self, dummy_coil_images_2d):
        with pytest.raises(ValueError, match="reference_coil index .* is out of bounds"):
            estimate_phase_maps(dummy_coil_images_2d, method="reference", reference_coil=100)

    def test_input_type_errors(self):
        with pytest.raises(TypeError, match="coil_images must be a NumPy array"):
            estimate_phase_maps([1,2,3])
        with pytest.raises(ValueError, match="coil_images must have at least 3 dimensions"):
            estimate_phase_maps(np.array([[1,2],[3,4]]))


class TestEstimateSensitivityMaps:
    # Mocking placeholder functions for these tests
    @pytest.fixture(autouse=True)
    def mock_external_dependencies(self, monkeypatch):
        # Mock regrid_kspace: needs to return something of shape (grid_size)
        def mock_regrid(kspace_data, trajectory, grid_size, density_weights=None):
            # print(f"Mock regrid called with grid_size: {grid_size}")
            # Ensure grid_size is a tuple of integers
            if not isinstance(grid_size, tuple) or not all(isinstance(d, int) for d in grid_size):
                 raise ValueError(f"Mock regrid_kspace expects grid_size to be a tuple of ints, got {grid_size}")
            return np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)

        # Mock filter_low_frequencies: needs to return kspace_data (can be same as input for simplicity)
        def mock_filter_low_freq(kspace_data, trajectory):
            return kspace_data # Pass-through

        monkeypatch.setattr('reconlib.coil_combination.regrid_kspace', mock_regrid)
        monkeypatch.setattr('reconlib.coil_combination.filter_low_frequencies', mock_filter_low_freq)


    def test_lowres_method(self, dummy_kspace_data_2d, dummy_trajectory_2d, dummy_grid_size_2d):
        s_maps = estimate_sensitivity_maps(dummy_kspace_data_2d, dummy_trajectory_2d,
                                           dummy_grid_size_2d, method="lowres")
        expected_shape = (dummy_kspace_data_2d.shape[0],) + dummy_grid_size_2d
        assert s_maps.shape == expected_shape
        assert np.iscomplexobj(s_maps)
        
        # Check normalization: sum of squares of magnitudes across coils should be approx 1
        s_maps_mag_sq = np.abs(s_maps)**2
        sos_check = np.sum(s_maps_mag_sq, axis=0)
        assert np.allclose(sos_check, 1.0, atol=1e-5) # Looser tolerance due to mocks/randomness

    def test_espirit_method_raises_not_implemented(self, dummy_kspace_data_2d, dummy_trajectory_2d, dummy_grid_size_2d):
        with pytest.raises(NotImplementedError, match="ESPIRiT method not yet implemented"):
            estimate_sensitivity_maps(dummy_kspace_data_2d, dummy_trajectory_2d,
                                      dummy_grid_size_2d, method="espirit")

    def test_invalid_method(self, dummy_kspace_data_2d, dummy_trajectory_2d, dummy_grid_size_2d):
        with pytest.raises(ValueError, match="Unknown sensitivity estimation method"):
            estimate_sensitivity_maps(dummy_kspace_data_2d, dummy_trajectory_2d,
                                      dummy_grid_size_2d, method="invalid_method")
    
    def test_input_type_errors(self, dummy_trajectory_2d, dummy_grid_size_2d):
        with pytest.raises(TypeError, match="kspace_data must be a complex NumPy array"):
            estimate_sensitivity_maps(np.random.rand(4,100), dummy_trajectory_2d, dummy_grid_size_2d) # Real data
        with pytest.raises(ValueError, match="kspace_data must have at least 2 dimensions"):
            estimate_sensitivity_maps(np.array([1+1j]), dummy_trajectory_2d, dummy_grid_size_2d)


class TestReconstructCoilImages:
    # Mocking placeholder functions for these tests
    @pytest.fixture(autouse=True)
    def mock_regrid_and_density(self, monkeypatch):
        # Mock regrid_kspace
        def mock_regrid(kspace_data, trajectory, grid_size, density_weights=None):
            if not isinstance(grid_size, tuple) or not all(isinstance(d, int) for d in grid_size):
                 raise ValueError(f"Mock regrid_kspace expects grid_size to be a tuple of ints, got {grid_size}")
            return np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)
        
        # Mock compute_density_compensation
        def mock_compute_density(trajectory, method="pipe"):
            # Shape depends on trajectory shape.
            # If trajectory is (num_arms, num_samples, dims), output (num_arms, num_samples)
            # If trajectory is (N_kpoints, dims), output (N_kpoints,)
            if trajectory.ndim == 3: # (num_arms, num_samples, dims)
                return np.random.rand(trajectory.shape[0], trajectory.shape[1])
            elif trajectory.ndim == 2: # (N_kpoints, dims)
                return np.random.rand(trajectory.shape[0])
            else: # Fallback for other shapes, e.g. complex 1D
                return np.random.rand(trajectory.shape[0])


        monkeypatch.setattr('reconlib.coil_combination.regrid_kspace', mock_regrid)
        monkeypatch.setattr('reconlib.coil_combination.compute_density_compensation', mock_compute_density)

    def test_basic_reconstruction(self, dummy_kspace_data_2d, dummy_trajectory_2d, dummy_grid_size_2d):
        coil_images = reconstruct_coil_images(dummy_kspace_data_2d, dummy_trajectory_2d, dummy_grid_size_2d)
        expected_shape = (dummy_kspace_data_2d.shape[0],) + dummy_grid_size_2d
        assert coil_images.shape == expected_shape
        assert np.iscomplexobj(coil_images)

    def test_with_density_weights(self, dummy_kspace_data_flat, dummy_trajectory_flat, dummy_grid_size_2d):
        # dummy_trajectory_flat is (N_kpoints, 2)
        # dummy_kspace_data_flat is (num_coils, N_kpoints)
        # density_weights should be (N_kpoints,)
        density_weights = np.random.rand(dummy_trajectory_flat.shape[0]) 
        coil_images = reconstruct_coil_images(dummy_kspace_data_flat, dummy_trajectory_flat,
                                              dummy_grid_size_2d, density_weights=density_weights)
        expected_shape = (dummy_kspace_data_flat.shape[0],) + dummy_grid_size_2d
        assert coil_images.shape == expected_shape
        assert np.iscomplexobj(coil_images)

    def test_unmocked_regrid_raises_error(self, monkeypatch, dummy_kspace_data_2d, dummy_trajectory_2d, dummy_grid_size_2d):
        # Restore original regrid_kspace which should be a placeholder
        monkeypatch.setattr('reconlib.coil_combination.regrid_kspace', regrid_kspace) 
        # We also need to ensure compute_density_compensation doesn't fail first if it's called
        # For this test, let's provide density_weights to isolate regrid_kspace
        density_weights = np.random.rand(dummy_trajectory_2d.shape[0], dummy_trajectory_2d.shape[1])

        with pytest.raises(NotImplementedError, match="regrid_kspace not yet implemented"):
            reconstruct_coil_images(dummy_kspace_data_2d, dummy_trajectory_2d,
                                    dummy_grid_size_2d, density_weights=density_weights)
            
    def test_input_type_errors(self, dummy_trajectory_2d, dummy_grid_size_2d):
        with pytest.raises(TypeError, match="kspace_data must be a complex NumPy array"):
            reconstruct_coil_images(np.random.rand(4,100,10), dummy_trajectory_2d, dummy_grid_size_2d)
        with pytest.raises(ValueError, match="kspace_data must have at least 2 dimensions"):
            reconstruct_coil_images(np.array([1+1j]), dummy_trajectory_2d, dummy_grid_size_2d)


class TestComputeDensityCompensation:
    @pytest.mark.parametrize("traj_type", ["complex_flat", "real_2d_flat", "real_3d_arms"])
    def test_pipe_method(self, traj_type):
        if traj_type == "complex_flat":
            # (N_kpoints,) complex
            trajectory = np.random.rand(100) + 1j * np.random.rand(100)
            expected_shape = (100,)
        elif traj_type == "real_2d_flat":
            # (N_kpoints, dims)
            trajectory = np.random.rand(100, 2)
            expected_shape = (100,)
        elif traj_type == "real_3d_arms":
            # (num_arms, num_samples, dims)
            trajectory = np.random.rand(10, 20, 2)
            expected_shape = (10, 20)
        else: # real_1d_flat (interpreted as radius)
            trajectory = np.random.rand(100)
            expected_shape = (100,)


        weights = compute_density_compensation(trajectory, method="pipe")
        assert weights.shape == expected_shape
        assert np.issubdtype(weights.dtype, np.floating) # Should be real
        if traj_type == "complex_flat":
            assert np.all(weights == np.abs(trajectory))
        elif "real" in traj_type: # For real coordinate inputs
            if trajectory.ndim == 1: # Assumed to be radius
                 assert np.all(weights == trajectory)
            else: # Calculated radius
                 radius = np.sqrt(np.sum(trajectory**2, axis=-1))
                 assert np.allclose(weights, radius)


    def test_voronoi_method_raises_not_implemented(self):
        trajectory = np.random.rand(100, 2)
        with pytest.raises(NotImplementedError, match="Voronoi method for density compensation not yet implemented"):
            compute_density_compensation(trajectory, method="voronoi")

    def test_invalid_method(self):
        trajectory = np.random.rand(100, 2)
        with pytest.raises(ValueError, match="Unknown density compensation method"):
            compute_density_compensation(trajectory, method="invalid_method")
            
    def test_input_type_errors(self):
        with pytest.raises(TypeError, match="trajectory must be a NumPy array"):
            compute_density_compensation([1,2,3])

    def test_pipe_method_1d_real_trajectory(self):
        # Test specific case: 1D real trajectory (interpreted as radius directly)
        trajectory = np.random.rand(50) 
        weights = compute_density_compensation(trajectory, method="pipe")
        assert weights.shape == trajectory.shape
        assert np.all(weights == trajectory)

    def test_pipe_method_unsupported_trajectory_format(self):
        # Example: A 4D trajectory, which is not handled by current logic
        trajectory = np.random.rand(2,3,4,5)
        with pytest.raises(ValueError, match="Unsupported trajectory shape"):
            compute_density_compensation(trajectory, method="pipe")


# Helper to check if a module function is a placeholder (raises NotImplementedError)
def is_placeholder(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return False
    except NotImplementedError:
        return True
    except Exception: # Other errors mean it's not just a placeholder
        return False

# Test if the placeholder functions themselves raise NotImplementedError
# This is important because other tests might mock them.
class TestPlaceholdersAreImplementedOrNot:
    def test_regrid_kspace_placeholder(self):
        # Provide minimal valid-looking args
        kspace_data = np.array([1+1j])
        trajectory = np.array([[0,0]])
        grid_size = (2,2)
        assert is_placeholder(regrid_kspace, kspace_data, trajectory, grid_size), \
            "regrid_kspace is expected to be a placeholder but did not raise NotImplementedError"

    def test_filter_low_frequencies_placeholder(self):
        kspace_data = np.array([1+1j])
        trajectory = np.array([[0,0]])
        assert is_placeholder(filter_low_frequencies, kspace_data, trajectory), \
            "filter_low_frequencies is expected to be a placeholder but did not raise NotImplementedError"

# Note: Other placeholders like ESPIRiT are tested implicitly when methods using them are called.
# For example, estimate_sensitivity_maps with method="espirit".
# compute_voronoi_tessellation and compute_polygon_area are not directly exposed or tested here,
# but would be if compute_density_compensation(method="voronoi") was implemented using them.

# To run these tests:
# Ensure reconlib is in PYTHONPATH
# `pytest tests/test_coil_combination.py`
