import unittest
import torch
import numpy as np
from reconlib.operators import IRadon

class TestIRadon(unittest.TestCase):

    def setUp(self):
        self.img_size = (64, 64)
        self.angles = np.linspace(0, np.pi, 90, endpoint=False) # 90 angles over 180 degrees
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu") # Forcing CPU for consistent testing if GPU issues arise
        
        self.iradon_op = IRadon(
            img_size=self.img_size,
            angles=self.angles,
            filter_type="ramp",
            device=self.device
        )
        
        # Estimate n_detector_pixels for sinogram creation, consistent with IRadon internal logic
        self.n_detector_pixels = int(np.ceil(np.sqrt(self.img_size[0]**2 + self.img_size[1]**2)))
        if self.n_detector_pixels % 2 == 0:
            self.n_detector_pixels += 1

    def test_instantiation(self):
        self.assertIsNotNone(self.iradon_op)
        self.assertEqual(self.iradon_op.img_size, self.img_size)
        self.assertEqual(len(self.iradon_op.angles), len(self.angles))
        self.assertEqual(self.iradon_op.filter_type, "ramp")
        self.assertEqual(self.iradon_op.device, self.device)

    def test_ramp_filter_creation(self):
        # _ramp_filter is called in __init__, result stored in self.iradon_op.filter
        internal_filter = self.iradon_op.filter
        self.assertIsNotNone(internal_filter)
        self.assertEqual(internal_filter.ndim, 2) # Shape (1, n_detector)
        self.assertEqual(internal_filter.shape[0], 1)
        
        # Check if the filter length is consistent with n_detector_pixels estimated in setUp
        # This assumes the default filter creation uses the img_size diagonal estimate
        expected_n_detector = int(np.ceil(np.sqrt(self.img_size[0]**2 + self.img_size[1]**2)))
        if expected_n_detector % 2 == 0:
            expected_n_detector += 1
        self.assertEqual(internal_filter.shape[1], expected_n_detector)
        self.assertEqual(internal_filter.device, self.device)

    def test_grid_creation(self):
        # _create_grid is called in __init__, result stored in self.iradon_op.grid
        grid = self.iradon_op.grid
        self.assertIsNotNone(grid)
        self.assertEqual(grid.ndim, 3) # Shape (num_angles, img_height, img_width)
        self.assertEqual(grid.shape[0], len(self.angles))
        self.assertEqual(grid.shape[1], self.img_size[0])
        self.assertEqual(grid.shape[2], self.img_size[1])
        self.assertEqual(grid.device, self.device)

    def test_op_adj_radon_transform(self):
        # Create a simple phantom: a centered square
        phantom = torch.zeros(self.img_size, device=self.device, dtype=torch.float32)
        center_x, center_y = self.img_size[1] // 2, self.img_size[0] // 2
        square_size = self.img_size[0] // 8
        phantom[center_y - square_size: center_y + square_size, 
                center_x - square_size: center_x + square_size] = 1.0

        sinogram = self.iradon_op.op_adj(phantom)
        
        self.assertIsNotNone(sinogram)
        self.assertEqual(sinogram.ndim, 2)
        self.assertEqual(sinogram.shape[0], len(self.angles))
        # The number of detector pixels in sinogram should match the filter's width if filter exists,
        # or the internally estimated n_detector_pixels if no filter (though IRadon always has one by default)
        self.assertEqual(sinogram.shape[1], self.iradon_op.filter.shape[1])
        self.assertEqual(sinogram.device, self.device)
        self.assertTrue(torch.sum(sinogram) > 0) # Basic check that projection did something

    def test_op_filtered_backprojection(self):
        # Create a simple phantom: a centered square
        phantom = torch.zeros(self.img_size, device=self.device, dtype=torch.float32)
        center_y, center_x = self.img_size[0] // 2, self.img_size[1] // 2
        square_size = self.img_size[0] // 8
        
        # Make the square slightly off-center to avoid perfect symmetry issues with few angles
        y_start, y_end = center_y - square_size, center_y + square_size
        x_start, x_end = center_x - square_size - 2, center_x + square_size - 2
        phantom[y_start:y_end, x_start:x_end] = 1.0

        # Generate sinogram using op_adj (Radon transform part of IRadon)
        # Note: op_adj itself applies the filter again. For FBP, the input sino should be raw.
        # This means we need a "true" Radon transform first, or use a sinogram that is already filtered.
        # The IRadon.op_adj is defined as A, and IRadon.op is A_adj.
        # A(x) = op_adj(x) -> this is Radon with filter
        # A_adj(y) = op(y) -> this is FBP (filter + backproject)
        # If op_adj already includes the filter, then op(op_adj(x)) is not quite right for testing FBP alone.
        
        # Let's create a "raw" sinogram for the phantom first.
        # We can use PETForwardProjection if available, or implement a simple Radon here.
        # For now, let's assume the sinogram for FBP should be "unfiltered" or "filtered once".
        # The IRadon.op method expects a sinogram and applies its internal filter.
        # The IRadon.op_adj method takes an image and produces a "filtered" sinogram.
        
        # For this test, let's make a sinogram that has the properties op() expects.
        # We can use op_adj to get a "filtered sinogram"
        # If IRadon.op expects a *non-filtered* sinogram which it then filters, this is tricky.
        # From the IRadon implementation:
        # op(sino): sino_fft = fft(sino); filtered_sino_fft = sino_fft * self.filter; ifft.
        # This implies `sino` should be the raw, unfiltered sinogram.
        
        # How to get a raw sinogram? Let's use a temporary IRadon with filter_type=None for op_adj.
        temp_radon_no_filter = IRadon(
            img_size=self.img_size, 
            angles=self.angles, 
            filter_type=None, 
            device=self.device
        )
        raw_sinogram = temp_radon_no_filter.op_adj(phantom) # This is Radon(phantom) * scaling_factor
        
        # Now apply FBP (which includes filtering) using the main iradon_op
        reconstructed_img = self.iradon_op.op(raw_sinogram)

        self.assertIsNotNone(reconstructed_img)
        self.assertEqual(reconstructed_img.shape, self.img_size)
        self.assertEqual(reconstructed_img.device, self.device)

        # Compare reconstructed_img to phantom. This is tricky.
        # FBP is not perfect, especially with limited angles and simple phantom.
        # We expect some resemblance. A common metric is MSE or SSIM.
        # For a unit test, a simpler check might be to see if max intensity is in the right place,
        # or if the sum of reconstructed image is positive.
        
        # Check that the reconstruction has energy
        self.assertTrue(torch.sum(reconstructed_img**2) > 1e-6)

        # A very basic check: the reconstruction should be somewhat correlated with the phantom.
        # Normalize both to have zero mean and unit variance for a more stable correlation
        phantom_norm = (phantom - torch.mean(phantom)) / (torch.std(phantom) + 1e-9)
        reconstructed_norm = (reconstructed_img - torch.mean(reconstructed_img)) / (torch.std(reconstructed_img) + 1e-9)
        
        correlation = torch.sum(phantom_norm * reconstructed_norm) / (phantom.numel())
        # This correlation can be low for simple phantoms and FBP.
        # For a centered square, it should be reasonably positive.
        self.assertTrue(correlation > 0.05, f"Correlation too low: {correlation.item()}")


    def test_adjoint_property(self):
        # A = op_adj (Radon transform with filter)
        # A_adj = op (FBP - filter and backproject)
        # We need to test <A(x), y> approx <x, A_adj(y)>
        # x: random image
        # y: random sinogram
        
        torch.manual_seed(0) # for reproducibility
        img_x = torch.rand(self.img_size, device=self.device, dtype=torch.float32)
        
        # Sinogram y should match the expected output of op_adj(img_x)
        # or input of op(sino_y)
        # op_adj outputs sino of shape (num_angles, n_detector_pixels_from_filter)
        # op takes sino of shape (num_angles, n_detector_pixels_sino)
        # The n_detector_pixels should match.
        n_detector_op = self.iradon_op.filter.shape[1] # from default filter in iradon_op
        sino_y = torch.rand((len(self.angles), n_detector_op), device=self.device, dtype=torch.float32)

        # Ax = self.iradon_op.op_adj(img_x)
        # This is Radon transform of img_x, WITH filtering, due to IRadon's op_adj structure.
        Ax = self.iradon_op.op_adj(img_x)
        
        # A_adj_y = self.iradon_op.op(sino_y)
        # This is FBP of sino_y, which means sino_y is filtered, then backprojected.
        A_adj_y = self.iradon_op.op(sino_y)

        # Dot products
        # <Ax, y>
        dot_Ax_y = torch.vdot(Ax.flatten(), sino_y.flatten())
        
        # <x, A_adj_y>
        dot_x_A_adj_y = torch.vdot(img_x.flatten(), A_adj_y.flatten())

        # print(f"Dot <Ax,y>: {dot_Ax_y.item()}")
        # print(f"Dot <x,A_adj_y>: {dot_x_A_adj_y.item()}")
        
        # The issue: IRadon.op applies a filter. IRadon.op_adj also applies a filter.
        # So, A = F_radon * Filter, and A_adj = Backproject * Filter.
        # This means the A_adj defined here is not the true adjoint of A.
        # <F_radon*Filter x, y> vs <x, Backproject*Filter y>
        # True adjoint of (F_radon * Filter) is (Filter_adj * F_radon_adj) = (Filter * Backproject)
        # So, op(y) should be Filter * Backproject(y) for the dot product to hold if op_adj is Radon * Filter.
        # The current IRadon.op(y) is Backproject(Filter(y)).
        # And IRadon.op_adj(x) is Filter(Radon(x)).
        # So we are testing <Filter(Radon(x)), y> vs <x, Backproject(Filter(y))>
        # This is not the standard adjoint test unless Filter is self-adjoint and commutes, or Radon=Backproject_adj.
        # Let R = Radon (no filter), B = Backproject (no filter, R_adj=B). F = Filter.
        # op_adj(x) = F(R(x))  (current IRadon.op_adj implementation)
        # op(y) = B(F(y))    (current IRadon.op implementation)
        # We need to test <F R x, y> =? <x, B F y>. This should hold if F is self-adjoint (real symmetric filter).
        # The ramp filter is real and symmetric in Fourier space, so F is self-adjoint.
        # So the dot product test should pass.

        torch.testing.assert_close(dot_Ax_y, dot_x_A_adj_y, rtol=1e-3, atol=1e-3) # Increased tolerance

from reconlib.operators import PETForwardProjection

class TestPETForwardProjection(unittest.TestCase):
    def setUp(self):
        self.img_size = (64, 64)
        # Fewer angles might be more typical for PET or to speed up tests
        self.angles = np.linspace(0, np.pi, 45, endpoint=False) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.pet_op = PETForwardProjection(
            img_size=self.img_size,
            angles=self.angles,
            device=self.device
        )
        # n_detector_pixels is determined internally by PETForwardProjection
        self.n_detector_pixels = self.pet_op.n_detector_pixels


    def test_instantiation_pet(self):
        self.assertIsNotNone(self.pet_op)
        self.assertEqual(self.pet_op.img_size, self.img_size)
        self.assertEqual(len(self.pet_op.angles), len(self.angles))
        self.assertEqual(self.pet_op.device, self.device)
        self.assertTrue(self.pet_op.n_detector_pixels > 0)

    def test_grid_creation_pet(self):
        # _create_grid is called in __init__, result stored in self.pet_op.grid
        grid = self.pet_op.grid
        self.assertIsNotNone(grid)
        self.assertEqual(grid.ndim, 3) # Shape (num_angles, img_height, img_width)
        self.assertEqual(grid.shape[0], len(self.angles))
        self.assertEqual(grid.shape[1], self.img_size[0])
        self.assertEqual(grid.shape[2], self.img_size[1])
        self.assertEqual(grid.device, self.device)

    def test_op_pet_forward_projection(self):
        # Create a simple phantom: a centered square
        phantom = torch.zeros(self.img_size, device=self.device, dtype=torch.float32)
        center_y, center_x = self.img_size[0] // 2, self.img_size[1] // 2
        square_size = self.img_size[0] // 8
        phantom[center_y - square_size: center_y + square_size, 
                center_x - square_size: center_x + square_size] = 1.0

        # op is PET forward projection (Radon transform)
        sinogram = self.pet_op.op(phantom) 
        
        self.assertIsNotNone(sinogram)
        self.assertEqual(sinogram.ndim, 2)
        self.assertEqual(sinogram.shape[0], len(self.angles))
        self.assertEqual(sinogram.shape[1], self.n_detector_pixels)
        self.assertEqual(sinogram.device, self.device)
        self.assertTrue(torch.sum(sinogram) > 0) # Basic check

    def test_op_adj_pet_backprojection(self):
        # Create a simple sinogram: e.g., a single active detector bin for all angles
        sino_input = torch.zeros((len(self.angles), self.n_detector_pixels), device=self.device, dtype=torch.float32)
        center_detector_bin = self.n_detector_pixels // 2
        sino_input[:, center_detector_bin] = 1.0 # All angles see something at the center detector

        # op_adj is simple backprojection
        backprojected_img = self.pet_op.op_adj(sino_input)

        self.assertIsNotNone(backprojected_img)
        self.assertEqual(backprojected_img.shape, self.img_size)
        self.assertEqual(backprojected_img.device, self.device)
        self.assertTrue(torch.sum(backprojected_img) > 0) # Basic check

        # For this specific sinogram, we expect higher intensity near the center of the image
        center_y, center_x = self.img_size[0] // 2, self.img_size[1] // 2
        patch_size = self.img_size[0] // 4
        center_patch = backprojected_img[
            center_y - patch_size // 2 : center_y + patch_size // 2,
            center_x - patch_size // 2 : center_x + patch_size // 2
        ]
        # Sum of center patch should be greater than sum of a corner patch of same size
        corner_patch = backprojected_img[0:patch_size, 0:patch_size]
        self.assertTrue(torch.sum(center_patch) > torch.sum(corner_patch) + 1e-6, 
                        f"Center patch sum {torch.sum(center_patch)} not greater than corner {torch.sum(corner_patch)}")


    def test_adjoint_property_pet(self):
        # For PETForwardProjection:
        # A = op (PET forward projection / Radon transform)
        # A_adj = op_adj (Simple backprojection)
        # Test <A(x), y> = <x, A_adj(y)>
        # x: random image
        # y: random sinogram
        
        torch.manual_seed(123) # for reproducibility
        img_x = torch.rand(self.img_size, device=self.device, dtype=torch.float32)
        sino_y = torch.rand((len(self.angles), self.n_detector_pixels), device=self.device, dtype=torch.float32)

        # Ax = A(img_x)
        Ax = self.pet_op.op(img_x)
        
        # A_adj_y = A_adj(sino_y)
        A_adj_y = self.pet_op.op_adj(sino_y)

        # Dot products
        dot_Ax_y = torch.vdot(Ax.flatten(), sino_y.flatten())
        dot_x_A_adj_y = torch.vdot(img_x.flatten(), A_adj_y.flatten())
        
        # print(f"PET Dot <Ax,y>: {dot_Ax_y.item()}")
        # print(f"PET Dot <x,A_adj_y>: {dot_x_A_adj_y.item()}")

        # PET op and op_adj do not involve filtering, so this should hold well.
        # The operations are linear interpolation and its adjoint.
        torch.testing.assert_close(dot_Ax_y, dot_x_A_adj_y, rtol=1e-4, atol=1e-4)

from reconlib.operators import PATForwardProjection

class TestPATForwardProjection(unittest.TestCase):
    def setUp(self):
        self.img_shape = (16, 16) # ny, nx
        # Sensor positions are relative to the center of the image.
        # Image pixel coords in PATForwardProjection are from -(N-1)/2 to (N-1)/2.
        # For N=16, range is -7.5 to 7.5. Sensors at +/-8 are just outside.
        self.sensor_positions = torch.tensor([
            [0.0, 8.0],  # Top, outside
            [8.0, 0.0],  # Right, outside
            [0.0, -8.0], # Bottom, outside
            [-8.0, 0.0]  # Left, outside
        ], dtype=torch.float32)
        self.num_sensors = self.sensor_positions.shape[0]
        
        self.sound_speed = 1.0
        # Time points: from 0 up to a time that allows sound to cross the image and reach sensors.
        # Max distance from center (0,0) to a corner (7.5, 7.5) is sqrt(7.5^2 + 7.5^2) approx 10.6.
        # Sensor at (0,8) to corner (-7.5, 7.5) is distance sqrt((-7.5-0)^2 + (7.5-8)^2) = sqrt(56.25 + 0.25) = sqrt(56.5) approx 7.5
        # Sensor at (0,8) to opposite corner (-7.5, -7.5) is distance sqrt((-7.5-0)^2 + (-7.5-8)^2) = sqrt(56.25 + 240.25) = sqrt(296.5) approx 17.2
        # So, time_points up to ~18-20 should be enough if sound_speed is 1.
        self.time_points = torch.linspace(0, 20, 40, dtype=torch.float32) # 40 time samples
        self.num_time_samples = len(self.time_points)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu") # Forcing CPU

        self.pat_op = PATForwardProjection(
            img_shape=self.img_shape,
            sensor_positions=self.sensor_positions,
            sound_speed=self.sound_speed,
            time_points=self.time_points,
            device=self.device
        )
        
        # Move local copies of tensors to device as well for direct use in tests
        self.sensor_positions = self.sensor_positions.to(self.device)
        self.time_points = self.time_points.to(self.device)


    def test_instantiation_pat(self):
        self.assertIsNotNone(self.pat_op)
        self.assertEqual(self.pat_op.img_shape, self.img_shape)
        torch.testing.assert_close(self.pat_op.sensor_positions, self.sensor_positions)
        self.assertEqual(self.pat_op.sound_speed, self.sound_speed)
        torch.testing.assert_close(self.pat_op.time_points, self.time_points)
        self.assertEqual(self.pat_op.device, self.device)
        self.assertIsNotNone(self.pat_op.pixel_x_coords) # Check if pixel grids were created
        self.assertIsNotNone(self.pat_op.pixel_y_coords)

    def test_op_forward_projection(self):
        phantom_image = torch.zeros(self.img_shape, device=self.device, dtype=torch.float32)
        # Place a single bright pixel at the center of the image.
        # For (16,16), center indices are (7,7) or (8,8).
        # Pixel coordinates are -(N-1)/2 to (N-1)/2. For N=16, this is -7.5 to 7.5.
        # The pixel (0,0) in PATForwardProjection internal coords is at img_idx (img_shape[0]//2 -1 , img_shape[1]//2 -1) if N is even.
        # Or more simply, the pixel whose world coordinate is (0,0).
        # Let's put a point at image index (cy, cx) corresponding to physical (0,0) if possible.
        # cy_idx, cx_idx = self.img_shape[0] // 2, self.img_shape[1] // 2 # This is one of the 4 center pixels
        # A single point source at the origin of the coordinate system (center of the image grid)
        # The PAT operator's internal grid has (0,0) at the center.
        # If img_shape is (16,16), indices 7 and 8 are around the center.
        # Pixel coord for index i is i - (N-1)/2. For N=16, (N-1)/2 = 7.5.
        # Index 7 -> 7 - 7.5 = -0.5. Index 8 -> 8 - 7.5 = 0.5.
        # So a source at pixel (7,7) has coords (-0.5, -0.5).
        # A source at (8,8) has coords (0.5, 0.5).
        # Let's use a small 2x2 square at the center for robustness.
        cy, cx = self.img_shape[0] // 2, self.img_shape[1] // 2
        phantom_image[cy-1:cy+1, cx-1:cx+1] = 1.0 # Small 2x2 square around center
        
        sensor_data = self.pat_op.op(phantom_image)
        
        self.assertEqual(sensor_data.shape, (self.num_sensors, self.num_time_samples))
        self.assertEqual(sensor_data.device, self.device)

        # Basic property check:
        # Source is near (0,0). Sensors are at distance 8. Sound speed 1.
        # Expected time of arrival: t = distance / sound_speed = 8.0 / 1.0 = 8.0.
        expected_time = 8.0
        
        # Find the time_points index closest to expected_time
        time_diffs = torch.abs(self.time_points - expected_time)
        closest_time_idx = torch.argmin(time_diffs)
        
        # Check if there's significant signal around this time index for all sensors
        # Due to shell thickness, signal might be spread over a few time points
        # And due to 2x2 source, it's not a perfect point.
        start_idx = max(0, closest_time_idx - 2)
        end_idx = min(self.num_time_samples, closest_time_idx + 3) # check a small window of 5 points

        for s_idx in range(self.num_sensors):
            signal_window = sensor_data[s_idx, start_idx:end_idx]
            self.assertTrue(torch.sum(signal_window) > 1e-3, 
                            f"Sensor {s_idx} should have received a signal around t={expected_time} (indices {start_idx}-{end_idx-1}). Sum was {torch.sum(signal_window)}")
        self.assertTrue(torch.sum(sensor_data) > 1e-3) # Overall check

    def test_op_adj_backprojection(self):
        sensor_data_input = torch.zeros((self.num_sensors, self.num_time_samples), device=self.device, dtype=torch.float32)
        
        # Simulate a signal from sensor 0 (at [0.0, 8.0]) at a time corresponding to a source at origin (0,0)
        # Distance = 8.0. Time = 8.0 / 1.0 = 8.0.
        expected_time = 8.0
        closest_time_idx = torch.argmin(torch.abs(self.time_points - expected_time))
        
        sensor_data_input[0, closest_time_idx] = 1.0 # Signal at sensor 0 at this specific time
        
        reconstructed_image = self.pat_op.op_adj(sensor_data_input)
        
        self.assertEqual(reconstructed_image.shape, self.img_shape)
        self.assertEqual(reconstructed_image.device, self.device)
        self.assertTrue(torch.sum(reconstructed_image) > 1e-3, "Reconstructed image should have some energy.")

        # The backprojection should create a circular/arc pattern.
        # The highest intensity should be on a circle of radius `expected_time * sound_speed`
        # centered at `sensor_positions[0]`.
        # For this test, just checking total energy is a start.
        # A more specific check: pixels around the true source location (0,0) should have some intensity.
        # The pixel at (0,0) relative to center is around img_idx (cy,cx).
        cy, cx = self.img_shape[0] // 2, self.img_shape[1] // 2
        # Check a small patch around the center. Because sensor 0 is at (0,8), the arc from this
        # specific sensor data point should pass through points like (0,0) if radius is 8.
        center_patch = reconstructed_image[cy-2:cy+2, cx-2:cx+2]
        self.assertTrue(torch.sum(center_patch) > 1e-4, 
                        f"Center patch of reconstructed image should have energy. Sum was {torch.sum(center_patch)}")

    def test_adjoint_property_pat(self):
        torch.manual_seed(42) # For reproducibility
        img_x = torch.rand(self.img_shape, device=self.device, dtype=torch.float32)
        sensor_data_y = torch.rand((self.num_sensors, self.num_time_samples), device=self.device, dtype=torch.float32)
        
        # Ax = A(img_x)
        Ax = self.pat_op.op(img_x)
        
        # A_adj_y = A_adj(sensor_data_y)
        A_adj_y = self.pat_op.op_adj(sensor_data_y)
        
        # Dot products
        dot_Ax_y = torch.vdot(Ax.flatten(), sensor_data_y.flatten())
        dot_x_A_adj_y = torch.vdot(img_x.flatten(), A_adj_y.flatten())
        
        # print(f"PAT Dot <Ax,y>: {dot_Ax_y.item()}")
        # print(f"PAT Dot <x,A_adj_y>: {dot_x_A_adj_y.item()}")
        
        # Due to the binary shell condition (pixel is either in or out), this might not be perfectly adjoint
        # without very fine discretization or a smoother shell.
        # The tolerance might need to be relatively loose.
        torch.testing.assert_close(dot_Ax_y, dot_x_A_adj_y, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
