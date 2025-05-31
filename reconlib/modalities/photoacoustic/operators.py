import torch
from reconlib.operators import Operator
import numpy as np

class PhotoacousticOperator(Operator):
    """
    Forward and Adjoint Operator for Photoacoustic Tomography.

    Models the generation and propagation of acoustic waves following light absorption.
    The forward operator simulates the acoustic signals received by transducers
    based on an initial pressure distribution (the image to be reconstructed),
    using a simplified time-of-flight model.
    The adjoint operator performs a corresponding back-projection.
    """
    def __init__(self,
                 image_shape: tuple[int, int],  # (Ny, Nx)
                 sensor_positions: torch.Tensor, # Shape (num_sensors, 2) for 2D
                 sound_speed: float = 1500.0,    # m/s
                 time_samples: int = 256,        # Number of time samples for sensor data
                 pixel_size: float = 0.0001,     # meters (e.g., 0.1 mm)
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.Ny, self.Nx = self.image_shape
        self.sensor_positions = sensor_positions.to(torch.device(device)) # (num_sensors, 2)
        self.num_sensors = self.sensor_positions.shape[0]
        self.sound_speed = sound_speed
        self.time_samples = time_samples
        self.pixel_size = pixel_size # Physical size of a pixel
        self.device = torch.device(device)

        # Create image pixel coordinates
        img_y_coords = torch.arange(self.Ny, device=self.device) * self.pixel_size
        img_x_coords = torch.arange(self.Nx, device=self.device) * self.pixel_size
        img_grid_y, img_grid_x = torch.meshgrid(img_y_coords, img_x_coords, indexing='ij')
        self.pixel_coords = torch.stack((img_grid_y.flatten(), img_grid_x.flatten()), dim=1) # (Ny*Nx, 2)

        # Calculate distance matrix: (num_pixels, num_sensors)
        # self.pixel_coords: (N_pixels, 2) -> (N_pixels, 1, 2)
        # self.sensor_positions: (N_sensors, 2) -> (1, N_sensors, 2)
        dist_sq = torch.sum(
            (self.pixel_coords.unsqueeze(1) - self.sensor_positions.unsqueeze(0))**2,
            dim=2
        )
        self.distances = torch.sqrt(dist_sq) # (N_pixels, N_sensors)

        # Calculate time-of-flight matrix: (N_pixels, N_sensors)
        self.tof_matrix = self.distances / self.sound_speed

        # Determine max time for time vector based on max distance
        max_dist = torch.max(self.distances)
        max_time_needed = max_dist / self.sound_speed
        self.time_vector = torch.linspace(0, max_time_needed * 1.1, self.time_samples, device=self.device) # (time_samples,)
        self.dt = self.time_vector[1] - self.time_vector[0] if self.time_samples > 1 else max_time_needed


        print(f"PhotoacousticOperator (Time-of-Flight) initialized.")
        print(f"  Image: {self.Ny}x{self.Nx} pixels, Pixel size: {self.pixel_size*1000:.2f} mm")
        print(f"  Sensors: {self.num_sensors}, Sound speed: {self.sound_speed} m/s")
        print(f"  Time samples: {self.time_samples}, dt: {self.dt*1e6:.2f} us, Max time: {self.time_vector[-1]*1e3:.2f} ms")


    def op(self, initial_pressure_map: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Initial pressure map to sensor data (time series).
        y_s(t) = sum_r P0(r) * delta(t - |r - r_s|/c)
        (Simplified: sum contributions from pixels arriving at sensor 's' at time 't')
        Args:
            initial_pressure_map (torch.Tensor): The initial pressure distribution at t=0.
                                                 Shape: self.image_shape (Ny, Nx).
        Returns:
            torch.Tensor: Simulated time-series data at sensor locations.
                          Shape: (num_sensors, num_time_samples).
        """
        if initial_pressure_map.shape != self.image_shape:
            raise ValueError(f"Input map shape {initial_pressure_map.shape} must match {self.image_shape}.")
        initial_pressure_vector = initial_pressure_map.to(self.device).flatten() # (N_pixels,)

        sensor_data = torch.zeros((self.num_sensors, self.time_samples), device=self.device)

        # For each sensor
        for s_idx in range(self.num_sensors):
            # tof_s for this sensor: (N_pixels,)
            tof_s = self.tof_matrix[:, s_idx]

            # For each pixel, find which time bin its signal falls into for this sensor
            # time_bin_indices = torch.floor(tof_s / self.dt).long() # Simple binning

            # More robust: find nearest time sample
            # tof_s.unsqueeze(1) -> (N_pixels, 1)
            # self.time_vector.unsqueeze(0) -> (1, time_samples)
            # Find index of time_vector closest to each tof_s value
            time_diffs = torch.abs(tof_s.unsqueeze(1) - self.time_vector.unsqueeze(0))
            time_bin_indices = torch.argmin(time_diffs, dim=1) # (N_pixels,)

            # Accumulate pressure signals into the correct time bins
            # initial_pressure_vector is (N_pixels,)
            # Need to handle cases where time_bin_indices are out of bounds for sensor_data[s_idx,:]
            valid_bins_mask = (time_bin_indices >= 0) & (time_bin_indices < self.time_samples)

            # Use index_add_ for safe accumulation if multiple pixels map to same time bin (unlikely with float TOF)
            # sensor_data[s_idx, :].index_add_(0, time_bin_indices[valid_bins_mask], initial_pressure_vector[valid_bins_mask])
            # A simpler loop for clarity, though index_add is better for performance/correctness with exact binning
            for p_idx in range(self.pixel_coords.shape[0]):
                if valid_bins_mask[p_idx]:
                    bin_idx = time_bin_indices[p_idx]
                    sensor_data[s_idx, bin_idx] += initial_pressure_vector[p_idx]

        # Optional: convolve with a short pulse to make signals less spiky if dt is small
        # For now, this is a very basic "sum at arrival time" model.
        # A small amount of blurring can make it more robust to time discretization.
        # Example: if self.time_samples > 10:
        #    blur_kernel = torch.tensor([0.5, 1.0, 0.5], device=self.device).view(1,1,-1) / 2.0
        #    sensor_data = torch.nn.functional.conv1d(sensor_data.unsqueeze(1), blur_kernel, padding='same').squeeze(1)


        return sensor_data

    def op_adj(self, sensor_time_data: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Sensor time data to reconstructed initial pressure map.
        This is a simple back-projection: P_adj(r) = sum_s y_s(t = |r - r_s|/c)

        Args:
            sensor_time_data (torch.Tensor): Time-series data from sensors.
                                             Shape: (num_sensors, num_time_samples).
        Returns:
            torch.Tensor: Reconstructed initial pressure map.
                          Shape: self.image_shape.
        """
        if sensor_time_data.shape != (self.num_sensors, self.time_samples):
            raise ValueError(f"Input data shape {sensor_time_data.shape} incorrect.")
        sensor_time_data = sensor_time_data.to(self.device)

        reconstructed_map_vector = torch.zeros(self.pixel_coords.shape[0], device=self.device)

        # For each pixel
        for p_idx in range(self.pixel_coords.shape[0]):
            # tof_p for this pixel: (N_sensors,)
            tof_p = self.tof_matrix[p_idx, :]

            # Find nearest time sample index in sensor_time_data for each sensor's TOF
            time_diffs = torch.abs(tof_p.unsqueeze(1) - self.time_vector.unsqueeze(0))
            time_bin_indices = torch.argmin(time_diffs, dim=1) # (N_sensors,)

            # Accumulate the sensor reading from that time bin for all sensors
            # reconstructed_map_vector[p_idx] = torch.sum(sensor_time_data[torch.arange(self.num_sensors), time_bin_indices])
            # Loop for clarity:
            val = 0.0
            for s_idx in range(self.num_sensors):
                bin_idx = time_bin_indices[s_idx]
                if 0 <= bin_idx < self.time_samples: # Check bounds
                    val += sensor_time_data[s_idx, bin_idx]
            reconstructed_map_vector[p_idx] = val

        return reconstructed_map_vector.reshape(self.image_shape)

if __name__ == '__main__':
    print("\nRunning basic PhotoacousticOperator (Time-of-Flight) checks...")
    device_pat_op = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_s = (16, 16) # Small image for faster test Ny, Nx
    n_sensors = 8
    # Circular sensor geometry
    center_x, center_y = (img_s[1]-1)*0.0001/2, (img_s[0]-1)*0.0001/2 # Image center
    radius = max(img_s) * 0.0001 * 0.7 # Radius slightly larger than half image diagonal
    angles_sens = torch.linspace(0, 2 * np.pi, n_sensors, endpoint=False, device=device_pat_op)
    sensor_pos = torch.stack([
        center_x + radius * torch.cos(angles_sens),
        center_y + radius * torch.sin(angles_sens)
    ], dim=1)

    sound_s = 1500.0
    time_samps = 64
    pix_s = 0.0001 # 0.1 mm

    try:
        pat_op = PhotoacousticOperator(
            image_shape=img_s,
            sensor_positions=sensor_pos,
            sound_speed=sound_s,
            time_samples=time_samps,
            pixel_size=pix_s,
            device=device_pat_op
        )
        print("PhotoacousticOperator (Time-of-Flight) instantiated.")

        phantom_pressure = torch.zeros(img_s, device=device_pat_op)
        phantom_pressure[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 1.0 # Square source
        # phantom_pressure[img_s[0]//2, img_s[1]//2] = 1.0 # Point source

        sensor_data_sim = pat_op.op(phantom_pressure)
        print(f"Forward op output shape: {sensor_data_sim.shape}")
        assert sensor_data_sim.shape == (n_sensors, time_samps)

        recon_pressure_map = pat_op.op_adj(sensor_data_sim)
        print(f"Adjoint op output shape: {recon_pressure_map.shape}")
        assert recon_pressure_map.shape == img_s

        # Basic dot product test
        x_dp = torch.randn_like(phantom_pressure)
        y_dp_rand = torch.randn_like(sensor_data_sim)

        Ax = pat_op.op(x_dp)
        Aty = pat_op.op_adj(y_dp_rand)

        lhs = torch.dot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.dot(x_dp.flatten(), Aty.flatten())

        print(f"PAT Dot product test: LHS={lhs.item():.6f}, RHS={rhs.item():.6f}")
        # This test is sensitive to the discretization of time and space.
        # For this simple model, it should pass reasonably well.
        assert np.isclose(lhs.item(), rhs.item(), rtol=1e-3), "Dot product test failed for PAT TOF operator."

        print("PhotoacousticOperator (Time-of-Flight) __main__ checks completed.")

        # Optional: visualize if matplotlib is available
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1,3, figsize=(12,4))
        # axes[0].imshow(phantom_pressure.cpu().numpy()); axes[0].set_title("Phantom")
        # axes[1].imshow(sensor_data_sim.cpu().numpy(), aspect='auto'); axes[1].set_title("Sensor Data")
        # axes[2].imshow(recon_pressure_map.cpu().numpy()); axes[2].set_title("Adjoint Recon")
        # plt.show()

    except Exception as e:
        print(f"Error in PhotoacousticOperator (Time-of-Flight) __main__ checks: {e}")
        import traceback
        traceback.print_exc()
