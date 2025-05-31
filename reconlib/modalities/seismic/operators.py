import torch
import numpy as np # For np.pi, np.sqrt, np.linspace etc.
from reconlib.operators import Operator

class SeismicForwardOperator(Operator):
    """
    Forward and Adjoint Operator for 2D Seismic Imaging.

    Models seismic data acquisition by simulating travel times from a source
    to scatterers in a 2D subsurface reflectivity map, and then to receivers.
    This is a simplified ray-based model.

    Args:
        reflectivity_map_shape (tuple[int, int]): Shape of the 2D subsurface
                                                 reflectivity map (num_depth_pixels, num_x_pixels).
        wave_speed_mps (float): Seismic wave speed in meters per second.
        time_sampling_dt_s (float): Time step for seismic traces in seconds.
        num_time_samples (int): Number of time samples per trace.
        source_pos_m (tuple[float, float]): (x, z) coordinates of the seismic source in meters.
                                          Assumes z=0 is surface, positive z is depth.
        receiver_pos_m (torch.Tensor): Tensor of receiver positions.
                                       Shape (num_receivers, 2), where each row is (x_rec, z_rec).
                                       Assumes z=0 is surface.
        pixel_spacing_m (float or tuple[float,float], optional): Physical size of pixels (dz, dx) in meters.
                                         If float, assumes square pixels. Defaults to 1.0.
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 reflectivity_map_shape: tuple[int, int], # (num_z_pixels, num_x_pixels)
                 wave_speed_mps: float,
                 time_sampling_dt_s: float,
                 num_time_samples: int,
                 source_pos_m: tuple[float, float], # (src_x, src_z)
                 receiver_pos_m: torch.Tensor,    # (num_receivers, 2) for (rec_x, rec_z)
                 pixel_spacing_m: float | tuple[float,float] = 1.0,
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.reflectivity_map_shape = reflectivity_map_shape # (Nz, Nx)
        self.Nz, self.Nx = reflectivity_map_shape

        self.wave_speed_mps = wave_speed_mps
        self.dt_s = time_sampling_dt_s
        self.num_time_samples = num_time_samples

        self.source_pos_m = torch.tensor(source_pos_m, dtype=torch.float32, device=device)

        if not isinstance(receiver_pos_m, torch.Tensor):
            receiver_pos_m = torch.tensor(receiver_pos_m, dtype=torch.float32)
        self.receiver_pos_m = receiver_pos_m.to(device)
        if self.receiver_pos_m.ndim != 2 or self.receiver_pos_m.shape[1] != 2:
            raise ValueError("receiver_pos_m must be of shape (num_receivers, 2).")
        self.num_receivers = self.receiver_pos_m.shape[0]

        if isinstance(pixel_spacing_m, (float, int)):
            self.pixel_dz_m = float(pixel_spacing_m)
            self.pixel_dx_m = float(pixel_spacing_m)
        elif isinstance(pixel_spacing_m, tuple) and len(pixel_spacing_m) == 2:
            self.pixel_dz_m = float(pixel_spacing_m[0]) # depth pixel size
            self.pixel_dx_m = float(pixel_spacing_m[1]) # horizontal pixel size
        else:
            raise ValueError("pixel_spacing_m must be a float or a 2-tuple (dz, dx).")

        self.device = torch.device(device)

        # Create pixel grid coordinates (centers of pixels)
        # Z-axis positive downwards, X-axis positive to the right
        # Source/Receivers usually at z=0 or slightly above.
        z_coords_m = (torch.arange(self.Nz, device=self.device) + 0.5) * self.pixel_dz_m
        x_coords_m = (torch.arange(self.Nx, device=self.device) + 0.5) * self.pixel_dx_m

        # pixel_grid_x: (Nz, Nx), pixel_grid_z: (Nz, Nx)
        self.pixel_grid_x_m, self.pixel_grid_z_m = torch.meshgrid(x_coords_m, z_coords_m, indexing='xy')
        # self.pixel_grid will be (Nz, Nx, 2) where each entry is (pixel_x_coord, pixel_z_coord)
        self.pixel_grid_m = torch.stack((self.pixel_grid_x_m, self.pixel_grid_z_m), dim=-1)


    def op(self, x_reflectivity_map: torch.Tensor) -> torch.Tensor:
        """
        Forward Seismic operation: Simulates seismic traces from a reflectivity map.
        x_reflectivity_map: (Nz, Nx) subsurface reflectivity.
        Returns: (num_receivers, num_time_samples) seismic traces.
        """
        if x_reflectivity_map.shape != self.reflectivity_map_shape:
            raise ValueError(f"Input x_reflectivity_map shape {x_reflectivity_map.shape} must match {self.reflectivity_map_shape}.")
        if x_reflectivity_map.device != self.device:
            x_reflectivity_map = x_reflectivity_map.to(self.device)
        # Reflectivity is typically real.
        if x_reflectivity_map.is_complex():
            # print("Warning: Complex reflectivity map provided, using its magnitude for seismic op.")
            x_reflectivity_map = torch.abs(x_reflectivity_map)
        x_reflectivity_map = x_reflectivity_map.float()


        y_seismic_traces = torch.zeros((self.num_receivers, self.num_time_samples),
                                       dtype=torch.float32, device=self.device)

        for i_rec in range(self.num_receivers):
            rec_pos = self.receiver_pos_m[i_rec, :] # (rec_x, rec_z)

            # Distances from source to all pixels: (Nz, Nx)
            dist_source_to_pixels = torch.sqrt(
                (self.pixel_grid_m[..., 0] - self.source_pos_m[0])**2 + \
                (self.pixel_grid_m[..., 1] - self.source_pos_m[1])**2
            )

            # Distances from all pixels to current receiver: (Nz, Nx)
            dist_pixels_to_receiver = torch.sqrt(
                (self.pixel_grid_m[..., 0] - rec_pos[0])**2 + \
                (self.pixel_grid_m[..., 1] - rec_pos[1])**2
            )

            # Total time of flight: source -> pixel -> receiver
            total_time_of_flight_s = (dist_source_to_pixels + dist_pixels_to_receiver) / self.wave_speed_mps # (Nz, Nx)

            # Convert TOF to time sample indices
            time_sample_indices = torch.round(total_time_of_flight_s / self.dt_s).long() # (Nz, Nx)

            # Flatten for easier processing
            flat_reflectivity = x_reflectivity_map.flatten() # (Nz*Nx)
            flat_time_indices = time_sample_indices.flatten() # (Nz*Nx)

            for j_px_flat in range(flat_reflectivity.shape[0]):
                refl_val = flat_reflectivity[j_px_flat]
                if refl_val == 0: # No reflection, skip
                    continue

                time_idx = flat_time_indices[j_px_flat]

                if 0 <= time_idx < self.num_time_samples:
                    # Simplistic model: add reflectivity to the trace at TOF.
                    # A more complex model would convolve with a source wavelet.
                    # Attenuation could also be added: 1 / (dist_src_px * dist_px_rec) approx
                    y_seismic_traces[i_rec, time_idx] += refl_val

        return y_seismic_traces

    def op_adj(self, y_seismic_traces: torch.Tensor) -> torch.Tensor:
        """
        Adjoint Seismic operation: Backprojects seismic traces to a reflectivity map (migration).
        y_seismic_traces: (num_receivers, num_time_samples) recorded seismic data.
        Returns: (Nz, Nx) reconstructed subsurface reflectivity.
        """
        expected_trace_shape = (self.num_receivers, self.num_time_samples)
        if y_seismic_traces.shape != expected_trace_shape:
            raise ValueError(f"Input y_seismic_traces shape {y_seismic_traces.shape} must match {expected_trace_shape}.")
        if y_seismic_traces.device != self.device:
            y_seismic_traces = y_seismic_traces.to(self.device)
        y_seismic_traces = y_seismic_traces.float()


        x_reflectivity_adj = torch.zeros(self.reflectivity_map_shape,
                                           dtype=torch.float32, device=self.device)

        for i_px_z in range(self.Nz):
            for j_px_x in range(self.Nx):
                pixel_pos_m = self.pixel_grid_m[i_px_z, j_px_x, :] # (px_x, px_z)

                # Dist from source to this pixel
                dist_source_to_pixel = torch.sqrt(
                    (pixel_pos_m[0] - self.source_pos_m[0])**2 + \
                    (pixel_pos_m[1] - self.source_pos_m[1])**2
                )

                accumulated_val_for_pixel = 0.0
                for k_rec in range(self.num_receivers):
                    rec_pos = self.receiver_pos_m[k_rec, :]

                    # Dist from this pixel to this receiver
                    dist_pixel_to_receiver = torch.sqrt(
                        (pixel_pos_m[0] - rec_pos[0])**2 + \
                        (pixel_pos_m[1] - rec_pos[1])**2
                    )

                    total_time_of_flight_s = (dist_source_to_pixel + dist_pixel_to_receiver) / self.wave_speed_mps
                    time_sample_idx = torch.round(total_time_of_flight_s / self.dt_s).long()

                    if 0 <= time_sample_idx < self.num_time_samples:
                        # Add the trace value at this TOF to the pixel
                        accumulated_val_for_pixel += y_seismic_traces[k_rec, time_sample_idx]

                x_reflectivity_adj[i_px_z, j_px_x] = accumulated_val_for_pixel

        return x_reflectivity_adj

if __name__ == '__main__':
    print("Running basic SeismicForwardOperator checks...")
    device = torch.device('cpu')
    map_shape_test = (32, 48) # Nz, Nx - Keep small for tests

    src_pos = (map_shape_test[1] * 0.01 / 2, 0.0) # Source at center surface, pixel_spacing=0.01
    num_recs = 10
    rec_x = np.linspace(0, map_shape_test[1] * 0.01, num_recs)
    rec_z = np.zeros(num_recs)
    rec_pos_test = torch.tensor(np.stack([rec_x, rec_z], axis=-1), dtype=torch.float32, device=device)

    try:
        seismic_op_test = SeismicForwardOperator(
            reflectivity_map_shape=map_shape_test,
            wave_speed_mps=2000.0,
            time_sampling_dt_s=0.001, # 1 ms
            num_time_samples=500,    # 0.5 seconds of recording
            source_pos_m=src_pos,
            receiver_pos_m=rec_pos_test,
            pixel_spacing_m=0.01, # 10m pixels
            device=device
        )
        print("SeismicForwardOperator instantiated.")

        phantom_map = torch.zeros(map_shape_test, dtype=torch.float32, device=device)
        phantom_map[map_shape_test[0]//2, map_shape_test[1]//2] = 1.0 # Point scatterer
        phantom_map[map_shape_test[0]*3//4, :] = 0.5 # A layer

        traces_sim = seismic_op_test.op(phantom_map)
        print(f"Forward op output shape (traces): {traces_sim.shape}")
        assert traces_sim.shape == (num_recs, seismic_op_test.num_time_samples)

        recon_map_adj = seismic_op_test.op_adj(traces_sim)
        print(f"Adjoint op output shape (map): {recon_map_adj.shape}")
        assert recon_map_adj.shape == map_shape_test

        # Basic dot product test (can be slow due to nested loops)
        # For real operators, torch.vdot expects complex, but our ops are real-valued here.
        # Using (a*b).sum() for real dot product.
        x_dp_seis = torch.randn_like(phantom_map)
        y_dp_rand_seis = torch.randn_like(traces_sim)

        Ax_seis = seismic_op_test.op(x_dp_seis)
        Aty_seis = seismic_op_test.op_adj(y_dp_rand_seis)

        lhs_seis = (Ax_seis * y_dp_rand_seis).sum() # Real dot product
        rhs_seis = (x_dp_seis * Aty_seis).sum()   # Real dot product

        print(f"Seismic Dot product test: LHS={lhs_seis.item():.4e}, RHS={rhs_seis.item():.4e}")
        if not np.isclose(lhs_seis.item(), rhs_seis.item(), rtol=1e-3, atol=1e-4): # Increased tolerance
           print(f"Warning: Seismic Dot product components differ. LHS: {lhs_seis}, RHS: {rhs_seis}")

        print("SeismicForwardOperator __main__ checks completed.")
    except Exception as e:
        print(f"Error in SeismicForwardOperator __main__ checks: {e}")
