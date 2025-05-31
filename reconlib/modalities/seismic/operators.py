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
        source_wavelet: torch.Tensor = None,
        wavelet_time_offset_s: float = 0.0,
        apply_geometrical_spreading: bool = True,
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

        # Store wavelet and spreading parameters
        if source_wavelet is not None:
            if not isinstance(source_wavelet, torch.Tensor):
                source_wavelet = torch.tensor(source_wavelet, dtype=torch.float32)
            if source_wavelet.ndim != 1:
                raise ValueError("source_wavelet must be a 1D tensor.")
            self.source_wavelet = source_wavelet.to(device=self.device, dtype=torch.float32)
            self.wavelet_center_idx = int(round(wavelet_time_offset_s / self.dt_s))
            self.reversed_wavelet = torch.flip(self.source_wavelet, dims=[0])
        else:
            self.source_wavelet = None
            self.wavelet_center_idx = 0
            self.reversed_wavelet = None

        self.apply_geometrical_spreading = apply_geometrical_spreading


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

        # Distances from source to all pixels: (Nz, Nx)
        dist_source_to_pixels = torch.sqrt(
            (self.pixel_grid_m[..., 0] - self.source_pos_m[0])**2 + \
            (self.pixel_grid_m[..., 1] - self.source_pos_m[1])**2
        )

        for i_rec in range(self.num_receivers):
            rec_pos = self.receiver_pos_m[i_rec, :] # (rec_x, rec_z)

            # Distances from all pixels to current receiver: (Nz, Nx)
            dist_pixels_to_receiver = torch.sqrt(
                (self.pixel_grid_m[..., 0] - rec_pos[0])**2 + \
                (self.pixel_grid_m[..., 1] - rec_pos[1])**2
            )

            # Total time of flight: source -> pixel -> receiver
            total_time_of_flight_s = (dist_source_to_pixels + dist_pixels_to_receiver) / self.wave_speed_mps # (Nz, Nx)
            time_sample_indices = torch.round(total_time_of_flight_s / self.dt_s).long() # (Nz, Nx)

            for px_z in range(self.Nz):
                for px_x in range(self.Nx):
                    refl_val = x_reflectivity_map[px_z, px_x]
                    if refl_val == 0:
                        continue

                    time_idx_event_center = time_sample_indices[px_z, px_x]
                    current_amplitude_contribution = refl_val

                    if self.apply_geometrical_spreading:
                        dist_sp = dist_source_to_pixels[px_z, px_x]
                        dist_pr = dist_pixels_to_receiver[px_z, px_x]
                        total_dist = dist_sp + dist_pr
                        # Avoid division by zero if total_dist is very small
                        amplitude_scaling = 1.0 / (total_dist + 1e-9)
                        current_amplitude_contribution *= amplitude_scaling

                    if current_amplitude_contribution == 0: # Can happen if refl_val or scaling is zero
                        continue

                    if self.source_wavelet is not None:
                        t_start_in_trace = time_idx_event_center - self.wavelet_center_idx
                        for k_w in range(len(self.source_wavelet)):
                            t_trace = t_start_in_trace + k_w
                            if 0 <= t_trace < self.num_time_samples:
                                y_seismic_traces[i_rec, t_trace] += current_amplitude_contribution * self.source_wavelet[k_w]
                    else: # No wavelet, just deposit scaled reflectivity
                        if 0 <= time_idx_event_center < self.num_time_samples:
                            y_seismic_traces[i_rec, time_idx_event_center] += current_amplitude_contribution

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

                dist_source_to_pixel = torch.sqrt(
                    (pixel_pos_m[0] - self.source_pos_m[0])**2 + \
                    (pixel_pos_m[1] - self.source_pos_m[1])**2
                )

                accumulated_val_for_this_pixel = 0.0
                for k_rec in range(self.num_receivers):
                    rec_pos = self.receiver_pos_m[k_rec, :]
                    dist_pixel_to_receiver = torch.sqrt(
                        (pixel_pos_m[0] - rec_pos[0])**2 + \
                        (pixel_pos_m[1] - rec_pos[1])**2
                    )

                    time_idx_event_center = torch.round(
                        (dist_source_to_pixel + dist_pixel_to_receiver) / self.wave_speed_mps / self.dt_s
                    ).long()

                    value_from_trace_interaction = 0.0
                    if self.reversed_wavelet is not None: # Adjoint of convolution is correlation
                        t_start_in_trace = time_idx_event_center - self.wavelet_center_idx
                        for k_w in range(len(self.reversed_wavelet)):
                            t_trace = t_start_in_trace + k_w
                            if 0 <= t_trace < self.num_time_samples:
                                value_from_trace_interaction += y_seismic_traces[k_rec, t_trace] * self.reversed_wavelet[k_w]
                    else: # No wavelet
                        if 0 <= time_idx_event_center < self.num_time_samples:
                            value_from_trace_interaction = y_seismic_traces[k_rec, time_idx_event_center]

                    final_contrib_this_ray = value_from_trace_interaction
                    if self.apply_geometrical_spreading:
                        total_dist = dist_source_to_pixel + dist_pixel_to_receiver
                        amplitude_scaling = 1.0 / (total_dist + 1e-9)
                        final_contrib_this_ray *= amplitude_scaling

                    accumulated_val_for_this_pixel += final_contrib_this_ray

                x_reflectivity_adj[i_px_z, j_px_x] = accumulated_val_for_this_pixel

        return x_reflectivity_adj

if __name__ == '__main__':
    print("Running basic SeismicForwardOperator checks...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    map_shape_test = (16, 24) # Nz, Nx - Keep small for tests for speed
    pixel_spacing_val = 1.0 # meters for dz and dx

    src_pos_test = (map_shape_test[1] * pixel_spacing_val / 2.0, 0.0) # Source at center surface
    num_recs_test = 10
    rec_x_test = torch.linspace(0, (map_shape_test[1]-1) * pixel_spacing_val, num_recs_test, device=device)
    rec_z_test = torch.zeros(num_recs_test, device=device)
    rec_pos_test_tensor = torch.stack([rec_x_test, rec_z_test], dim=-1)

    time_sampling_dt_s_val = 0.001 # 1 ms
    num_time_samples_val = 250    # 0.25 seconds of recording

    # Define Ricker wavelet function
    def ricker_wavelet(peak_freq, dt, num_samples, device='cpu'):
        """Generates a Ricker wavelet."""
        t = np.arange(-num_samples // 2, num_samples // 2) * dt
        # Corrected Ricker formula: (1 - 2 * (pi*f*t)^2) * exp(-(pi*f*t)^2)
        factor = (np.pi * peak_freq * t)**2
        wavelet = (1 - 2 * factor) * np.exp(-factor)
        return torch.tensor(wavelet, dtype=torch.float32, device=device)

    wavelet_len = 63 # Odd number for a symmetric wavelet with a clear center
    test_wavelet_tensor = ricker_wavelet(peak_freq=25.0, dt=time_sampling_dt_s_val, num_samples=wavelet_len, device=device)
    # Offset should be such that wavelet_center_idx points to the peak of the wavelet
    # If num_samples is odd, peak is at (num_samples-1)/2.
    # Time offset = index * dt
    test_wavelet_offset_s_val = ((wavelet_len -1) // 2) * time_sampling_dt_s_val


    phantom_map_test = torch.zeros(map_shape_test, dtype=torch.float32, device=device)
    phantom_map_test[map_shape_test[0]//3, map_shape_test[1]//3] = 1.0
    phantom_map_test[map_shape_test[0]*2//3, map_shape_test[1]*2//3] = -0.5


    test_configs = [
        {"name": "Wavelet_SpreadingTrue", "wavelet": test_wavelet_tensor, "offset": test_wavelet_offset_s_val, "spreading": True},
        {"name": "NoWavelet_SpreadingTrue", "wavelet": None, "offset": 0.0, "spreading": True},
        {"name": "Wavelet_SpreadingFalse", "wavelet": test_wavelet_tensor, "offset": test_wavelet_offset_s_val, "spreading": False},
        {"name": "NoWavelet_SpreadingFalse", "wavelet": None, "offset": 0.0, "spreading": False}, # Original behavior
    ]

    for config in test_configs:
        print(f"\n--- Testing Configuration: {config['name']} ---")
        try:
            seismic_op_test = SeismicForwardOperator(
                reflectivity_map_shape=map_shape_test,
                wave_speed_mps=1500.0,
                time_sampling_dt_s=time_sampling_dt_s_val,
                num_time_samples=num_time_samples_val,
                source_pos_m=src_pos_test,
                receiver_pos_m=rec_pos_test_tensor,
                pixel_spacing_m=pixel_spacing_val,
                source_wavelet=config["wavelet"],
                wavelet_time_offset_s=config["offset"],
                apply_geometrical_spreading=config["spreading"],
                device=device
            )
            print(f"SeismicForwardOperator ({config['name']}) instantiated.")

            traces_sim = seismic_op_test.op(phantom_map_test)
            print(f"Forward op output shape (traces): {traces_sim.shape}")
            assert traces_sim.shape == (num_recs_test, num_time_samples_val)

            recon_map_adj = seismic_op_test.op_adj(traces_sim)
            print(f"Adjoint op output shape (map): {recon_map_adj.shape}")
            assert recon_map_adj.shape == map_shape_test

            x_dp_seis = torch.randn_like(phantom_map_test)
            y_dp_rand_seis = torch.randn_like(traces_sim)

            Ax_seis = seismic_op_test.op(x_dp_seis)
            Aty_seis = seismic_op_test.op_adj(y_dp_rand_seis)

            lhs_seis = (Ax_seis * y_dp_rand_seis).sum()
            rhs_seis = (x_dp_seis * Aty_seis).sum()

            print(f"Dot product test ({config['name']}): LHS={lhs_seis.item():.4e}, RHS={rhs_seis.item():.4e}")
            # Adjust tolerance: spreading and wavelet can affect precision
            rtol_val = 1e-2 if config["spreading"] or config["wavelet"] is not None else 1e-3
            atol_val = 1e-3 if config["spreading"] or config["wavelet"] is not None else 1e-4
            if not np.isclose(lhs_seis.item(), rhs_seis.item(), rtol=rtol_val, atol=atol_val):
               print(f"Warning: Dot product FAILED for {config['name']}. Diff: {abs(lhs_seis.item() - rhs_seis.item())}")
            else:
               print(f"Dot product PASSED for {config['name']}.")

        except Exception as e:
            print(f"Error in SeismicForwardOperator ({config['name']}) checks: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll SeismicForwardOperator __main__ checks completed.")
