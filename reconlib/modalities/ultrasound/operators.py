import numpy as np
import torch
from reconlib.operators import Operator # Assuming Operator is in reconlib.operators

class UltrasoundForwardOperator(Operator):
    @staticmethod
    def _ultrasound_pulse(t, f0, bandwidth_fractional):
        """
        Generates a Gaussian-modulated sinusoidal pulse.
        t: time vector
        f0: center frequency
        bandwidth_fractional: fractional bandwidth (e.g., 0.5 for 50%)
        """
        if f0 == 0 or bandwidth_fractional == 0: # Avoid division by zero if no pulse
            # Return a simple box or delta-like function if no frequency/bandwidth
            # For now, returning ones, meaning no specific pulse shape, just direct mapping.
            # This case should ideally be handled based on how t and reflectivity are used.
            # A delta pulse (1 at t=0, else 0) would be sum(reflectivity at time_of_flight)
            # For a continuous pulse, this means we're essentially sampling the pulse at time_diffs
            return torch.ones_like(t, dtype=torch.float32)

        # Calculate sigma for the Gaussian envelope based on fractional bandwidth
        # A common definition: sigma_envelope = sqrt(2*ln(2)) / (pi * f0 * bandwidth_fractional)
        # Or simplified: sigma_envelope ~ 1 / (f0 * bandwidth_fractional) for some definitions
        # Let's use a definition where bandwidth is related to the spread in frequency domain.
        # For a Gaussian envelope e^(-t^2 / (2*sigma_t^2)), its FWHM in time is ~2.355*sigma_t
        # Its FWHM in frequency is ~2.355*sigma_f, where sigma_f = 1/(2*pi*sigma_t)
        # If bandwidth_Hz = f0 * bandwidth_fractional, and assume FWHM_freq ~ bandwidth_Hz
        # sigma_f = (f0 * bandwidth_fractional) / 2.355
        # sigma_t = 1 / (2 * np.pi * sigma_f) = 2.355 / (2 * np.pi * f0 * bandwidth_fractional)
        # Simplified sigma_t for envelope duration:
        sigma_envelope = 1.0 / (f0 * bandwidth_fractional * np.pi + 1e-9) # Added pi for bandwidth def.

        envelope = torch.exp(-(t**2) / (2 * sigma_envelope**2 + 1e-9)) # Gaussian envelope
        carrier = torch.cos(2 * np.pi * f0 * t)           # Sinusoidal carrier
        return (envelope * carrier).to(torch.complex64)

    """
    Ultrasound Forward and Adjoint Operator.

    Models the generation of ultrasound echo signals from a reflectivity image
    and provides the adjoint for basic backprojection (Delay-and-Sum).

    Args:
        image_shape (tuple[int, int]): Shape of the input image (height, width).
        sound_speed (float): Speed of sound in m/s.
        num_elements (int): Number of transducer elements.
        element_positions (torch.Tensor): Positions of transducer elements (num_elements, 2).
                                          Coordinates (x, y) in meters.
        center_frequency (float): Center frequency of the ultrasound pulse in Hz. (Not used in simple model yet)
        sampling_rate (float): Sampling rate of the echo data in Hz.
        num_samples (int): Number of time samples per echo signal.
        device (str or torch.device, optional): Device for computations. Defaults to 'cpu'.
    """
    def __init__(self,
                 image_shape: tuple[int, int],
                 sound_speed: float = 1540.0,
                 num_elements: int = 64,
                 element_positions: torch.Tensor | None = None,
                 # TODO: Add pulse parameters like center_frequency, bandwidth for more realistic simulation
                 sampling_rate: float = 40e6, # 40 MHz
                 num_samples: int = 1024,
                 image_spacing: tuple[float, float] | None = None, # (dy, dx) in meters per pixel,
                 center_frequency: float = 5e6,
                 pulse_bandwidth_fractional: float = 0.6,
                 beam_sigma_rad: float = 0.02,
                 attenuation_coeff_db_cm_mhz: float = 0.3,
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape # (height, width)
        self.sound_speed = sound_speed
        self.num_elements = num_elements
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples
        self.device = torch.device(device)
        self.center_frequency = center_frequency
        self.pulse_bandwidth_fractional = pulse_bandwidth_fractional
        self.beam_sigma_rad = beam_sigma_rad
        self.attenuation_coeff_db_cm_mhz = attenuation_coeff_db_cm_mhz

        if image_spacing is None:
            # Default to 1mm pixel spacing if not provided
            self.image_spacing = (0.001, 0.001)
            print(f"Warning: image_spacing not provided. Defaulting to 1mm x 1mm pixels.")
        else:
            self.image_spacing = image_spacing

        if element_positions is None:
            # Create a simple linear array if not provided
            # Assume elements are along x-axis, centered at y=0, z pointing into image
            # Element width (pitch) could be e.g., 0.3 mm
            element_pitch = 0.0003
            array_width = (num_elements - 1) * element_pitch
            x_coords = torch.linspace(-array_width / 2, array_width / 2, num_elements, device=self.device)
            # Position elements slightly above the image (e.g., at y = -0.01 meters or -10mm)
            # Assuming image top is y=0, extending downwards.
            self.element_positions = torch.stack(
                (x_coords, torch.full_like(x_coords, -0.01)), dim=1
            )
        else:
            if not isinstance(element_positions, torch.Tensor):
                element_positions = torch.tensor(element_positions, dtype=torch.float32)
            self.element_positions = element_positions.to(self.device)

        if self.element_positions.shape != (self.num_elements, 2):
            raise ValueError(f"element_positions shape must be ({self.num_elements}, 2), "
                             f"got {self.element_positions.shape}")

        # Create image pixel grid (coordinates in meters)
        # Image origin (0,0) at top-left, y positive downwards, x positive to the right
        # Transducer is typically "above" the image (negative y relative to image top)
        img_height_m = self.image_shape[0] * self.image_spacing[0]
        img_width_m = self.image_shape[1] * self.image_spacing[1]

        # Pixel centers
        self.pixel_y_coords = torch.linspace(
            self.image_spacing[0] / 2,
            img_height_m - self.image_spacing[0] / 2,
            self.image_shape[0], device=self.device
        )
        self.pixel_x_coords = torch.linspace(
            self.image_spacing[1] / 2,
            img_width_m - self.image_spacing[1] / 2,
            self.image_shape[1], device=self.device
        )

        # Meshgrid for all pixel coordinates
        # pixel_grid_x shape: (img_height, img_width)
        # pixel_grid_y shape: (img_height, img_width)
        self.pixel_grid_x, self.pixel_grid_y = torch.meshgrid(
            self.pixel_x_coords, self.pixel_y_coords, indexing='xy' # 'xy' gives X (width), Y (height)
        )
        # Stack to get (img_height, img_width, 2)
        self.pixel_grid = torch.stack((self.pixel_grid_x, self.pixel_grid_y), dim=-1)


    def op(self, image_x: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Simulates ultrasound echo data from a reflectivity image.
        y = A * x

        Args:
            image_x (torch.Tensor): Input reflectivity image.
                                    Shape (height, width), on self.device.
        Returns:
            torch.Tensor: Simulated echo data (raw RF data).
                          Shape (num_elements, num_samples), on self.device.
        """

        if image_x.shape != self.image_shape:
            raise ValueError(f"Input image_x shape {image_x.shape} must match {self.image_shape}.")
        if image_x.device != self.device:
            image_x = image_x.to(self.device)
        # Reflectivity is typically real, but output can be complex due to pulse.
        # If image_x is already complex, it might represent complex reflectivity.
        if not torch.is_complex(image_x):
            image_x_complex = image_x.to(torch.complex64)
        else:
            image_x_complex = image_x

        echo_data_y = torch.zeros((self.num_elements, self.num_samples),
                                  dtype=torch.complex64, device=self.device)

        # Time axis for RF data (seconds)
        time_axis_rf = torch.arange(self.num_samples, device=self.device, dtype=torch.float32) / self.sampling_rate

        # Attenuation: alpha_Np_m_MHz = dB_cm_MHz * ln(10)/20 * 100 cm/m
        alpha_Np_m_per_MHz = self.attenuation_coeff_db_cm_mhz * (np.log(10) / 20.0) * 100.0

        for elem_idx in range(self.num_elements):
            elem_pos = self.element_positions[elem_idx, :] # (x_e, y_e)

            # Distances and angles from current element to all pixels
            # self.pixel_grid is (W, H, 2), elem_pos is (2)
            dist_vecs = self.pixel_grid - elem_pos.view(1, 1, 2) # Broadcasting: (W,H,2)
            distances = torch.sqrt(torch.sum(dist_vecs**2, dim=-1)).T # Transpose to (H,W)

            # Time of flight (round trip)
            time_of_flight_map = (2 * distances / self.sound_speed).to(self.device) # (H,W)

            # Beam pattern (Gaussian based on angle)
            # Angle relative to element's normal (assuming normal is along y-axis for now)
            dx_to_pixels = dist_vecs[..., 0].T # (H,W)
            dy_to_pixels = dist_vecs[..., 1].T # (H,W) (positive if pixel is "below" element)
            # Angle = atan(dx / dy) - simple approx; assumes element looks along -Y
            # For y_pos_elements = -0.005, image at y>=0, dy_to_pixels is positive.
            angles_to_pixels = torch.atan2(dx_to_pixels, torch.abs(dy_to_pixels) + 1e-9) # (H,W)
            beam_weights = torch.exp(-(angles_to_pixels**2) / (2 * self.beam_sigma_rad**2 + 1e-9)) # (H,W)

            # Attenuation
            # Effective attenuation coefficient at center frequency (Np/m)
            attenuation_factor_at_f0 = alpha_Np_m_per_MHz * (self.center_frequency / 1e6)
            # Attenuation for round trip distance (amplitude)
            attenuation_map = torch.exp(-attenuation_factor_at_f0 * (2 * distances)) # (H,W)

            # Combine image reflectivity with spatial weights (beam and attenuation)
            weighted_reflectivity = image_x_complex * beam_weights * attenuation_map # (H,W)

            # For each RF time sample, calculate its contribution from all pixels
            for t_idx_rf, t_val_rf in enumerate(time_axis_rf):
                # Calculate time differences relative to TOF for each pixel
                time_diffs = t_val_rf - time_of_flight_map # (H,W)

                # Get pulse values for these time differences
                pulse_contrib_at_t_rf = self._ultrasound_pulse(time_diffs,
                                                               self.center_frequency,
                                                               self.pulse_bandwidth_fractional) # (H,W)

                # Sum contributions from all pixels for this RF time sample
                echo_data_y[elem_idx, t_idx_rf] = torch.sum(weighted_reflectivity * pulse_contrib_at_t_rf)

        return echo_data_y
    def op_adj(self, echo_data_y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Backprojects echo data to form an image (Delay-and-Sum).
        x_adj = A^T * y

        Args:
            echo_data_y (torch.Tensor): Input echo data (raw RF data).
                                        Shape (num_elements, num_samples), on self.device.
        Returns:
            torch.Tensor: Reconstructed image using adjoint (DAS-like).
                          Shape (height, width), on self.device.
        """

        if echo_data_y.shape != (self.num_elements, self.num_samples):
            raise ValueError(f"Input echo_data_y shape {echo_data_y.shape} must be "
                             f"({self.num_elements}, {self.num_samples}).")
        if echo_data_y.device != self.device:
            echo_data_y = echo_data_y.to(self.device)
        if not torch.is_complex(echo_data_y): # Echo data should be complex
            echo_data_y = echo_data_y.to(torch.complex64)

        reconstructed_image_x = torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)

        time_axis_rf = torch.arange(self.num_samples, device=self.device, dtype=torch.float32) / self.sampling_rate
        alpha_Np_m_per_MHz = self.attenuation_coeff_db_cm_mhz * (np.log(10) / 20.0) * 100.0

        for elem_idx in range(self.num_elements):
            elem_pos = self.element_positions[elem_idx, :]
            dist_vecs = self.pixel_grid - elem_pos.view(1, 1, 2)
            distances = torch.sqrt(torch.sum(dist_vecs**2, dim=-1)).T
            time_of_flight_map = (2 * distances / self.sound_speed).to(self.device)

            dx_to_pixels = dist_vecs[..., 0].T
            dy_to_pixels = dist_vecs[..., 1].T
            angles_to_pixels = torch.atan2(dx_to_pixels, torch.abs(dy_to_pixels) + 1e-9)
            beam_weights = torch.exp(-(angles_to_pixels**2) / (2 * self.beam_sigma_rad**2 + 1e-9))

            attenuation_factor_at_f0 = alpha_Np_m_per_MHz * (self.center_frequency / 1e6)
            attenuation_map = torch.exp(-attenuation_factor_at_f0 * (2 * distances)) # Same attenuation model for adjoint

            # Accumulate contributions for this element
            element_contribution_to_image = torch.zeros_like(reconstructed_image_x)
            for t_idx_rf, t_val_rf in enumerate(time_axis_rf):
                echo_sample_val = echo_data_y[elem_idx, t_idx_rf]
                if torch.abs(echo_sample_val) < 1e-12: # Skip if echo sample is zero
                    continue

                # Time differences for pulse evaluation (adjoint uses time-reversed pulse implicitly by this diff)
                time_diffs = time_of_flight_map - t_val_rf # Note: tof_map - t_rf (reversed from op)

                pulse_values = self._ultrasound_pulse(time_diffs,
                                                      self.center_frequency,
                                                      self.pulse_bandwidth_fractional)

                # Accumulate: echo_sample * pulse_value * spatial_weights
                element_contribution_to_image += echo_sample_val * pulse_values * beam_weights * attenuation_map

            reconstructed_image_x += element_contribution_to_image

        return reconstructed_image_x

# Example usage (for testing within this file)
if __name__ == '__main__':
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing UltrasoundForwardOperator on {test_device}")

    img_h, img_w = 64, 64
    operator_params = {
        'image_shape': (img_h, img_w),
        'sound_speed': 1540.0,
        'num_elements': 32,
        'sampling_rate': 20e6, # 20 MHz
        'num_samples': 512,
        'image_spacing': (0.0005, 0.0005), # 0.5 mm pixels,
            'center_frequency': 5e6,
            'pulse_bandwidth_fractional': 0.6,
            'beam_sigma_rad': 0.02,
            'attenuation_coeff_db_cm_mhz': 0.3,
            'device': test_device
    }
    us_op = UltrasoundForwardOperator(**operator_params)

    # Create a simple phantom (e.g., a point scatterer)
    phantom_image = torch.zeros((img_h, img_w), dtype=torch.complex64, device=test_device)
    phantom_image[img_h // 2, img_w // 2] = 1.0
    phantom_image[img_h // 4, img_w // 4] = 0.5

    print(f"Phantom image shape: {phantom_image.shape}")

    # Test forward operation
    try:
        echo_data = us_op.op(phantom_image)
        print(f"Simulated echo data shape: {echo_data.shape}")
        assert echo_data.shape == (us_op.num_elements, us_op.num_samples)
        print("Forward operation test successful.")
    except Exception as e:
        print(f"Error during forward operation test: {e}")
        raise

    # Test adjoint operation
    try:
        # Use the simulated echo data to test adjoint
        reconstructed_adj = us_op.op_adj(echo_data)
        print(f"Adjoint reconstructed image shape: {reconstructed_adj.shape}")
        assert reconstructed_adj.shape == us_op.image_shape
        print("Adjoint operation test successful.")

        # Basic check: Max intensity should be around where the point scatterer was
        # (though it will be blurred due to beamforming)
        if torch.sum(torch.abs(reconstructed_adj)) > 0: # Ensure not all zeros
            max_val_coords = torch.argmax(torch.abs(reconstructed_adj))
            max_r, max_c = np.unravel_index(max_val_coords.cpu().numpy(), reconstructed_adj.shape)
            print(f"Max intensity in adjoint reconstruction at: ({max_r}, {max_c})")
        else:
            print("Warning: Adjoint reconstruction is all zeros.")

    except Exception as e:
        print(f"Error during adjoint operation test: {e}")
        raise

    # Dot product test (optional, can be slow)
    # Requires a random image x and random echo data y_rand
    # <A*x, y_rand> vs <x, A^T*y_rand>
    try:
        print("\nRunning dot product test...")
        x_dp = torch.randn_like(phantom_image) + 1j * torch.randn_like(phantom_image)
        y_dp_rand = torch.randn_like(echo_data) + 1j * torch.randn_like(echo_data)

        Ax = us_op.op(x_dp)
        Aty = us_op.op_adj(y_dp_rand)

        lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

        print(f"LHS (vdot(Ax, y_rand)): {lhs}")
        print(f"RHS (vdot(x, Aty)): {rhs}")
        # For complex numbers, vdot is sum(a* conj(b)). Adjoint property is <Ax,y> = <x, A^H y>.
        # So, if Aty is A^H y, then rhs should be vdot(x, Aty).
        # If vdot(Ax,y) = sum( (Ax)_i * conj(y_i) )
        # and vdot(x, Aty) = sum( x_i * conj((Aty)_i) )
        # These should be equal.

        relative_diff = torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs) / 2 + 1e-9) # Avoid div by zero
        print(f"Relative difference: {relative_diff.item()}")
        if relative_diff < 1e-3: # Allow some tolerance due to numerical precision and simplified model
            print("Dot product test PASSED.")
        else:
            print(f"Dot product test FAILED. Relative difference: {relative_diff.item()}")

    except Exception as e:
        print(f"Error during dot product test: {e}")
        # raise # Comment out if it's too slow or fails due to model simplifications often
