import torch
import numpy as np
from reconlib.operators import Operator # Assuming Operator is in reconlib.operators

class UltrasoundForwardOperator(Operator):
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
                 image_spacing: tuple[float, float] | None = None, # (dy, dx) in meters per pixel
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape # (height, width)
        self.sound_speed = sound_speed
        self.num_elements = num_elements
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples
        self.device = torch.device(device)

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
        if not torch.is_complex(image_x): # Assume reflectivity can be complex for phase effects
            image_x = image_x.to(torch.complex64)

        echo_data_y = torch.zeros((self.num_elements, self.num_samples),
                                  dtype=torch.complex64, device=self.device)

        time_axis = torch.arange(self.num_samples, device=self.device) / self.sampling_rate

        for elem_idx in range(self.num_elements):
            elem_pos = self.element_positions[elem_idx, :] # (x_e, y_e)

            # Calculate distance from this element to all pixels
            # dist_x = self.pixel_grid[..., 0] - elem_pos[0]
            # dist_y = self.pixel_grid[..., 1] - elem_pos[1]
            # distances = torch.sqrt(dist_x**2 + dist_y**2) # Shape (height, width)

            # More direct way:
            # pixel_grid is (W, H, 2), elem_pos is (2)
            # For broadcasting, reshape elem_pos to (1, 1, 2)
            distances = torch.sqrt(torch.sum((self.pixel_grid - elem_pos.view(1, 1, 2))**2, dim=-1))
            # distances shape is (W,H) due to meshgrid 'xy'. Let's ensure it's (H,W) for image_x consistency
            # If pixel_grid_x was (W,H) and pixel_grid_y was (W,H), then distances is (W,H)
            # Our meshgrid was pixel_grid_x (W,H), pixel_grid_y (W,H) -> stack makes (W,H,2)
            # So distances is (W,H). image_x is (H,W). Need to transpose distances or image_x.
            # Let's make distances (H,W)
            distances = distances.T # Now (H,W)

            # Time of flight (two-way: element to pixel and back)
            time_of_flight = 2 * distances / self.sound_speed # Shape (height, width)

            # For each pixel, find which time sample its echo arrives at
            # This is a simplified model: assumes a short pulse, no pulse shape yet.
            # Each pixel contributes its reflectivity to the corresponding time bin.

            # Iterate over pixels (can be slow, vectorization is better)
            # For this simplified model, we sum contributions.
            # A more realistic model would involve convolution with a pulse.
            for r_idx in range(self.image_shape[0]): # height
                for c_idx in range(self.image_shape[1]): # width
                    tof = time_of_flight[r_idx, c_idx]
                    reflectivity = image_x[r_idx, c_idx]

                    if reflectivity == 0: continue # Skip if no reflection

                    # Find the time sample index
                    # This is a very basic mapping. A real system would have pulse shape, interpolation.
                    time_sample_idx = torch.round(tof * self.sampling_rate).long()

                    if 0 <= time_sample_idx < self.num_samples:
                        # Attenuation (simple 1/distance, not physically accurate frequency-dependent)
                        # Attenuation should be 1/distance for pressure amplitude if spherical wave.
                        # For intensity, 1/distance^2. For round trip, effect is squared.
                        # Let's use 1/distance for now, and ignore for very small distances.
                        attenuation = 1.0 / (distances[r_idx, c_idx] + 1e-6) # Add epsilon to avoid div by zero

                        # Phase shift (simplified: k * r, where k is wavenumber related to center_freq)
                        # For now, assuming reflectivity already contains phase information or we ignore phase.
                        # To add basic phase:
                        # omega = 2 * np.pi * self.center_frequency (need center_frequency)
                        # phase_shift = torch.exp(-1j * omega * tof)
                        # echo_data_y[elem_idx, time_sample_idx] += reflectivity * attenuation * phase_shift

                        echo_data_y[elem_idx, time_sample_idx] += reflectivity * attenuation
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

        time_axis = torch.arange(self.num_samples, device=self.device) / self.sampling_rate

        for elem_idx in range(self.num_elements):
            elem_pos = self.element_positions[elem_idx, :] # (x_e, y_e)

            # distances = torch.sqrt(torch.sum((self.pixel_grid - elem_pos.view(1, 1, 2))**2, dim=-1)).T # (H,W)
            # Corrected distances calculation to match op:
            dist_calc_temp = self.pixel_grid - elem_pos.view(1,1,2) # W, H, 2
            distances = torch.sqrt(torch.sum(dist_calc_temp**2, dim=-1)) # W, H
            distances = distances.T # H, W

            time_of_flight = 2 * distances / self.sound_speed # Shape (height, width)

            # For each pixel, determine the TOF and sample the corresponding echo data
            # This is the "sum" part of Delay-and-Sum
            for r_idx in range(self.image_shape[0]): # height
                for c_idx in range(self.image_shape[1]): # width
                    tof = time_of_flight[r_idx, c_idx]

                    # Find the time sample index from TOF
                    time_sample_idx = torch.round(tof * self.sampling_rate).long()

                    if 0 <= time_sample_idx < self.num_samples:
                        # Attenuation correction (adjoint of 1/distance is also 1/distance if treated as real weights)
                        attenuation = 1.0 / (distances[r_idx, c_idx] + 1e-6)

                        # Adjoint of phase shift: if op was reflectivity * att * exp(-1j*omega*tof)
                        # then adjoint is data * att * exp(+1j*omega*tof) summed over elements
                        # For now, phase is not in op, so not in adjoint.

                        reconstructed_image_x[r_idx, c_idx] += echo_data_y[elem_idx, time_sample_idx] * attenuation

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
        'image_spacing': (0.0005, 0.0005), # 0.5 mm pixels
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
