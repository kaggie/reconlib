"""Module for PET/CT simulation utilities, including phantom generation and projection simulation."""

import torch
import numpy as np
from typing import Union, Optional, Tuple, Any, Dict

from reconlib.geometry import ScannerGeometry, SystemMatrix
from reconlib.projectors import ForwardProjector

class PhantomGenerator:
    """
    Generates various types of digital phantoms for PET and CT simulation.
    """
    def __init__(self, device: str = 'cpu'):
        """
        Initializes a phantom generator.

        Args:
            device (str, optional): The computational device ('cpu' or 'cuda')
                                    where the phantom tensor will be created. Defaults to 'cpu'.
        """
        self.device = device

    def generate(self,
                 size: Tuple[int, ...],
                 phantom_type: str = 'shepp-logan-pet',
                 **kwargs: Any) -> torch.Tensor:
        """
        Generates a digital phantom (e.g., Shepp-Logan for PET/CT).

        The generated phantom will be a PyTorch tensor on the device specified during initialization.

        Args:
            size (Tuple[int, ...]): The dimensions of the phantom.
                                    Can be (H, W) for 2D or (D, H, W) for 3D.
            phantom_type (str, optional): Type of phantom to generate.
                                          Examples: 'shepp-logan-pet', 'shepp-logan-ct',
                                          'uniform-cylinder', 'spheres', 'custom'.
                                          Defaults to 'shepp-logan-pet'.
            **kwargs (Any): Additional keyword arguments specific to the phantom type.
                            For 'shepp-logan-*', this might include 'modified' (bool).
                            For 'uniform-cylinder', 'radius', 'height', 'value'.
                            For 'spheres', 'sphere_definitions' (list of dicts with 'center', 'radius', 'value').

        Returns:
            torch.Tensor: A tensor representing the generated phantom.

        Raises:
            NotImplementedError: This method is a placeholder and needs to be implemented
                                 for specific phantom types.
            ValueError: If an unsupported phantom_type is requested or size is invalid.
        """
        # Example for size validation:
        if not (len(size) == 2 or len(size) == 3):
            raise ValueError("Phantom size must be 2D (H, W) or 3D (D, H, W).")

        print(f"Placeholder: Would generate a '{phantom_type}' phantom of size {size} on device '{self.device}'.")
        print(f"Additional parameters received: {kwargs}")

        # Comments on different phantom types:
        # - 'shepp-logan-pet': Standard Shepp-Logan, but ellipse values represent PET activity (e.g., higher in "tumors").
        # - 'shepp-logan-ct': Standard Shepp-Logan, ellipse values represent CT attenuation coefficients (Hounsfield Units or 1/cm).
        #   Often, a modified Shepp-Logan (e.g., with different ellipse parameters or values) is used.
        # - 'uniform-cylinder': A simple cylinder with a uniform value.
        # - 'spheres': Multiple spheres at specified locations with given radii and values.
        # - 'custom': Could load from a file or use a user-defined function passed via kwargs.

        # The actual implementation would involve creating a grid of coordinates,
        # defining shapes (ellipses, cylinders, spheres), and assigning values based on whether
        # coordinates fall within those shapes. Libraries like `skimage.data.shepp_logan_phantom`
        # (NumPy based) or custom PyTorch implementations would be used here.

        if phantom_type.lower() == 'circles_pet':
            if len(size) == 2:
                H, W = size
                phantom = torch.zeros(size, dtype=torch.float32, device=self.device)
                # Add a large circle
                y_coords, x_coords = torch.ogrid[-H//2:H//2, -W//2:W//2] # Create open grids
                # Ensure y_coords and x_coords are on the correct device
                y_coords = y_coords.to(self.device)
                x_coords = x_coords.to(self.device)

                mask1 = x_coords*x_coords + y_coords*y_coords <= (min(H,W)//3)**2
                phantom[mask1] = 1.0
                # Add a smaller circle with different intensity
                mask2 = (x_coords-W//8)*(x_coords-W//8) + (y_coords-H//8)*(y_coords-H//8) <= (min(H,W)//6)**2
                phantom[mask2] = 0.5
                # Add batch and channel dimension: (B, C, H, W)
                return phantom.unsqueeze(0).unsqueeze(0) 
            else: # 3D
                raise NotImplementedError(f"3D phantom generation for type '{phantom_type}' is not yet implemented.")
        elif phantom_type.lower() == 'shepp-logan-pet': # Minimal 2D Shepp-Logan for PET
            if len(size) == 2:
                # This is a very simplified Shepp-Logan, actual implementation would use ellipse parameters
                # For demonstration, using a similar structure to circles_pet
                H, W = size
                phantom = torch.zeros(size, dtype=torch.float32, device=self.device)
                y_coords, x_coords = torch.ogrid[-H//2:H//2, -W//2:W//2]
                y_coords = y_coords.to(self.device)
                x_coords = x_coords.to(self.device)

                # Background ellipse (representing soft tissue)
                bg_ellipse = ((x_coords / (W*0.45))**2 + (y_coords / (H*0.4))**2) <= 1
                phantom[bg_ellipse] = 0.2 # Low activity

                # "Tumor" 1 (higher activity)
                tumor1_ellipse = (((x_coords - W*0.1) / (W*0.1))**2 + ((y_coords + H*0.05) / (H*0.15))**2) <= 1
                phantom[tumor1_ellipse] = 1.0

                # "Tumor" 2 (medium activity)
                tumor2_ellipse = (((x_coords + W*0.15) / (W*0.12))**2 + ((y_coords - H*0.1) / (H*0.08))**2) <= 1
                phantom[tumor2_ellipse] = 0.75
                
                # Add batch and channel dimension: (B, C, H, W)
                return phantom.unsqueeze(0).unsqueeze(0)
            else: # 3D
                raise NotImplementedError(f"3D phantom generation for type '{phantom_type}' is not yet implemented.")
        else:
            raise NotImplementedError(f"Phantom generation for type '{phantom_type}' is not yet implemented.")


def simulate_projection_data(phantom: torch.Tensor,
                             projector: Union[ForwardProjector, SystemMatrix],
                             noise_model: Optional[str] = None,
                             **noise_params: Any) -> torch.Tensor:
    """
    Simulates projection data from a phantom using a given forward projector or system matrix.
    Optionally adds noise to the simulated data.

    Args:
        phantom (torch.Tensor): The input phantom image (e.g., activity map for PET,
                                attenuation map for CT). Expected to be on the same device
                                as the projector or will be moved.
        projector (Union[ForwardProjector, SystemMatrix]): An instantiated ForwardProjector
                                                           or SystemMatrix object.
        noise_model (Optional[str], optional): Type of noise to add.
                                               Examples: 'poisson', 'gaussian'.
                                               If None, no noise is added. Defaults to None.
        **noise_params (Any): Keyword arguments for the noise model.
                              For 'poisson': `sensitivity` (overall scaling factor for counts).
                              For 'gaussian': `mean`, `std`.

    Returns:
        torch.Tensor: The simulated projection data (e.g., sinogram).

    Raises:
        NotImplementedError: This method is a placeholder for the core projection and noise addition.
        ValueError: If an unsupported projector type or noise model is provided.
    """
    if not isinstance(phantom, torch.Tensor):
        raise ValueError("Phantom must be a PyTorch tensor.")

    # Ensure phantom is on the same device as the projector's components if possible
    projector_device = None
    if hasattr(projector, 'device'):
        projector_device = projector.device
    elif isinstance(projector, SystemMatrix) and hasattr(projector.projector_op, 'device'):
        projector_device = projector.projector_op.device
    
    if projector_device and phantom.device.type != projector_device:
        print(f"Moving phantom from {phantom.device} to {projector_device} to match projector.")
        phantom = phantom.to(projector_device)

    print(f"Placeholder: Simulating projection data for phantom of shape {phantom.shape}.")

    # 1. Perform forward projection
    # The actual call depends on whether 'projector' is a ForwardProjector or SystemMatrix
    # ForwardProjector has a 'project' method.
    # SystemMatrix has 'forward_project' (or 'op' which calls it).
    # Assuming the intent is to get the "forward view" of the phantom.
    if isinstance(projector, ForwardProjector):
        # ideal_projection_data = projector.project(phantom)
        print("Projector is ForwardProjector. Would call projector.project(phantom).")
    elif isinstance(projector, SystemMatrix):
        # ideal_projection_data = projector.forward_project(phantom)
        print("Projector is SystemMatrix. Would call projector.forward_project(phantom).")
    else:
        raise ValueError(f"Unsupported projector type: {type(projector)}. "
                         "Must be ForwardProjector or SystemMatrix.")

    # This is where the actual projection would happen:
    # Example:
    # if isinstance(projector, ForwardProjector):
    #     ideal_projection_data = projector.project(phantom)
    # elif isinstance(projector, SystemMatrix):
    #     ideal_projection_data = projector.forward_project(phantom) # or projector.op(phantom)
    # else: ...

    # Ensure phantom is on the correct device
    projector_device_to_check = None
    if hasattr(projector, 'device'): # SystemMatrix, ForwardProjector directly
        projector_device_to_check = projector.device
    elif hasattr(projector, 'system_matrix') and hasattr(projector.system_matrix, 'device'): # ForwardProjector wraps SystemMatrix
        projector_device_to_check = projector.system_matrix.device
    
    if projector_device_to_check and phantom.device != projector_device_to_check:
        phantom = phantom.to(projector_device_to_check)

    if isinstance(projector, SystemMatrix):
        projection_data = projector.op(phantom)
    elif isinstance(projector, ForwardProjector):
        projection_data = projector.project(phantom) # project method of ForwardProjector
    else:
        raise TypeError("projector must be an instance of SystemMatrix or ForwardProjector")

    # 2. Add noise (if specified)
    if noise_model:
        if noise_model.lower() == 'poisson':
            intensity_scale = noise_params.get('intensity_scale', 1000.0)
            # Ensure projection_data is non-negative before scaling for Poisson noise
            # And scale it to simulate counts
            max_abs_val = torch.max(torch.abs(projection_data))
            if max_abs_val == 0: max_abs_val = torch.tensor(1.0) # Avoid division by zero if projection_data is all zero

            scaled_projections = torch.relu(projection_data / max_abs_val * intensity_scale)
            
            noisy_projections = torch.poisson(scaled_projections)
            # Scale back to original data range
            noisy_projections = noisy_projections / intensity_scale * max_abs_val
            return noisy_projections
        elif noise_model.lower() == 'gaussian':
            sigma = noise_params.get('sigma', 0.1)
            # Scale sigma relative to data magnitude if desired, or use absolute sigma
            # For simplicity, using sigma as a fraction of max intensity if not directly interpretable
            # actual_sigma = sigma * torch.max(torch.abs(projection_data)) if torch.max(torch.abs(projection_data)) > 0 else sigma
            # Or, assume sigma is absolute. Let's assume sigma is absolute for now.
            noise = torch.randn_like(projection_data) * sigma
            noisy_projections = projection_data + noise
            return noisy_projections
        else:
            raise ValueError(f"Unsupported noise model: {noise_model}. Choose 'poisson' or 'gaussian'.")
    else: # No noise model
        return projection_data

# Example Usage (commented out, requires actual implementations)
# if __name__ == '__main__':
#     # 1. Setup Scanner Geometry (example)
#     angles_pet = np.linspace(0, np.pi, 180, endpoint=False)
#     n_det_pixels_pet = 128
#     pet_scanner_geom = ScannerGeometry(
#         detector_positions=np.zeros((n_det_pixels_pet, 2)), # Dummy
#         angles=angles_pet,
#         detector_size=np.array([4.0, 4.0]),
#         geometry_type='cylindrical_pet',
#         n_detector_pixels=n_det_pixels_pet
#     )
#     img_size_2d = (128, 128)
#
#     # 2. Create a Phantom
#     phantom_gen = PhantomGenerator(device='cpu')
#     try:
#         pet_phantom = phantom_gen.generate(size=img_size_2d, phantom_type='shepp-logan-pet')
#     except NotImplementedError as e:
#         print(e)
#         pet_phantom = torch.rand(1, 1, *img_size_2d) # Dummy if not implemented
#
#     # 3. Setup a Forward Projector
#     # Using SystemMatrix directly as projector for this example
#     # system_matrix_pet = SystemMatrix(scanner_geometry=pet_scanner_geom, img_size=img_size_2d)
#     # Using ForwardProjector class
#     forward_proj = ForwardProjector(scanner_geometry=pet_scanner_geom, img_size=img_size_2d)
#
#
#     # 4. Simulate Projection Data
#     try:
#         # projection_data_ideal = simulate_projection_data(pet_phantom, system_matrix_pet)
#         projection_data_ideal = simulate_projection_data(pet_phantom, forward_proj)
#         print(f"Ideal projection data shape: {projection_data_ideal.shape}")
#
#         # projection_data_noisy = simulate_projection_data(pet_phantom, system_matrix_pet,
#         #                                                  noise_model='poisson', sensitivity=1000.0)
#         # print(f"Noisy projection data shape: {projection_data_noisy.shape}")
#
#     except NotImplementedError as e:
#         print(e)
#
#     # Further steps could include plotting the phantom and projections
#     # from reconlib.plotting import visualize_reconstruction, plot_projection_data
#     # if pet_phantom.ndim == 4 and pet_phantom.shape[0]==1 and pet_phantom.shape[1]==1: # B,C,H,W
#     #      visualize_reconstruction(pet_phantom.squeeze().cpu().numpy(), main_title="PET Phantom")
#     # else:
#     #      visualize_reconstruction(pet_phantom.cpu().numpy(), main_title="PET Phantom") # if 2D/3D directly
#
#     # if 'projection_data_ideal' in locals() and projection_data_ideal is not None:
#     #     if projection_data_ideal.ndim == 4: # B,C,NumAngles,NumDet
#     #          plot_projection_data(projection_data_ideal.squeeze().cpu().numpy(), title="Ideal Sinogram")
#     #     elif projection_data_ideal.ndim == 2: # NumAngles,NumDet
#     #          plot_projection_data(projection_data_ideal.cpu().numpy(), title="Ideal Sinogram")
