import torch
import torch.nn as nn # Generally good to have, Operator might use it.
from typing import Tuple, Any

# Attempt to import actual base classes and types, fall back to placeholders if not found
try:
    from reconlib.operators import Operator, NUFFTOperator
except ImportError:
    print("Warning: reconlib.operators.Operator or reconlib.operators.NUFFTOperator not found. Using placeholders.")
    class Operator: # Placeholder
        def __init__(self):
            self.device = torch.device('cpu') # Mock device attribute
        def op(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        def op_adj(self, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
    
    NUFFTOperator = Any # Placeholder type for type hinting

class MultiCoilNUFFTOperator(Operator):
    """
    A multi-coil NUFFT operator that wraps a single-coil NUFFT operator.
    It applies the single-coil NUFFT operation independently to each coil.
    """
    def __init__(self, single_coil_nufft_op: NUFFTOperator):
        """
        Args:
            single_coil_nufft_op: An instance of a single-coil NUFFT operator.
                                  This operator should implement .op(), .op_adj(),
                                  and have .device and .image_shape attributes.
        """
        super().__init__()
        self.single_coil_nufft_op = single_coil_nufft_op
        
        if not hasattr(single_coil_nufft_op, 'device'):
            raise AttributeError("single_coil_nufft_op must have a 'device' attribute.")
        self.device = single_coil_nufft_op.device
        
        if not hasattr(single_coil_nufft_op, 'image_shape'):
            raise AttributeError("single_coil_nufft_op must have an 'image_shape' attribute.")
        self.image_shape = single_coil_nufft_op.image_shape # Shape of a single coil image (e.g., H, W or D, H, W)

        if hasattr(single_coil_nufft_op, 'k_trajectory'):
            self.k_trajectory = single_coil_nufft_op.k_trajectory
        else:
            # k_trajectory is often essential for understanding the NUFFT setup,
            # but not strictly required for the op/op_adj calls if single_coil_nufft_op handles it internally.
            # For consistency with other operators, it's good to expose it if available.
            # If single_coil_nufft_op doesn't expose it, this multi-coil wrapper cannot.
            # Depending on NUFFTOperator definition, this might be None or raise error.
            self.k_trajectory = None 
            # print("Warning: single_coil_nufft_op does not have a 'k_trajectory' attribute. It will not be exposed.")


    def op(self, multi_coil_image_data: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward NUFFT operation to each coil image.
        Args:
            multi_coil_image_data: A tensor of multi-coil image data.
                                   Shape: (num_coils, *image_shape), e.g., (C, H, W) or (C, D, H, W).
        Returns:
            A tensor of multi-coil k-space data.
            Shape: (num_coils, num_kpoints).
        """
        # Expected shape: (num_coils, *image_shape)
        expected_ndim = len(self.image_shape) + 1
        if multi_coil_image_data.ndim != expected_ndim:
            # If it's a single image matching image_shape, add a coil dimension
            if multi_coil_image_data.ndim == len(self.image_shape) and multi_coil_image_data.shape == self.image_shape:
                multi_coil_image_data = multi_coil_image_data.unsqueeze(0) # Add dummy coil dim
            else:
                raise ValueError(
                    f"Input multi_coil_image_data has incorrect ndim ({multi_coil_image_data.ndim}). "
                    f"Expected {expected_ndim} (for shape (num_coils, *image_shape)) "
                    f"or {len(self.image_shape)} (for shape (*image_shape) to be unsqueezed)."
                )

        # After potential unsqueeze, check spatial dimensions
        if multi_coil_image_data.shape[1:] != self.image_shape:
            raise ValueError(f"Image dimensions of input data {multi_coil_image_data.shape[1:]} "
                             f"do not match operator's image_shape {self.image_shape}.")

        multi_coil_image_data = multi_coil_image_data.to(self.device)
        
        output_kspace_list = []
        for i in range(multi_coil_image_data.shape[0]): # Iterate over coils
            single_coil_image = multi_coil_image_data[i]
            single_coil_kspace = self.single_coil_nufft_op.op(single_coil_image)
            output_kspace_list.append(single_coil_kspace)
        
        return torch.stack(output_kspace_list, dim=0)

    def op_adj(self, multi_coil_kspace_data: torch.Tensor) -> torch.Tensor:
        """
        Applies the adjoint NUFFT operation to each coil's k-space data.
        Args:
            multi_coil_kspace_data: A tensor of multi-coil k-space data.
                                    Shape: (num_coils, num_kpoints).
        Returns:
            A tensor of multi-coil image data.
            Shape: (num_coils, *image_shape).
        """
        # Example k-space shape: (num_coils, num_kpoints)
        # single_coil_kspace_data shape for single_coil_nufft_op.op_adj would be (num_kpoints,)
        if multi_coil_kspace_data.ndim < 2:
            if multi_coil_kspace_data.ndim == 1 : # Allow single coil k-space (K,)
                multi_coil_kspace_data = multi_coil_kspace_data.unsqueeze(0)
            else:
                raise ValueError(f"Input multi_coil_kspace_data must have shape (num_coils, num_kpoints). "
                                 f"Got {multi_coil_kspace_data.shape}.")
        
        # Cannot easily assert shape[1] (num_kpoints) without knowing it from single_coil_nufft_op
        # if self.k_trajectory is not None and multi_coil_kspace_data.shape[1] != self.k_trajectory.shape[0]:
        #     raise ValueError(f"Number of k-points in input data {multi_coil_kspace_data.shape[1]} "
        #                      f"does not match operator's k_trajectory length {self.k_trajectory.shape[0]}.")


        multi_coil_kspace_data = multi_coil_kspace_data.to(self.device)
        
        output_image_list = []
        for i in range(multi_coil_kspace_data.shape[0]): # Iterate over coils
            single_coil_kspace = multi_coil_kspace_data[i]
            single_coil_image = self.single_coil_nufft_op.op_adj(single_coil_kspace)
            output_image_list.append(single_coil_image)
            
        return torch.stack(output_image_list, dim=0)

if __name__ == '__main__':
    print("--- Testing MultiCoilNUFFTOperator ---")
    
    # Use the placeholder Operator if the real one wasn't imported
    BaseOpForMock = Operator 
    
    class MockSingleCoilNUFFTOp(BaseOpForMock):
        def __init__(self, image_shape: Tuple[int, ...], k_points_count: int, device: torch.device):
            super().__init__() # Call super if Operator has an __init__
            self.image_shape = image_shape
            self.k_points_count = k_points_count 
            self.device = device
            # Dummy trajectory, shape (num_kpoints, num_dims_image_space)
            self.k_trajectory = torch.zeros((k_points_count, len(image_shape)), device=device) 

        def op(self, image_data: torch.Tensor) -> torch.Tensor:
            assert image_data.shape == self.image_shape, \
                f"Mock op shape mismatch: expected {self.image_shape}, got {image_data.shape}"
            assert image_data.device == self.device
            return torch.randn(self.k_points_count, dtype=torch.complex64, device=self.device)

        def op_adj(self, kspace_data: torch.Tensor) -> torch.Tensor:
            assert kspace_data.shape == (self.k_points_count,), \
                f"Mock op_adj shape mismatch: expected ({self.k_points_count},), got {kspace_data.shape}"
            assert kspace_data.device == self.device
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device)

    # Test parameters
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_shape_2d = (32, 32)
    k_count_2d = 100
    num_coils_test = 4

    print(f"Using device: {test_device}")

    # Instantiate mock single-coil operator
    mock_sc_op = MockSingleCoilNUFFTOp(image_shape=img_shape_2d, k_points_count=k_count_2d, device=test_device)

    # Instantiate MultiCoilNUFFTOperator
    mc_nufft = MultiCoilNUFFTOperator(mock_sc_op)
    print("MultiCoilNUFFTOperator instantiated successfully.")
    assert mc_nufft.device == test_device
    assert mc_nufft.image_shape == img_shape_2d
    if mc_nufft.k_trajectory is not None: # k_trajectory is optional in the wrapper
         assert mc_nufft.k_trajectory.shape == (k_count_2d, len(img_shape_2d))

    # Test op method
    test_multi_coil_image = torch.randn(num_coils_test, *img_shape_2d, dtype=torch.complex64, device=test_device)
    print(f"Input image shape for op: {test_multi_coil_image.shape}")
    k_out = mc_nufft.op(test_multi_coil_image)
    expected_k_shape = (num_coils_test, k_count_2d)
    print(f"Output k-space shape from op: {k_out.shape}, Expected: {expected_k_shape}")
    assert k_out.shape == expected_k_shape, f"op output shape mismatch."
    assert k_out.device == test_device, f"op output device mismatch."
    print("op method test passed.")

    # Test op_adj method
    test_multi_coil_kspace = torch.randn(num_coils_test, k_count_2d, dtype=torch.complex64, device=test_device)
    print(f"Input k-space shape for op_adj: {test_multi_coil_kspace.shape}")
    img_out = mc_nufft.op_adj(test_multi_coil_kspace)
    expected_img_shape = (num_coils_test, *img_shape_2d)
    print(f"Output image shape from op_adj: {img_out.shape}, Expected: {expected_img_shape}")
    assert img_out.shape == expected_img_shape, f"op_adj output shape mismatch."
    assert img_out.device == test_device, f"op_adj output device mismatch."
    print("op_adj method test passed.")
    
    # Test with 3D image shape (e.g., D, H, W)
    img_shape_3d = (16, 32, 32) # D, H, W
    k_count_3d = 150
    mock_sc_op_3d = MockSingleCoilNUFFTOp(image_shape=img_shape_3d, k_points_count=k_count_3d, device=test_device)
    mc_nufft_3d = MultiCoilNUFFTOperator(mock_sc_op_3d)
    print("\nMultiCoilNUFFTOperator (3D) instantiated successfully.")
    
    test_multi_coil_image_3d = torch.randn(num_coils_test, *img_shape_3d, dtype=torch.complex64, device=test_device)
    k_out_3d = mc_nufft_3d.op(test_multi_coil_image_3d)
    expected_k_shape_3d = (num_coils_test, k_count_3d)
    print(f"Output k-space shape from op (3D): {k_out_3d.shape}, Expected: {expected_k_shape_3d}")
    assert k_out_3d.shape == expected_k_shape_3d
    
    test_multi_coil_kspace_3d = torch.randn(num_coils_test, k_count_3d, dtype=torch.complex64, device=test_device)
    img_out_3d = mc_nufft_3d.op_adj(test_multi_coil_kspace_3d)
    expected_img_shape_3d = (num_coils_test, *img_shape_3d)
    print(f"Output image shape from op_adj (3D): {img_out_3d.shape}, Expected: {expected_img_shape_3d}")
    assert img_out_3d.shape == expected_img_shape_3d
    print("3D image shape tests passed.")

    print("\nAll MultiCoilNUFFTOperator tests passed successfully.")

