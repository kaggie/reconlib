import torch
import torch.nn as nn
# Adjust paths as necessary
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Go up three levels for reconlib

from reconlib.operators import NUFFTOperator
# Assuming ResNetDenoiser is in a sibling directory 'models' if this file is also in 'models'
# If this file (modl_network.py) is in reconlib/deeplearning/models, then:
from .resnet_denoiser import SimpleResNetDenoiser 
# And DataConsistencyMoDL is in reconlib/deeplearning/layers:
from ..layers.data_consistency import DataConsistencyMoDL 


class MoDLNet(nn.Module):
    """
    MoDL (Model-based Deep Learning) network for MRI reconstruction.
    This network unrolls an iterative optimization algorithm that alternates between
    a learned regularization (denoising via CNN) and data consistency steps.
    """
    def __init__(self,
                 nufft_op: NUFFTOperator,
                 denoiser_cnn: nn.Module, # Pass an instance of the denoiser
                 num_iterations: int = 5,
                 lambda_dc_initial: float = 0.05,
                 learnable_lambda_dc: bool = False,
                 # shared_weights for denoiser is implicit if only one denoiser_cnn instance is passed and used.
                 # If separate denoisers per iteration were needed, a list of denoisers would be passed.
                 num_cg_iterations_dc: int = 5 
                ):
        super().__init__()
        self.num_iterations = num_iterations
        self.nufft_op = nufft_op # Used by DC layers
        self.device = nufft_op.device

        # Denoiser (Regularization block)
        # The same denoiser instance will be used across iterations (shared weights)
        self.denoiser_cnn = denoiser_cnn 

        # Data Consistency blocks
        # Each iteration can have its own lambda_dc if learnable_lambda_dc is True,
        # otherwise, they share the same lambda_dc value.
        self.dc_layers = nn.ModuleList()
        for _ in range(num_iterations):
            if learnable_lambda_dc:
                # Make lambda_dc a learnable parameter for each DC block
                # Initialize with lambda_dc_initial, ensure it stays positive (e.g., via softplus or exp)
                # For simplicity, we'll start with a fixed lambda passed to each,
                # but a nn.Parameter could be used here.
                # dc_lambda = nn.Parameter(torch.tensor(lambda_dc_initial, device=self.device, dtype=torch.float32))
                # For now, let's assume lambda_dc_initial is either a float (fixed for all) or a list/tuple of floats.
                # The DataConsistencyMoDL class handles scalar tensor for lambda_dc correctly.
                current_lambda = torch.tensor(lambda_dc_initial, device=self.device, dtype=torch.float32)
                if learnable_lambda_dc: # Make it a parameter if learnable
                    current_lambda = nn.Parameter(current_lambda)

            else: # Fixed lambda for all iterations
                current_lambda = lambda_dc_initial # This will be converted to tensor in DataConsistencyMoDL

            self.dc_layers.append(
                DataConsistencyMoDL(nufft_op=self.nufft_op, 
                                    lambda_dc=current_lambda, 
                                    num_cg_iterations=num_cg_iterations_dc)
            )
            
        # Store lambda_dc for external access if needed (especially if not learnable)
        # If learnable, they are in self.dc_layers[i].lambda_dc
        if not learnable_lambda_dc:
            self.lambda_dc_fixed = torch.tensor(lambda_dc_initial, device=self.device, dtype=torch.float32)


    def forward(self, 
                observed_k_space_y: torch.Tensor, 
                initial_image_x0: torch.Tensor | None = None
               ) -> torch.Tensor:
        """
        Forward pass of the MoDL network.

        Args:
            observed_k_space_y: Undersampled k-space data. Shape: (batch_size, num_k_points) or (num_k_points).
                                 Batching needs to be handled carefully if k-space/image shapes vary per batch item.
                                 For now, assume batch_size=1 or op is per-example.
            initial_image_x0: Optional initial image estimate. Shape: (batch_size, *image_shape) or (*image_shape).
                              If None, it's computed using A^H y.
        
        Returns:
            Reconstructed image. Shape: (batch_size, *image_shape) or (*image_shape).
        """
        
        # Handle potential batch dimension for k-space and image
        # For simplicity, let's assume inputs are single examples for now, or that
        # NUFFTOperator and denoiser can handle batch inputs if k_traj is shared.
        # This example assumes k_traj is fixed in NUFFTOperator, so batching applies to image/k-space data.

        if initial_image_x0 is None:
            # If k-space is batched (N, num_k_points), op_adj needs to handle it or loop.
            # Assume op_adj takes (num_k_points,) and returns (*image_shape)
            if observed_k_space_y.ndim == 2 and observed_k_space_y.shape[0] > 1 : # Batched k-space
                # Need to process each item in batch, then stack. This complicates things.
                # For now, let's assume single (num_k_points) k-space input for A^H y.
                if observed_k_space_y.shape[0] != 1: # Make sure it's not (1, N_k) which is fine
                    # Or handle batch in op_adj of NUFFTOperator
                    raise NotImplementedError("Batched k-space for A^H y not directly handled in this example's initial_image_x0 calculation. Assume single k-space input or op_adj handles batches.")
                current_x_k = self.nufft_op.op_adj(observed_k_space_y.squeeze(0)) # Remove batch dim if (1,Nk)
                current_x_k = current_x_k.unsqueeze(0) # Add batch dim for network
            else: # Assumed (num_k_points,)
                current_x_k = self.nufft_op.op_adj(observed_k_space_y)
                # Add batch dimension if denoiser expects (N,C,H,W) and image_shape is (H,W)
                if current_x_k.ndim == len(self.nufft_op.image_shape):
                     current_x_k = current_x_k.unsqueeze(0) # (1, H, W) or (1, D, H, W)
        else:
            current_x_k = initial_image_x0
            if current_x_k.ndim == len(self.nufft_op.image_shape): # Ensure batch dim
                 current_x_k = current_x_k.unsqueeze(0)


        # Ensure observed_k_space_y also has a batch dim if it's single and x is batched, or vice-versa
        # This part is tricky depending on how batching is handled by NUFFTOperator vs CNN
        # For now, assume DC layer handles single y and single x_k, and batching is external if needed.
        # If current_x_k is (1, ...), and y is (Nk), then DC layer should get y.
        # If current_x_k is (N, ...), then y should be (N, Nk) and DC layer needs to loop or be batch-aware.
        # Let's assume for now that forward takes single instance inputs, batching is handled by a trainer.
        if current_x_k.shape[0] > 1:
            raise NotImplementedError("Batch processing in MoDLNet forward pass needs careful implementation for DC consistency. Assuming single instance for now.")
        
        # If k-space had a batch dim (1, Nk), squeeze it for DC layer if DC layer expects (Nk)
        if observed_k_space_y.ndim == 2 and observed_k_space_y.shape[0] == 1:
            observed_k_space_y_for_dc = observed_k_space_y.squeeze(0)
        else:
            observed_k_space_y_for_dc = observed_k_space_y


        for i in range(self.num_iterations):
            # Regularization / Denoising step
            # Denoiser expects (N,C,H,W) or (N,H,W) which it unsqueezes.
            # If current_x_k is (1, D, H, W) for 3D, denoiser needs to handle 3D.
            # Current SimpleResNetDenoiser is 2D. This implies MoDLNet needs a 2D/3D aware denoiser or reshaping.
            # For now, assume denoiser matches dimensionality of x_k.
            # And if x_k is (1,H,W), denoiser input is fine. If (1,D,H,W), denoiser needs to be 3D.
            
            # Let's assume current_x_k is (1, *image_shape)
            # If denoiser is 2D and image_shape is 3D, this is an issue.
            # For this example, let's assume denoiser can handle the shape.
            denoiser_input = current_x_k # This should be (N, C, H, W) or (N, C, D, H, W) for denoiser
                                        # Our current SimpleResNetDenoiser is 2D (N,C,H,W)
                                        # If image_shape is 3D, we need a 3D denoiser or process slices.
                                        # This example will assume image_shape is 2D for now for simplicity with current denoiser.
            if len(self.nufft_op.image_shape) == 3 and self.denoiser_cnn.initial_conv.in_channels <=2 : # HACK: check if denoiser is 2D
                 # Process 3D data slice by slice or use a 3D Denoiser
                 # This is a placeholder for proper 3D handling.
                 # For now, if 3D input and 2D denoiser, it will likely fail or do something weird.
                 # We will need a SimpleResNetDenoiser3D or similar.
                 print("Warning: MoDLNet using a 2D denoiser for 3D data. This needs a 3D denoiser.")
            
            denoised_zk = self.denoiser_cnn(denoiser_input) # z_k in paper
            
            # Data Consistency step
            # dc_layer expects current_image_estimate_zk to be (*image_shape)
            # and observed_k_space_y to be (num_k_points,)
            current_x_k = self.dc_layers[i](denoised_zk.squeeze(0), observed_k_space_y_for_dc) # x_{k+1} in paper
            
            # Re-add batch dim if it was squeezed for DC.
            if current_x_k.ndim == len(self.nufft_op.image_shape): # If (H,W) or (D,H,W)
                 current_x_k = current_x_k.unsqueeze(0) # (1, H, W) or (1, D, H, W)

        return current_x_k.squeeze(0) # Return single image result, remove batch dim

if __name__ == '__main__':
    print("Running MoDLNet basic example...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup for a 2D case
    img_s = 32
    ishape = (img_s, img_s)
    k_pts = 100
    dims = 2
    
    # Dummy NUFFTOperator (using Mock or Real as in DataConsistencyMoDL example)
    try:
        k_traj_dummy = torch.rand(k_pts, dims, device=device) - 0.5
        oversamp_factor_dummy = tuple([2.0]*dims)
        kb_J_dummy = tuple([4]*dims)
        kb_alpha_dummy = tuple([2.34 * J for J in kb_J_dummy])
        Ld_dummy = tuple([2**8]*dims)
        nufft_op_example = NUFFTOperator(
            image_shape=ishape, k_trajectory=k_traj_dummy,
            oversamp_factor=oversamp_factor_dummy, kb_J=kb_J_dummy,
            kb_alpha=kb_alpha_dummy, Ld=Ld_dummy, device=device
        )
        print("Using actual NUFFTOperator for MoDLNet example.")
    except Exception as e:
        print(f"Could not init actual NUFFTOperator ({e}), MoDLNet example may not be fully representative.")
        # Fallback to a mock if needed, but previous DC layer example set one up.
        # For MoDLNet, it's better if NUFFTOperator is functional.
        # This example will fail if NUFFTOperator cannot be created.
        raise

    # Denoiser - assuming 2D data, 1 channel input/output
    # If input image is complex (e.g. 2 channels for real/imag), adjust in_channels/out_channels
    denoiser = SimpleResNetDenoiser(in_channels=1, out_channels=1, num_internal_channels=32, num_blocks=2).to(device)

    # MoDLNet
    modl_network = MoDLNet(nufft_op=nufft_op_example,
                           denoiser_cnn=denoiser,
                           num_iterations=3, # K value in paper
                           lambda_dc_initial=0.05,
                           num_cg_iterations_dc=3).to(device)

    # Dummy data
    # Assume initial_image_x0 is (H,W) or (D,H,W) for single batch
    # And observed_k_space_y is (num_k_points,)
    initial_image = torch.randn(ishape, dtype=torch.complex64, device=device)
    k_space_obs = torch.randn(k_pts, dtype=torch.complex64, device=device)
    
    # If initial image is None, MoDLNet will compute A^H y
    # For this test, let's provide one.
    
    print(f"Input initial_image shape: {initial_image.shape}")
    print(f"Input k_space_obs shape: {k_space_obs.shape}")

    # Run forward pass
    reconstructed_image = modl_network(k_space_obs, initial_image)
    print(f"Output reconstructed_image shape: {reconstructed_image.shape}")
    assert reconstructed_image.shape == ishape, "Output shape mismatch!"
    
    # Test with initial_image_x0 = None
    reconstructed_image_no_x0 = modl_network(k_space_obs, None)
    print(f"Output reconstructed_image_no_x0 shape: {reconstructed_image_no_x0.shape}")
    assert reconstructed_image_no_x0.shape == ishape, "Output shape mismatch (no x0)!"
    
    print("MoDLNet basic example run completed.")
