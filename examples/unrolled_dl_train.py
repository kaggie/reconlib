import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import argparse
import os
import sys

# This allows running the example directly from the 'examples' folder.
# For general use, it's recommended to install reconlib (e.g., `pip install -e .` from root).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.operators import NUFFTOperator
from reconlib.deeplearning.models.resnet_denoiser import SimpleResNetDenoiser
from reconlib.deeplearning.models.modl_network import MoDLNet
from reconlib.deeplearning.datasets import MoDLDataset
# Assuming iternufft.py functions are accessible for dataset
try:
    from iternufft import generate_phantom_2d, generate_radial_trajectory_2d, \
                          generate_phantom_3d, generate_radial_trajectory_3d
except ImportError:
    # Define dummy functions if iternufft is not found for script to be parsable
    print("Warning: iternufft.py not found or its functions are not accessible.")
    def generate_phantom_2d(size, device): return torch.zeros((size,size), device=device)
    def generate_phantom_3d(shape, device): return torch.zeros(shape, device=device)
    def generate_radial_trajectory_2d(**kwargs): return torch.zeros((100,2), device=kwargs.get('device','cpu'))
    def generate_radial_trajectory_3d(**kwargs): return torch.zeros((100,3), device=kwargs.get('device','cpu'))


def train_modl_network(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")

    # --- Data Setup ---
    if args.dim == 2:
        image_shape = (args.image_size, args.image_size)
        phantom_func = generate_phantom_2d
        phantom_params = {'size': args.image_size}
        k_traj_func = generate_radial_trajectory_2d
        k_traj_params = {'num_spokes': args.num_spokes, 'samples_per_spoke': args.image_size}
    elif args.dim == 3:
        image_shape = (args.image_size // 2, args.image_size, args.image_size) # Smaller Z dim
        phantom_func = generate_phantom_3d
        phantom_params = {'shape': image_shape}
        k_traj_func = generate_radial_trajectory_3d
        k_traj_params = {'num_profiles_z': args.num_profiles_z, 
                         'num_spokes_per_profile': args.num_spokes_per_profile, 
                         'samples_per_spoke': args.image_size,
                         'shape': image_shape}
    else:
        raise ValueError("Dimension must be 2 or 3.")

    # NUFFT Operator parameters (shared for dataset and model)
    oversamp_factor = tuple([args.oversamp_factor] * args.dim)
    kb_J = tuple([args.kb_width] * args.dim)
    kb_alpha = tuple([args.kb_alpha_scale * J for J in kb_J]) # e.g. 2.34 * J
    # Ld: Table length for NUFFT. Default args.table_oversamp is 256.
    # For higher accuracy, especially with larger kernels or higher precision needs,
    # larger table lengths (e.g., 1024 for 2D, 512 for 3D) can be used,
    # consistent with defaults in reconlib.nufft.NUFFT2D/3D.
    Ld = tuple([args.table_oversamp] * args.dim)
    # Kd (oversampled grid size) can be derived by NUFFTOperator if None
    Kd = tuple(int(N * os) for N, os in zip(image_shape, oversamp_factor))
    n_shift = tuple([0.0] * args.dim)


    nufft_op_params = {
        'oversamp_factor': oversamp_factor,
        'kb_J': kb_J,
        'kb_alpha': kb_alpha,
        'Ld': Ld,
        'Kd': Kd,
        'kb_m': tuple([0.0]*args.dim), # Default m=0
        'n_shift': n_shift,
        'nufft_type_3d': 'table' if args.dim == 3 else None # Relevant for NUFFTOperator
    }
    if args.dim == 3: # Add interpolation_order for 3D NUFFT in NUFFTOperator via nufft_impl
        nufft_op_params['interpolation_order'] = args.interpolation_order
    
    # Create k-trajectory for NUFFTOperator (used by MoDLNet and Dataset)
    # This k-trajectory is fixed for all data items in this basic setup
    k_traj_fixed = k_traj_func(device=device, **k_traj_params)

    # Dataset and DataLoader
    # Pass all nufft_op_params to dataset which will create its own NUFFTOperator
    dataset_nufft_params = nufft_op_params.copy()
    if args.dim == 2 and 'interpolation_order' in dataset_nufft_params: # NUFFT2D doesn't take this
        del dataset_nufft_params['interpolation_order']
    if args.dim == 2 and 'nufft_type_3d' in dataset_nufft_params:
        del dataset_nufft_params['nufft_type_3d']


    modl_dataset = MoDLDataset(
        dataset_size=args.dataset_size,
        image_shape=image_shape,
        k_trajectory_func=k_traj_func, # Will re-generate, but NUFFTOp in dataset will use its own k_traj_fixed
        k_trajectory_params=k_traj_params, # So this is somewhat redundant here if dataset uses fixed op
        nufft_op_params=dataset_nufft_params,   # Pass all params for dataset's internal NUFFTOperator
        phantom_func=phantom_func,
        phantom_params=phantom_params,
        noise_level_kspace=args.noise_level,
        device=device
    )
    # Override the dataset's nufft_op.k_trajectory if we want it truly fixed from outside
    # This ensures the k_trajectory is identical for all samples if k_traj_func has randomness
    modl_dataset.k_traj = k_traj_fixed 
    modl_dataset.nufft_op.k_trajectory = k_traj_fixed 


    dataloader = DataLoader(modl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # --- Model Setup ---
    # NUFFT Operator for the MoDL network
    model_nufft_op_params = nufft_op_params.copy()
    if args.dim == 2 and 'interpolation_order' in model_nufft_op_params:
        del model_nufft_op_params['interpolation_order']
    if args.dim == 2 and 'nufft_type_3d' in model_nufft_op_params: # NUFFT2D doesn't use this
        del model_nufft_op_params['nufft_type_3d']


    nufft_op_model = NUFFTOperator(
        k_trajectory=k_traj_fixed, # Use the same fixed trajectory
        image_shape=image_shape,
        device=device,
        **model_nufft_op_params
    )

    # Denoiser CNN
    denoiser_channels = args.denoiser_channels 
    if args.dim == 3 and denoiser_channels <=2: # Assuming denoiser_channels=1 for mag, 2 for real/imag complex
        print("Warning: Using a 2D-style denoiser (in_channels<=2) for 3D data. Ensure denoiser is 3D compatible or processes slices.")
    
    denoiser = SimpleResNetDenoiser(
        in_channels=denoiser_channels, 
        out_channels=denoiser_channels,
        num_internal_channels=args.denoiser_internal_channels,
        num_blocks=args.denoiser_num_blocks
    ).to(device)

    # MoDL Network
    modl_network = MoDLNet(
        nufft_op=nufft_op_model,
        denoiser_cnn=denoiser,
        num_iterations=args.modl_iterations,
        lambda_dc_initial=args.lambda_dc,
        learnable_lambda_dc=args.learnable_lambda,
        num_cg_iterations_dc=args.cg_iterations
    ).to(device)

    # --- Training ---
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(modl_network.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        modl_network.train()
        epoch_loss = 0.0
        for i, (x0_batch, y_observed_batch, x_true_batch) in enumerate(dataloader):
            # Current MoDLDataset and MoDLNet are designed for single instance (batch_size=1 implicitly)
            # If batch_size > 1, we loop through the batch. This is inefficient.
            # A truly batched implementation would require NUFFTOperator and MoDLNet to handle batch dims.
            
            current_batch_size = x0_batch.shape[0]
            batch_loss = 0.0

            for b_idx in range(current_batch_size):
                x0, y_observed, x_true = x0_batch[b_idx], y_observed_batch[b_idx], x_true_batch[b_idx]
                x0, y_observed, x_true = x0.to(device), y_observed.to(device), x_true.to(device)

                # Prepare x_true for loss calculation based on denoiser output type
                if denoiser_channels == 2: # Denoiser outputs 2 channels (real/imag)
                    x_true_for_loss = torch.stack([x_true.real, x_true.imag], dim=0) # Shape (2, *spatial_dims)
                elif denoiser_channels == 1: # Denoiser outputs 1 channel (e.g. magnitude)
                    x_true_for_loss = torch.abs(x_true).unsqueeze(0) # Shape (1, *spatial_dims)
                else: # Should match denoiser output format if more complex
                    x_true_for_loss = x_true 
                
                optimizer.zero_grad()
                
                # MoDLNet.forward expects initial_image_x0 to be (*image_shape), complex.
                reconstructed_x = modl_network(y_observed, x0) # x0 is complex from dataset

                # Prepare reconstructed_x for loss calculation
                if denoiser_channels == 2:
                    # If denoiser output is 2-channel real/imag, and MoDLNet returns complex, convert
                    reconstructed_x_for_loss = torch.stack([reconstructed_x.real, reconstructed_x.imag], dim=0)
                elif denoiser_channels == 1:
                    reconstructed_x_for_loss = torch.abs(reconstructed_x).unsqueeze(0)
                else:
                    reconstructed_x_for_loss = reconstructed_x

                loss = criterion(reconstructed_x_for_loss, x_true_for_loss)
                loss.backward()
                batch_loss += loss.item()
            
            optimizer.step() # Step after processing the entire batch (or mini-batch)
            epoch_loss += (batch_loss / current_batch_size)


            if (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Avg Item Loss: {loss.item():.4f}") # loss.item() is last item's loss

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Epoch Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            save_path = os.path.join(args.checkpoint_dir, f"modl_recon_epoch_{epoch+1}.pth")
            torch.save(modl_network.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoDL Network Training Script")
    # Data params
    parser.add_argument('--dim', type=int, default=2, help="Dimension of NUFFT (2 or 3)")
    parser.add_argument('--image_size', type=int, default=64, help="Base image size (e.g., 64 for 64x64 or for 3D like 32x64x64)")
    parser.add_argument('--dataset_size', type=int, default=100, help="Number of on-the-fly samples for dataset")
    # K-traj params (specific to example generators)
    parser.add_argument('--num_spokes', type=int, default=32, help="For 2D radial: number of spokes")
    parser.add_argument('--num_profiles_z', type=int, default=16, help="For 3D stack-of-stars: profiles in Z")
    parser.add_argument('--num_spokes_per_profile', type=int, default=16, help="For 3D stack-of-stars: spokes per Z profile")
    # NUFFT params
    parser.add_argument('--oversamp_factor', type=float, default=2.0, help="NUFFT oversampling factor")
    parser.add_argument('--kb_width', type=int, default=4, help="Kaiser-Bessel kernel width (J)")
    parser.add_argument('--kb_alpha_scale', type=float, default=2.34, help="Scale for KB alpha (alpha = scale * J)")
    parser.add_argument('--table_oversamp', type=int, default=2**8, help="Table oversampling factor (Ld)")
    parser.add_argument('--interpolation_order', type=int, default=1, help="Interpolation order for 3D table NUFFT (0 for NN, 1 for Linear)")
    # MoDL params
    parser.add_argument('--denoiser_channels', type=int, default=1, help="Channels for denoiser (1 for mag, 2 for complex as real/imag)")
    parser.add_argument('--denoiser_internal_channels', type=int, default=32, help="Internal channels in denoiser")
    parser.add_argument('--denoiser_num_blocks', type=int, default=3, help="Number of ResNet blocks in denoiser")
    parser.add_argument('--modl_iterations', type=int, default=5, help="Number of MoDL unrolled iterations (K)")
    parser.add_argument('--lambda_dc', type=float, default=0.05, help="Lambda for data consistency term")
    parser.add_argument('--learnable_lambda', action='store_true', help="Make lambda_dc learnable")
    parser.add_argument('--cg_iterations', type=int, default=3, help="Number of CG iterations in DC step")
    # Training params
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size") # Changed from 1 to allow user setting
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--noise_level', type=float, default=0.01, help="Relative k-space noise level")
    parser.add_argument('--log_interval', type=int, default=10, help="Log training loss every N steps")
    parser.add_argument('--save_interval', type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument('--checkpoint_dir', type=str, default='./modl_checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--num_workers', type=int, default=0, help="Num workers for DataLoader (0 for main process)")
    parser.add_argument('--use_cuda', action='store_true', help="Enable CUDA if available")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train_modl_network(args)
