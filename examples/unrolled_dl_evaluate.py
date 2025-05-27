import torch
import numpy as np
import math
import argparse
import os
import sys
import matplotlib.pyplot as plt

# Adjust path to import from reconlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.operators import NUFFTOperator
from reconlib.deeplearning.models.resnet_denoiser import SimpleResNetDenoiser
from reconlib.deeplearning.models.modl_network import MoDLNet
from reconlib.deeplearning.datasets import MoDLDataset # For data generation consistency
# Assuming iternufft.py functions are accessible
try:
    from iternufft import generate_phantom_2d, generate_radial_trajectory_2d, \
                          generate_phantom_3d, generate_radial_trajectory_3d
except ImportError:
    print("Warning: iternufft.py not found or its functions are not accessible.")
    # Define dummy functions if iternufft is not found for script to be parsable
    def generate_phantom_2d(size, device): return torch.zeros((size,size), device=device)
    def generate_phantom_3d(shape, device): return torch.zeros(shape, device=device)
    def generate_radial_trajectory_2d(**kwargs): return torch.zeros((100,2), device=kwargs.get('device','cpu'))
    def generate_radial_trajectory_3d(**kwargs): return torch.zeros((100,3), device=kwargs.get('device','cpu'))

# Basic PSNR and SSIM (structural similarity) functions for evaluation
# For SSIM, a library like scikit-image might be preferred for a robust implementation,
# but here's a simple version for demonstration if skimage is not a direct dependency.

def psnr_torch(gt, pred, data_range=None):
    if data_range is None:
        data_range = gt.max() - gt.min()
    mse = torch.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def ssim_torch_simple(gt, pred, data_range=None, C1=0.01**2, C2=0.03**2):
    """ Simplified SSIM for single channel images, assumes gt/pred are (H,W) """
    if data_range is None:
        data_range = gt.max() - gt.min()
    
    mu_x = torch.mean(pred)
    mu_y = torch.mean(gt)
    sigma_x_sq = torch.mean((pred - mu_x)**2)
    sigma_y_sq = torch.mean((gt - mu_y)**2)
    sigma_xy = torch.mean((pred - mu_x)*(gt - mu_y))
    
    ssim_num = (2*mu_x*mu_y + C1*data_range**2) * (2*sigma_xy + C2*data_range**2)
    ssim_den = (mu_x**2 + mu_y**2 + C1*data_range**2) * (sigma_x_sq + sigma_y_sq + C2*data_range**2)
    
    return ssim_num / ssim_den


def evaluate_modl_network(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")

    # --- Data Setup (Similar to training, but for a test set) ---
    if args.dim == 2:
        image_shape = (args.image_size, args.image_size)
        phantom_func = generate_phantom_2d
        phantom_params = {'size': args.image_size}
        k_traj_func = generate_radial_trajectory_2d
        k_traj_params = {'num_spokes': args.num_spokes, 'samples_per_spoke': args.image_size}
    elif args.dim == 3:
        image_shape = (args.image_size // 2, args.image_size, args.image_size)
        phantom_func = generate_phantom_3d
        phantom_params = {'shape': image_shape}
        k_traj_func = generate_radial_trajectory_3d
        k_traj_params = {'num_profiles_z': args.num_profiles_z, 
                         'num_spokes_per_profile': args.num_spokes_per_profile, 
                         'samples_per_spoke': args.image_size,
                         'shape': image_shape}
    else:
        raise ValueError("Dimension must be 2 or 3.")

    oversamp_factor = tuple([args.oversamp_factor] * args.dim)
    kb_J = tuple([args.kb_width] * args.dim)
    kb_alpha = tuple([args.kb_alpha_scale * J for J in kb_J])
    Ld = tuple([args.table_oversamp] * args.dim)
    Kd = tuple(int(N * os) for N, os in zip(image_shape, oversamp_factor))
    n_shift = tuple([0.0] * args.dim)

    nufft_op_params = {
        'oversamp_factor': oversamp_factor, 'kb_J': kb_J, 'kb_alpha': kb_alpha,
        'Ld': Ld, 'Kd': Kd, 'kb_m': tuple([0.0]*args.dim), 'n_shift': n_shift,
        'nufft_type_3d': 'table' if args.dim == 3 else None
    }
    if args.dim == 3: # Add interpolation_order for 3D NUFFT in NUFFTOperator via nufft_impl
        nufft_op_params['interpolation_order'] = args.interpolation_order
    
    k_traj_fixed = k_traj_func(device=device, **k_traj_params)

    # Test Dataset
    # Using MoDLDataset to generate test samples easily
    test_dataset_size = args.test_dataset_size
    
    # Ensure nufft_op_params for dataset is compatible with its NUFFTOperator
    dataset_nufft_params = nufft_op_params.copy()
    if args.dim == 2 and 'interpolation_order' in dataset_nufft_params:
        del dataset_nufft_params['interpolation_order']
    if args.dim == 2 and 'nufft_type_3d' in dataset_nufft_params:
        del dataset_nufft_params['nufft_type_3d']

    test_dataset = MoDLDataset(
        dataset_size=test_dataset_size, image_shape=image_shape,
        k_trajectory_func=k_traj_func, k_trajectory_params=k_traj_params,
        nufft_op_params=dataset_nufft_params, 
        phantom_func=phantom_func, phantom_params=phantom_params,
        noise_level_kspace=args.noise_level, device=device
    )
    test_dataset.k_traj = k_traj_fixed # Ensure same trajectory as model was trained on
    test_dataset.nufft_op.k_trajectory = k_traj_fixed
    
    # No DataLoader needed if evaluating one by one, or can use batch_size=1
    
    # --- Model Setup ---
    model_nufft_op_params = nufft_op_params.copy()
    if args.dim == 2 and 'interpolation_order' in model_nufft_op_params:
        del model_nufft_op_params['interpolation_order']
    if args.dim == 2 and 'nufft_type_3d' in model_nufft_op_params:
        del model_nufft_op_params['nufft_type_3d']


    nufft_op_model = NUFFTOperator(
        k_trajectory=k_traj_fixed, image_shape=image_shape, device=device, **model_nufft_op_params
    )
    denoiser_channels = args.denoiser_channels
    denoiser = SimpleResNetDenoiser(
        in_channels=denoiser_channels, out_channels=denoiser_channels,
        num_internal_channels=args.denoiser_internal_channels, num_blocks=args.denoiser_num_blocks
    ).to(device)
    
    modl_network = MoDLNet(
        nufft_op=nufft_op_model, denoiser_cnn=denoiser,
        num_iterations=args.modl_iterations, lambda_dc_initial=args.lambda_dc,
        # For evaluation, learnable_lambda is determined by the loaded model, not an arg here.
        # num_cg_iterations_dc is part of the model architecture.
        num_cg_iterations_dc=args.cg_iterations 
    ).to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist.")
        return
    
    modl_network.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    modl_network.eval()
    print(f"Loaded model from {args.checkpoint_path}")

    # --- Evaluation Loop ---
    total_psnr = 0.0
    total_ssim = 0.0
    
    for i in range(test_dataset_size):
        x0, y_observed, x_true = test_dataset[i]
        x0, y_observed, x_true = x0.to(device), y_observed.to(device), x_true.to(device)

        with torch.no_grad():
            # MoDLNet forward pass expects single instance inputs: x0 (*image_shape), y (num_k_points,)
            reconstructed_x = modl_network(y_observed, x0)

        # Ensure reconstructed_x and x_true are on CPU for metrics/plotting if functions expect numpy
        rec_np = reconstructed_x.abs().cpu() # Often evaluate on magnitude
        gt_np = x_true.abs().cpu()
        
        current_psnr = psnr_torch(gt_np, rec_np)
        current_ssim = ssim_torch_simple(gt_np, rec_np) # Simple SSIM for single slice
        
        total_psnr += current_psnr
        total_ssim += current_ssim
        
        print(f"Sample {i+1}/{test_dataset_size} - PSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}")

        if args.save_images_dir and (i < args.num_images_to_save): # Save a few examples
            if not os.path.exists(args.save_images_dir):
                os.makedirs(args.save_images_dir)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            if args.dim == 2:
                axs[0].imshow(gt_np.numpy(), cmap='gray')
                axs[0].set_title("Ground Truth")
                axs[1].imshow(x0.abs().cpu().numpy(), cmap='gray') # Initial/Zero-filled
                axs[1].set_title("Initial Recon (A^H y)")
                axs[2].imshow(rec_np.numpy(), cmap='gray')
                axs[2].set_title("MoDL Reconstructed")
            elif args.dim == 3:
                center_slice = gt_np.shape[0] // 2
                axs[0].imshow(gt_np[center_slice].numpy(), cmap='gray')
                axs[0].set_title(f"GT (Slice {center_slice})")
                axs[1].imshow(x0.abs().cpu()[center_slice].numpy(), cmap='gray')
                axs[1].set_title(f"A^H y (Slice {center_slice})")
                axs[2].imshow(rec_np[center_slice].numpy(), cmap='gray')
                axs[2].set_title(f"MoDL (Slice {center_slice})")
            
            for ax_ in axs: ax_.axis('off')
            plt.savefig(os.path.join(args.save_images_dir, f"eval_sample_{i+1}.png"))
            plt.close(fig)
            print(f"  Saved image for sample {i+1}")

    avg_psnr = total_psnr / test_dataset_size
    avg_ssim = total_ssim / test_dataset_size
    print(f"\nAverage PSNR over {test_dataset_size} samples: {avg_psnr:.2f} dB")
    print(f"Average SSIM over {test_dataset_size} samples: {avg_ssim:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoDL Network Evaluation Script")
    # Inherit most args from training script for consistency
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument('--dim', type=int, default=2, help="Dimension of NUFFT (2 or 3)")
    parser.add_argument('--image_size', type=int, default=64, help="Base image size")
    parser.add_argument('--test_dataset_size', type=int, default=10, help="Number of on-the-fly test samples")
    
    parser.add_argument('--num_spokes', type=int, default=32, help="For 2D radial: number of spokes")
    parser.add_argument('--num_profiles_z', type=int, default=16, help="For 3D stack-of-stars: profiles in Z")
    parser.add_argument('--num_spokes_per_profile', type=int, default=16, help="For 3D stack-of-stars: spokes per Z profile")
    
    parser.add_argument('--oversamp_factor', type=float, default=2.0)
    parser.add_argument('--kb_width', type=int, default=4)
    parser.add_argument('--kb_alpha_scale', type=float, default=2.34)
    parser.add_argument('--table_oversamp', type=int, default=2**8)
    parser.add_argument('--interpolation_order', type=int, default=1, help="Interpolation order for 3D table NUFFT (0 for NN, 1 for Linear)")
    
    parser.add_argument('--denoiser_channels', type=int, default=1)
    parser.add_argument('--denoiser_internal_channels', type=int, default=32)
    parser.add_argument('--denoiser_num_blocks', type=int, default=3)
    parser.add_argument('--modl_iterations', type=int, default=5)
    parser.add_argument('--lambda_dc', type=float, default=0.05)
    # Learnable lambda not relevant for eval if loading checkpoint that didn't have it
    parser.add_argument('--cg_iterations', type=int, default=3)
    
    parser.add_argument('--noise_level', type=float, default=0.01, help="Relative k-space noise level for test data")
    parser.add_argument('--use_cuda', action='store_true', help="Enable CUDA if available")
    
    parser.add_argument('--save_images_dir', type=str, default=None, help="Directory to save example reconstructed images. If None, images are not saved.")
    parser.add_argument('--num_images_to_save', type=int, default=3, help="Number of sample images to save if save_images_dir is provided.")

    args = parser.parse_args()
    evaluate_modl_network(args)
