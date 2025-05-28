import torch
import numpy as np
import matplotlib.pyplot as plt
from reconlib.operators import NUFFTOperator, CoilSensitivityOperator, MRIForwardOperator
from reconlib.csm import estimate_espirit_maps # Placeholder
from reconlib.regularizers import L1Regularizer
from reconlib.reconstructors import ProximalGradientReconstructor
from reconlib.plotting import plot_phase_image # For plotting complex image parts

def run_espirit_nufft_example():
    print("--- Running ESPIRiT-NUFFT Regularized Reconstruction Example (Placeholders) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Define Parameters
    image_shape = (64, 64)
    num_coils = 4
    num_k_points = image_shape[0] * image_shape[1] // 2 # Undersampled
    
    # NUFFT parameters (example for NUFFT2D via NUFFTOperator)
    oversamp_factor = (2.0, 2.0)
    kb_J = (6, 6) 
    kb_alpha = tuple(val * 2.34 for val in kb_J) # Common heuristic
    Ld = tuple(int(ims * ovs) for ims, ovs in zip(image_shape, oversamp_factor))


    # 2. Create Dummy Data
    #   a. True Image (simple phantom)
    true_image_np = np.zeros(image_shape, dtype=np.complex64)
    true_image_np[image_shape[0]//4:3*image_shape[0]//4, image_shape[1]//4:3*image_shape[1]//4] = 1 + 0.5j
    true_image_torch = torch.from_numpy(true_image_np).to(device)

    #   b. K-space Trajectory (e.g., radial-like, normalized to [-0.5, 0.5])
    angles = np.linspace(0, np.pi, image_shape[0], endpoint=False)
    # Calculate how many radii points are needed for the given num_k_points
    num_spokes = image_shape[0]
    points_per_spoke = num_k_points // num_spokes
    if points_per_spoke == 0: points_per_spoke = 1 # Ensure at least one point per spoke
    
    radii = np.linspace(-0.5, 0.5, points_per_spoke, endpoint=True)
    k_coords_x = []
    k_coords_y = []
    for ang in angles:
        for r in radii:
            k_coords_x.append(r * np.cos(ang))
            k_coords_y.append(r * np.sin(ang))
    
    k_trajectory_np = np.stack([np.array(k_coords_y), np.array(k_coords_x)], axis=-1)[:num_k_points]
    k_trajectory_torch = torch.from_numpy(k_trajectory_np).float().to(device)
    
    if k_trajectory_torch.shape[0] != num_k_points: # Adjust if stacking resulted in more points due to discrete spokes
        k_trajectory_torch = k_trajectory_torch[:num_k_points,:]

    if k_trajectory_torch.shape[1] != len(image_shape): # Ensure correct dimension
         k_trajectory_torch = k_trajectory_torch[:, :len(image_shape)]


    #   c. Coil Sensitivity Maps (ESPIRiT placeholder)
    print("\nEstimating ESPIRiT maps (using placeholder)...")
    # estimate_espirit_maps is a placeholder, will return zeros.
    # We need non-zero maps for the example to run.
    s_maps_np = np.zeros((num_coils,) + image_shape, dtype=np.complex64)
    for c in range(num_coils):
        # Simple phase variation across coils
        s_maps_np[c, ...] = np.exp(1j * c * np.pi / num_coils) 
        # Add a Gaussian amplitude profile for spatial variation
        y, x = np.ogrid[-image_shape[0]//2:image_shape[0]//2, -image_shape[1]//2:image_shape[1]//2]
        gauss_profile = np.exp(-(x**2 + y**2) / (2 * (image_shape[0]/3)**2))
        s_maps_np[c, ...] *= gauss_profile
    s_maps_torch = torch.from_numpy(s_maps_np).to(device)
    s_maps_torch = s_maps_torch / (torch.sqrt(torch.sum(torch.abs(s_maps_torch)**2, dim=0, keepdim=True)) + 1e-9)


    #   d. Synthesize Multi-Coil K-space Data
    temp_nufft_op = NUFFTOperator(k_trajectory=k_trajectory_torch, image_shape=image_shape,
                                  oversamp_factor=oversamp_factor, kb_J=kb_J, kb_alpha=kb_alpha, Ld=Ld, device=device)
    temp_coil_op = CoilSensitivityOperator(s_maps_torch)
    temp_mri_fwd_op = MRIForwardOperator(nufft_operator=temp_nufft_op, coil_operator=temp_coil_op)
    
    kspace_data_torch = temp_mri_fwd_op.op(true_image_torch)
    noise_level = 0.05 * torch.norm(kspace_data_torch) / np.sqrt(float(np.prod(kspace_data_torch.shape)))
    kspace_data_torch += noise_level * torch.randn_like(kspace_data_torch)
    print(f"Synthesized k-space data shape: {kspace_data_torch.shape}")


    # 3. Setup Reconstruction Components
    nufft_op = NUFFTOperator(k_trajectory=k_trajectory_torch, image_shape=image_shape,
                             oversamp_factor=oversamp_factor, kb_J=kb_J, kb_alpha=kb_alpha, Ld=Ld, device=device)
    coil_sens_op = CoilSensitivityOperator(coil_sensitivities_tensor=s_maps_torch)
    mri_forward_op = MRIForwardOperator(nufft_operator=nufft_op, coil_operator=coil_sens_op)
    
    lambda_l1 = 0.01 
    l1_regularizer = L1Regularizer(lambda_reg=lambda_l1)

    pg_reconstructor = ProximalGradientReconstructor(
        iterations=30, 
        step_size=0.1, 
        verbose=True
    )

    # 4. Perform Reconstruction
    print("\nStarting reconstruction...")
    def forward_fn_wrapper(image, smaps_ignored): 
        return mri_forward_op.op(image)

    def adjoint_fn_wrapper(kspace, smaps_ignored):
        return mri_forward_op.op_adj(kspace)

    x_initial = adjoint_fn_wrapper(kspace_data_torch, None)

    reconstructed_image_torch = pg_reconstructor.reconstruct(
        kspace_data=kspace_data_torch,
        forward_op_fn=forward_fn_wrapper,
        adjoint_op_fn=adjoint_fn_wrapper,
        regularizer_prox_fn=l1_regularizer.proximal_operator,
        x_init=x_initial.clone()
    )
    reconstructed_image_np = reconstructed_image_torch.cpu().numpy()

    # 5. Plotting
    is_interactive = hasattr(plt, 'isinteractive') and plt.isinteractive()
    fn_true_mag = "espirit_true_mag.png" if not is_interactive else None
    fn_true_phase = "espirit_true_phase.png" if not is_interactive else None
    fn_adj = "espirit_adjoint.png" if not is_interactive else None
    fn_recon = "espirit_recon.png" if not is_interactive else None

    plt.figure(figsize=(12, 10)) # Adjusted for 2x2 layout
    
    # True Image Magnitude
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(true_image_np), cmap='gray')
    plt.title("True Image (Magnitude)")
    plt.colorbar()
    plt.axis('off')
    if fn_true_mag: plt.savefig(fn_true_mag)

    # True Image Phase
    plt.subplot(2, 2, 2)
    # Using plot_phase_image for phase part
    plot_phase_image(np.angle(true_image_np), title="True Image (Phase)", filename=fn_true_phase)
    # The plot_phase_image function handles its own figure saving if filename is provided,
    # so we don't call plt.savefig for this subplot if using that helper.
    # For consistency, if plot_phase_image is used in a subplot, ensure it doesn't create a new figure.
    # The current reconlib.plotting.plot_phase_image creates a new figure.
    # For subplots, it's better to use plt.imshow directly for phase as well or modify plot_phase_image
    # to accept an `ax` argument. For simplicity here, let's use imshow.
    # Re-doing subplot for true phase with imshow:
    if fn_true_phase: plt.close() # Close figure from plot_phase_image if it made one
    plt.subplot(2, 2, 2) # Re-claim subplot
    plt.imshow(np.angle(true_image_np), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.title("True Image (Phase)")
    plt.colorbar(label="Phase (radians)")
    plt.axis('off')
    if fn_true_phase: plt.savefig(fn_true_phase)


    # Initial Estimate (Adjoint Recon)
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(x_initial.cpu().numpy()), cmap='gray')
    plt.title("Initial Estimate (Adjoint Recon)")
    plt.colorbar()
    plt.axis('off')
    if fn_adj: plt.savefig(fn_adj)
    
    # Reconstructed Image
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(reconstructed_image_np), cmap='gray')
    plt.title(f"Reconstructed Image (L1 Reg, {pg_reconstructor.iterations} iters)")
    plt.colorbar()
    plt.axis('off')
    if fn_recon: plt.savefig(fn_recon)
    
    plt.suptitle("ESPIRiT-NUFFT Regularized Reconstruction Example")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if is_interactive:
        plt.show()
    else:
        plt.savefig("espirit_recon_summary.png")
        plt.close('all') # Close all figures
        print(f"Plots saved: {fn_true_mag}, {fn_true_phase}, {fn_adj}, {fn_recon}, espirit_recon_summary.png (if not interactive)")

    print("\n--- ESPIRiT-NUFFT Example Finished ---")

if __name__ == "__main__":
    run_espirit_nufft_example()
