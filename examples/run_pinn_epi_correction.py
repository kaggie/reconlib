import sys
import os
import torch
import numpy as np # For dummy data generation

# Add project root to sys.path to ensure reconlib and modalities can be found
# Assumes this script is in examples/ and project root is one level up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration for Fast Demo Mode ---
FAST_DEMO_MODE = True
# If True, uses placeholder NUFFT ops defined locally for speed.
# If False, attempts to use reconlib.nufft.NUFFT2D.

try:
    from reconlib.modalities.MRI.pinn_reconstructor import PINNReconstructor, SimpleCNN
    # NUFFT3DAdapter is defined in pinn_reconstructor, so if we need it explicitly, it would be:
    # from reconlib.modalities.MRI.pinn_reconstructor import NUFFT3DAdapter
    from reconlib.nufft import NUFFT2D # Try to import NUFFT2D for non-fast mode
    from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator
    from reconlib.modalities.MRI.physics_loss import BlochResidualLoss, GIRFErrorLoss, B0OffResonanceLoss # Import new classes
    RECONLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all modules from reconlib. Error: {e}")
    RECONLIB_AVAILABLE = False
    if not FAST_DEMO_MODE:
        print("FATAL: reconlib modules required but not found, and FAST_DEMO_MODE is False. Exiting.")
        sys.exit(1)

# --- Placeholder NUFFT Definitions for Fast Demo Mode ---
class PlaceholderNUFFT2D:
    def __init__(self, image_shape, k_trajectory, device='cpu', **kwargs):
        self.image_shape = image_shape
        self.k_trajectory = k_trajectory
        self.device = device
        print("INFO: Using PlaceholderNUFFT2D.")

    def forward(self, x): # Image (Y,X) -> K-space (K,)
        # Minimal op to keep autograd happy if x requires grad
        return (torch.sum(x) * 0.0 + torch.zeros(self.k_trajectory.shape[0], dtype=torch.complex64, device=self.device))

    def adjoint(self, y): # K-space (K,) -> Image (Y,X)
        return (torch.sum(y) * 0.0 + torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device))

class PlaceholderMultiCoilNUFFTOperator:
    def __init__(self, single_coil_nufft_op):
        self.single_coil_nufft_op = single_coil_nufft_op
        self.device = single_coil_nufft_op.device
        self.image_shape = single_coil_nufft_op.image_shape
        print("INFO: Using PlaceholderMultiCoilNUFFTOperator.")

    def op(self, multi_coil_image_data): # (C,Y,X) -> (C,K)
        output_kspace_list = []
        for i in range(multi_coil_image_data.shape[0]):
            single_coil_image = multi_coil_image_data[i]
            output_kspace_list.append(self.single_coil_nufft_op.forward(single_coil_image))
        return torch.stack(output_kspace_list, dim=0)

    def op_adj(self, multi_coil_kspace_data): # (C,K) -> (C,Y,X)
        output_image_list = []
        for i in range(multi_coil_kspace_data.shape[0]):
            single_coil_kspace = multi_coil_kspace_data[i]
            output_image_list.append(self.single_coil_nufft_op.adjoint(single_coil_kspace))
        return torch.stack(output_image_list, dim=0)


def generate_simple_epi_trajectory(image_shape_spatial, num_etl, partial_fourier=0.5, undersampling_factor=1.0):
    """Generates a simplified 3D EPI-like trajectory, normalized to [-0.5, 0.5]."""
    nz, ny, nx = image_shape_spatial

    # Number of phase encoding lines (ky) after partial Fourier
    ny_acquired = int(ny * partial_fourier)

    # Number of kx points (readout)
    kx_points = np.linspace(-0.5, 0.5, nx, endpoint=True)

    # Phase encoding steps (ky)
    ky_steps = np.linspace(-0.5, 0.5 * (ny_acquired / (ny * 0.5)), ny_acquired, endpoint=True) # Adjust for partial Fourier coverage

    # Slice encoding steps (kz) - fully sampled for this example
    kz_steps = np.linspace(-0.5, 0.5, nz, endpoint=True)

    traj_points = []
    for i_z, kz_val in enumerate(kz_steps):
        # Potentially undersample slices
        if np.random.rand() > (1.0 - undersampling_factor) and i_z % int(1/undersampling_factor if undersampling_factor > 0 else 1) != 0 : # Basic random undersampling
             if undersampling_factor < 1.0 : continue


        for i_y, ky_val in enumerate(ky_steps):
            # Readout direction alternates for EPI
            current_kx_points = kx_points if i_y % 2 == 0 else kx_points[::-1]
            for kx_val in current_kx_points:
                traj_points.append([kz_val, ky_val, kx_val]) # Kz, Ky, Kx order

    traj = torch.tensor(np.array(traj_points), dtype=torch.float32)

    # If num_etl is specified, simulate echo train length (group kx lines per ky/kz)
    # This simplified version just ensures the total number of points is controlled.
    # A real EPI would have specific ETL grouping. Here, we might just truncate/tile.
    if len(traj_points) > num_etl * ny_acquired * nz :
        traj = traj[:num_etl * ny_acquired * nz, :] # Simplistic truncation based on "effective ETL"

    print(f"Generated EPI trajectory with {traj.shape[0]} points.")
    return traj


def main():
    print("--- Running PINN EPI Correction Example ---")

    # 1. Setup Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_shape_spatial = (16, 16)  # Y, X - Switched to 2D for speed
    num_coils = 2
    # Actual number of points will be determined by trajectory generation.

    if FAST_DEMO_MODE:
        print("***********************************************************")
        print("*** RUNNING IN FAST DEMO MODE WITH PLACEHOLDER NUFFT    ***")
        print("***********************************************************")
    # Actual number of points will be determined by trajectory generation.

    # EPI Scan Parameters (for B0 loss)
    scan_parameters_epi = {
        'echo_spacing_ms': 0.7,  # Typical echo spacing for EPI
        'phase_encoding_lines': image_shape_spatial[0], # Ny, number of phase encoding lines
        # 'gamma_hz_t' will use default if not specified here
    }
    # General Scan Parameters (for Bloch loss, if used)
    scan_parameters_general = {"TE": 0.03, "TR": 2.0, "flip_angle": 30, "T1_assumed": 1.0, "T2_assumed": 0.08}


    pinn_config = {
        "learning_rate": 1e-3,
        "data_fidelity_weight": 1.0,
        "num_epochs": 1, # Drastically reduced for speed
        "device": device
    }

    # NUFFT parameters (must be 2D tuples for 2D NUFFT)
    nufft_params_reconlib = {
        'oversamp_factor': (1.5, 1.5),
        'kb_J': (4, 4), # Kaiser-Bessel kernel width
        'Ld': (32, 32)  # Table length for NUFFT2D if it were table based, or general detail param.
                        # reconlib.NUFFT2D does not use Ld for table, but might use for other things or ignore.
    }

    # 2. Generate/Load Dummy Data
    print("Generating dummy data...")

    # Ideal k-space trajectory (2D)
    # Using a simplified random trajectory for 2D to avoid complex EPI gen for now
    num_k_points_actual = image_shape_spatial[0] * image_shape_spatial[1] # e.g. 16*16 = 256
    trajectory_ideal = (torch.rand(num_k_points_actual, 2, device=device) - 0.5).float()
    print(f"Ideal trajectory shape: {trajectory_ideal.shape}")

    # Actual k-space trajectory (simulating off-resonance for 2D)
    off_resonance_factor = 0.05 # Small shift
    trajectory_actual = trajectory_ideal.clone()
    # For 2D (ky, kx): apply shift to ky based on kx
    trajectory_actual[:, 0] += off_resonance_factor * trajectory_ideal[:, 1] # ky_actual = ky_ideal + factor * kx_ideal
    trajectory_actual = torch.clamp(trajectory_actual, -0.5, 0.5)

    # B0 Map (2D)
    b0_freq_max_hz = 30.0 # Max off-resonance in Hz for a small FOV
    y_coords = torch.linspace(-1, 1, image_shape_spatial[0], device=device)
    x_coords = torch.linspace(-1, 1, image_shape_spatial[1], device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    b0_map = b0_freq_max_hz * yy  # Simple linear gradient in y for b0 map
    b0_map = b0_map.to(device)
    print(f"B0 map shape: {b0_map.shape}")

    # Coil Sensitivity Maps (2D)
    coil_sensitivities = torch.zeros(num_coils, *image_shape_spatial, dtype=torch.complex64, device=device)
    for c in range(num_coils):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, image_shape_spatial[0], device=device),
            torch.linspace(-1, 1, image_shape_spatial[1], device=device),
            indexing='ij'
        )
        coil_sensitivities[c] = (1.0 - 0.5 * torch.abs(xx - (c * 0.5 - 0.25))) * \
                                (1.0 - 0.5 * torch.abs(yy + (c * 0.5 - 0.25)))
        coil_sensitivities[c] = coil_sensitivities[c].to(torch.complex64)
    print(f"Coil sensitivities shape: {coil_sensitivities.shape}")

    # Ground Truth Image (simple phantom: a square for 2D)
    gt_image = torch.zeros(image_shape_spatial, dtype=torch.complex64, device=device)
    s_y, e_y = image_shape_spatial[0]//4, 3*image_shape_spatial[0]//4
    s_x, e_x = image_shape_spatial[1]//4, 3*image_shape_spatial[1]//4
    gt_image[s_y:e_y, s_x:e_x] = 1.0 + 0.5j # Complex image
    print(f"Ground truth image shape: {gt_image.shape}")

    # Simulate True k-space data (now off-resonant)
    print("Simulating off-resonant true k-space data...")
    # Simplified effective TE for phase simulation on image
    effective_te_for_simulation_s = (scan_parameters_epi['echo_spacing_ms'] / 1000.0) * \
                                    (scan_parameters_epi['phase_encoding_lines'] / 2.0)
    phase_shift = 2 * torch.pi * b0_map * effective_te_for_simulation_s
    gt_image_offresonant = gt_image.to(torch.complex64) * torch.exp(1j * phase_shift)

    gt_coil_images_offresonant = gt_image_offresonant.unsqueeze(0) * coil_sensitivities # (num_coils, Y, X)

    # Use NUFFT2D for simulation (using ideal trajectory for k-space generation from off-resonant image)
    if not FAST_DEMO_MODE and RECONLIB_AVAILABLE:
        nufft_sim_single_coil = NUFFT2D(
            image_shape=image_shape_spatial,
            k_trajectory=trajectory_ideal, # Simulate signal at ideal sampling points from off-resonant image
            device=device,
            **nufft_params_reconlib
        )
        class ReconlibNUFFT2DAdapter:
            def __init__(self, nufft2d_instance: NUFFT2D):
                self.nufft_instance = nufft2d_instance
                self.device = nufft2d_instance.device
                self.image_shape = nufft2d_instance.image_shape
                self.k_trajectory = nufft2d_instance.k_trajectory
            def op(self, x: torch.Tensor) -> torch.Tensor: return self.nufft_instance.forward(x)
            def op_adj(self, y: torch.Tensor) -> torch.Tensor: return self.nufft_instance.adjoint(y)

        adapter_for_simulation = ReconlibNUFFT2DAdapter(nufft_sim_single_coil)
        mc_nufft_for_simulation = MultiCoilNUFFTOperator(adapter_for_simulation)
        print("INFO: Using reconlib.NUFFT2D for k-space simulation.")
    else:
        nufft_sim_single_coil = PlaceholderNUFFT2D(
            image_shape=image_shape_spatial,
            k_trajectory=trajectory_ideal, # Placeholders use ideal trajectory for this simulation step
            device=device
        )
        mc_nufft_for_simulation = PlaceholderMultiCoilNUFFTOperator(nufft_sim_single_coil)
        if not RECONLIB_AVAILABLE and not FAST_DEMO_MODE:
            print("INFO: reconlib not available, using PlaceholderNUFFT for k-space simulation.")
        else:
            print("INFO: Using PlaceholderNUFFT for k-space simulation (FAST_DEMO_MODE).")

    true_kspace_data_mc = mc_nufft_for_simulation.op(gt_coil_images_offresonant) # k-space from off-resonant image
    print(f"Simulated k-space data shape: {true_kspace_data_mc.shape}")

    # 3. Instantiate NUFFT, CNN, and Reconstructor for PINN
    print("Initializing PINN Reconstructor components...")
    # NUFFT operator for reconstruction will use the *ideal* trajectory for data consistency term
    # or it could use the *actual* trajectory if the goal is to reconstruct on that grid.
    # For off-resonance correction, the data term usually uses the actual (measured) trajectory.
    # Let's assume the PINN tries to map from data acquired with `trajectory_actual` to an image
    # that would look good if it *had been* acquired with `trajectory_ideal`.
    # Or, more commonly, the data fidelity term uses `trajectory_actual` and the physics terms enforce priors.
    # For this example, let's have the main NUFFT op for data fidelity use `trajectory_actual`.
    if not FAST_DEMO_MODE and RECONLIB_AVAILABLE:
        nufft_recon_single_coil = NUFFT2D(
            image_shape=image_shape_spatial,
            k_trajectory=trajectory_actual,
            device=device,
            **nufft_params_reconlib
        )
        # Assuming ReconlibNUFFT2DAdapter is defined above if this path is taken
        adapter_recon = ReconlibNUFFT2DAdapter(nufft_recon_single_coil)
        mc_nufft_recon = MultiCoilNUFFTOperator(adapter_recon)
        print("INFO: Using reconlib.NUFFT2D for PINN reconstruction.")
    else:
        # FAST_DEMO_MODE or RECONLIB_UNAVAILABLE
        nufft_recon_single_coil = PlaceholderNUFFT2D(
            image_shape=image_shape_spatial,
            k_trajectory=trajectory_actual,
            device=device
        )
        mc_nufft_recon = PlaceholderMultiCoilNUFFTOperator(nufft_recon_single_coil)
        if not RECONLIB_AVAILABLE and not FAST_DEMO_MODE:
            print("INFO: reconlib not available, using PlaceholderNUFFT for PINN reconstruction.")
        else:
            print("INFO: Using PlaceholderNUFFT for PINN reconstruction (FAST_DEMO_MODE).")

    # Instantiate Physics Loss Terms
    physics_terms_list = []
    # Optional: Add BlochResidualLoss if desired (using general scan_params)
    # physics_terms_list.append(BlochResidualLoss(weight=0.0001, name="Bloch"))

    # Optional: Add GIRFErrorLoss if trajectory_actual is meaningful beyond B0 simulation
    # physics_terms_list.append(GIRFErrorLoss(weight=0.001, name="GIRF"))

    # Add B0OffResonanceLoss
    if B0OffResonanceLoss is not None : # Check if class was imported (it should be)
        b0_loss_term = B0OffResonanceLoss(
            b0_map=b0_map,
            scan_parameters_epi=scan_parameters_epi,
            weight=0.01 # Example weight
        )
        physics_terms_list.append(b0_loss_term)
    else: # Should not happen if imports are correct
        print("Warning: B0OffResonanceLoss class not available for PINN.")


    # CNN: input 1 channel, output num_coils channels, now for 2D
    cnn_model = SimpleCNN(n_channels_in=1, n_channels_out=num_coils, n_spatial_dims=2).to(device)

    reconstructor = PINNReconstructor(
        nufft_op=mc_nufft_recon, # This NUFFT uses actual_trajectory for data fidelity
        cnn_model=cnn_model,
        config=pinn_config,
        physics_terms=physics_terms_list
    )
    print("PINNReconstructor instantiated.")

    # 4. Run Reconstruction
    print("Starting PINN reconstruction...")
    # loss_fn_kwargs needs to supply all data required by the active physics_terms
    loss_function_kwargs = {
        "trajectory_ideal": trajectory_ideal,   # For GIRFErrorLoss (if used)
        "trajectory_actual": trajectory_actual, # For GIRFErrorLoss (if used)
        "scan_parameters": scan_parameters_general, # For BlochResidualLoss (if used)
        "b0_map": b0_map,                       # For B0OffResonanceLoss
        "scan_parameters_epi": scan_parameters_epi # For B0OffResonanceLoss
    }
    reconstructed_img = reconstructor.reconstruct(
        initial_kspace_data_mc=true_kspace_data_mc, # This is k-space from off-resonant image
        num_epochs=pinn_config["num_epochs"],
        loss_fn_kwargs=loss_function_kwargs
    )
    print("Reconstruction finished.")

    # 5. Output/Verification
    print(f"Reconstructed image shape: {reconstructed_img.shape}") # Expected (num_coils, Y, X)

    # Combine coils for viewing (e.g., Root-Sum-of-Squares)
    final_reconstructed_image_rss = torch.sqrt(torch.sum(torch.abs(reconstructed_img)**2, dim=0))
    print(f"RSS Reconstructed image shape: {final_reconstructed_image_rss.shape}") # Expected (Y, X)

    print("--- Plotting skipped for this test run ---")
    # # Optional: Plotting (requires matplotlib)
    # (Plotting is still commented out to ensure timely completion)
    # If enabled, one might plot: gt_image, gt_image_offresonant, final_reconstructed_image_rss
    # try:
    #     import matplotlib.pyplot as plt
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # Now 3 plots
    #     axes[0].imshow(torch.abs(gt_image).cpu().numpy(), cmap='gray')
    #     axes[0].set_title(f"Ground Truth (Corrected)")
    #     axes[0].axis('off')

    #     axes[1].imshow(torch.abs(gt_image_offresonant).cpu().numpy(), cmap='gray')
    #     axes[1].set_title(f"Simulated Off-Resonant Image")
    #     axes[1].axis('off')

    #     axes[2].imshow(torch.abs(final_reconstructed_image_rss).cpu().numpy(), cmap='gray')
    #     axes[2].set_title(f"PINN Reconstructed")
    #     axes[2].axis('off')
    #     plt.suptitle("PINN B0 Off-Resonance Correction Example (2D)")
    #     # ... savefig ...
    # except ImportError:
    #     print("Matplotlib not found. Skipping plot generation.")
    # except Exception as e:
    #     print(f"Error during plotting: {e}")

    print("--- PINN EPI Correction Example Completed ---")

if __name__ == '__main__':
    main()
