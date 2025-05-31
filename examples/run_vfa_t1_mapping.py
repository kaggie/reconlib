import torch
import time
import sys
import os
import math

# Add project root to sys.path to allow importing reconlib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from reconlib.modalities.MRI.T1_mapping import spgr_signal, fit_t1_vfa

# Optional: For phantom generation and plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    # from skimage.data import shepp_logan_phantom # If scikit-image is available
    # For a simple built-in phantom:
    def simple_phantom(size=128, device='cpu'):
        img = torch.zeros((size, size), device=device, dtype=torch.float32)
        # Create a square region
        img[size//4:size*3//4, size//4:size*3//4] = 1
        # Create another smaller square region with different value
        img[size//3:size*2//3, size//3:size*2//3] = 2
        return img
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib and NumPy found. Plotting will be enabled.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib and/or NumPy not available. Plotting will be skipped.")
    # Define a dummy simple_phantom if numpy is not available for the main logic to run
    if 'np' not in locals() and 'np' not in globals(): # Check if numpy failed to import
        def simple_phantom(size=128, device='cpu'):
            img = torch.zeros((size, size), device=device, dtype=torch.float32)
            img[size//4:size*3//4, size//4:size*3//4] = 1
            img[size//3:size*2//3, size//3:size*2//3] = 2
            return img


if __name__ == '__main__':
    print("--- Running VFA T1 Mapping Example ---")

    # 1. Setup Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 64 # Keep it small for faster demo
    TR_ms = 20.0
    flip_angles_deg_np = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    flip_angles_deg = torch.tensor(flip_angles_deg_np, device=device, dtype=torch.float32)

    noise_level_percent = 2.0 # 2% noise
    use_b1_map_simulation = True # Simulate with B1 variation
    use_b1_map_fitting = True    # Try to correct with B1 map in fitting

    # 2. Generate Ground Truth Data
    print("Generating ground truth T1 and M0 maps...")
    true_T1_ms = torch.ones((img_size, img_size), device=device, dtype=torch.float32) * 1200.0 # Background T1
    true_M0 = torch.ones((img_size, img_size), device=device, dtype=torch.float32) * 2000.0    # Background M0

    ph_structure = simple_phantom(img_size, device=device)
    true_T1_ms[ph_structure == 1] = 800.0  # Region 1 T1
    true_M0[ph_structure == 1] = 2500.0   # Region 1 M0
    true_T1_ms[ph_structure == 2] = 1800.0 # Region 2 T1
    true_M0[ph_structure == 2] = 2200.0   # Region 2 M0

    b1_map_actual = torch.ones((img_size, img_size), device=device, dtype=torch.float32)
    if use_b1_map_simulation:
        print("Simulating with B1+ field variations.")
        y_coords = torch.linspace(-1, 1, img_size, device=device)
        x_coords = torch.linspace(-1, 1, img_size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # Simple linear gradient for B1 map (0.8 to 1.2 variation)
        b1_map_actual = 0.8 + 0.4 * (yy + 1) / 2

    b1_map_for_fitting = b1_map_actual if use_b1_map_fitting else None
    if use_b1_map_fitting and b1_map_for_fitting is None and use_b1_map_simulation :
        print("Warning: Simulating with B1 variation, but not providing B1 map to fitting algorithm.")
    elif not use_b1_map_simulation and use_b1_map_fitting:
        print("Warning: Not simulating B1 variation, but providing a B1 map to fitting. Using ideal B1 for fitting.")
        b1_map_for_fitting = torch.ones((img_size, img_size), device=device, dtype=torch.float32)


    # 3. Simulate VFA Signals
    print("Simulating VFA signals...")
    # Reshape for broadcasting: T1/M0 (1, H, W), FA (N_fa, 1, 1), B1_actual (1,H,W)
    _T1 = true_T1_ms.unsqueeze(0)
    _M0 = true_M0.unsqueeze(0)
    _FA_rad_nominal = (flip_angles_deg * math.pi / 180.0).view(-1, 1, 1)
    _B1_actual_bc = b1_map_actual.unsqueeze(0) # for broadcasting with FAs

    effective_FA_rad = _FA_rad_nominal * _B1_actual_bc # Element-wise if B1 map is used

    signals_clean = spgr_signal(_T1, _M0, effective_FA_rad, TR_ms)

    # Add Gaussian noise
    max_signal_val = torch.max(torch.abs(signals_clean))
    if max_signal_val == 0: max_signal_val = 1.0 # Avoid division by zero if signal is all zero
    noise_std_dev = (noise_level_percent / 100.0) * max_signal_val

    signals_noisy = signals_clean + torch.randn_like(signals_clean) * noise_std_dev
    print(f"Noisy signals shape: {signals_noisy.shape}")

    # 4. Run VFA T1 Fitting
    print("Running VFA T1 fitting...")
    start_time = time.time()

    # Note: fit_t1_vfa expects b1_map to have spatial_dims, not (1,H,W)
    # The unsqueeze for b1_map_bc happens inside fit_t1_vfa
    fitted_T1_map, fitted_M0_map = fit_t1_vfa(
        signals=signals_noisy,
        flip_angles_deg=flip_angles_deg,
        TR=TR_ms,
        b1_map=b1_map_for_fitting, # Pass the B1 map used for fitting (could be ideal or actual)
        initial_T1_ms_guess=900.0,
        initial_M0_guess=-1, # Auto-derive
        num_iterations=100, # Keep low for demo speed
        learning_rate=0.1, # Adam default often 0.001, but for these problems can be higher
        optimizer_type='adam',
        verbose=True, # Print loss updates
        device=device
    )
    end_time = time.time()
    print(f"Fitting completed in {end_time - start_time:.2f} seconds.")

    # 5. Display Results
    print("\n--- Results ---")
    print(f"Mean True T1: {torch.mean(true_T1_ms).item():.2f} ms")
    print(f"Mean Fitted T1: {torch.mean(fitted_T1_map).item():.2f} ms")
    print(f"Mean True M0: {torch.mean(true_M0).item():.2f}")
    print(f"Mean Fitted M0: {torch.mean(fitted_M0_map).item():.2f}")

    if MATPLOTLIB_AVAILABLE:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        im = axs[0,0].imshow(true_T1_ms.cpu().numpy(), cmap='viridis', vmin=0, vmax=2000)
        axs[0,0].set_title("True T1 (ms)")
        plt.colorbar(im, ax=axs[0,0], fraction=0.046, pad=0.04)

        im = axs[0,1].imshow(fitted_T1_map.cpu().numpy(), cmap='viridis', vmin=0, vmax=2000)
        axs[0,1].set_title("Fitted T1 (ms)")
        plt.colorbar(im, ax=axs[0,1], fraction=0.046, pad=0.04)

        t1_error_map = torch.abs(fitted_T1_map - true_T1_ms)
        im = axs[0,2].imshow(t1_error_map.cpu().numpy(), cmap='magma', vmin=0, vmax=500)
        axs[0,2].set_title("T1 Error Map (ms)")
        plt.colorbar(im, ax=axs[0,2], fraction=0.046, pad=0.04)

        im = axs[1,0].imshow(true_M0.cpu().numpy(), cmap='viridis', vmin=0, vmax=3000)
        axs[1,0].set_title("True M0")
        plt.colorbar(im, ax=axs[1,0], fraction=0.046, pad=0.04)

        im = axs[1,1].imshow(fitted_M0_map.cpu().numpy(), cmap='viridis', vmin=0, vmax=3000)
        axs[1,1].set_title("Fitted M0")
        plt.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)

        m0_error_map = torch.abs(fitted_M0_map - true_M0)
        im = axs[1,2].imshow(m0_error_map.cpu().numpy(), cmap='magma', vmin=0, vmax=500)
        axs[1,2].set_title("M0 Error Map")
        plt.colorbar(im, ax=axs[1,2], fraction=0.046, pad=0.04)

        if use_b1_map_simulation:
            fig_b1, ax_b1 = plt.subplots(1,1, figsize=(5,5))
            im_b1 = ax_b1.imshow(b1_map_actual.cpu().numpy(), cmap='coolwarm', vmin=0.8, vmax=1.2)
            ax_b1.set_title("Simulated B1+ Map (Actual FA / Nominal FA)")
            plt.colorbar(im_b1, ax=ax_b1, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        print("Plots displayed.")
    else:
        print("Plotting skipped as matplotlib is not available.")

    print("--- VFA T1 Mapping Example Completed ---")
