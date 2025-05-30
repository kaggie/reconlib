import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.modalities.ultrasound.operators import UltrasoundForwardOperator
from reconlib.modalities.ultrasound.reconstructors import das_reconstruction, inverse_reconstruction_pg
# from reconlib.modalities.ultrasound.utils import compute_and_apply_voronoi_weights_to_echo_data # If used

def generate_simple_phantom(image_shape, device='cpu'):
    """ Creates a simple phantom with a few point scatterers. """
    phantom = torch.zeros(image_shape, dtype=torch.complex64, device=device)
    h, w = image_shape
    # Add some scatterers
    phantom[h // 2, w // 2] = 1.0
    phantom[h // 4, w // 4] = 0.8
    phantom[h // 2 + h // 8, w // 2 - w // 8] = 0.6j # Complex reflectivity
    phantom[h // 3, int(w * 0.75)] = 0.7 - 0.5j
    return phantom

def visualize_results(original, das_recon, inv_recon_l1, inv_recon_l2, inv_recon_settings):
    """ Basic visualization of original phantom and reconstructions. """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Determine global min/max for consistent scaling across reconstructions if desired
    # For simplicity, individual scaling for now.

    axes[0,0].imshow(torch.abs(original).cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0,0].set_title('Original Phantom (Magnitude)')
    axes[0,0].axis('off')

    axes[0,1].imshow(torch.abs(das_recon).cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0,1].set_title('DAS Recon (Adjoint)')
    axes[0,1].axis('off')

    axes[1,0].imshow(torch.abs(inv_recon_l1).cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1,0].set_title(f"Inverse Recon (L1, {inv_recon_settings['l1']['iters']} iters, $\lambda$={inv_recon_settings['l1']['lambda']})")
    axes[1,0].axis('off')

    axes[1,1].imshow(torch.abs(inv_recon_l2).cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1,1].set_title(f"Inverse Recon (L2, {inv_recon_settings['l2']['iters']} iters, $\lambda$={inv_recon_settings['l2']['lambda']})")
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Phantom
    image_h, image_w = 128, 128 # Increased size for better visual
    phantom = generate_simple_phantom((image_h, image_w), device=device)

    # 2. Ultrasound Operator Setup
    # These parameters should ideally be chosen to suit the phantom size and desired resolution
    operator_params = {
        'image_shape': (image_h, image_w),
        'sound_speed': 1540.0,
        'num_elements': 64, # Number of transducer elements
        'sampling_rate': 40e6, # 40 MHz
        'num_samples': 1024,   # Number of time samples per echo
        'image_spacing': (0.0002, 0.0002), # 0.2 mm pixels (adjust based on image_h, image_w)
        'device': device
    }
    # Example element_positions: linear array centered above image
    elem_pitch = 0.0003 # 0.3mm
    array_width = (operator_params['num_elements'] - 1) * elem_pitch
    x_coords = torch.linspace(-array_width / 2, array_width / 2, operator_params['num_elements'], device=device)
    # Position elements slightly "above" the image region (e.g., at y = -5mm)
    # Assuming image spans from y=0 to y_max.
    y_pos_elements = -0.005
    operator_params['element_positions'] = torch.stack(
        (x_coords, torch.full_like(x_coords, y_pos_elements)), dim=1
    )

    us_operator = UltrasoundForwardOperator(**operator_params)
    print("UltrasoundForwardOperator initialized.")

    # 3. Simulate Echo Data
    print("Simulating echo data...")
    echo_data = us_operator.op(phantom)
    print(f"Simulated echo data shape: {echo_data.shape}")

    # Add some noise (optional)
    noise_level = 0.05 * torch.mean(torch.abs(echo_data)) # 5% of mean signal amplitude
    echo_data_noisy = echo_data + noise_level * (
        torch.randn_like(echo_data) + 1j * torch.randn_like(echo_data)
    )
    print("Added complex Gaussian noise to echo data.")

    target_echo_data = echo_data_noisy

    # 4. Adjoint Reconstruction (DAS)
    print("\nPerforming DAS reconstruction...")
    das_image = das_reconstruction(target_echo_data, us_operator)
    print(f"DAS reconstructed image shape: {das_image.shape}")

    # 5. Inverse Reconstruction (Proximal Gradient)
    # Common settings
    pg_iterations = 30 # More iterations for potentially better results
    pg_step_size = 0.05 # May need tuning

    # L1 Regularization
    lambda_l1 = 0.001 # May need tuning
    print(f"\nPerforming Inverse Reconstruction (L1, lambda={lambda_l1})...")
    inv_recon_l1 = inverse_reconstruction_pg(
        echo_data=target_echo_data,
        ultrasound_operator=us_operator,
        regularizer_type='l1',
        lambda_reg=lambda_l1,
        iterations=pg_iterations,
        step_size=pg_step_size,
        verbose=True
    )
    print(f"L1 Inverse reconstructed image shape: {inv_recon_l1.shape}")

    # L2 Regularization
    lambda_l2 = 0.01 # May need tuning
    print(f"\nPerforming Inverse Reconstruction (L2, lambda={lambda_l2})...")
    inv_recon_l2 = inverse_reconstruction_pg(
        echo_data=target_echo_data,
        ultrasound_operator=us_operator,
        regularizer_type='l2',
        lambda_reg=lambda_l2,
        iterations=pg_iterations,
        step_size=pg_step_size,
        verbose=True
    )
    print(f"L2 Inverse reconstructed image shape: {inv_recon_l2.shape}")

    # Store settings for visualization title
    inv_recon_settings = {
        'l1': {'iters': pg_iterations, 'lambda': lambda_l1},
        'l2': {'iters': pg_iterations, 'lambda': lambda_l2}
    }

    # 6. Visualization
    print("\nVisualizing results...")
    visualize_results(phantom, das_image, inv_recon_l1, inv_recon_l2, inv_recon_settings)

    print("\nUltrasound reconstruction example finished.")

if __name__ == '__main__':
    main()
