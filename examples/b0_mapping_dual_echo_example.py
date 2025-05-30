import torch
import numpy as np
import matplotlib.pyplot as plt
from reconlib.b0_mapping import calculate_b0_map_dual_echo, calculate_b0_map_multi_echo_linear_fit
from reconlib.b0_mapping.utils import create_mask_from_magnitude
from reconlib.plotting import plot_phase_image, plot_b0_field_map

def run_b0_mapping_example():
    print("--- Running B0 Mapping Dual Echo Example ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Create Dummy Data
    image_size = (64, 64)
    num_echoes = 3
    echo_times_sec = [0.002, 0.004, 0.006]  # seconds

    # Create a simple magnitude image (e.g., a circle)
    magnitude = np.zeros(image_size)
    center_y, center_x = image_size[0] // 2, image_size[1] // 2
    radius = image_size[0] // 3
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    magnitude[(y - center_y)**2 + (x - center_x)**2 <= radius**2] = 1.0
    
    # Create a simple B0 field (e.g., a linear gradient in Hz)
    b0_true_hz = np.zeros(image_size)
    b0_true_hz += (np.linspace(-50, 50, image_size[1]).reshape(1, -1)) # Gradient along x
    b0_true_hz[magnitude == 0] = 0 # Apply only within the object

    # Simulate phase images: phase = 2 * pi * B0_hz * TE
    phase_images_np = np.zeros((num_echoes,) + image_size)
    for i in range(num_echoes):
        phase_images_np[i,...] = (2 * np.pi * b0_true_hz * echo_times_sec[i]) * magnitude
    
    phase_images_torch = torch.from_numpy(phase_images_np).float().to(device)
    echo_times_torch = torch.tensor(echo_times_sec).float().to(device)
    # magnitude_torch = torch.from_numpy(magnitude).float().to(device) # Not directly used by b0 funcs

    # 2. Create Mask
    mask_np = create_mask_from_magnitude(magnitude, threshold_factor=0.05)
    mask_torch = torch.from_numpy(mask_np).bool().to(device)

    # 3. Calculate B0 map using dual-echo method (first two echoes)
    print("\nCalculating B0 map using dual-echo method...")
    # The calculate_b0_map_dual_echo function specifically requires two echoes.
    # We select the first two phase images and corresponding echo times.
    b0_map_dual_echo_torch = calculate_b0_map_dual_echo(
        phase_images_torch[:2,...], echo_times_torch[:2], mask=mask_torch
    )
    b0_map_dual_echo_np = b0_map_dual_echo_torch.cpu().numpy()

    # 4. Calculate B0 map using multi-echo linear fit
    print("\nCalculating B0 map using multi-echo linear fit...")
    b0_map_multi_echo_torch = calculate_b0_map_multi_echo_linear_fit(
        phase_images_torch, echo_times_torch, mask=mask_torch
    )
    b0_map_multi_echo_np = b0_map_multi_echo_torch.cpu().numpy()

    # 5. Plotting
    is_interactive = hasattr(plt, 'isinteractive') and plt.isinteractive()
    
    plt.figure(figsize=(18, 12))
    fn_p1 = "b0_phase1.png" if not is_interactive else None
    fn_p2 = "b0_phase2.png" if not is_interactive else None
    fn_t = "b0_true.png" if not is_interactive else None
    fn_d = "b0_dual_echo.png" if not is_interactive else None
    fn_m = "b0_multi_echo.png" if not is_interactive else None
    
    plt.subplot(2, 3, 1)
    plot_phase_image(phase_images_np[0,...] * mask_np, title="Phase Image (TE1)", filename=fn_p1)
    
    plt.subplot(2, 3, 2)
    plot_phase_image(phase_images_np[1,...] * mask_np, title="Phase Image (TE2)", filename=fn_p2)

    plt.subplot(2, 3, 3)
    plot_b0_field_map(b0_true_hz * mask_np, title="True B0 Map (masked)", filename=fn_t)

    plt.subplot(2, 3, 4)
    plot_b0_field_map(b0_map_dual_echo_np * mask_np, title="B0 Map (Dual-Echo, masked)", filename=fn_d)
    
    plt.subplot(2, 3, 5)
    plot_b0_field_map(b0_map_multi_echo_np * mask_np, title="B0 Map (Multi-Echo Fit, masked)", filename=fn_m)
    
    plt.suptitle("B0 Mapping Example Results")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if is_interactive:
        plt.show()
    else:
        plt.savefig("b0_mapping_summary.png")
        plt.close()
        print(f"Plots saved: {fn_p1}, {fn_p2}, {fn_t}, {fn_d}, {fn_m}, b0_mapping_summary.png (if not interactive)")

    print("\n--- B0 Mapping Example Finished ---")

if __name__ == "__main__":
    run_b0_mapping_example()
