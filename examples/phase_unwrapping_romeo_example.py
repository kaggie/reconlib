import torch
import numpy as np
import matplotlib.pyplot as plt
from reconlib.phase_unwrapping import unwrap_phase_romeo # Placeholder
from reconlib.phase_unwrapping.utils import generate_mask_for_unwrapping
from reconlib.plotting import plot_phase_image, plot_unwrapped_phase_map # Using plot_unwrapped_phase_map for unwrapped

def run_phase_unwrapping_example():
    print("--- Running Phase Unwrapping ROMEO Example (Placeholder) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Create Dummy Data
    image_size = (64, 64)
    
    # Create a simple magnitude image (e.g., a circle)
    magnitude_np = np.zeros(image_size)
    center_y, center_x = image_size[0] // 2, image_size[1] // 2
    radius = image_size[0] // 3
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    magnitude_np[(y - center_y)**2 + (x - center_x)**2 <= radius**2] = 1.0

    # Create a simple wrapped phase image (e.g., a ramp that wraps multiple times)
    true_phase_np = np.linspace(-3*np.pi, 3*np.pi, image_size[1]).reshape(1, -1) # Range allowing multiple wraps
    true_phase_np = np.repeat(true_phase_np, image_size[0], axis=0)
    true_phase_np += np.linspace(-1*np.pi, 1*np.pi, image_size[0]).reshape(-1, 1) # Add some variation in y
    
    wrapped_phase_np = (true_phase_np + np.pi) % (2 * np.pi) - np.pi
    wrapped_phase_np *= magnitude_np # Apply mask to phase as well

    wrapped_phase_torch = torch.from_numpy(wrapped_phase_np).float().to(device)
    magnitude_torch = torch.from_numpy(magnitude_np).float().to(device)

    # 2. Generate Mask
    mask_np = generate_mask_for_unwrapping(magnitude_np, method='threshold', threshold_factor=0.05)
    mask_torch = torch.from_numpy(mask_np).bool().to(device) if mask_np is not None else None

    # 3. Call ROMEO phase unwrapping (placeholder)
    print("\nCalling ROMEO phase unwrapping (placeholder function)...")
    # Since unwrap_phase_romeo is a placeholder, it will just return the input.
    unwrapped_phase_torch = unwrap_phase_romeo(
        wrapped_phase_torch, 
        magnitude=magnitude_torch, 
        mask=mask_torch
    )
    unwrapped_phase_np = unwrapped_phase_torch.cpu().numpy()

    # 4. Plotting
    is_interactive = hasattr(plt, 'isinteractive') and plt.isinteractive()
    fn_w = "romeo_wrapped_phase.png" if not is_interactive else None
    fn_t = "romeo_true_unwrapped.png" if not is_interactive else None
    fn_p = "romeo_placeholder_unwrapped.png" if not is_interactive else None

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plot_phase_image(wrapped_phase_np * mask_np, title="Wrapped Phase (Masked)", filename=fn_w)
    
    plt.subplot(1, 3, 2)
    # True unwrapped phase for comparison
    plot_unwrapped_phase_map(true_phase_np * mask_np, title="True Unwrapped Phase (Masked)", filename=fn_t)

    plt.subplot(1, 3, 3)
    # Since ROMEO is a placeholder, this will look like the wrapped phase.
    plot_unwrapped_phase_map(unwrapped_phase_np * mask_np, title="ROMEO Unwrapped (Placeholder, Masked)", filename=fn_p)
    
    plt.suptitle("Phase Unwrapping ROMEO Example Results")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if is_interactive:
        plt.show()
    else:
        plt.savefig("romeo_unwrapping_summary.png")
        plt.close()
        print(f"Plots saved: {fn_w}, {fn_t}, {fn_p}, romeo_unwrapping_summary.png (if not interactive)")

    print("\n--- Phase Unwrapping ROMEO Example Finished ---")

if __name__ == "__main__":
    run_phase_unwrapping_example()
