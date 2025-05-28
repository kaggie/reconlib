# reconlib/plotting.py
"""Module for visualization tasks in MRI reconstruction."""

import numpy as np
import matplotlib.pyplot as plt

def plot_phase_image(phase_image: np.ndarray, title: str = "Phase Image", cmap: str = "twilight", vmin: float = -np.pi, vmax: float = np.pi, filename: str = None):
    """
    Displays or saves a 2D phase image.

    Args:
        phase_image (np.ndarray): The 2D phase data (in radians).
        title (str, optional): Title of the plot. Defaults to "Phase Image".
        cmap (str, optional): Colormap for the plot. Defaults to "twilight".
        vmin (float, optional): Minimum value for the color scale. Defaults to -np.pi.
        vmax (float, optional): Maximum value for the color scale. Defaults to np.pi.
        filename (str, optional): If provided, saves the figure to this path instead of showing.
    """
    if not isinstance(phase_image, np.ndarray) or phase_image.ndim != 2:
        raise ValueError("phase_image must be a 2D NumPy array.")

    plt.figure()
    plt.imshow(phase_image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Phase (radians)")
    plt.title(title)
    plt.axis('off') # Optional: to turn off axis numbers and ticks
    if filename:
        plt.savefig(filename)
        plt.close() # Close the figure to free memory when saving
    else:
        plt.show()

def plot_unwrapped_phase_map(unwrapped_phase_map: np.ndarray, title: str = "Unwrapped Phase Map", cmap: str = "viridis", filename: str = None):
    """
    Displays or saves a 2D unwrapped phase map.

    Args:
        unwrapped_phase_map (np.ndarray): The 2D unwrapped phase data.
        title (str, optional): Title of the plot. Defaults to "Unwrapped Phase Map".
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
        filename (str, optional): If provided, saves the figure to this path instead of showing.
    """
    if not isinstance(unwrapped_phase_map, np.ndarray) or unwrapped_phase_map.ndim != 2:
        raise ValueError("unwrapped_phase_map must be a 2D NumPy array.")

    plt.figure()
    plt.imshow(unwrapped_phase_map, cmap=cmap)
    plt.colorbar(label="Unwrapped Phase (radians)")
    plt.title(title)
    plt.axis('off')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_b0_field_map(b0_map: np.ndarray, title: str = "B0 Field Map", cmap: str = "coolwarm", center_zero: bool = True, filename: str = None):
    """
    Displays or saves a 2D B0 field map.

    Args:
        b0_map (np.ndarray): The 2D B0 field map data (e.g., in Hz).
        title (str, optional): Title of the plot. Defaults to "B0 Field Map".
        cmap (str, optional): Colormap for the plot. Defaults to "coolwarm".
        center_zero (bool, optional): If True, centers the colormap around zero. Defaults to True.
        filename (str, optional): If provided, saves the figure to this path instead of showing.
    """
    if not isinstance(b0_map, np.ndarray) or b0_map.ndim != 2:
        raise ValueError("b0_map must be a 2D NumPy array.")

    plt.figure()
    
    vmin, vmax = None, None
    if center_zero:
        if b0_map.size > 0: # Ensure b0_map is not empty
            abs_max = np.max(np.abs(b0_map))
            if abs_max > 1e-9: # Avoid issues if map is effectively all zeros
                vmin, vmax = -abs_max, abs_max
            else: # Default for an all-zero or near-zero map
                vmin, vmax = -1, 1 
        else: # Handle empty array case
            vmin, vmax = -1, 1


    im = plt.imshow(b0_map, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="B0 offset (e.g., Hz)")
    plt.title(title)
    plt.axis('off')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
