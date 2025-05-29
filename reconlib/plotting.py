# reconlib/plotting.py
"""Module for visualization tasks in MRI reconstruction."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import Union, List, Optional, Dict, Any # Added for new functions

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


def plot_projection_data(projection_data: np.ndarray, title: str = "Projection Data",
                         aspect_ratio: str = 'auto', cmap: str = 'viridis',
                         filename: Optional[str] = None):
    """
    Displays or saves 2D projection data (e.g., a sinogram).
    Assumes projection_data is a 2D NumPy array.

    Args:
        projection_data (np.ndarray): The 2D projection data.
        title (str, optional): Title of the plot. Defaults to "Projection Data".
        aspect_ratio (str, optional): Aspect ratio for imshow. Defaults to 'auto'.
                                      Use 'equal' for square pixels.
        cmap (str, optional): Colormap for the plot. Defaults to 'viridis'.
        filename (Optional[str], optional): If provided, saves the figure. Otherwise, shows it.
    """
    if not isinstance(projection_data, np.ndarray) or projection_data.ndim != 2:
        raise ValueError("projection_data must be a 2D NumPy array.")

    plt.figure()
    plt.imshow(projection_data, cmap=cmap, aspect=aspect_ratio)
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.xlabel("Detector Bin")
    plt.ylabel("Angle/View")
    # plt.gca().set_aspect(aspect_ratio) # Covered by imshow's aspect argument

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _get_slice_indices(num_total_slices: int, num_slices_to_display: Optional[int] = None, specific_indices: Optional[List[int]] = None) -> List[int]:
    """Helper function to determine slice indices for display."""
    if specific_indices:
        # Validate specific indices
        valid_indices = [idx for idx in specific_indices if 0 <= idx < num_total_slices]
        if not valid_indices:
            # Fallback if no valid specific indices are provided
            print(f"Warning: Provided specific_indices {specific_indices} are out of range for {num_total_slices} slices. Defaulting.")
            return list(np.linspace(0, num_total_slices - 1, min(3, num_total_slices), dtype=int))
        return sorted(list(set(valid_indices))) # Ensure unique and sorted

    if num_slices_to_display is None or num_slices_to_display <= 0:
        num_slices_to_display = min(3, num_total_slices) # Default to 3 or fewer if not enough slices

    num_slices_to_display = min(num_slices_to_display, num_total_slices)
    if num_slices_to_display == 0:
        return []
        
    # Linspace handles cases where num_total_slices < num_slices_to_display by returning fewer points
    indices = np.linspace(0, num_total_slices - 1, num_slices_to_display, dtype=int).tolist()
    return sorted(list(set(indices))) # Ensure unique and sorted, handles num_slices_to_display > num_total_slices


def visualize_reconstruction(image: np.ndarray,
                             slice_info: Optional[Union[int, List[int], Dict[int, Union[int, List[int]]]]] = None,
                             cmap: str = 'gray',
                             main_title: str = "Reconstructed Image",
                             row_titles: Optional[List[str]] = None, # e.g., ["Axial", "Sagittal", "Coronal"]
                             filename: Optional[str] = None):
    """
    Displays or saves slices of a 2D or 3D reconstructed image.

    Args:
        image (np.ndarray): The 2D or 3D image data. Assumes (H, W) for 2D,
                            and (Depth, Height, Width) or (Slice, H, W) for 3D.
                            Order of axes for 3D (e.g. ZYX, XYZ) should be consistent.
        slice_info (Optional[Union[int, List[int], Dict[int, Union[int, List[int]]]]], optional):
            Determines how slices are displayed for 3D images.
            - If None (default for 3D): Show N=3 evenly spaced slices for each of the 3 axes (0, 1, 2).
            - If int (e.g., `slice_info=5`): Show 5 evenly spaced slices for each of the 3 axes.
            - If List[int] (e.g., `slice_info=[10, 20, 30]`): Interpreted as slice indices for axis 0.
            - If Dict[int, Union[int, List[int]]] (e.g., `{0: 10, 1: [15, 25], 2: 3}`):
                Keys are axis indices (0, 1, 2).
                Values can be an int (number of evenly spaced slices) or a List[int] (specific slice indices for that axis).
            For 2D images, this parameter is ignored.
        cmap (str, optional): Colormap for the images. Defaults to 'gray'.
        main_title (str, optional): Overall title for the figure. Defaults to "Reconstructed Image".
        row_titles (Optional[List[str]], optional): Titles for rows if multiple axes are plotted.
                                                   Defaults to ["Slices along Axis 0", ...].
        filename (Optional[str], optional): If provided, saves the figure. Otherwise, shows it.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if image.ndim == 2:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap=cmap)
        plt.colorbar()
        plt.title(main_title)
        plt.axis('off')
    elif image.ndim == 3:
        num_axes_to_plot = 0
        slices_to_plot_per_axis: Dict[int, List[int]] = {}
        axis_titles_map = ["Slices along Axis 0 (e.g., Axial)",
                           "Slices along Axis 1 (e.g., Sagittal)",
                           "Slices along Axis 2 (e.g., Coronal)"]
        if row_titles:
            axis_titles_map = [row_titles[i] if i < len(row_titles) else axis_titles_map[i] for i in range(3)]


        if slice_info is None: # Default: 3 slices for each of the 3 axes
            num_axes_to_plot = 3
            for axis_idx in range(3):
                if image.shape[axis_idx] > 0: # Check if axis has size
                    slices_to_plot_per_axis[axis_idx] = _get_slice_indices(image.shape[axis_idx], 3)
                else: # Axis has size 0, cannot plot
                    slices_to_plot_per_axis[axis_idx] = []

        elif isinstance(slice_info, int): # N evenly spaced slices for each of the 3 axes
            num_axes_to_plot = 3
            for axis_idx in range(3):
                if image.shape[axis_idx] > 0:
                    slices_to_plot_per_axis[axis_idx] = _get_slice_indices(image.shape[axis_idx], slice_info)
                else:
                    slices_to_plot_per_axis[axis_idx] = []

        elif isinstance(slice_info, list): # Specific slices for axis 0
            num_axes_to_plot = 1
            if image.shape[0] > 0:
                 slices_to_plot_per_axis[0] = _get_slice_indices(image.shape[0], specific_indices=slice_info)
            else:
                 slices_to_plot_per_axis[0] = []


        elif isinstance(slice_info, dict):
            num_axes_to_plot = len(slice_info)
            for axis_idx, s_info in slice_info.items():
                if not (0 <= axis_idx < 3):
                    print(f"Warning: Invalid axis index {axis_idx} in slice_info. Skipping.")
                    num_axes_to_plot -=1
                    continue
                if image.shape[axis_idx] == 0:
                    slices_to_plot_per_axis[axis_idx] = []
                    continue

                if isinstance(s_info, int):
                    slices_to_plot_per_axis[axis_idx] = _get_slice_indices(image.shape[axis_idx], s_info)
                elif isinstance(s_info, list):
                    slices_to_plot_per_axis[axis_idx] = _get_slice_indices(image.shape[axis_idx], specific_indices=s_info)
                else:
                    print(f"Warning: Invalid slice info for axis {axis_idx}. Defaulting.")
                    slices_to_plot_per_axis[axis_idx] = _get_slice_indices(image.shape[axis_idx], 3)
        else:
            raise ValueError("Invalid slice_info format.")

        # Filter out axes with no valid slices
        active_axes_to_plot = sorted([axis for axis, slices in slices_to_plot_per_axis.items() if slices])
        num_active_axes = len(active_axes_to_plot)

        if num_active_axes == 0:
            print("Warning: No valid slices found to display for the 3D image based on slice_info.")
            # Optionally plot a placeholder or return
            plt.figure()
            plt.text(0.5, 0.5, "No slices to display", ha='center', va='center')
            plt.title(main_title)
            if filename: plt.savefig(filename); plt.close()
            else: plt.show()
            return

        max_cols = 0
        for axis_idx in active_axes_to_plot:
            max_cols = max(max_cols, len(slices_to_plot_per_axis[axis_idx]))
        
        if max_cols == 0: # Should be caught by num_active_axes == 0, but as safeguard
            print("Warning: Max columns for subplots is 0. No slices to display.")
            # ... (similar placeholder plot as above)
            plt.figure()
            plt.text(0.5, 0.5, "No slices to display (max_cols is 0)", ha='center', va='center')
            plt.title(main_title)
            if filename: plt.savefig(filename); plt.close()
            else: plt.show()
            return


        fig, axes = plt.subplots(num_active_axes, max_cols, figsize=(max_cols * 4, num_active_axes * 4), squeeze=False)
        fig.suptitle(main_title, fontsize=16)

        for row, axis_idx in enumerate(active_axes_to_plot):
            current_axis_slices = slices_to_plot_per_axis[axis_idx]
            axes[row, 0].set_ylabel(axis_titles_map[axis_idx], fontsize=12) # Row title

            for col, slice_idx in enumerate(current_axis_slices):
                ax = axes[row, col]
                if axis_idx == 0: # Slice along Z (depth) -> (H, W)
                    slice_data = image[slice_idx, :, :]
                elif axis_idx == 1: # Slice along Y (height) -> (D, W), transpose for display (W, D)
                    slice_data = image[:, slice_idx, :].T
                elif axis_idx == 2: # Slice along X (width) -> (D, H), transpose for display (H, D)
                    slice_data = image[:, :, slice_idx].T
                else:
                    continue # Should not happen

                im = ax.imshow(slice_data, cmap=cmap, origin='lower') # origin='lower' is common for medical images
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')
                # Add a colorbar to each subplot. For shared colorbar, it's more complex.
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


            # Hide unused subplots in the row
            for col in range(len(current_axis_slices), max_cols):
                axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}. Must be 2 or 3.")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_voronoi_diagram_2d(points: np.ndarray, ax: plt.Axes = None, 
                            show_points: bool = True, line_colors='k', 
                            line_width: float = 1.0, point_kwargs: dict = None, 
                            **kwargs_plot_2d) -> plt.Axes:
    """
    Plots a 2D Voronoi diagram for a given set of points.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) representing N points in 2D.
        ax (plt.Axes, optional): Matplotlib axes object to plot on. If None, a new
                                 figure and axes are created. Defaults to None.
        show_points (bool, optional): Whether to display the input points on the diagram.
                                      Defaults to True.
        line_colors (str or sequence, optional): Color(s) for the Voronoi cell edges.
                                                 Defaults to 'k' (black).
        line_width (float, optional): Width of the Voronoi cell edges. Defaults to 1.0.
        point_kwargs (dict, optional): Dictionary of keyword arguments to pass to `ax.plot`
                                       for customizing the appearance of points if `show_points`
                                       is True and these kwargs are provided.
                                       Example: {'marker':'x', 's':50, 'c':'red'}.
                                       Note: `voronoi_plot_2d` uses `plt.plot` for points, so
                                       `s` (size) is not a valid kwarg here, use `markersize`.
                                       If point_kwargs are given, points are plotted manually.
        **kwargs_plot_2d: Additional keyword arguments to pass to `scipy.spatial.voronoi_plot_2d`.
                          Common arguments include `show_vertices` (bool).

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.

    Raises:
        ValueError: If `points` is not a 2D NumPy array with at least 3 points.
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Input 'points' must be a NumPy array.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input 'points' must be a 2D array with shape (N, 2).")
    if points.shape[0] < 3: # Qhull requires at least D+1 points for D-dimensional space. For 2D, this is 3 points.
        raise ValueError("Input 'points' must have at least 3 points to compute a Voronoi diagram.")

    # Compute Voronoi diagram
    vor = Voronoi(points)

    if ax is None:
        fig, ax = plt.subplots()

    # Default for show_vertices if not in kwargs_plot_2d
    if 'show_vertices' not in kwargs_plot_2d:
        kwargs_plot_2d['show_vertices'] = False
        
    # Handle custom point plotting
    plot_points_manually = show_points and point_kwargs is not None
    
    # If plotting points manually, tell voronoi_plot_2d not to plot them initially.
    effective_show_points_for_voronoi_plot = show_points and not plot_points_manually

    voronoi_plot_2d(vor, ax=ax, 
                    show_points=effective_show_points_for_voronoi_plot, 
                    line_colors=line_colors, 
                    line_width=line_width, 
                    **kwargs_plot_2d)

    if plot_points_manually:
        # Ensure 'marker' is set if not provided, 'o' is a common default.
        # `voronoi_plot_2d` defaults to 'o' if it plots points.
        # `ax.plot` also defaults to solid lines if no marker is specified, so we add 'o'.
        current_point_kwargs = point_kwargs.copy()
        if 'marker' not in current_point_kwargs and 'linestyle' not in current_point_kwargs:
             # if no marker and no linestyle, ax.plot defaults to a line, we want points.
             if not any(key in current_point_kwargs for key in ['ls', 'linestyle']):
                current_point_kwargs.setdefault('linestyle', 'None') # Prevent lines
             current_point_kwargs.setdefault('marker', 'o') # Default to 'o' if not specified and no line

        ax.plot(vor.points[:,0], vor.points[:,1], **current_point_kwargs)
    
    return ax

def plot_density_weights_2d(weights_matrix: np.ndarray, ax: plt.Axes = None, 
                              title: str = "Density Compensation Weights", 
                              cmap: str = "viridis", 
                              colorbar_label: str = "Weight", 
                              **kwargs_imshow) -> plt.Axes:
    """
    Plots a 2D density compensation weights matrix.

    Args:
        weights_matrix (np.ndarray): The 2D NumPy array of weights to plot.
        ax (plt.Axes, optional): Matplotlib axes object to plot on. If None, a new
                                 figure and axes are created. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Density Compensation Weights".
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
        colorbar_label (str, optional): Label for the colorbar. Defaults to "Weight".
        **kwargs_imshow: Additional keyword arguments to pass to `ax.imshow()`.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.

    Raises:
        ValueError: If `weights_matrix` is not a 2D NumPy array.
    """
    if not isinstance(weights_matrix, np.ndarray):
        raise ValueError("Input 'weights_matrix' must be a NumPy array.")
    if weights_matrix.ndim != 2:
        raise ValueError("Input 'weights_matrix' must be a 2D array.")

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(weights_matrix, cmap=cmap, **kwargs_imshow)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.axis('off')  # Turn off axis ticks and labels

    return ax

def plot_voronoi_diagram_3d_slice(points: np.ndarray, slice_axis: int = 2, slice_coord: float = 0.0, 
                                  ax: plt.Axes = None, show_points: bool = False, 
                                  line_color: str = 'k', line_width: float = 1.0, 
                                  point_marker: str = 'o', point_color: str = 'r', 
                                  point_size: float = 20) -> plt.Axes:
    """
    Plots a 2D slice of a 3D Voronoi diagram.

    Args:
        points (np.ndarray): Shape (N, 3) for N points in 3D.
        slice_axis (int): Axis to slice along (0 for X, 1 for Y, 2 for Z). Defaults to 2.
        slice_coord (float): Coordinate value on the slice_axis for the slice. Defaults to 0.0.
        ax (plt.Axes, optional): Matplotlib axes to plot on. Defaults to None.
        show_points (bool): Whether to show original 3D points that lie on the slice. Defaults to False.
        line_color (str): Color for the Voronoi cell edges on the slice. Defaults to 'k'.
        line_width (float): Width of the Voronoi cell edges. Defaults to 1.0.
        point_marker (str): Marker for the projected points. Defaults to 'o'.
        point_color (str): Color for the projected points. Defaults to 'r'.
        point_size (float): Size for the projected points. Defaults to 20.

    Returns:
        plt.Axes: The Matplotlib axes object with the plot.

    Raises:
        ValueError: For invalid inputs.
    """
    from matplotlib.collections import LineCollection # Specific import

    EPSILON = 1e-7 # Small value for float comparisons

    if not isinstance(points, np.ndarray):
        raise ValueError("Input 'points' must be a NumPy array.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input 'points' must be a 3D array with shape (N, 3).")
    if points.shape[0] < 4: # Need at least D+1 points for Voronoi in D dimensions
        raise ValueError("Input 'points' must have at least 4 points for a 3D Voronoi diagram.")
    if slice_axis not in [0, 1, 2]:
        raise ValueError("Input 'slice_axis' must be 0, 1, or 2.")
    if not isinstance(slice_coord, (int, float)):
        raise ValueError("Input 'slice_coord' must be a float or integer.")

    vor = Voronoi(points)
    segments_2d = []

    # Axes for 2D plot
    plot_axes_indices = [i for i in range(3) if i != slice_axis]

    for ridge_vertex_indices in vor.ridge_vertices:
        if -1 in ridge_vertex_indices: # Skip ridges involving vertices at infinity
            continue

        ridge_3d_vertices = vor.vertices[ridge_vertex_indices]
        
        # Store 3D intersection points for the current ridge polygon
        current_ridge_intersections_3d = []

        for i in range(len(ridge_3d_vertices)):
            v0 = ridge_3d_vertices[i]
            v1 = ridge_3d_vertices[(i + 1) % len(ridge_3d_vertices)] # Wrap around

            val0 = v0[slice_axis] - slice_coord
            val1 = v1[slice_axis] - slice_coord

            # Case 1: Edge lies in the plane
            if abs(val0) < EPSILON and abs(val1) < EPSILON:
                current_ridge_intersections_3d.append(v0)
                current_ridge_intersections_3d.append(v1)
                continue # This edge is fully on the plane

            # Case 2: One vertex on the plane, the other not (and edge not parallel)
            # (Handled by Case 3 if the other point makes the edge cross,
            #  or if it's an endpoint of an in-plane segment)
            # Add points that are exactly on the plane
            if abs(val0) < EPSILON:
                current_ridge_intersections_3d.append(v0)
            
            # Case 3: Edge crosses the plane (val0 and val1 have opposite signs)
            # Ensure not both are zero (already handled by Case 1)
            # Ensure edge is not parallel to the plane but outside it
            if val0 * val1 < -EPSILON * EPSILON : # Strictly opposite signs
                # Denominator for t: (v1[slice_axis] - v0[slice_axis])
                # This is (val1 + slice_coord) - (val0 + slice_coord) = val1 - val0
                denominator = v1[slice_axis] - v0[slice_axis]
                if abs(denominator) < EPSILON: # Should not happen if val0*val1 < 0
                    continue 
                
                t = (slice_coord - v0[slice_axis]) / denominator
                if 0 <= t <= 1: # Intersection point lies within the segment v0-v1
                    intersection_3d = v0 + t * (v1 - v0)
                    current_ridge_intersections_3d.append(intersection_3d)
        
        # Remove duplicate 3D points (simple method, might need refinement for precision)
        if not current_ridge_intersections_3d:
            continue
            
        unique_intersections_3d_tuples = sorted(list(set(tuple(row) for row in current_ridge_intersections_3d)))
        unique_intersections_3d = [np.array(pt) for pt in unique_intersections_3d_tuples]


        if len(unique_intersections_3d) >= 2:
            # Project these 3D intersection points to 2D
            projected_points_2d = [np.delete(pt, slice_axis) for pt in unique_intersections_3d]

            # If exactly 2 points, that's our segment
            if len(projected_points_2d) == 2:
                segments_2d.append(projected_points_2d)
            elif len(projected_points_2d) > 2:
                # More than 2 points: they form a polygon on the slice.
                # We need its edges as segments.
                # Sort points by angle around centroid to form a convex polygon
                # This is a common way to order points of a convex polygon.
                centroid_2d = np.mean(projected_points_2d, axis=0)
                
                def sort_key_angle(point):
                    # Angle of point relative to centroid
                    return np.arctan2(point[1] - centroid_2d[1], point[0] - centroid_2d[0])
                
                sorted_projected_points_2d = sorted(projected_points_2d, key=sort_key_angle)
                
                for i in range(len(sorted_projected_points_2d)):
                    p_start = sorted_projected_points_2d[i]
                    p_end = sorted_projected_points_2d[(i + 1) % len(sorted_projected_points_2d)]
                    segments_2d.append([p_start, p_end])

    if ax is None:
        fig, ax = plt.subplots()

    # Set axis labels
    axis_labels = ['X', 'Y', 'Z']
    ax.set_xlabel(axis_labels[plot_axes_indices[0]])
    ax.set_ylabel(axis_labels[plot_axes_indices[1]])
    ax.set_title(f"Voronoi Diagram Slice at {axis_labels[slice_axis]} = {slice_coord:.2f}")

    if segments_2d:
        lc = LineCollection(segments_2d, colors=line_color, linewidths=line_width)
        ax.add_collection(lc)
        ax.autoscale_view() # Rescale view to include the lines
    else: # Handle case with no segments found
        ax.autoscale_view()


    if show_points:
        # Filter points that are close to the slice_coord along slice_axis
        on_slice_mask = np.abs(points[:, slice_axis] - slice_coord) < EPSILON
        points_on_slice_3d = points[on_slice_mask]
        
        if points_on_slice_3d.shape[0] > 0:
            points_on_slice_2d = np.delete(points_on_slice_3d, slice_axis, axis=1)
            ax.scatter(points_on_slice_2d[:, 0], points_on_slice_2d[:, 1], 
                       marker=point_marker, color=point_color, s=point_size, zorder=5) # zorder to plot on top

    ax.set_aspect('equal') # Ensure aspect ratio is equal for correct visualization
    return ax

def plot_density_weights_3d_slice(weights_volume: np.ndarray, slice_axis: int = 2, 
                                  slice_index: int = 0, ax: plt.Axes = None, 
                                  title: str = "3D Density Weights Slice", 
                                  cmap: str = "viridis", 
                                  colorbar_label: str = "Weight", 
                                  **kwargs_imshow) -> plt.Axes:
    """
    Plots a 2D slice of a 3D density compensation weights volume.

    Args:
        weights_volume (np.ndarray): The 3D NumPy array of weights.
        slice_axis (int): Axis to slice along (0 for X, 1 for Y, 2 for Z). Defaults to 2.
        slice_index (int): Index along the slice_axis for the slice. Defaults to 0.
        ax (plt.Axes, optional): Matplotlib axes object to plot on. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "3D Density Weights Slice".
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
        colorbar_label (str, optional): Label for the colorbar. Defaults to "Weight".
        **kwargs_imshow: Additional keyword arguments to pass to `ax.imshow()`.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.

    Raises:
        ValueError: If inputs are invalid (e.g., not a 3D NumPy array, invalid slice_axis or slice_index).
    """
    if not isinstance(weights_volume, np.ndarray):
        raise ValueError("Input 'weights_volume' must be a NumPy array.")
    if weights_volume.ndim != 3:
        raise ValueError("Input 'weights_volume' must be a 3D array.")
    
    if slice_axis not in [0, 1, 2]:
        raise ValueError("Input 'slice_axis' must be 0, 1, or 2.")
    
    if not (0 <= slice_index < weights_volume.shape[slice_axis]):
        raise ValueError(f"Input 'slice_index' {slice_index} is out of bounds for axis {slice_axis} "
                         f"with size {weights_volume.shape[slice_axis]}.")

    # Extract the 2D slice
    if slice_axis == 0:
        slice_2d = weights_volume[slice_index, :, :]
        axis_labels_on_plot = ['Y', 'Z']
    elif slice_axis == 1:
        slice_2d = weights_volume[:, slice_index, :]
        axis_labels_on_plot = ['X', 'Z']
    else: # slice_axis == 2
        slice_2d = weights_volume[:, :, slice_index]
        axis_labels_on_plot = ['X', 'Y']

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(slice_2d, cmap=cmap, **kwargs_imshow)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    
    # Dynamic title
    full_title = f"{title} (Axis {['X', 'Y', 'Z'][slice_axis]}={slice_index})"
    ax.set_title(full_title)
    
    ax.set_xlabel(axis_labels_on_plot[0])
    ax.set_ylabel(axis_labels_on_plot[1])
    
    # Unlike other plots, we might want to keep axis ticks for image slices
    # ax.axis('off') # Optional: uncomment to turn off axis ticks and labels

    return ax

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
