# reconlib/plotting.py
"""Module for visualization tasks in MRI reconstruction."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

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
