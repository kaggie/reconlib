# reconlib/plotting.py
"""Module for visualization tasks in MRI reconstruction."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import torch 
from mpl_toolkits.mplot3d import Axes3D 
# Voronoi is already imported by plot_voronoi_diagram_2d
from .voronoi_utils import ConvexHull # Added for plot_3d_voronoi_with_hull

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

def plot_3d_delaunay(points: torch.Tensor, 
                     tetrahedra: torch.Tensor, 
                     convex_hull: ConvexHull = None, # Accepts a precomputed ConvexHull object
                     show_points=True, show_tetrahedra=True, show_hull=True, 
                     alpha_tetra: float = 0.1, alpha_hull: float = 0.2):
    """ 
    Plots the 3D Delaunay triangulation, points, and optionally the convex hull.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing 3D points.
        tetrahedra (torch.Tensor): Tensor of shape (M, 4) with tetrahedra indices
                                   (indices into the `points` tensor).
        convex_hull (ConvexHull, optional): Precomputed ConvexHull object for the `points`.
                                            If None, it will be computed internally if `show_hull` is True.
        show_points (bool): Whether to plot input points.
        show_tetrahedra (bool): Whether to plot tetrahedra.
        show_hull (bool): Whether to plot the convex hull.
        alpha_tetra (float): Transparency for tetrahedra.
        alpha_hull (float): Transparency for hull faces.
    """
    assert points.dim() == 2 and points.shape[1] == 3, "Points must be (N, 3) tensor"

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    points_cpu = points.cpu() # Ensure points are on CPU for plotting and hull computation if needed

    # Plot points
    if show_points:
        points_np = points_cpu.numpy()
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='red', s=50, label='Points', depthshade=True)

    # Plot tetrahedra
    # Each tetrahedron is plotted as a collection of 4 triangular faces
    # To make them slightly transparent, we plot each face.
    plotted_tetra_label = False
    if show_tetrahedra and tetrahedra.numel() > 0:
        # Add a single label for all tetrahedra faces
        ax.plot_trisurf([], [], [], color='blue', alpha=alpha_tetra, edgecolor='darkblue', label='Delaunay Tetrahedra (Faces)')
        plotted_tetra_label = True # Set to true as we've added the label

        for tetra_indices in tetrahedra: # tetra_indices is [p0_idx, p1_idx, p2_idx, p3_idx]
            verts_tetra = points_cpu[tetra_indices] # Get the 4 vertices of the tetrahedron
            
            # Define faces of a tetrahedron: (0,1,2), (0,1,3), (0,2,3), (1,2,3) using local indices
            local_faces_indices = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
            ]
            
            for face_idx_list in local_faces_indices:
                face_verts_tensor = verts_tetra[face_idx_list]
                face_np = face_verts_tensor.numpy()
                ax.plot_trisurf(face_np[:, 0], face_np[:, 1], face_np[:, 2],
                                color='blue', alpha=alpha_tetra, edgecolor='darkblue', linewidth=0.3, 
                                shade=True)
    
    # Plot convex hull
    if show_hull:
        if convex_hull is None:
            # Compute hull if not provided, using the reconlib.voronoi_utils.ConvexHull
            # This ConvexHull is PyTorch-native.
            try:
                # ConvexHull expects points on the device it will run on.
                # If points were originally on GPU, pass them as is.
                # The ConvexHull class itself handles points.cpu() if its internal methods need it (e.g. SciPy fallback)
                # but the PyTorch native parts should work on the original device.
                # For plotting, we use points_cpu anyway.
                convex_hull = ConvexHull(points) 
            except Exception: # Removed "as e" to avoid unused variable if print is commented out
                # print(f"Could not compute convex hull for plotting: {e}")
                convex_hull = None # Ensure it's None if computation fails
        
        if convex_hull is not None and convex_hull.simplices is not None and convex_hull.simplices.numel() > 0:
            # points_cpu is already defined
            # Add a single label for the hull if it was shown
            ax.plot_trisurf([], [], [], color='lightgreen', alpha=alpha_hull, edgecolor='green', label='Convex Hull (Faces)')
            for simplex_face in convex_hull.simplices: # simplices are indices into original points
                tri_verts_np = points_cpu[simplex_face].numpy()
                ax.plot_trisurf(tri_verts_np[:, 0], tri_verts_np[:, 1], tri_verts_np[:, 2],
                                color='lightgreen', alpha=alpha_hull, edgecolor='green', linewidth=0.5, 
                                shade=True)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Delaunay Triangulation and Convex Hull')

    # Consolidate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
    plt.show()

def plot_voronoi_kspace(kspace_points: torch.Tensor, 
                        weights: torch.Tensor = None, 
                        bounds: torch.Tensor = None, 
                        ax: plt.Axes = None, # Allow passing an existing Axes object
                        title: str = 'K-space Voronoi Diagram',
                        show_points: bool = True,
                        point_size: float = 10,
                        point_color_map: str = 'viridis', # For when points are colored by weights
                        line_color: str = 'gray',
                        line_width: float = 0.8,
                        show_legend: bool = True):
    """
    Plots Voronoi cells for k-space samples, optionally colored by weights and bounded.
    Currently supports 2D k-space points.

    Args:
        kspace_points (torch.Tensor): Shape (N, 2) for N k-space points in 2D.
        weights (torch.Tensor, optional): Shape (N,). If provided, k-space points
                                          can be colored by these weights.
        bounds (torch.Tensor, optional): Shape (2, 2) [[min_x, min_y], [max_x, max_y]]
                                         for bounding the Voronoi diagram.
        ax (plt.Axes, optional): Matplotlib axes to plot on. If None, new figure/axes created.
        title (str, optional): Plot title.
        show_points (bool, optional): Whether to draw the k-space sample points.
        point_size (float, optional): Size of the k-space sample points.
        point_color_map (str, optional): Colormap for points if colored by weights.
        line_color (str, optional): Color of Voronoi cell edges.
        line_width (float, optional): Width of Voronoi cell edges.
        show_legend (bool, optional): Whether to show legend (e.g., for color bar if weights are used).
    """
    if not isinstance(kspace_points, torch.Tensor):
        raise TypeError("kspace_points must be a PyTorch tensor.")
    
    dim = kspace_points.shape[1]
    if dim != 2:
        raise NotImplementedError(f"Plotting is currently implemented for 2D k-space points only. Got {dim}D.")

    kspace_points_np = kspace_points.cpu().numpy()
    
    weights_np = None
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a PyTorch tensor if provided.")
        if weights.shape[0] != kspace_points.shape[0]:
            raise ValueError("weights must have the same number of elements as kspace_points.")
        weights_np = weights.cpu().numpy()

    bounds_np = None
    if bounds is not None:
        if not isinstance(bounds, torch.Tensor):
            raise TypeError("bounds must be a PyTorch tensor if provided.")
        if bounds.shape != (2, 2):
            raise ValueError("bounds must have shape (2, 2) for 2D: [[min_x, min_y], [max_x, max_y]].")
        bounds_np = bounds.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8)) # Create a new figure and axes
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(kspace_points_np, qhull_options='Qbb Qc Qz')
    except Exception as e: # Catches scipy.spatial.qhull.QhullError
        ax.text(0.5, 0.5, f"Voronoi computation failed:\n{e}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, color='red')
        if show_points: # Still plot points if Voronoi fails
            ax.scatter(kspace_points_np[:, 0], kspace_points_np[:, 1], s=point_size, color='blue', zorder=3, 
                       label='K-space Points (Voronoi Failed)' if show_legend else None)
        ax.set_title(title + " (Voronoi Failed)")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        if show_legend and ax.get_legend_handles_labels()[0]: # Check if there are any labels
             ax.legend()
        ax.set_aspect('equal', adjustable='box')
        return ax # Return ax even on failure

    # Plot Voronoi cell edges (finite ridges)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, 
                    line_colors=line_color, linewidth=line_width)

    # Plot K-space Points
    if show_points:
        if weights_np is not None:
            scatter = ax.scatter(kspace_points_np[:, 0], kspace_points_np[:, 1], 
                                 s=point_size, c=weights_np, cmap=point_color_map, zorder=3)
            if show_legend:
                plt.colorbar(scatter, ax=ax, label='Weights')
        else:
            ax.scatter(kspace_points_np[:, 0], kspace_points_np[:, 1], 
                       s=point_size, color='blue', zorder=3, 
                       label='K-space Points' if show_legend else None)

    # Draw Bounding Box
    legend_elements_exist = bool(ax.get_legend_handles_labels()[0]) # Check before adding more labels

    if bounds_np is not None:
        min_x, min_y = bounds_np[0, 0], bounds_np[0, 1]
        max_x, max_y = bounds_np[1, 0], bounds_np[1, 1]
        # Plot rectangle lines
        ax.plot([min_x, max_x, max_x, min_x, min_x], 
                [min_y, min_y, max_y, max_y, min_y], 
                color='red', linestyle='--', linewidth=1.5, 
                label='Bounds' if show_legend else None)
        
        # Adjust plot limits slightly outside the bounds
        padding_x = (max_x - min_x) * 0.05 
        padding_y = (max_y - min_y) * 0.05
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
    else:
        # Autoscale if no bounds, may already be handled by voronoi_plot_2d, but can be explicit
        ax.autoscale_view()


    # Styling
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    
    # Update legend_elements_exist after potentially adding bounds label
    if not legend_elements_exist and (bounds_np is not None or (show_points and weights_np is None)):
        legend_elements_exist = bool(ax.get_legend_handles_labels()[0])

    if show_legend and legend_elements_exist:
        # Consolidate legend if multiple labels were added (e.g. 'K-space Points' and 'Bounds')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
    ax.set_aspect('equal', adjustable='box')
    
    return ax # Return the axes object for further modification or display by caller

def plot_3d_voronoi_with_hull(points: torch.Tensor, 
                              # vertices: torch.Tensor, # This was in the issue's signature, but seems unused if simplices are for the main hull
                              simplices: torch.Tensor, # Simplices for the main convex hull of 'points'
                              show_points=True, show_voronoi=True, 
                              show_hull=True, alpha=0.3, voronoi_alpha=0.1):
    """
    Plots 3D points, their overall convex hull, and their Voronoi cells (bounded by individual convex hulls).

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing input points.
        simplices (torch.Tensor): Indices of triangular faces for the convex hull of ALL `points`.
                                  Shape (K, 3), indices into `points`.
        show_points (bool): Whether to plot input points.
        show_voronoi (bool): Whether to plot Voronoi cells.
        show_hull (bool): Whether to plot the overall convex hull of `points`.
        alpha (float): Transparency for the main convex hull faces.
        voronoi_alpha (float): Transparency for individual Voronoi cell hull faces.
    """
    points_np = points.cpu().numpy() # For SciPy Voronoi and Matplotlib
    
    fig = plt.figure(figsize=(12, 9)) # Slightly larger figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot input points
    if show_points:
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='red', s=50, label='Input Points', depthshade=True)

    # Plot Voronoi cells
    if show_voronoi:
        vor = Voronoi(points_np) # Compute Voronoi diagram from original points
        vor_vertices_torch = torch.tensor(vor.vertices, dtype=points.dtype, device=points.device)

        plotted_voronoi_label = False
        for i, region_indices in enumerate(vor.regions):
            if not region_indices or -1 in region_indices: # Skip empty or open regions
                continue
            
            # Vertices of the current Voronoi cell
            current_voronoi_cell_verts_torch = vor_vertices_torch[region_indices]

            if current_voronoi_cell_verts_torch.shape[0] < 4: # Need at least 4 points for 3D hull
                # print(f"Skipping Voronoi region {i} for point {vor.point_region[i]}: not enough unique vertices for 3D hull ({current_voronoi_cell_verts_torch.shape[0]})")
                continue
            
            try:
                # Use the reconlib.voronoi_utils.ConvexHull for each Voronoi cell
                # This ConvexHull is now PyTorch-native.
                region_hull = ConvexHull(current_voronoi_cell_verts_torch)
                
                # Ensure region_hull.simplices is not empty and is a tensor
                if region_hull.simplices is not None and region_hull.simplices.numel() > 0:
                    # Get points for these simplices from current_voronoi_cell_verts_torch
                    cell_points_for_plot = current_voronoi_cell_verts_torch.cpu()
                    for simplex_face in region_hull.simplices: # simplex_face is [idx1, idx2, idx3] into current_voronoi_cell_verts_torch
                        tri_verts_np = cell_points_for_plot[simplex_face].numpy()
                        
                        label_to_use = None
                        if not plotted_voronoi_label:
                            label_to_use = 'Voronoi Cells (Hulls)' # This label will be set on the last iteration due to loop structure
                            # This approach for single label is not ideal. Better to plot one dummy element for label.
                        
                        ax.plot_trisurf(tri_verts_np[:, 0], tri_verts_np[:, 1], tri_verts_np[:, 2],
                                        color='cyan', alpha=voronoi_alpha, edgecolor='blue', linewidth=0.5, 
                                        shade=True) 
                if not plotted_voronoi_label and region_hull.simplices is not None and region_hull.simplices.numel() > 0 : # Add label after first successful plot
                    ax.plot_trisurf([], [], [], color='cyan', alpha=voronoi_alpha, edgecolor='blue', label='Voronoi Cells (Hulls)')
                    plotted_voronoi_label = True

            except Exception: # Removed "as e" to avoid unused variable warning if print is commented out
                # print(f"Could not compute or plot hull for Voronoi region {i} (point {vor.point_region.get(i, 'N/A')}): {e}")
                pass # Continue if a single Voronoi cell hull fails

    # Plot overall convex hull of the input points
    if show_hull and simplices is not None and simplices.numel() > 0:
        points_cpu = points.cpu() # Ensure points are on CPU for indexing
        # Plot one dummy element for the label
        ax.plot_trisurf([], [], [], color='lightgreen', alpha=alpha, edgecolor='green', label='Overall Convex Hull (Faces)')
        for simplex_face in simplices: # These are indices into the original 'points' tensor
            tri_verts_np = points_cpu[simplex_face].numpy()
            ax.plot_trisurf(tri_verts_np[:, 0], tri_verts_np[:, 1], tri_verts_np[:, 2],
                            color='lightgreen', alpha=alpha, edgecolor='green', linewidth=0.5, 
                            shade=True)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voronoi Diagram with Convex Hulls')

    # Consolidate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
    plt.show()

def plot_3d_hull(points: torch.Tensor, vertices: torch.Tensor, simplices: torch.Tensor, 
                 show_points=True, show_hull=True, alpha=0.3):
    """ Plots the 3D convex hull and input points.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing input points.
        vertices (torch.Tensor): Indices of points on the convex hull.
                                 (Note: this arg is not directly used in the provided plot_3d_hull
                                  if simplices already contain all necessary vertex info from points tensor,
                                  but kept for signature consistency with example.)
        simplices (torch.Tensor): Indices of triangular faces, shape (K, 3).
                                  Each row contains indices into the `points` tensor.
        show_points (bool): Whether to plot input points.
        show_hull (bool): Whether to plot the convex hull.
        alpha (float): Transparency for hull faces.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    if show_points:
        # Ensure points are on CPU and NumPy for Matplotlib
        points_np = points.cpu().numpy()
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='red', s=50, label='Points')

    # Plot convex hull
    if show_hull and simplices.numel() > 0: # Check if simplices is not empty
        # Ensure points are on CPU for indexing and plotting
        points_cpu = points.cpu()
        for simplex in simplices: # simplex is a row [idx1, idx2, idx3]
            # Collect the 3 vertices for this face from the points tensor
            tri_vertices = points_cpu[simplex] 
            tri_np = tri_vertices.numpy() # Convert to NumPy for plot_trisurf

            ax.plot_trisurf(
                tri_np[:, 0], tri_np[:, 1], tri_np[:, 2],
                color='green', alpha=alpha, edgecolor='k' # Removed label='Convex Hull' from here to avoid multiple legend entries
            )
    
    # Add a single label for the hull if it was shown
    if show_hull and simplices.numel() > 0:
        ax.plot_trisurf([], [], [], color='green', alpha=alpha, edgecolor='k', label='Convex Hull (Faces)')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Consolidate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    if by_label: # Only show legend if there's something to show
        ax.legend(by_label.values(), by_label.keys())
    
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
