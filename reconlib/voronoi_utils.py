import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import sys # For stderr

EPSILON = 1e-9

def compute_polygon_area(vertices: np.ndarray) -> float:
    """
    Computes the area of a 2D polygon using the shoelace formula.

    Args:
        vertices: A NumPy array of shape (N, 2) representing the polygon's vertices,
                  where N is the number of vertices.

    Returns:
        The area of the polygon.

    Raises:
        ValueError: If the input vertices are invalid (not a NumPy array,
                      less than 3 vertices, or not 2D coordinates).
    """
    if not isinstance(vertices, np.ndarray):
        raise ValueError("Input 'vertices' must be a NumPy array.")
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("Input 'vertices' must be a 2D array with 2 columns (x, y coordinates).")
    if vertices.shape[0] < 3:
        raise ValueError("Input 'vertices' must have at least 3 vertices to form a polygon.")

    x = vertices[:, 0]
    y = vertices[:, 1]

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    if area < EPSILON:
        return 0.0
    return area

def compute_convex_hull_volume(vertices: np.ndarray) -> float:
    """
    Computes the volume of the convex hull of a set of 3D points.

    Args:
        vertices: A NumPy array of shape (N, 3) representing the 3D points,
                  where N is the number of points.

    Returns:
        The volume of the convex hull.

    Raises:
        ValueError: If the input vertices are invalid (not a NumPy array,
                      less than 4 vertices, or not 3D coordinates).
    """
    if not isinstance(vertices, np.ndarray):
        raise ValueError("Input 'vertices' must be a NumPy array.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Input 'vertices' must be a 2D array with 3 columns (x, y, z coordinates).")
    if vertices.shape[0] < 4:
        raise ValueError("Input 'vertices' must have at least 4 vertices to form a 3D convex hull.")

    volume = 0.0
    try:
        hull = ConvexHull(vertices)
        volume = hull.volume
    except QhullError:
        print("Warning: QhullError encountered. This might be due to degenerate input points (e.g., collinear or coplanar). Setting volume to 0.0.", file=sys.stderr)
        volume = 0.0
    except Exception as e:
        # Catch any other potential errors during ConvexHull computation
        print(f"An unexpected error occurred during ConvexHull computation: {e}", file=sys.stderr)
        volume = 0.0


    if volume < EPSILON:
        return 0.0
    return volume

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalizes an array of weights to sum to 1.

    Args:
        weights: A NumPy array of weights.

    Returns:
        A NumPy array of normalized weights.
        If the sum of input weights is close to zero, returns the original weights.
    """
    if not isinstance(weights, np.ndarray):
        raise ValueError("Input 'weights' must be a NumPy array.")

    total = np.sum(weights)

    if np.abs(total) < EPSILON:
        print("Warning: Sum of weights is zero or near-zero; returning unnormalized weights.", file=sys.stderr)
        return weights

    normalized_weights = weights / total
    
    # Clamp negative values to 0.0 after normalization
    # This is important if original weights could be negative and sum to a positive total,
    # or if precision issues result in tiny negative numbers.
    normalized_weights[normalized_weights < 0] = 0.0
    
    # Optional: Re-normalize if clamping changed the sum significantly, though typically not needed if original weights are non-negative.
    # current_sum_after_clamping = np.sum(normalized_weights)
    # if np.abs(current_sum_after_clamping) > EPSILON and np.abs(current_sum_after_clamping - 1.0) > EPSILON : # if sum is not zero and not 1
    #    normalized_weights = normalized_weights / current_sum_after_clamping


    return normalized_weights
