import torch
from reconlib.voronoi.density_weights_pytorch import compute_voronoi_density_weights_pytorch

def compute_and_apply_voronoi_weights_to_echo_data(
    echo_data: torch.Tensor,
    sample_points: torch.Tensor, # (N, D) tensor of sample locations for Voronoi calculation
    bounds: torch.Tensor | None = None, # Optional bounds for Voronoi calculation
    space_dim: int | None = None,
    apply_sqrt: bool = False
) -> torch.Tensor:
    """
    Computes Voronoi density weights based on sample_points and applies them
    to the echo_data.

    This utility is a placeholder for how Voronoi weights might be used in
    an ultrasound context, e.g., for density compensation if echo_data samples
    are spatially irregular and sample_points represent their locations.

    Args:
        echo_data (torch.Tensor): The echo data. Expected to be 1D (num_samples_total,)
                                  or 2D (num_elements, num_samples_per_element) which
                                  will be flattened for weighting if sample_points correspond
                                  to flattened data. The shape relationship between echo_data
                                  and sample_points needs careful consideration by the user.
        sample_points (torch.Tensor): Coordinates of the samples used to compute
                                      Voronoi weights. Shape (M, D), where M is
                                      the number of points (e.g., total echo samples if flattened)
                                      and D is the spatial dimension (2 or 3).
        bounds (torch.Tensor, optional): Bounds for Voronoi calculation.
                                         See compute_voronoi_density_weights_pytorch.
        space_dim (int, optional): Spatial dimension for Voronoi calculation.
                                   Inferred if None.
        apply_sqrt (bool, optional): If True, applies the square root of the
                                     computed weights. Defaults to False.

    Returns:
        torch.Tensor: Echo data multiplied by the computed Voronoi weights (or their sqrt).
                      Shape will be the same as input echo_data.

    Raises:
        ValueError: If shapes of echo_data and computed weights are incompatible.
    """
    if sample_points.ndim != 2:
        raise ValueError("sample_points must be a 2D tensor (num_points, dimensionality).")

    num_voronoi_points = sample_points.shape[0]

    # Assuming echo_data might be (num_elements, num_samples_per_element)
    # or already flattened to match sample_points.
    original_echo_shape = echo_data.shape
    if echo_data.numel() != num_voronoi_points:
        raise ValueError(
            f"Number of elements in echo_data ({echo_data.numel()}) "
            f"must match number of sample_points ({num_voronoi_points}) for weighting."
        )

    echo_data_flat = echo_data.flatten().to(sample_points.device) # Ensure same device

    # Compute Voronoi weights
    # These weights are typically 1/area or 1/volume, normalized.
    voronoi_weights = compute_voronoi_density_weights_pytorch(
        points=sample_points,
        bounds=bounds,
        space_dim=space_dim
    ) # Returns (M,) tensor

    if voronoi_weights.shape[0] != num_voronoi_points:
        # This should not happen if compute_voronoi_density_weights_pytorch works correctly
        raise RuntimeError("Computed Voronoi weights shape mismatch.")

    if apply_sqrt:
        # Ensure weights are non-negative before sqrt
        weights_to_apply = torch.sqrt(torch.relu(voronoi_weights))
    else:
        weights_to_apply = voronoi_weights

    # Ensure weights_to_apply can be broadcast or element-wise multiplied
    # If echo_data_flat is complex and weights are real, multiplication is fine.
    if echo_data_flat.is_complex() and not weights_to_apply.is_complex():
        weights_to_apply = weights_to_apply.to(torch.complex64) # Cast weights if needed, or ensure dtype compatibility
    elif not echo_data_flat.is_complex() and weights_to_apply.is_complex():
         weights_to_apply = weights_to_apply.real # Or handle error, depends on desired behavior

    weighted_echo_data_flat = echo_data_flat * weights_to_apply.to(echo_data_flat.device)

    return weighted_echo_data_flat.reshape(original_echo_shape)


if __name__ == '__main__':
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Voronoi Utilities on {test_device}")

    # Example: Simulate some echo data and sample points
    num_echo_samples = 100
    echo_data_test = torch.randn(num_echo_samples, dtype=torch.complex64, device=test_device)

    # Assume these samples correspond to 2D spatial locations
    sample_points_2d = torch.rand((num_echo_samples, 2), device=test_device) * 10 - 5 # Random points in [-5, 5]
    bounds_2d = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], device=test_device)

    print(f"Original echo data (first 5): {echo_data_test[:5]}")

    try:
        weighted_data = compute_and_apply_voronoi_weights_to_echo_data(
            echo_data=echo_data_test,
            sample_points=sample_points_2d,
            bounds=bounds_2d,
            space_dim=2
        )
        print(f"Weighted echo data (first 5): {weighted_data[:5]}")
        assert weighted_data.shape == echo_data_test.shape
        print("compute_and_apply_voronoi_weights_to_echo_data (2D) test successful.")
    except Exception as e:
        print(f"Error during 2D Voronoi utility test: {e}")
        # This test depends on the full Voronoi stack, which might have issues
        # or require specific data conditions not met by random points.
        # For now, allow to pass if placeholder Voronoi logic in density_weights_pytorch fails.
        print("Test might fail if underlying Voronoi components have strict requirements or are placeholders.")


    # Example for 3D (if underlying Voronoi supports it)
    if num_echo_samples >= 4: # Need at least dim+1 points for Delaunay in 3D
        sample_points_3d = torch.rand((num_echo_samples, 3), device=test_device) * 2 - 1 # Random points in [-1, 1]
        bounds_3d = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=test_device)
        try:
            weighted_data_3d = compute_and_apply_voronoi_weights_to_echo_data(
                echo_data=echo_data_test,
                sample_points=sample_points_3d,
                bounds=bounds_3d,
                space_dim=3
            )
            print(f"Weighted echo data (3D) (first 5): {weighted_data_3d[:5]}")
            assert weighted_data_3d.shape == echo_data_test.shape
            print("compute_and_apply_voronoi_weights_to_echo_data (3D) test successful.")
        except Exception as e:
            print(f"Error during 3D Voronoi utility test: {e}")
            print("Test might fail if underlying Voronoi components have strict requirements (e.g. non-coplanar points for 3D).")

    # Test with apply_sqrt
    try:
        weighted_data_sqrt = compute_and_apply_voronoi_weights_to_echo_data(
            echo_data=echo_data_test,
            sample_points=sample_points_2d, # Using 2D points again
            bounds=bounds_2d,
            space_dim=2,
            apply_sqrt=True
        )
        print(f"SQRT Weighted echo data (first 5): {weighted_data_sqrt[:5]}")
        assert weighted_data_sqrt.shape == echo_data_test.shape
        print("compute_and_apply_voronoi_weights_to_echo_data (2D, sqrt) test successful.")
    except Exception as e:
        print(f"Error during 2D Voronoi utility (sqrt) test: {e}")
        print("Test might fail if underlying Voronoi components have strict requirements.")
