import torch
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
from reconlib.regularizers.common import L1Regularizer, L2Regularizer
# Assuming UltrasoundForwardOperator will be imported from .operators or a higher level
# For now, to make this file potentially runnable standalone for tests, we might need a mock.
# from .operators import UltrasoundForwardOperator

class MockUltrasoundOperator:
    """ Minimal mock for testing reconstructors if operators.py isn't directly available. """
    def __init__(self, device='cpu', image_shape=(64,64)):
        self.device = device
        self.image_shape = image_shape
        self.num_elements = 32
        self.num_samples = 512

    def op(self, image_x, sensitivity_maps=None): # sensitivity_maps is for ProximalGradientReconstructor compatibility
        # Mock forward: return dummy kspace of correct expected shape by ProximalGradientReconstructor
        # This mock doesn't need to be accurate, just produce right shapes.
        # Real forward op for ultrasound returns (num_elements, num_samples)
        # ProximalGradientReconstructor typically expects kspace_data that matches output of forward_op_fn
        # The forward_op_fn for ProxGrad is A(x), so it should return something like kspace data.
        # Let's assume the "kspace_data" for ultrasound context is the echo_data.
        return torch.zeros((self.num_elements, self.num_samples), dtype=torch.complex64, device=image_x.device)

    def op_adj(self, echo_data_y, sensitivity_maps=None): # sensitivity_maps for compatibility
        # Mock adjoint: return dummy image of correct shape
        return torch.zeros(self.image_shape, dtype=torch.complex64, device=echo_data_y.device)

def das_reconstruction(
    echo_data: torch.Tensor,
    ultrasound_operator: 'UltrasoundForwardOperator' # Type hint with quotes for forward reference
) -> torch.Tensor:
    """
    Performs Delay-and-Sum (DAS) reconstruction using the adjoint of the
    ultrasound operator.

    Args:
        echo_data (torch.Tensor): The acquired ultrasound echo data.
                                  Shape (num_elements, num_samples).
        ultrasound_operator (UltrasoundForwardOperator): An instance of the
                                                       UltrasoundForwardOperator.
    Returns:
        torch.Tensor: The reconstructed image. Shape (height, width).
    """
    if not hasattr(ultrasound_operator, 'op_adj'):
        raise TypeError("ultrasound_operator must have an 'op_adj' method.")

    # The sensitivity_maps argument is part of ProximalGradientReconstructor's
    # adjoint_op_fn signature. If UltrasoundForwardOperator.op_adj doesn't
    # accept it, we might need a wrapper lambda if used directly with it.
    # However, for DAS, we call it directly.
    reconstructed_image = ultrasound_operator.op_adj(echo_data)
    return reconstructed_image

def inverse_reconstruction_pg(
    echo_data: torch.Tensor,
    ultrasound_operator: 'UltrasoundForwardOperator', # Type hint
    regularizer_type: str, # 'l1' or 'l2'
    lambda_reg: float,
    iterations: int = 10,
    step_size: float = 0.01,
    initial_estimate_fn: callable = None, # Optional: fn(kspace_data, sensitivity_maps, adjoint_op_fn) -> initial_image
    verbose: bool = False,
    # x_init: Optional initial image guess.
    # image_shape_for_zero_init: if x_init and initial_estimate_fn are None.
) -> torch.Tensor:
    """
    Performs inverse reconstruction using the Proximal Gradient algorithm.

    Solves: argmin_x { || A(x) - y ||_2^2 + lambda * R(x) }
    where A is the ultrasound_operator, y is echo_data, R is the regularizer.

    Args:
        echo_data (torch.Tensor): Acquired ultrasound echo data.
                                  Shape (num_elements, num_samples).
        ultrasound_operator (UltrasoundForwardOperator): Configured ultrasound operator.
        regularizer_type (str): Type of regularizer ('l1' or 'l2').
        lambda_reg (float): Regularization strength.
        iterations (int, optional): Number of iterations. Defaults to 10.
        step_size (float, optional): Step size for gradient descent. Defaults to 0.01.
        initial_estimate_fn (callable, optional): Function to compute initial image.
                                                  If None, defaults to zero or adjoint.
        verbose (bool, optional): Print iteration progress. Defaults to False.

    Returns:
        torch.Tensor: The reconstructed image. Shape (height, width).
    """
    device = echo_data.device

    # Select regularizer
    if regularizer_type.lower() == 'l1':
        regularizer = L1Regularizer(lambda_reg=lambda_reg)
    elif regularizer_type.lower() == 'l2':
        regularizer = L2Regularizer(lambda_reg=lambda_reg)
    else:
        raise ValueError(f"Unsupported regularizer_type: {regularizer_type}. Choose 'l1' or 'l2'.")

    # ProximalGradientReconstructor expects forward_op_fn(image, sensitivity_maps)
    # and adjoint_op_fn(kspace, sensitivity_maps).
    # Our UltrasoundForwardOperator.op and op_adj don't use sensitivity_maps.
    # We can wrap them in lambdas.

    forward_op_fn_wrapper = lambda image_estimate, smaps: ultrasound_operator.op(image_estimate)
    adjoint_op_fn_wrapper = lambda kspace_coils, smaps: ultrasound_operator.op_adj(kspace_coils)

    # The regularizer_prox_fn signature is fn(image, steplength)
    # lambda_reg is already part of the regularizer object.
    regularizer_prox_fn_wrapper = lambda image, sl: regularizer.proximal_operator(image, sl)


    pg_reconstructor = ProximalGradientReconstructor(
        iterations=iterations,
        step_size=step_size,
        initial_estimate_fn=initial_estimate_fn, # User can pass a custom one
        verbose=verbose,
        log_fn=None # No logging function by default
    )

    # Determine initial guess strategy if initial_estimate_fn is None
    x_init_arg = None
    image_shape_for_zero_init_arg = None

    if initial_estimate_fn is None:
        # Default initial estimate: A^H(y)
        # This matches one of the options for ProximalGradientReconstructor's internal init logic
        # if its initial_estimate_fn is set. Or we can provide it as x_init.
        # Let's try providing it as x_init for clarity here.
        print("inverse_reconstruction_pg: Using adjoint of echo_data as initial estimate.")
        x_init_arg = ultrasound_operator.op_adj(echo_data)
        # Alternatively, if we want ProximalGradientReconstructor to do zero init:
        # image_shape_for_zero_init_arg = ultrasound_operator.image_shape

    reconstructed_image = pg_reconstructor.reconstruct(
        kspace_data=echo_data, # This is our 'y'
        forward_op_fn=forward_op_fn_wrapper,
        adjoint_op_fn=adjoint_op_fn_wrapper,
        regularizer_prox_fn=regularizer_prox_fn_wrapper,
        sensitivity_maps=None, # Ultrasound model here doesn't use coil sensitivities
        x_init=x_init_arg,
        image_shape_for_zero_init=image_shape_for_zero_init_arg
    )

    return reconstructed_image

# Example usage (for testing within this file)
if __name__ == '__main__':
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Ultrasound Reconstructors on {test_device}")

    # Use the Mock Operator for standalone testing
    mock_op_params = {'device': test_device, 'image_shape': (32, 32)}
    mock_us_operator = MockUltrasoundOperator(**mock_op_params)

    # Create dummy echo data
    dummy_echo_data = torch.randn(
        (mock_us_operator.num_elements, mock_us_operator.num_samples),
        dtype=torch.complex64, device=test_device
    )
    print(f"Dummy echo data shape: {dummy_echo_data.shape}")

    # Test DAS Reconstruction
    try:
        das_image = das_reconstruction(dummy_echo_data, mock_us_operator)
        print(f"DAS reconstructed image shape: {das_image.shape}")
        assert das_image.shape == mock_us_operator.image_shape
        print("DAS reconstruction test successful (execution only).")
    except Exception as e:
        print(f"Error during DAS reconstruction test: {e}")
        raise

    # Test Inverse Reconstruction (Proximal Gradient) - L2
    try:
        inv_image_l2 = inverse_reconstruction_pg(
            echo_data=dummy_echo_data,
            ultrasound_operator=mock_us_operator,
            regularizer_type='l2',
            lambda_reg=0.01,
            iterations=5, # Keep iterations low for quick test
            step_size=0.1,
            verbose=True
        )
        print(f"Inverse (L2) reconstructed image shape: {inv_image_l2.shape}")
        assert inv_image_l2.shape == mock_us_operator.image_shape
        print("Inverse reconstruction (L2, PG) test successful (execution only).")
    except Exception as e:
        print(f"Error during Inverse (L2, PG) reconstruction test: {e}")
        raise

    # Test Inverse Reconstruction (Proximal Gradient) - L1
    try:
        inv_image_l1 = inverse_reconstruction_pg(
            echo_data=dummy_echo_data,
            ultrasound_operator=mock_us_operator,
            regularizer_type='l1',
            lambda_reg=0.005,
            iterations=5,
            step_size=0.1,
            verbose=True
        )
        print(f"Inverse (L1) reconstructed image shape: {inv_image_l1.shape}")
        assert inv_image_l1.shape == mock_us_operator.image_shape
        print("Inverse reconstruction (L1, PG) test successful (execution only).")
    except Exception as e:
        print(f"Error during Inverse (L1, PG) reconstruction test: {e}")
        raise

    print("\nAll reconstructor tests completed (execution checks).")
