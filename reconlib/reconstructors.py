"""Module for defining Reconstructor classes."""

import torch
from abc import ABC, abstractmethod
# MRIData would be imported if type hinting mri_data_obj, but not strictly needed for implementation
# from reconlib.data import MRIData 

class Reconstructor(ABC):
    """
    Abstract base class for reconstructors.

    Defines the interface for the reconstruct method.
    """
    @abstractmethod
    def reconstruct(self, mri_data_obj, initial_guess=None, verbose=False):
        """
        Performs the reconstruction.

        Args:
            mri_data_obj: An MRIData object containing the data to reconstruct.
            initial_guess: Optional initial guess for the image.
            verbose: Optional flag for optimizer verbosity.

        Returns:
            The reconstructed image (PyTorch tensor).
        """
        pass

class IterativeReconstructor(Reconstructor):
    """
    Orchestrates iterative image reconstruction using a forward operator,
    regularizer, and optimizer.
    """
    def __init__(self, forward_operator, regularizer, optimizer):
        """
        Initializes the IterativeReconstructor.

        Args:
            forward_operator: An instance of a forward operator (e.g., MRIForwardOperator).
            regularizer: An instance of a regularizer (e.g., L1Regularizer).
            optimizer: An instance of an optimizer (e.g., FISTA).
        """
        self.forward_operator = forward_operator
        self.regularizer = regularizer
        self.optimizer = optimizer

    def reconstruct(self, mri_data_obj, initial_guess=None, verbose=False):
        """
        Performs the iterative reconstruction.

        Args:
            mri_data_obj: An MRIData object containing k_space_data and image_shape.
            initial_guess (torch.Tensor, optional): An initial guess for the image. 
                                                    Defaults to None, which triggers adjoint reconstruction.
            verbose (bool, optional): If True, enables verbose output from the optimizer,
                                      if the optimizer supports it. Defaults to False.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        
        # Determine the device from the forward operator
        # This assumes forward_operator has a 'device' attribute
        if not hasattr(self.forward_operator, 'device'):
            raise AttributeError("Forward operator must have a 'device' attribute.")
        device = self.forward_operator.device

        # Extract k-space data and ensure it's a PyTorch tensor on the correct device
        # MRIData stores k_space_data as NumPy array.
        # k_space_data is typically complex.
        k_space_data_np = mri_data_obj.k_space_data
        k_space_data = torch.from_numpy(k_space_data_np).to(dtype=torch.complex64, device=device)

        # Handle initial guess
        if initial_guess is not None:
            if not isinstance(initial_guess, torch.Tensor):
                initial_guess = torch.tensor(initial_guess, device=device) # Convert if not tensor
            
            initial_guess = initial_guess.to(device=device, dtype=torch.complex64) # Ensure device and dtype
            
            # Verify shape if possible (forward_operator.image_shape should be available)
            if hasattr(self.forward_operator, 'image_shape') and initial_guess.shape != self.forward_operator.image_shape:
                raise ValueError(f"Provided initial_guess shape {initial_guess.shape} does not match "
                                 f"forward_operator.image_shape {self.forward_operator.image_shape}.")
        else:
            # Default initial guess: Adjoint reconstruction
            # op_adj should return a tensor on the correct device and dtype
            # print("IterativeReconstructor: No initial_guess provided, using adjoint reconstruction as default.")
            initial_guess = self.forward_operator.op_adj(k_space_data)
            # Ensure it's complex, as op_adj might return real if k_space_data was accidentally real
            initial_guess = initial_guess.to(dtype=torch.complex64)


        # Set optimizer verbosity if supported
        if hasattr(self.optimizer, 'verbose') and verbose is not None:
            self.optimizer.verbose = verbose
        
        # Perform reconstruction using the optimizer
        # The optimizer's solve method is expected to handle device consistency internally
        # for the image estimate, given k_space_data and initial_guess are on device.
        reconstructed_image = self.optimizer.solve(
            k_space_data=k_space_data,
            forward_op=self.forward_operator,
            regularizer=self.regularizer,
            initial_guess=initial_guess
        )

        return reconstructed_image
