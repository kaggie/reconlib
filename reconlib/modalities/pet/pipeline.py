"""Module for PET/CT reconstruction workflow utilities, including pipelines and monitors."""

import torch
import numpy as np
from typing import Callable, List, Optional, Any, Dict

from reconlib.optimizers import Optimizer
# Attempt to import specific metrics, but don't fail if not fully implemented yet
try:
    from reconlib.metrics.image_metrics import ssim, psnr # Common metrics
except ImportError:
    print("Warning: Could not import ssim, psnr from reconlib.metrics.image_metrics. Metrics calculator will be limited.")
    ssim = None
    psnr = None

class ReconstructionPipeline:
    """
    A flexible pipeline for tomographic reconstruction, encompassing preprocessing,
    the main reconstruction algorithm, and postprocessing steps.
    """
    def __init__(self,
                 preprocessing_steps: Optional[List[Callable[[Any], Any]]] = None,
                 reconstruction_algorithm: Optional[Optimizer] = None, # Expects an instantiated Optimizer
                 postprocessing_steps: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None):
        """
        Initializes a reconstruction pipeline.

        Args:
            preprocessing_steps (Optional[List[Callable[[Any], Any]]]):
                A list of functions to be applied sequentially to the raw data before reconstruction.
                Each function takes data (Any type) and returns processed data (Any type).
            reconstruction_algorithm (Optional[Optimizer]):
                An instantiated optimizer object (e.g., FISTA, ADMM, OSEM) that has a `solve` method.
            postprocessing_steps (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]):
                A list of functions to be applied sequentially to the reconstructed image.
                Each function takes a torch.Tensor and returns a torch.Tensor.
        """
        self.preprocessing_steps: List[Callable[[Any], Any]] = preprocessing_steps if preprocessing_steps is not None else []
        self.reconstruction_algorithm: Optional[Optimizer] = reconstruction_algorithm
        self.postprocessing_steps: List[Callable[[torch.Tensor], torch.Tensor]] = postprocessing_steps if postprocessing_steps is not None else []

    def run(self, raw_data: Any, initial_image_guess: Optional[torch.Tensor] = None, verbose: bool = False) -> torch.Tensor:
        """
        Executes the full reconstruction pipeline from raw data to a processed image.

        Args:
            raw_data (Any): The initial raw data to be processed (e.g., sinogram, k-space data).
            initial_image_guess (Optional[torch.Tensor]): An initial guess for the image,
                                                          passed to the reconstruction algorithm if provided.
            verbose (bool): If True, print statements about pipeline progress may be enabled.

        Returns:
            torch.Tensor: The reconstructed (and potentially postprocessed) image.

        Raises:
            NotImplementedError: If the core reconstruction algorithm step is not yet fully implemented
                                 to handle the data flow correctly.
            ValueError: If no reconstruction algorithm is set in the pipeline.
        """
        if verbose:
            print("Starting reconstruction pipeline...")

        processed_data = raw_data
        if verbose and not self.preprocessing_steps:
            print("No preprocessing steps defined.")
        for i, step_func in enumerate(self.preprocessing_steps):
            if verbose:
                print(f"Running preprocessing step {i+1}/{len(self.preprocessing_steps)}: {getattr(step_func, '__name__', 'Unnamed function')}...")
            processed_data = step_func(processed_data)
            if verbose:
                print(f"Data shape/type after step: {processed_data.shape if hasattr(processed_data, 'shape') else type(processed_data)}")

        if self.reconstruction_algorithm is None:
            if isinstance(processed_data, torch.Tensor):
                print("Warning: No reconstruction algorithm set. Returning preprocessed data as is.")
                reconstructed_image = processed_data
            else:
                raise ValueError("No reconstruction algorithm is set in the pipeline, and preprocessed data is not a tensor.")

        else:
            if verbose:
                print(f"Running reconstruction algorithm: {self.reconstruction_algorithm.__class__.__name__}...")

            # The core challenge: Optimizer.solve() has a specific signature:
            # solve(self, k_space_data, forward_op, regularizer, initial_guess=None)
            # OSEM.solve() uses k_space_data (as projection_data), its own system_matrix (as forward_op), ignores regularizer.
            # FISTA/ADMM need k_space_data, an explicit forward_op, and a regularizer.
            # PenalizedLikelihoodReconstruction also needs these.
            #
            # This pipeline's `run` method needs to correctly supply these.
            # For now, we assume that if `self.reconstruction_algorithm` is an OSEM instance,
            # its `system_matrix` is already configured.
            # If it's FISTA/ADMM/PenalizedLikelihood, they must have been initialized with their
            # respective SystemMatrix (as forward_op) and Regularizer, or these need to be
            # passed/accessible in a different way.
            #
            # This current placeholder assumes the optimizer's `solve` method can be called
            # with just the data and initial guess, implying other components like
            # forward_op and regularizer are pre-configured within the optimizer instance.
            # This is true for OSEM if its `system_matrix` is used as `forward_op`.
            # For FISTA/ADMM within PenalizedLikelihoodReconstruction, those are attributes.

            # A more robust pipeline might require the optimizer to expose a method that
            # clearly states its dependencies, or the pipeline might need to manage
            # forward_op and regularizer explicitly.

            # Example for OSEM (which has system_matrix internally):
            # if isinstance(self.reconstruction_algorithm, OrderedSubsetsExpectationMaximization):
            #    reconstructed_image = self.reconstruction_algorithm.solve(
            #        k_space_data=processed_data, # This is projection_data
            #        forward_op=self.reconstruction_algorithm.system_matrix, # OSEM uses its own
            #        initial_guess=initial_image_guess
            #    )
            # else:
            # For FISTA/ADMM/PenalizedLikelihood, they need `forward_op` and `regularizer`.
            # If these are stored within the optimizer (e.g. PenalizedLikelihoodReconstruction stores them),
            # the call could be:
            #    reconstructed_image = self.reconstruction_algorithm.solve(
            #        k_space_data=processed_data,
            #        forward_op=self.reconstruction_algorithm.system_matrix, # Assuming it has it
            #        regularizer=self.reconstruction_algorithm.regularizer, # Assuming it has it
            #        initial_guess=initial_image_guess
            #    )

            print("Placeholder: Core reconstruction algorithm execution.")
            print("The `solve` method of the configured optimizer needs to be called here with appropriate arguments.")
            print("Current `Optimizer` interface expects: k_space_data, forward_op, regularizer, initial_guess.")
            print("This pipeline's `run` method needs to manage how `forward_op` and `regularizer` are provided "
                  "if they are not part of the `reconstruction_algorithm` instance itself (like OSEM's system_matrix).")
            raise NotImplementedError("Core reconstruction step in pipeline.run() needs to correctly call "
                                      "the optimizer's solve() method with all necessary arguments "
                                      "(e.g., forward_op, regularizer if not encapsulated).")
            # Dummy assignment for structure:
            # reconstructed_image = initial_image_guess if initial_image_guess is not None else torch.zeros_like(processed_data) # Incorrect

        image_to_postprocess = reconstructed_image
        if verbose and not self.postprocessing_steps:
            print("No postprocessing steps defined.")
        for i, step_func in enumerate(self.postprocessing_steps):
            if verbose:
                print(f"Running postprocessing step {i+1}/{len(self.postprocessing_steps)}: {getattr(step_func, '__name__', 'Unnamed function')}...")
            image_to_postprocess = step_func(image_to_postprocess)
            if verbose:
                print(f"Image shape after step: {image_to_postprocess.shape}")

        if verbose:
            print("Reconstruction pipeline finished.")
        return image_to_postprocess


def convergence_monitor(iteration: int,
                        current_image: torch.Tensor,
                        previous_image: Optional[torch.Tensor] = None,
                        objective_value: Optional[float] = None,
                        tolerance: float = 1e-4) -> bool:
    """
    Monitors convergence criteria (e.g., change in image norm, objective value).
    Placeholder implementation.

    Args:
        iteration (int): The current iteration number.
        current_image (torch.Tensor): The image at the current iteration.
        previous_image (Optional[torch.Tensor]): The image from the previous iteration.
        objective_value (Optional[float]): The value of the objective function at the current iteration.
        tolerance (float): The tolerance level for convergence. If the change is below this,
                           convergence is considered reached.

    Returns:
        bool: True if convergence criteria are met, False otherwise.
    """
    print(f"Convergence monitor called at iteration {iteration}. Objective: {objective_value}")
    if previous_image is not None:
        diff_norm = torch.linalg.norm((current_image - previous_image).flatten())
        prev_norm = torch.linalg.norm(previous_image.flatten()) + 1e-9 # Avoid division by zero
        relative_change = diff_norm / prev_norm
        print(f"Relative change from previous image: {relative_change.item():.2e}")
        if relative_change < tolerance:
            print(f"Convergence criterion (relative image change < {tolerance:.1e}) met.")
            # return True # This would stop it
    raise NotImplementedError("`convergence_monitor` logic is not fully implemented. "
                              "Specific criteria (e.g., max iterations, objective change) should be added.")
    # return False # Default to not converged

def metrics_calculator(image: torch.Tensor,
                       reference_image: torch.Tensor,
                       metrics: List[str] = ['rmse', 'psnr', 'ssim']) -> Dict[str, float]:
    """
    Calculates image quality metrics (e.g., RMSE, PSNR, SSIM) between a reconstructed image
    and a reference (ground truth) image. Placeholder implementation.

    Args:
        image (torch.Tensor): The reconstructed image.
        reference_image (torch.Tensor): The reference (ground truth) image.
        metrics (List[str]): A list of metric names to calculate.
                             Supported: 'rmse', 'psnr', 'ssim' (if available).

    Returns:
        Dict[str, float]: A dictionary where keys are metric names and values are the calculated scores.
    """
    results: Dict[str, float] = {}
    print(f"Metrics calculator called for metrics: {metrics}")

    if image.shape != reference_image.shape:
        # Try to adjust if one is (C,H,W) and other is (H,W) by unsqueezing/squeezing
        if image.ndim == reference_image.ndim + 1 and image.shape[0] == 1:
            image = image.squeeze(0)
        elif reference_image.ndim == image.ndim + 1 and reference_image.shape[0] == 1:
            reference_image = reference_image.squeeze(0)
        
        if image.shape != reference_image.shape: # Check again
            raise ValueError(f"Image shapes must match for metrics calculation. "
                             f"Got {image.shape} and {reference_image.shape}")

    if image.device != reference_image.device:
        reference_image = reference_image.to(image.device)
    
    image = image.float()
    reference_image = reference_image.float()

    # Ensure images are in a comparable range (e.g. 0-1 or 0-255) for PSNR/SSIM if they expect it.
    # This is a placeholder; actual range depends on the metric functions.
    # For SSIM from reconlib, it might handle ranges internally or expect specific ranges.

    for metric_name in metrics:
        if metric_name == 'rmse':
            mse = torch.mean((image - reference_image) ** 2)
            results['rmse'] = torch.sqrt(mse).item()
            print(f"RMSE calculated: {results['rmse']:.4f}")
        elif metric_name == 'psnr':
            if psnr is not None:
                # psnr_val = psnr(reference_image, image, data_range=...) # data_range might be needed
                # This is a placeholder, actual psnr function might vary
                # For now, a basic one assuming data_range is max value of reference_image
                mse = torch.mean((image - reference_image) ** 2)
                if mse == 0: # Perfect match
                    results['psnr'] = float('inf')
                else:
                    data_range = torch.max(reference_image) - torch.min(reference_image)
                    if data_range == 0 : data_range = torch.max(image) # if ref is flat
                    if data_range == 0 : data_range = 1.0 # if both are flat
                    results['psnr'] = 20 * torch.log10(data_range / torch.sqrt(mse)).item()
                print(f"PSNR calculated: {results['psnr']:.2f} dB")
            else:
                print(f"PSNR calculation skipped (reconlib.metrics.image_metrics.psnr not available).")
        elif metric_name == 'ssim':
            if ssim is not None:
                # ssim_val = ssim(reference_image.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0), data_range=...)
                # The ssim function from reconlib.metrics.image_metrics likely expects (N, C, H, W)
                # and might have a data_range parameter.
                # This is a placeholder for the actual call.
                # For now, assume images are (H,W) or (C,H,W) and need batch/channel dims.
                ref_for_ssim = reference_image
                img_for_ssim = image
                if ref_for_ssim.ndim == 2: # H, W -> N, C, H, W
                    ref_for_ssim = ref_for_ssim.unsqueeze(0).unsqueeze(0)
                    img_for_ssim = img_for_ssim.unsqueeze(0).unsqueeze(0)
                elif ref_for_ssim.ndim == 3: # C, H, W -> N, C, H, W
                    ref_for_ssim = ref_for_ssim.unsqueeze(0)
                    img_for_ssim = img_for_ssim.unsqueeze(0)
                
                # results['ssim'] = ssim(ref_for_ssim, img_for_ssim, data_range=...).item()
                # Replace with actual call if ssim function is known.
                # For now, also a placeholder.
                print(f"SSIM calculation would use shapes: ref {ref_for_ssim.shape}, img {img_for_ssim.shape}")
            else:
                print(f"SSIM calculation skipped (reconlib.metrics.image_metrics.ssim not available).")
        else:
            print(f"Warning: Unknown metric '{metric_name}' requested.")

    raise NotImplementedError("`metrics_calculator` is a placeholder. "
                              "Actual metric calculations (esp. PSNR, SSIM using reconlib.metrics) need to be correctly implemented.")
    # return results # This would be the actual return
