"""Module for PET and CT specific data preprocessing utilities."""

import torch
import numpy as np

def normalize_counts(projection_data: torch.Tensor, decay_factors: torch.Tensor, acquisition_time: float) -> torch.Tensor:
    """
    Normalizes PET count data to correct for radioactive decay and acquisition time.
    Placeholder implementation.

    Args:
        projection_data (torch.Tensor): Raw PET projection data (counts).
        decay_factors (torch.Tensor): Tensor containing decay correction factors for each projection bin or event.
                                      This could be precomputed based on time stamps and isotope half-life.
        acquisition_time (float): Total acquisition time or effective time per projection/event.

    Returns:
        torch.Tensor: Normalized PET projection data.
    """
    # Basic device checks (optional for functions, but good practice if they become complex)
    # if projection_data.device.type != decay_factors.device.type:
    #     raise ValueError("projection_data and decay_factors must be on the same device.")

    print(f"Placeholder: Would normalize counts for data of shape {projection_data.shape} "
          f"using decay factors of shape {decay_factors.shape} and acquisition time {acquisition_time}.")
    raise NotImplementedError("`normalize_counts` function is not yet implemented.")

def randoms_correction(projection_data: torch.Tensor, randoms_estimate: torch.Tensor) -> torch.Tensor:
    """
    Subtracts estimated random coincidences from PET projection data.
    Placeholder implementation.

    Args:
        projection_data (torch.Tensor): Raw or partially processed PET projection data.
        randoms_estimate (torch.Tensor): Estimated randoms coincidences, matching the shape of projection_data.

    Returns:
        torch.Tensor: Projection data corrected for randoms.
    """
    # if projection_data.shape != randoms_estimate.shape:
    #     raise ValueError("projection_data and randoms_estimate must have the same shape.")
    # if projection_data.device.type != randoms_estimate.device.type:
    #     raise ValueError("projection_data and randoms_estimate must be on the same device.")

    print(f"Placeholder: Would perform randoms correction on data of shape {projection_data.shape} "
          f"using randoms estimate of shape {randoms_estimate.shape}.")
    raise NotImplementedError("`randoms_correction` function is not yet implemented.")

# Example Usage (commented out)
# if __name__ == '__main__':
#     # normalize_counts example
#     dummy_pet_data = torch.rand(1, 1, 180, 128) * 100 # Counts
#     dummy_decay_factors = torch.rand(1, 1, 180, 128) * 0.5 + 0.5 # Factors around 1
#     acq_time = 600.0 # seconds
#     try:
#         norm_counts = normalize_counts(dummy_pet_data, dummy_decay_factors, acq_time)
#     except NotImplementedError as e:
#         print(e)

    # randoms_correction example
#     dummy_randoms = torch.rand(1, 1, 180, 128) * 10
#     try:
#         corrected_for_randoms = randoms_correction(dummy_pet_data, dummy_randoms)
#     except NotImplementedError as e:
#         print(e)

    # CT function (normalize_projection_data) was moved to ct/preprocessing.py
    # So, its example usage is also removed from here.
