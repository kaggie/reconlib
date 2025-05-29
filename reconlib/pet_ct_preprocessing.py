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

def normalize_projection_data(projection_data: torch.Tensor, detector_sensitivity_map: torch.Tensor) -> torch.Tensor:
    """
    Normalizes CT projection data for detector sensitivity variations (e.g., air scan normalization).
    Also known as flat-field correction. For CT, this often involves taking the negative log.
    Placeholder implementation.

    Args:
        projection_data (torch.Tensor): Raw CT projection data (intensity values).
        detector_sensitivity_map (torch.Tensor): A map representing detector sensitivities,
                                                 often obtained from an air scan (I_0).
                                                 Should match the shape of projection_data.

    Returns:
        torch.Tensor: Normalized CT projection data (often in units of attenuation).
    """
    # if projection_data.shape != detector_sensitivity_map.shape:
    #     raise ValueError("projection_data and detector_sensitivity_map must have the same shape.")
    # if projection_data.device.type != detector_sensitivity_map.device.type:
    #     raise ValueError("projection_data and detector_sensitivity_map must be on the same device.")

    print(f"Placeholder: Would normalize CT projection data of shape {projection_data.shape} "
          f"using sensitivity map of shape {detector_sensitivity_map.shape}.")
    raise NotImplementedError("`normalize_projection_data` function for CT is not yet implemented.")

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

    # normalize_projection_data (CT) example
#     dummy_ct_projections = torch.rand(1, 1, 360, 256) * 4096 # Raw intensities
#     dummy_air_scan = torch.rand(1, 1, 360, 256) * 4000 + 100 # I_0, should be higher than attenuated signal
#     try:
#         norm_ct_data = normalize_projection_data(dummy_ct_projections, dummy_air_scan)
#     except NotImplementedError as e:
#         print(e)
