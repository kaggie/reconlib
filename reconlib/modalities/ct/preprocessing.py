# reconlib/modalities/ct/preprocessing.py
"""
This module will contain CT-specific preprocessing functions.
"""
import torch
import numpy as np # normalize_projection_data uses np
from typing import Optional # normalize_projection_data uses Optional

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
#     # normalize_projection_data (CT) example
#     dummy_ct_projections = torch.rand(1, 1, 360, 256) * 4096 # Raw intensities
#     dummy_air_scan = torch.rand(1, 1, 360, 256) * 4000 + 100 # I_0, should be higher than attenuated signal
#     try:
#         norm_ct_data = normalize_projection_data(dummy_ct_projections, dummy_air_scan)
#     except NotImplementedError as e:
#         print(e)
