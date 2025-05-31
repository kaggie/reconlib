"""
reconlib.modalities
===================

This package contains modules for different imaging modalities within reconlib.
Each submodule (e.g., pcct, pet, ct, ultrasound, oct, em, xray_phase_contrast, seismic, astronomical)
provides specialized tools for that particular modality.
"""

from . import pcct
from . import pet
from . import ct
from . import spect # Uncommented and added
# Example:
# from . import ultrasound

__all__ = [
    'pcct',
    'pet',
    'ct',
    'spect', # Added
    # 'ultrasound',
]
