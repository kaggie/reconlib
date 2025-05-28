# reconlib/phase_unwrapping/__init__.py
"""
Phase unwrapping algorithms and utilities for 3D MRI data.

This module provides several PyTorch-based implementations for 3D phase unwrapping:
- `unwrap_phase_3d_quality_guided`: A path-following algorithm guided by a quality map.
- `unwrap_phase_3d_least_squares`: Solves for the unwrapped phase by formulating it as a 
                                   Poisson equation, solved with FFTs.
- `unwrap_phase_3d_goldstein`: A simplified Goldstein-style algorithm using k-space filtering.
- `unwrap_multi_echo_masked_reference`: Unwraps multi-echo phase data using a reference echo
                                        and a provided spatial unwrapper.

Additionally, it includes placeholders for other common algorithms:
- `unwrap_phase_puror` (PUROR)
- `unwrap_phase_romeo` (ROMEO)
- `unwrap_phase_deep_learning` (Deep Learning-based)

Utility functions, such as mask generation, are also available.
"""

from .puror import unwrap_phase_puror
from .romeo import unwrap_phase_romeo
from .deep_learning_unwrap import unwrap_phase_deep_learning
from .utils import generate_mask_for_unwrapping
from .quality_guided_unwrap import unwrap_phase_3d_quality_guided
from .least_squares_unwrap import unwrap_phase_3d_least_squares
from .goldstein_unwrap import unwrap_phase_3d_goldstein
from .reference_echo_unwrap import unwrap_multi_echo_masked_reference


__all__ = [
    "unwrap_phase_puror",
    "unwrap_phase_romeo",
    "unwrap_phase_deep_learning",
    "generate_mask_for_unwrapping",
    "unwrap_phase_3d_quality_guided",
    "unwrap_phase_3d_least_squares",
    "unwrap_phase_3d_goldstein",
    "unwrap_multi_echo_masked_reference",
]
