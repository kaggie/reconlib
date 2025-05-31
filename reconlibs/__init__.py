"""
ReconLibs Package
"""
# This file makes reconlibs a Python package.
# You can import submodules directly if they are added to __all__ here
# or by using their full path, e.g., from reconlibs.modality import ...

# For now, let's make it easy to import the modality module
from . import modality

__version__ = "0.1.0" # Example version

# You might want to expose certain things directly at the reconlibs level
# For example, if there's a very common class or function:
# from .modality.epr import EPRImaging

# Or define __all__ for `from reconlibs import *`
# __all__ = ['modality'] # Example
