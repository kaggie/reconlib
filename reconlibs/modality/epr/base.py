import abc

class EPRImaging(abc.ABC):
  """Abstract base class for EPR imaging techniques."""

  def __init__(self, metadata: dict, data: dict):
    """Initializes the EPR imaging class.

    Args:
      metadata: A dictionary containing metadata for the EPR experiment.
      data: A dictionary containing the EPR data.
    """
    self.metadata = metadata
    self.data = data

  @abc.abstractmethod
  def get_physics_model(self):
    """Returns the physics model for the EPR technique.

    This method should be implemented by subclasses to define the specific
    physics model used for reconstruction.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def reconstruct(self, *args, **kwargs):
    """Performs image reconstruction from the EPR data.

    This method should be implemented by subclasses to define the specific
    reconstruction algorithm used for the EPR technique.
    """
    raise NotImplementedError
