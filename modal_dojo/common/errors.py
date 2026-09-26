class TrainingDojoError(ValueError):
    """Base error for Modal Dojo."""


class TrainingDojoConfigError(TrainingDojoError):
    """Raised when a training or deploy config is invalid."""


class GpuAllocationError(TrainingDojoConfigError):
    """Raised when a recipe's cluster or parallelism settings are invalid."""
