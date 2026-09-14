class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""


class DataValidationError(DataPipelineError):
    """Raised when input data fails validation."""


class SampleAlignmentError(DataPipelineError):
    """Raised when expression and metadata samples do not align."""