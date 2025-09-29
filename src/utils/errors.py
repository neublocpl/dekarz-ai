class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DownloadError(PipelineError):
    """Raised when a file download fails."""
    pass

class ConversionError(PipelineError):
    """Raised when a file conversion fails."""
    pass