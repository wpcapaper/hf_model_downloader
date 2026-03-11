"""Custom exceptions for hf_model_downloader."""


class HFModelDownloaderError(Exception):
    """Base exception for hf_model_downloader."""

    pass


class DownloadError(HFModelDownloaderError):
    """Error during model download."""

    pass


class ConfigurationError(HFModelDownloaderError):
    """Error in configuration."""

    pass


class ProfileNotFoundError(HFModelDownloaderError):
    """Requested profile not found."""

    pass


class ValidationError(HFModelDownloaderError):
    """Validation error."""

    pass
