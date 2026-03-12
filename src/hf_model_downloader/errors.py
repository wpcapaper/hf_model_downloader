"""Custom exceptions for hf_model_downloader."""

from typing import Tuple


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


try:
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
except ImportError:
    # Fallback for testing without huggingface_hub installed
    HfHubHTTPError = None
    RepositoryNotFoundError = None
    RevisionNotFoundError = None
    EntryNotFoundError = None
    GatedRepoError = None

try:
    import requests.exceptions
    from urllib3.exceptions import TimeoutError as URLLIB3TimeoutError
except ImportError:
    requests = None
    URLLIB3TimeoutError = None

def classify_error(exc: Exception) -> Tuple[bool, str]:
    """
    Classify an exception as retriable or non-retriable.

    Args:
        exc: The exception to classify

    Returns:
        Tuple of (is_retriable: bool, reason: str)
        - is_retriable: True if the error is transient and worth retrying
        - reason: Human-readable explanation of the classification
    """
    # Check for disk/storage errors (non-retriable)
    if isinstance(exc, (OSError, PermissionError)):
        # Disk full, permission denied, etc.
        error_msg = str(exc).lower()
        if "no space left" in error_msg or "disk full" in error_msg:
            return False, "Disk full - no space available"
        if "permission" in error_msg or "access" in error_msg:
            return False, "Permission denied - check file/directory permissions"
        return False, f"Local system error: {type(exc).__name__}"

    # Check for repository not found (non-retriable)
    if RepositoryNotFoundError is not None and isinstance(exc, RepositoryNotFoundError):
        return False, "Repository not found - check repo_id"

    # Check for revision not found (non-retriable)
    if RevisionNotFoundError is not None and isinstance(exc, RevisionNotFoundError):
        return False, "Revision not found - check branch/tag/commit"

    # Check for entry not found (non-retriable)
    if EntryNotFoundError is not None and isinstance(exc, EntryNotFoundError):
        return False, "File not found in repository"

    # Check for gated repo errors (non-retriable)
    if GatedRepoError is not None and isinstance(exc, GatedRepoError):
        return False, "Gated repository - authentication or access approval required"

    # Check for HTTP errors with status codes
    # First check for known HfHubHTTPError, then fall back to duck typing
    is_http_error = False
    if HfHubHTTPError is not None and isinstance(exc, HfHubHTTPError):
        is_http_error = True
    elif hasattr(exc, "response") and exc.response is not None:
        # Duck typing: if it has a response object, treat as HTTP error
        is_http_error = True
    if is_http_error:
        # Try to extract status code from response
        status_code = None
        if hasattr(exc, "response") and exc.response is not None:
            status_code = getattr(exc.response, "status_code", None)

        if status_code is not None:
            # Rate limiting (retriable)
            if status_code == 429:
                return True, "Rate limited - wait and retry"

            # Authentication errors (non-retriable)
            if status_code == 401:
                return False, "Authentication failed - check HF_TOKEN"

            # Authorization errors (non-retriable)
            if status_code == 403:
                return False, "Access forbidden - check permissions or gated repo access"

            # Not found (non-retriable)
            if status_code == 404:
                return False, "Resource not found - check repo_id and file path"

            # Server errors (retriable)
            if 500 <= status_code < 600:
                return True, f"Server error ({status_code}) - transient, retry recommended"

        # Unknown HTTP error - be conservative and retry
        return True, "HTTP error - potentially transient, retry recommended"

    # Check for timeout errors (retriable)
    if requests is not None:
        if isinstance(exc, requests.exceptions.Timeout):
            return True, "Request timeout - transient, retry recommended"

        if isinstance(exc, requests.exceptions.ConnectionError):
            error_msg = str(exc).lower()
            # DNS failures
            if "dns" in error_msg or "name resolution" in error_msg or "getaddrinfo" in error_msg:
                return True, "DNS resolution failure - transient, retry recommended"
            # Connection reset, refused, etc.
            return True, "Connection error - transient, retry recommended"

    # Check for urllib3 timeout (retriable)
    if URLLIB3TimeoutError is not None and isinstance(exc, URLLIB3TimeoutError):
        return True, "Network timeout - transient, retry recommended"

    # Check for any ReadError by name (duck typing for different httpx versions)
    # ReadError happens when connection is interrupted during file transfer
    if type(exc).__name__ == "ReadError":
        return True, "Connection interrupted during transfer - transient, retry recommended"
    # Check for our own validation errors (non-retriable)
    if isinstance(exc, ValidationError):
        return False, "Validation error - fix input parameters"

    # Check for configuration errors (non-retriable)
    if isinstance(exc, ConfigurationError):
        return False, "Configuration error - check settings"

    # Unknown error - be conservative and don't retry to avoid infinite loops
    return False, f"Unknown error type ({type(exc).__name__}) - not retrying by default"
