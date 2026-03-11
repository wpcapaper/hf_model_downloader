"""Unit tests for the retry classifier."""

import pytest

from hf_model_downloader.errors import (
    ConfigurationError,
    ValidationError,
    classify_error,
)

try:
    from huggingface_hub.utils import HfHubHTTPError

    HAS_HF_HUB = True
except ImportError:
    HfHubHTTPError = None
    HAS_HF_HUB = False


class MockResponse:
    """Mock HTTP response object."""

    def __init__(self, status_code: int):
        self.status_code = status_code


def create_hf_http_error(message: str, status_code: int | None = None) -> Exception:
    """
    Create a mock HfHubHTTPError for testing.

    We use a mock because creating a real HfHubHTTPError requires complex setup
    with httpx.Response objects. The mock has the same attributes that classify_error
    checks for (response.status_code).
    """

    class MockHfHubHTTPError(Exception):
        """Mock HfHubHTTPError for testing."""

        def __init__(self, msg: str, status: int | None = None):
            super().__init__(msg)
            # Always create a response object for duck typing
            # If status is None, create response without status_code attribute
            if status is not None:
                self.response = MockResponse(status)
            else:
                # Response without status_code - simulates malformed HTTP error
                class MockResponseWithoutStatus:
                    pass

                self.response = MockResponseWithoutStatus()

    return MockHfHubHTTPError(message, status_code)


class TestRetriableErrors:
    """Tests for retriable error classification."""

    def test_rate_limit_429(self) -> None:
        """Test that 429 rate limit errors are retriable."""
        exc = create_hf_http_error("Rate limit exceeded", status_code=429)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Rate limited" in reason

    def test_server_error_500(self) -> None:
        """Test that 500 server errors are retriable."""
        exc = create_hf_http_error("Internal server error", status_code=500)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Server error" in reason
        assert "500" in reason

    def test_server_error_502(self) -> None:
        """Test that 502 bad gateway errors are retriable."""
        exc = create_hf_http_error("Bad gateway", status_code=502)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Server error" in reason

    def test_connection_error_dns_failure(self) -> None:
        """Test that DNS resolution failures are retriable."""
        try:
            import requests.exceptions

            exc = requests.exceptions.ConnectionError(
                "Failed to resolve 'huggingface.co': DNS failure"
            )
            is_retriable, reason = classify_error(exc)
            assert is_retriable is True
            assert "DNS" in reason
        except ImportError:
            pytest.skip("requests not available")

    def test_timeout_error(self) -> None:
        """Test that timeout errors are retriable."""
        try:
            import requests.exceptions

            exc = requests.exceptions.Timeout("Connection timed out")
            is_retriable, reason = classify_error(exc)
            assert is_retriable is True
            assert "timeout" in reason.lower()
        except ImportError:
            pytest.skip("requests not available")

    def test_http_error_without_status_code(self) -> None:
        """Test that HTTP errors without status codes are retriable (conservative)."""
        exc = create_hf_http_error("Unknown HTTP error", status_code=None)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "HTTP error" in reason


class TestNonRetriableErrors:
    """Tests for non-retriable error classification."""

    def test_authentication_error_401(self) -> None:
        """Test that 401 authentication errors are non-retriable."""
        exc = create_hf_http_error("Unauthorized", status_code=401)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Authentication" in reason or "401" in reason

    def test_forbidden_error_403(self) -> None:
        """Test that 403 forbidden errors are non-retriable."""
        exc = create_hf_http_error("Forbidden", status_code=403)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "forbidden" in reason.lower() or "403" in reason

    def test_not_found_error_404(self) -> None:
        """Test that 404 not found errors are non-retriable."""
        exc = create_hf_http_error("Not found", status_code=404)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "not found" in reason.lower() or "404" in reason

    def test_disk_full_error(self) -> None:
        """Test that disk full errors are non-retriable."""
        exc = OSError("[Errno 28] No space left on device")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Disk full" in reason or "space" in reason.lower()

    def test_permission_denied_error(self) -> None:
        """Test that permission errors are non-retriable."""
        exc = PermissionError("[Errno 13] Permission denied: '/cache/models'")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Permission" in reason

    def test_validation_error(self) -> None:
        """Test that validation errors are non-retriable."""
        exc = ValidationError("Invalid repository ID format")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Validation" in reason

    def test_configuration_error(self) -> None:
        """Test that configuration errors are non-retriable."""
        exc = ConfigurationError("Invalid configuration")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Configuration" in reason


class TestHuggingFaceHubExceptions:
    """Tests for huggingface_hub specific exceptions (if available)."""

    def test_repository_not_found(self) -> None:
        """Test that RepositoryNotFoundError is non-retriable."""
        try:
            from huggingface_hub import RepositoryNotFoundError

            exc = RepositoryNotFoundError("Repo 'nonexistent/model' not found")
            is_retriable, reason = classify_error(exc)
            assert is_retriable is False
            assert "Repository not found" in reason
        except ImportError:
            pytest.skip("huggingface_hub not available")

    def test_revision_not_found(self) -> None:
        """Test that RevisionNotFoundError is non-retriable."""
        try:
            from huggingface_hub import RevisionNotFoundError

            exc = RevisionNotFoundError("Revision 'bad-branch' not found")
            is_retriable, reason = classify_error(exc)
            assert is_retriable is False
            assert "Revision not found" in reason
        except ImportError:
            pytest.skip("huggingface_hub not available")

    def test_gated_repo_error(self) -> None:
        """Test that GatedRepoError is non-retriable."""
        try:
            from huggingface_hub import GatedRepoError

            exc = GatedRepoError("Access to gated repo denied")
            is_retriable, reason = classify_error(exc)
            assert is_retriable is False
            assert "Gated repository" in reason or "gated" in reason.lower()
        except ImportError:
            pytest.skip("huggingface_hub not available")


class TestEdgeCases:
    """Tests for edge cases and unknown errors."""

    def test_unknown_error_is_non_retriable(self) -> None:
        """Test that unknown errors are non-retriable by default."""
        exc = RuntimeError("Something unexpected happened")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Unknown error" in reason

    def test_generic_oserror(self) -> None:
        """Test that generic OSError is non-retriable."""
        exc = OSError("Some system error")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "system error" in reason.lower()
