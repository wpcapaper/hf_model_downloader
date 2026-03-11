"""Unit tests for error classification and backoff calculation."""

import pytest

from hf_model_downloader.downloader import _calculate_backoff
from hf_model_downloader.errors import (
    ConfigurationError,
    ValidationError,
    classify_error,
)


class MockResponse:
    """Mock HTTP response object."""

    def __init__(self, status_code: int):
        self.status_code = status_code


def create_http_error(message: str, status_code: int | None = None) -> Exception:
    """
    Create a mock HTTP error for testing.

    Uses duck typing - the mock has the same attributes that classify_error
    checks for (response.status_code).
    """

    class MockHTTPError(Exception):
        """Mock HTTP error for testing."""

        def __init__(self, msg: str, status: int | None = None):
            super().__init__(msg)
            if status is not None:
                self.response = MockResponse(status)
            else:
                # Response without status_code attribute
                class MockResponseWithoutStatus:
                    pass

                self.response = MockResponseWithoutStatus()

    return MockHTTPError(message, status_code)


class TestErrorClassifierRetriable:
    """Tests for retriable error classification."""

    def test_rate_limit_429(self) -> None:
        """Test that 429 rate limit errors are retriable."""
        exc = create_http_error("Rate limit exceeded", status_code=429)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Rate limited" in reason

    def test_server_error_500(self) -> None:
        """Test that 500 server errors are retriable."""
        exc = create_http_error("Internal server error", status_code=500)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Server error" in reason
        assert "500" in reason

    def test_server_error_502(self) -> None:
        """Test that 502 bad gateway errors are retriable."""
        exc = create_http_error("Bad gateway", status_code=502)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Server error" in reason

    def test_server_error_503(self) -> None:
        """Test that 503 service unavailable errors are retriable."""
        exc = create_http_error("Service unavailable", status_code=503)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "Server error" in reason

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

    def test_connection_error(self) -> None:
        """Test that connection errors are retriable."""
        try:
            import requests.exceptions

            exc = requests.exceptions.ConnectionError("Connection failed")
            is_retriable, reason = classify_error(exc)
            assert is_retriable is True
            assert "Connection error" in reason or "transient" in reason.lower()
        except ImportError:
            pytest.skip("requests not available")

    def test_dns_failure(self) -> None:
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

    def test_urllib3_timeout(self) -> None:
        """Test that urllib3 timeout errors are retriable."""
        try:
            from urllib3.exceptions import TimeoutError as URLLIB3TimeoutError

            exc = URLLIB3TimeoutError("Read timed out")
            is_retriable, reason = classify_error(exc)
            assert is_retriable is True
            assert "timeout" in reason.lower()
        except ImportError:
            pytest.skip("urllib3 not available")

    def test_http_error_without_status_code(self) -> None:
        """Test that HTTP errors without status codes are retriable."""
        exc = create_http_error("Unknown HTTP error", status_code=None)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is True
        assert "HTTP error" in reason


class TestErrorClassifierNonRetriable:
    """Tests for non-retriable error classification."""

    def test_authentication_error_401(self) -> None:
        """Test that 401 authentication errors are non-retriable."""
        exc = create_http_error("Unauthorized", status_code=401)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Authentication" in reason or "401" in reason

    def test_forbidden_error_403(self) -> None:
        """Test that 403 forbidden errors are non-retriable."""
        exc = create_http_error("Forbidden", status_code=403)
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "forbidden" in reason.lower() or "403" in reason

    def test_not_found_error_404(self) -> None:
        """Test that 404 not found errors are non-retriable."""
        exc = create_http_error("Not found", status_code=404)
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

    def test_generic_oserror(self) -> None:
        """Test that generic OSError is non-retriable."""
        exc = OSError("Some system error")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "system error" in reason.lower()

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

    def test_unknown_error(self) -> None:
        """Test that unknown errors are non-retriable by default."""
        exc = RuntimeError("Something unexpected happened")
        is_retriable, reason = classify_error(exc)
        assert is_retriable is False
        assert "Unknown error" in reason





class TestBackoffCalculation:
    """Tests for exponential backoff calculation with jitter disabled."""

    def test_backoff_attempt_0(self) -> None:
        """Test backoff for attempt 0 (first retry)."""
        # With jitter=0, results are deterministic
        wait = _calculate_backoff(attempt=0, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 1.0  # 1.0 * 2^0 = 1.0

    def test_backoff_attempt_1(self) -> None:
        """Test backoff for attempt 1."""
        wait = _calculate_backoff(attempt=1, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 2.0  # 1.0 * 2^1 = 2.0

    def test_backoff_attempt_2(self) -> None:
        """Test backoff for attempt 2."""
        wait = _calculate_backoff(attempt=2, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 4.0  # 1.0 * 2^2 = 4.0

    def test_backoff_attempt_3(self) -> None:
        """Test backoff for attempt 3."""
        wait = _calculate_backoff(attempt=3, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 8.0  # 1.0 * 2^3 = 8.0

    def test_backoff_attempt_4(self) -> None:
        """Test backoff for attempt 4."""
        wait = _calculate_backoff(attempt=4, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 16.0  # 1.0 * 2^4 = 16.0

    def test_backoff_attempt_5(self) -> None:
        """Test backoff for attempt 5."""
        wait = _calculate_backoff(attempt=5, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 32.0  # 1.0 * 2^5 = 32.0

    def test_backoff_attempt_6(self) -> None:
        """Test backoff for attempt 6."""
        wait = _calculate_backoff(attempt=6, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 60.0  # 1.0 * 2^6 = 64.0, but capped at max_wait=60.0

    def test_backoff_respects_max_wait(self) -> None:
        """Test that backoff is capped at max_wait."""
        # Even for high attempts
        wait = _calculate_backoff(attempt=10, base_wait=1.0, max_wait=60.0, jitter=0.0)
        assert wait == 60.0  # Capped at max_wait

    def test_backoff_custom_base_wait(self) -> None:
        """Test backoff with custom base_wait."""
        wait = _calculate_backoff(attempt=0, base_wait=2.0, max_wait=60.0, jitter=0.0)
        assert wait == 2.0  # 2.0 * 2^0 = 2.0

        wait = _calculate_backoff(attempt=1, base_wait=2.0, max_wait=60.0, jitter=0.0)
        assert wait == 4.0  # 2.0 * 2^1 = 4.0

        wait = _calculate_backoff(attempt=2, base_wait=2.0, max_wait=60.0, jitter=0.0)
        assert wait == 8.0  # 2.0 * 2^2 = 8.0

    def test_backoff_custom_max_wait(self) -> None:
        """Test backoff with custom max_wait."""
        wait = _calculate_backoff(attempt=0, base_wait=1.0, max_wait=30.0, jitter=0.0)
        assert wait == 1.0

        wait = _calculate_backoff(attempt=5, base_wait=1.0, max_wait=30.0, jitter=0.0)
        assert wait == 30.0  # Would be 32.0, capped at 30.0

    def test_backoff_with_jitter_enabled(self) -> None:
        """Test that jitter adds randomness (non-deterministic)."""
        # With jitter, results vary - just verify they're in reasonable range
        base_wait = 10.0
        jitter = 0.2

        waits = [
            _calculate_backoff(attempt=0, base_wait=base_wait, max_wait=60.0, jitter=jitter)
            for _ in range(10)
        ]

        # All should be in range (1-jitter)*10 to (1+jitter)*10 = 8.0 to 12.0
        for wait in waits:
            assert 8.0 <= wait <= 12.0

        # Should have some variation (not all the same)
        # This test might occasionally fail due to randomness, but very unlikely
        assert len(set(waits)) > 1 or len(waits) == 1  # Allow for edge case

    def test_backoff_zero_jitter_deterministic(self) -> None:
        """Test that jitter=0 produces deterministic results."""
        waits = [
            _calculate_backoff(attempt=2, base_wait=1.0, max_wait=60.0, jitter=0.0)
            for _ in range(10)
        ]

        # All should be exactly 4.0
        assert all(w == 4.0 for w in waits)

    def test_backoff_sequence(self) -> None:
        """Test a sequence of backoff attempts."""
        base_wait = 1.0
        max_wait = 60.0
        jitter = 0.0

        expected_sequence = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]

        for attempt, expected in enumerate(expected_sequence):
            wait = _calculate_backoff(attempt, base_wait, max_wait, jitter)
            assert wait == expected, f"Attempt {attempt}: expected {expected}, got {wait}"
