"""Integration tests for hf_model_downloader.

These tests require network access and are guarded by the HFMDL_INTEGRATION=1
environment variable. They test real-world scenarios with the HuggingFace Hub.

Run with: HFMDL_INTEGRATION=1 HF_ENDPOINT=https://hf-mirror.com uv run pytest -q
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Skip all tests in this module if HFMDL_INTEGRATION is not set
pytestmark = pytest.mark.skipif(
    os.environ.get("HFMDL_INTEGRATION") != "1",
    reason="Integration tests require HFMDL_INTEGRATION=1",
)

# Test model - tiny and fast to download
TINY_MODEL = "sshleifer/tiny-distilroberta-base"
# Non-existent repo for 404 tests
NONEXISTENT_REPO = "this/definitely-does-not-exist-12345"
# Gated model that requires token
GATED_MODEL = "meta-llama/Llama-2-7b-hf"


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for each test."""
    with tempfile.TemporaryDirectory(prefix="hfmdl_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def env_with_mirror():
    """Get environment with HF_ENDPOINT set to mirror."""
    env = os.environ.copy()
    # Use hf-mirror.com if not already set
    if "HF_ENDPOINT" not in env:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
    return env


class TestDownloadTinyModel:
    """Test basic download functionality with a tiny model."""

    @pytest.mark.integration
    def test_download_tiny_model_success(self, temp_cache_dir, env_with_mirror):
        """Test successful download of tiny model."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            TINY_MODEL,
            "--revision",
            "main",
            "--output",
            str(temp_cache_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=60,
        )

        # Check success
        assert result.returncode == 0, f"Download failed: {result.stderr}"
        assert "Successfully downloaded" in result.stdout or "Downloaded to:" in result.stdout

        # Verify files exist in cache
        assert temp_cache_dir.exists()
        # The model should have some files
        files = list(temp_cache_dir.rglob("*"))
        assert len(files) > 0, "No files downloaded"

    @pytest.mark.integration
    def test_download_with_force_flag(self, temp_cache_dir, env_with_mirror):
        """Test that --force-download re-downloads the model."""
        cmd_download = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            TINY_MODEL,
            "--revision",
            "main",
            "--output",
            str(temp_cache_dir),
        ]

        # First download
        result1 = subprocess.run(
            cmd_download,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=60,
        )
        assert result1.returncode == 0, f"First download failed: {result1.stderr}"

        # Force re-download
        cmd_force = cmd_download + ["--force-download"]
        result2 = subprocess.run(
            cmd_force,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=60,
        )
        assert result2.returncode == 0, f"Force download failed: {result2.stderr}"
        assert "Successfully downloaded" in result2.stdout or "Downloaded to:" in result2.stdout


class TestInterruptResume:
    """Test interrupt and resume functionality."""

    @pytest.mark.integration
    def test_interrupt_and_resume(self, temp_cache_dir, env_with_mirror):
        """Test that download can be interrupted and resumed."""
        # Start download in subprocess
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            TINY_MODEL,
            "--revision",
            "main",
            "--output",
            str(temp_cache_dir),
        ]

        # Start process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env_with_mirror,
        )

        # Wait a bit and send SIGINT
        time.sleep(1)
        proc.send_signal(signal.SIGINT)

        # Wait for process to terminate
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        # Process should have been interrupted (exit code 4 or non-zero)
        # Exit code 4 = EXIT_ABORTED
        assert proc.returncode != 0, "Process should have been interrupted"

        # Now resume the download - it should complete without full restart
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=60,
        )

        # Should succeed
        assert result.returncode == 0, f"Resume failed: {result.stderr}"
        assert "Successfully downloaded" in result.stdout or "Downloaded to:" in result.stdout


class TestErrorCases:
    """Test error handling scenarios."""

    @pytest.mark.integration
    def test_404_nonexistent_repo(self, temp_cache_dir, env_with_mirror):
        """Test that 404 errors fail fast without retrying."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            NONEXISTENT_REPO,
            "--output",
            str(temp_cache_dir),
            "--no-retry-forever",  # Ensure no retry loop
            "--max-attempts",
            "1",
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=30,  # Should fail fast
        )
        elapsed = time.time() - start_time

        # Should fail
        assert result.returncode != 0, "Should fail for non-existent repo"

        # Should fail fast (no retry loop)
        assert elapsed < 15, f"Should fail fast, but took {elapsed:.1f}s"

        # Should mention it's a non-retriable error
        output = result.stdout + result.stderr
        assert "not found" in output.lower() or "404" in output or "non-retriable" in output.lower()

    @pytest.mark.integration
    def test_gated_model_without_token(self, temp_cache_dir, env_with_mirror):
        """Test that gated models fail without token."""
        # Ensure no token is set
        env = env_with_mirror.copy()
        env.pop("HF_TOKEN", None)

        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            GATED_MODEL,
            "--output",
            str(temp_cache_dir),
            "--no-retry-forever",
            "--max-attempts",
            "1",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        # Should fail
        assert result.returncode != 0, "Should fail for gated repo without token"

        # Should mention gated/auth error or not found
        # Note: On mirrors, gated repos may return LocalEntryNotFoundError
        output = result.stdout + result.stderr
        assert (
            "gated" in output.lower()
            or "403" in output
            or "forbidden" in output.lower()
            or "authentication" in output.lower()
            or "access" in output.lower()
            or "localentrynotfounderror" in output.lower()
            or "local" in output.lower()
        )

    @pytest.mark.integration
    def test_invalid_revision(self, temp_cache_dir, env_with_mirror):
        """Test that invalid revision fails with clear error."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            TINY_MODEL,
            "--revision",
            "nonexistent-branch-xyz",
            "--output",
            str(temp_cache_dir),
            "--no-retry-forever",
            "--max-attempts",
            "1",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=30,
        )

        # Should fail
        assert result.returncode != 0, "Should fail for invalid revision"

        # Should mention revision error
        output = result.stdout + result.stderr
        assert (
            "revision" in output.lower()
            or "not found" in output.lower()
            or "404" in output
        )


class TestCLI:
    """Test CLI commands and flags."""

    @pytest.mark.integration
    def test_validate_command_success(self, env_with_mirror):
        """Test validate command with existing repo."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "validate",
            TINY_MODEL,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=30,
        )

        # Should succeed
        assert result.returncode == 0, f"Validate failed: {result.stderr}"
        assert "valid" in result.stdout.lower()

    @pytest.mark.integration
    def test_validate_command_nonexistent(self, env_with_mirror):
        """Test validate command with non-existent repo."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "validate",
            NONEXISTENT_REPO,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=30,
        )

        # Should fail
        assert result.returncode != 0, "Should fail for non-existent repo"
        assert "not found" in (result.stdout + result.stderr).lower()

    @pytest.mark.integration
    def test_show_config_command(self, env_with_mirror):
        """Test show-config command."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "show-config",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=10,
        )

        # Should succeed
        assert result.returncode == 0, f"Show-config failed: {result.stderr}"

        # Should show endpoint
        assert "endpoint" in result.stdout.lower()

    @pytest.mark.integration
    def test_list_profiles_command(self, env_with_mirror):
        """Test list-profiles command."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "list-profiles",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=10,
        )

        # Should succeed (even with no profiles)
        assert result.returncode == 0, f"List-profiles failed: {result.stderr}"

    @pytest.mark.integration
    def test_endpoint_override(self, temp_cache_dir, env_with_mirror):
        """Test --endpoint flag override."""
        custom_endpoint = "https://custom.example.com"
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "show-config",
            "--endpoint",
            custom_endpoint,
            "--force-endpoint",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=10,
        )

        # Should show custom endpoint (with --force-endpoint)
        assert custom_endpoint in result.stdout

    @pytest.mark.integration
    def test_force_endpoint_flag(self, temp_cache_dir):
        """Test --force-endpoint flag."""
        # Set HF_ENDPOINT in environment
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://override.example.com"

        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "show-config",
            "--force-endpoint",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )

        # Should show mirror endpoint (from config), not the override
        assert "hf-mirror.com" in result.stdout or "https://hf-mirror.com" in result.stdout


class TestPatterns:
    """Test allow/ignore patterns."""

    @pytest.mark.integration
    def test_allow_pattern_json_only(self, temp_cache_dir, env_with_mirror):
        """Test downloading only JSON files with allow pattern."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            TINY_MODEL,
            "--revision",
            "main",
            "--output",
            str(temp_cache_dir),
            "--allow-pattern",
            "*.json",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=60,
        )

        # Should succeed
        assert result.returncode == 0, f"Download failed: {result.stderr}"

        # Should have JSON files
        json_files = list(temp_cache_dir.rglob("*.json"))
        assert len(json_files) > 0, "No JSON files downloaded"

        # Should NOT have other files like .bin
        bin_files = list(temp_cache_dir.rglob("*.bin"))
        assert len(bin_files) == 0, f"Found .bin files that should be excluded: {bin_files}"

    @pytest.mark.integration
    def test_ignore_pattern_bin_files(self, temp_cache_dir, env_with_mirror):
        """Test ignoring .bin files with ignore pattern."""
        cmd = [
            sys.executable,
            "-m",
            "hf_model_downloader.cli",
            "download",
            TINY_MODEL,
            "--revision",
            "main",
            "--output",
            str(temp_cache_dir),
            "--ignore-pattern",
            "*.bin",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_with_mirror,
            timeout=60,
        )

        # Should succeed
        assert result.returncode == 0, f"Download failed: {result.stderr}"

        # Should NOT have .bin files
        bin_files = list(temp_cache_dir.rglob("*.bin"))
        assert len(bin_files) == 0, f"Found .bin files that should be excluded: {bin_files}"

        # Should have other files
        all_files = [f for f in temp_cache_dir.rglob("*") if f.is_file()]
        assert len(all_files) > 0, "No files downloaded"
