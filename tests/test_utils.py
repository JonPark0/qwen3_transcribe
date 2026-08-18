"""Tests for utility functions."""

import pytest
from pathlib import Path
from core.utils import format_timestamp, format_duration, validate_file_path


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_zero_seconds(self):
        """Test formatting of zero seconds."""
        assert format_timestamp(0) == "00:00:00"

    def test_seconds_only(self):
        """Test formatting of seconds only."""
        assert format_timestamp(45) == "00:00:45"

    def test_minutes_and_seconds(self):
        """Test formatting of minutes and seconds."""
        assert format_timestamp(125) == "00:02:05"

    def test_hours_minutes_seconds(self):
        """Test formatting of hours, minutes, and seconds."""
        assert format_timestamp(3661) == "01:01:01"

    def test_large_value(self):
        """Test formatting of large time values."""
        assert format_timestamp(7384) == "02:03:04"

    def test_none_value(self):
        """Test formatting of None value."""
        assert format_timestamp(None) == "00:00:00"


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_milliseconds(self):
        """Test formatting of sub-second durations."""
        assert format_duration(0.5) == "500ms"

    def test_seconds(self):
        """Test formatting of second-range durations."""
        assert format_duration(5.7) == "5.7s"
        assert format_duration(45.2) == "45.2s"

    def test_minutes(self):
        """Test formatting of minute-range durations."""
        result = format_duration(125.5)
        assert result == "2m 5.5s"

    def test_large_duration(self):
        """Test formatting of large durations."""
        result = format_duration(3661.2)
        assert result == "61m 1.2s"


class TestValidateFilePath:
    """Tests for validate_file_path function."""

    def test_valid_absolute_path(self, tmp_path):
        """Test validation of valid absolute path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        result = validate_file_path(str(test_file), must_exist=True)
        assert isinstance(result, Path)
        assert result.exists()

    def test_valid_path_no_exist_check(self):
        """Test validation without existence check."""
        result = validate_file_path("/tmp/nonexistent.txt", must_exist=False)
        assert isinstance(result, Path)

    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path("../../../etc/passwd")

    def test_nonexistent_file_with_must_exist(self):
        """Test error when file doesn't exist and must_exist=True."""
        with pytest.raises(ValueError, match="File does not exist"):
            validate_file_path("/tmp/definitely_does_not_exist_12345.txt", must_exist=True)

    def test_pathlib_path_input(self, tmp_path):
        """Test that Path objects are accepted as input."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        result = validate_file_path(test_file, must_exist=True)
        assert isinstance(result, Path)
        assert result.exists()
