"""Tests for utility functions."""

import shutil

import numpy as np
import pytest
from pathlib import Path
from core.utils import format_timestamp, format_duration, load_audio_ffmpeg, validate_file_path


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestLoadAudioFfmpeg:
    """Tests for the single-call ffmpeg decoder."""

    @pytest.fixture
    def wav_path(self, tmp_path):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "tone.wav"
        # 10s stereo 44.1kHz - exercises downmix + resample
        t = np.arange(10 * 44100) / 44100.0
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(path), np.stack([tone, tone], axis=1), 44100)
        return path

    def test_full_file_is_16k_mono(self, wav_path):
        audio = load_audio_ffmpeg(str(wav_path))
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert abs(len(audio) - 10 * 16000) < 160

    def test_range_is_decoded_only(self, wav_path):
        audio = load_audio_ffmpeg(str(wav_path), start_time=2.0, end_time=5.5)
        assert abs(len(audio) - int(3.5 * 16000)) < 160

    def test_missing_file_returns_none(self, tmp_path):
        assert load_audio_ffmpeg(str(tmp_path / "missing.mp3")) is None
