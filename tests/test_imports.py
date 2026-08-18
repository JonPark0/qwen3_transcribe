"""Test that all modules can be imported successfully."""

import pytest


def test_import_core_modules():
    """Test that core modules can be imported."""
    try:
        from core import Qwen3ASRTranscriber, TranscriptEnhancer
        assert Qwen3ASRTranscriber is not None
        assert TranscriptEnhancer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_import_utils():
    """Test that utility functions can be imported."""
    try:
        from core import format_duration, format_timestamp, validate_file_path
        assert format_duration is not None
        assert format_timestamp is not None
        assert validate_file_path is not None
    except ImportError as e:
        pytest.fail(f"Failed to import utils: {e}")


def test_import_convert():
    """Test that convert module can be imported."""
    try:
        import convert
        assert convert is not None
        assert hasattr(convert, 'main')
    except ImportError as e:
        pytest.fail(f"Failed to import convert: {e}")


def test_import_enhance():
    """Test that enhance module can be imported."""
    try:
        import enhance
        assert enhance is not None
        assert hasattr(enhance, 'main')
    except ImportError as e:
        pytest.fail(f"Failed to import enhance: {e}")
