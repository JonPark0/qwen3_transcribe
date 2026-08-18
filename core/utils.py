"""Utility functions for whisper_transcribe."""

import os
from pathlib import Path
from typing import Union


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and sanitize file path to prevent directory traversal attacks.

    Args:
        file_path: Path to validate
        must_exist: If True, raises error if file doesn't exist

    Returns:
        Validated absolute Path object

    Raises:
        ValueError: If path contains suspicious patterns or doesn't exist (when must_exist=True)
    """
    try:
        path = Path(file_path).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}")

    # Check for suspicious patterns
    path_str = str(path)
    if '..' in Path(file_path).parts:
        raise ValueError(f"Path traversal detected in: {file_path}")

    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    return path


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS timestamp format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (HH:MM:SS)
    """
    if seconds is None:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
