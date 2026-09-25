#!/usr/bin/env python3
"""Setup script for qwen3_transcribe package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies (required)
# Note: qwen-asr pins transformers==4.57.6 and accelerate==1.12.0.
core_requirements = [
    "qwen-asr>=0.0.6",
    "torch>=2.2.0",
    "librosa>=0.10.0",
    "ffmpeg-python>=0.2.0",
    "soundfile>=0.12.0",
    "pydub>=0.25.0",
]

# Optional dependencies for Flash Attention
flash_attn_requirements = [
    "ninja>=1.11.0",
    "psutil>=7.0.0",
    "flash-attn>=2.7.4",
]

# Optional dependencies for enhancement
enhancement_requirements = [
    "google-genai>=1.0.0",
    "python-dotenv>=1.0.0",
]

setup(
    name="qwen3-transcribe",
    version="1.0.0",
    description="Audio transcription using Qwen3-ASR-1.7B with optional Gemini enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="JonPark0",
    author_email="jonpark0@users.noreply.github.com",
    url="https://github.com/JonPark0/qwen3_transcribe",
    packages=find_packages(),
    py_modules=['convert', 'enhance'],
    python_requires=">=3.10",  # qwen-asr -> accelerate==1.12.0 needs 3.10+
    install_requires=core_requirements,
    extras_require={
        "flash-attn": flash_attn_requirements,
        "enhancement": enhancement_requirements,
        "all": flash_attn_requirements + enhancement_requirements,
    },
    entry_points={
        "console_scripts": [
            "qwen3-transcribe=convert:main",
            "qwen3-enhance=enhance:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="qwen qwen3-asr transcription audio speech-to-text ai gemini",
    include_package_data=True,
    zip_safe=False,
)
