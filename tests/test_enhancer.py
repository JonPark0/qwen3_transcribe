"""Tests for TranscriptEnhancer class."""

import pytest
from core.enhancer import TranscriptEnhancer


class TestTranscriptEnhancer:
    """Tests for TranscriptEnhancer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        enhancer = TranscriptEnhancer()
        assert enhancer.verbose is False
        assert enhancer.target_language is None
        assert enhancer._client is None
        assert enhancer.MAX_INPUT_TOKENS == 1_048_576
        assert enhancer.MAX_OUTPUT_TOKENS == 65_536

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        enhancer = TranscriptEnhancer(verbose=True, target_language="ko")
        assert enhancer.verbose is True
        assert enhancer.target_language == "ko"

    def test_log_verbose_enabled(self, capsys):
        """Test logging with verbose enabled."""
        enhancer = TranscriptEnhancer(verbose=True)
        enhancer.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_verbose_disabled(self, capsys):
        """Test logging with verbose disabled."""
        enhancer = TranscriptEnhancer(verbose=False)
        enhancer.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" not in captured.out

    def test_estimate_tokens(self):
        """Test token estimation."""
        enhancer = TranscriptEnhancer()
        text = "This is a test string"
        tokens = enhancer.estimate_tokens(text)
        # Estimation is len(text) // 4
        assert tokens == len(text) // 4

    def test_format_token_count(self):
        """Test token count formatting."""
        enhancer = TranscriptEnhancer()
        assert enhancer.format_token_count(1000) == "1,000"
        assert enhancer.format_token_count(1000000) == "1,000,000"

    def test_estimate_output_requirements(self):
        """Test output token estimation."""
        enhancer = TranscriptEnhancer()
        input_tokens = 1000
        output_tokens = enhancer.estimate_output_requirements(input_tokens)
        # Should be 1.3x the input
        assert output_tokens == int(input_tokens * 1.3)

    def test_validate_input_size_valid(self):
        """Test input size validation with valid input."""
        enhancer = TranscriptEnhancer()
        content = "Short content"
        prompt = "Short prompt"
        is_valid, error = enhancer.validate_input_size(content, prompt)
        assert is_valid is True
        assert error == ""

    def test_validate_input_size_too_large(self):
        """Test input size validation with oversized input."""
        enhancer = TranscriptEnhancer()
        # Create a very large content (exceeding token limit)
        content = "x" * (enhancer.MAX_INPUT_TOKENS * 5)  # Way over limit
        prompt = "Short prompt"
        is_valid, error = enhancer.validate_input_size(content, prompt)
        assert is_valid is False
        assert "exceeds token limit" in error

    def test_validate_output_capacity_valid(self):
        """Test output capacity validation with valid input."""
        enhancer = TranscriptEnhancer()
        input_tokens = 1000
        is_valid, error = enhancer.validate_output_capacity(input_tokens)
        assert is_valid is True
        assert error == ""

    def test_validate_output_capacity_too_large(self):
        """Test output capacity validation with oversized input."""
        enhancer = TranscriptEnhancer()
        # Input that would exceed output limit
        input_tokens = enhancer.MAX_OUTPUT_TOKENS  # Will exceed when multiplied by 1.3
        is_valid, error = enhancer.validate_output_capacity(input_tokens)
        assert is_valid is False
        assert "may exceed token limit" in error

    def test_get_default_prompt_no_translation(self):
        """Test default prompt generation without translation."""
        enhancer = TranscriptEnhancer()
        prompt = enhancer.get_default_prompt()
        assert "expert transcript editor" in prompt.lower()
        assert "grammar" in prompt.lower()
        assert "Additionally, translate" not in prompt

    def test_get_default_prompt_with_translation(self):
        """Test default prompt generation with translation."""
        enhancer = TranscriptEnhancer(target_language="ko")
        prompt = enhancer.get_default_prompt()
        assert "expert transcript editor" in prompt.lower()
        assert "translate" in prompt.lower()
        assert "ko" in prompt

    # Note: We skip API tests as they require valid API keys
    # Those should be integration tests or mocked tests.
