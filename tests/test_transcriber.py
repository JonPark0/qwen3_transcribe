"""Tests for Qwen3ASRTranscriber class."""

import pytest
from core.transcriber import Qwen3ASRTranscriber, _join_words, _is_han_char, _is_han_word


class TestQwen3ASRTranscriber:
    """Tests for Qwen3ASRTranscriber class."""

    def test_initialization_default(self):
        """Test default initialization."""
        transcriber = Qwen3ASRTranscriber()
        assert transcriber.verbose is False
        assert transcriber.batch_size == 4
        assert transcriber.use_flash_attn is False
        assert transcriber.model_id == "Qwen/Qwen3-ASR-1.7B"
        assert transcriber.aligner_model_id == "Qwen/Qwen3-ForcedAligner-0.6B"
        assert transcriber.language is None
        assert transcriber.context == ""
        assert transcriber.model is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        transcriber = Qwen3ASRTranscriber(
            verbose=True,
            batch_size=8,
            use_flash_attn=True,
            language="Korean",
            context="Vocabulary: Quilter, apostle",
        )
        assert transcriber.verbose is True
        assert transcriber.batch_size == 8
        assert transcriber.use_flash_attn is True
        assert transcriber.language == "Korean"
        assert transcriber.context == "Vocabulary: Quilter, apostle"

    def test_log_verbose_enabled(self, capsys):
        """Test logging with verbose enabled."""
        transcriber = Qwen3ASRTranscriber(verbose=True)
        transcriber.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_verbose_disabled(self, capsys):
        """Test logging with verbose disabled."""
        transcriber = Qwen3ASRTranscriber(verbose=False)
        transcriber.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" not in captured.out

    def test_transcribe_audio_without_model_returns_error(self):
        """transcribe_audio() should return a failed result (not raise) if
        load_model() was never called."""
        transcriber = Qwen3ASRTranscriber()
        result = transcriber.transcribe_audio("nonexistent.mp3")
        assert result['success'] is False
        assert result['error'] is not None

    # Note: We skip actual model loading/inference tests as they require
    # large downloads and significant resources. Those should be integration
    # tests, run manually via `python convert.py --timestamp -v ...`.


class TestIsHanChar:
    """Tests for the Han-ideograph detection used by the segment joiner."""

    def test_chinese_char_is_han(self):
        assert _is_han_char("中") is True

    def test_korean_char_is_not_han(self):
        assert _is_han_char("한") is False

    def test_japanese_kana_is_not_han(self):
        assert _is_han_char("あ") is False

    def test_latin_char_is_not_han(self):
        assert _is_han_char("a") is False


class TestIsHanWord:
    def test_all_han_word(self):
        assert _is_han_word("中文") is True

    def test_mixed_word_is_not_han(self):
        assert _is_han_word("中a") is False

    def test_empty_word_is_not_han(self):
        assert _is_han_word("") is False


class TestJoinWords:
    """Tests for the CJK-aware word joiner."""

    def test_latin_words_get_spaces(self):
        assert _join_words(["hello", "world"]) == "hello world"

    def test_han_chars_get_no_spaces(self):
        assert _join_words(["中", "文"]) == "中文"

    def test_mixed_han_and_latin(self):
        # Han characters should not gain surrounding spaces, but the Latin
        # run around them keeps normal word spacing.
        result = _join_words(["hello", "中", "文", "world"])
        assert result == "hello中文world"

    def test_empty_list(self):
        assert _join_words([]) == ""

    def test_single_word(self):
        assert _join_words(["hello"]) == "hello"
