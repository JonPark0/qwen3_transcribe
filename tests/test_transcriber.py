"""Tests for Qwen3ASRTranscriber class."""

from types import SimpleNamespace

import numpy as np
import pytest
from core.transcriber import (
    Qwen3ASRTranscriber,
    _is_han_char,
    _is_han_word,
    _join_chunk_texts,
    _join_words,
)


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


class TestJoinChunkTexts:
    """Tests for stitching independently decoded pieces back together."""

    def test_korean_seam_gets_space(self):
        assert _join_chunk_texts(["반갑습니다.", "안녕하세요."]) == "반갑습니다. 안녕하세요."

    def test_han_seam_gets_no_space(self):
        assert _join_chunk_texts(["你好。", "谢谢。"]) == "你好。谢谢。"

    def test_kana_seam_gets_no_space(self):
        assert _join_chunk_texts(["こんにちは", "ありがとう"]) == "こんにちはありがとう"

    def test_empty_pieces_are_skipped(self):
        assert _join_chunk_texts(["", "  hello ", None, "world"]) == "hello world"


class TestMaxNewTokens:
    """max_new_tokens must scale with the piece length or output truncates."""

    def test_default_scales_with_chunk_length(self):
        assert Qwen3ASRTranscriber(max_chunk_sec=120).max_new_tokens == 1200

    def test_floor_for_short_chunks(self):
        assert Qwen3ASRTranscriber(max_chunk_sec=5).max_new_tokens == 256

    def test_explicit_value_wins(self):
        assert Qwen3ASRTranscriber(max_new_tokens=333).max_new_tokens == 333


class _FakeModel:
    """Stands in for Qwen3ASRModel: records batch sizes, returns one word
    per piece with a timestamp at the piece-local time 1.0-2.0s."""

    def __init__(self, fail_alignment=False):
        self.calls = []
        self.fail_alignment = fail_alignment

    def transcribe(self, audio, context, language, return_time_stamps):
        self.calls.append((len(audio), return_time_stamps))
        if return_time_stamps and self.fail_alignment:
            raise RuntimeError("aligner rejected language")
        outs = []
        for i, _ in enumerate(audio):
            idx = len(self.calls) * 100 + i
            stamps = None
            if return_time_stamps:
                item = SimpleNamespace(text=f"w{idx}", start_time=1.0, end_time=2.0)
                stamps = SimpleNamespace(items=[item])
            outs.append(SimpleNamespace(text=f"piece{idx}.", time_stamps=stamps))
        return outs


class TestChunkedTranscription:
    """transcribe_audio() pre-splits long audio and decodes it in batches."""

    @pytest.fixture
    def transcriber(self, monkeypatch):
        pytest.importorskip("qwen_asr")
        t = Qwen3ASRTranscriber(batch_size=2, max_chunk_sec=10)
        # 35s of low-level noise -> 4 pieces of <=10s
        audio = (np.random.RandomState(0).randn(35 * 16000) * 0.01).astype(np.float32)
        monkeypatch.setattr(t, "load_audio_segment", lambda *a, **k: (audio, 35.0))
        monkeypatch.setattr(t, "_ensure_aligner", lambda: None)
        return t

    def test_pieces_are_batched(self, transcriber):
        transcriber.model = _FakeModel()
        result = transcriber.transcribe_audio("x.wav")
        assert result["success"] is True
        assert transcriber.model.calls == [(2, False), (2, False)]
        assert result["text"].count("piece") == 4

    def test_progress_advances_per_batch(self, transcriber):
        transcriber.model = _FakeModel()
        seen = []
        transcriber.transcribe_audio("x.wav", progress_callback=lambda u: seen.append(u["progress"]))
        assert seen == sorted(seen)
        assert len(seen) >= 5  # load, start, 2 batches, processing, complete

    def test_timestamps_are_offset_per_piece(self, transcriber):
        transcriber.model = _FakeModel()
        transcriber.max_word_gap_sec = 0.1  # one segment per word
        result = transcriber.transcribe_audio("x.wav", enable_timestamps=True, start_time=100.0)
        starts = [c["start"] for c in result["chunks"]]
        # piece-local 1.0s + piece offset (~0/10/20/30s) + requested start_time
        assert len(starts) == 4
        assert starts == sorted(starts)
        assert starts[0] == pytest.approx(101.0, abs=0.01)
        assert 125.0 < starts[-1] < 137.0

    def test_alignment_failure_drops_timestamps_for_whole_file(self, transcriber):
        transcriber.model = _FakeModel(fail_alignment=True)
        result = transcriber.transcribe_audio("x.wav", enable_timestamps=True)
        assert result["success"] is True
        assert result["chunks"] == []
        assert "[" not in result["text"]
        assert result["text"].count("piece") == 4
