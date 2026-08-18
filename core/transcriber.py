"""Core transcription module with API and progress tracking support.

Backed by Qwen3-ASR-1.7B (via the `qwen-asr` PyPI package). Mirrors the public
interface of whisper_transcribe's WhisperTranscriber so callers (e.g.
whisper_webui) can swap engines with minimal changes.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import torch
import librosa
import numpy as np

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

from .utils import format_timestamp

logger = logging.getLogger(__name__)

# Chinese/Cantonese Han ideograph ranges. Qwen3-ForcedAligner tokenizes these
# scripts character-by-character (no inter-character spacing), while every
# other supported language (Korean, Japanese, English, ...) is tokenized as
# whitespace-separated words. Mirrors qwen_asr.inference.qwen3_forced_aligner
# .Qwen3ForceAlignProcessor.is_cjk_char, which intentionally excludes Hangul
# and Kana - only Han ideographs are joined without spaces.
_HAN_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)

def _is_han_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _HAN_RANGES)


def _is_han_word(word: str) -> bool:
    """A "word" token counts as Han if every character in it is a Han ideograph."""
    return bool(word) and all(_is_han_char(ch) for ch in word)


def _join_words(words: List[str]) -> str:
    """
    Join word-level tokens into display text, matching the spacing convention
    used by Qwen3-ForcedAligner's own tokenizer: no space is inserted directly
    before/after a Han-ideograph token, but every other language keeps normal
    whitespace-separated word boundaries.
    """
    parts: List[str] = []
    prev_is_han = False
    for i, word in enumerate(words):
        if not word:
            continue
        cur_is_han = _is_han_word(word)
        if i > 0 and parts and not prev_is_han and not cur_is_han:
            parts.append(" ")
        parts.append(word)
        prev_is_han = cur_is_han
    return "".join(parts)


class Qwen3ASRTranscriber:
    """
    Qwen3ASRTranscriber provides audio transcription using Qwen3-ASR-1.7B.

    This class can be used as a library (with progress callbacks) or through
    CLI. Its public interface mirrors WhisperTranscriber from
    whisper_transcribe so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
        verbose: bool = False,
        batch_size: int = 4,
        use_flash_attn: bool = False,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        aligner_model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: Optional[str] = None,
        context: str = "",
        max_new_tokens: int = 1024,
        max_segment_sec: float = 15.0,
        max_segment_chars: int = 80,
        max_word_gap_sec: float = 0.8,
        target_language: Optional[str] = None,
    ):
        """
        Initialize Qwen3ASRTranscriber.

        Args:
            verbose: Enable detailed logging
            batch_size: Batch size limit for inference (max_inference_batch_size
                        in qwen-asr). Lower values reduce VRAM usage.
            use_flash_attn: Enable Flash Attention 2 for faster GPU processing
            model_id: HuggingFace repo id for the ASR model
            aligner_model_id: HuggingFace repo id for the forced aligner model,
                        used only when timestamps are requested
            language: Force recognition language (canonical Qwen3-ASR name,
                        e.g. "Korean", "English"). None = auto-detect.
            context: Free-form context/hotwords to bias transcription
                        (domain vocabulary, names, etc.)
            max_new_tokens: Maximum tokens to generate per chunk
            max_segment_sec: Max duration of a reconstructed timestamp segment
            max_segment_chars: Max character count of a reconstructed segment
            max_word_gap_sec: Silence gap that forces a new segment
            target_language: Unused; kept for interface parity with
                        WhisperTranscriber (translation is handled by the
                        Gemini enhancer, not the ASR model, in this project).
        """
        self.verbose = verbose
        self.batch_size = batch_size
        self.use_flash_attn = use_flash_attn
        self.model_id = model_id
        self.aligner_model_id = aligner_model_id
        self.language = language
        self.context = context
        self.max_new_tokens = max_new_tokens
        self.max_segment_sec = max_segment_sec
        self.max_segment_chars = max_segment_chars
        self.max_word_gap_sec = max_word_gap_sec
        self.target_language = target_language

        self.model = None
        self.device = None
        self.model_dtype = None
        self._aligner_lock_message_shown = False

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def load_model(self):
        """Load Qwen3-ASR-1.7B model with appropriate device configuration."""
        self.log(f"Loading Qwen3-ASR model ({self.model_id})...")

        # Import here so the heavy `qwen_asr` -> `transformers` import chain
        # only happens once load_model() is actually called.
        from qwen_asr import Qwen3ASRModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "low_cpu_mem_usage": True,
        }

        if self.use_flash_attn and torch.cuda.is_available():
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.log("Flash Attention 2 enabled")
            except Exception as e:
                self.log(f"Flash Attention 2 not available, falling back to standard attention: {e}")

        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            max_inference_batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            **model_kwargs,
        )

        self.model_dtype = dtype
        self.device = device

        self.log(f"Model loaded successfully on {device} (batch_size={self.batch_size})")

    def _ensure_aligner(self):
        """
        Lazily load the forced aligner and attach it to the loaded ASR model.

        Qwen3ASRModel.forced_aligner is a plain instance attribute (not
        constructor-only), so it can be assigned after the fact without
        re-instantiating the ASR model.
        """
        if self.model is None:
            raise RuntimeError("Transcriber model is not initialized. Call load_model() first.")

        if getattr(self.model, "forced_aligner", None) is not None:
            return

        from qwen_asr import Qwen3ForcedAligner

        self.log(f"Loading forced aligner ({self.aligner_model_id})...")
        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = self.model_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        aligner = Qwen3ForcedAligner.from_pretrained(
            self.aligner_model_id,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.forced_aligner = aligner
        self.log("Forced aligner loaded successfully")

    def load_audio_segment(
        self,
        audio_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Load audio file with optional segment selection.

        Identical to WhisperTranscriber.load_audio_segment: pydub -> soundfile
        -> librosa fallback chain, always producing mono float32 @ 16kHz.

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds (None = from beginning)
            end_time: End time in seconds (None = to end)

        Returns:
            Tuple of (audio_array, duration_in_seconds)
        """
        self.log(f"Loading audio: {audio_path}")

        audio = None
        original_duration = 0

        # Try pydub first for better M4A/AAC support
        if HAS_PYDUB:
            try:
                self.log("Trying pydub for audio loading...")
                audio_segment = AudioSegment.from_file(audio_path)
                original_duration = float(len(audio_segment) / 1000.0)  # milliseconds to seconds

                # Apply segment selection if specified
                if start_time is not None or end_time is not None:
                    start_ms = int(start_time * 1000) if start_time is not None else 0
                    end_ms = int(end_time * 1000) if end_time is not None else len(audio_segment)

                    self.log(f"Extracting segment: {start_time or 0:.2f}s - {end_time or original_duration:.2f}s")
                    audio_segment = audio_segment[start_ms:end_ms]

                # Convert to mono 16kHz
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

                # Normalize based on sample width
                if audio_segment.sample_width == 2:
                    audio = audio / 32768.0
                elif audio_segment.sample_width == 4:
                    audio = audio / 2147483648.0

                duration = len(audio) / 16000.0
                self.log(f"Successfully loaded audio with pydub. Duration: {duration:.2f}s")

            except Exception as e:
                self.log(f"Pydub failed: {e}")
                audio = None

        # Fallback to librosa if pydub fails
        if audio is None:
            try:
                # Use soundfile backend to avoid audioread deprecation
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)

                # Calculate original duration
                original_duration = len(audio_data) / sample_rate

                # Apply segment selection if specified
                if start_time is not None or end_time is not None:
                    start_sample = int(start_time * sample_rate) if start_time is not None else 0
                    end_sample = int(end_time * sample_rate) if end_time is not None else len(audio_data)

                    self.log(f"Extracting segment: {start_time or 0:.2f}s - {end_time or original_duration:.2f}s")
                    audio_data = audio_data[start_sample:end_sample]

                # Resample if needed
                if sample_rate != 16000:
                    audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                else:
                    audio = audio_data

                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                duration = len(audio) / 16000.0
                self.log(f"Successfully loaded audio with soundfile. Duration: {duration:.2f}s")

            except Exception as e:
                self.log(f"Soundfile failed: {e}")
                # Final fallback: librosa (may show deprecation warnings)
                try:
                    audio, sample_rate = librosa.load(audio_path, sr=16000)

                    # For librosa fallback, we need to handle segment selection differently
                    if start_time is not None or end_time is not None:
                        self.log("Warning: Segment selection with librosa fallback may be less accurate")
                        start_sample = int(start_time * 16000) if start_time is not None else 0
                        end_sample = int(end_time * 16000) if end_time is not None else len(audio)
                        audio = audio[start_sample:end_sample]

                    duration = len(audio) / 16000.0
                    self.log(f"Successfully loaded audio with librosa (with warnings). Duration: {duration:.2f}s")

                except Exception as e2:
                    raise Exception(
                        f"All audio loading methods failed. "
                        f"Pydub: {e if HAS_PYDUB else 'Not available'}, "
                        f"Soundfile: {e}, Librosa: {e2}"
                    )

        if audio is None or len(audio) == 0:
            raise Exception("Audio file appears to be empty or corrupted")

        duration = len(audio) / 16000.0
        return audio, duration

    def _group_words_into_segments(self, items: List[Any]) -> List[Dict[str, Any]]:
        """
        Group word/character-level forced-alignment items into sentence-ish
        segments, similar in spirit to Whisper's 30s pipeline chunks.

        A new segment starts whenever any of these trigger:
          - accumulated segment duration would exceed max_segment_sec
          - accumulated segment character count would exceed max_segment_chars
          - the gap between the previous item's end and this item's start
            exceeds max_word_gap_sec

        Note: there is no punctuation-based sentence-boundary check.
        Qwen3-ForcedAligner's word tokenizer (Qwen3ForceAlignProcessor
        .clean_token) strips everything outside Unicode letter/number
        categories, so `ForcedAlignItem.text` never contains punctuation -
        checking for it here would always be a no-op.

        Args:
            items: List of ForcedAlignItem-like objects with .text/.start_time/.end_time

        Returns:
            List of {'start': float, 'end': float, 'text': str} segments
        """
        if not items:
            return []

        segments: List[Dict[str, Any]] = []
        cur_words: List[str] = []
        cur_start: Optional[float] = None
        cur_end: Optional[float] = None

        def flush():
            if cur_words:
                segments.append({
                    'start': cur_start,
                    'end': cur_end,
                    'text': _join_words(cur_words),
                })

        for item in items:
            text = (item.text or "").strip()
            start = float(item.start_time)
            end = float(item.end_time)

            if not text:
                continue

            if cur_start is None:
                cur_start, cur_end, cur_words = start, end, [text]
                continue

            gap = start - cur_end
            duration_if_added = end - cur_start
            chars_if_added = sum(len(w) for w in cur_words) + len(text)

            should_break = (
                gap > self.max_word_gap_sec
                or duration_if_added > self.max_segment_sec
                or chars_if_added > self.max_segment_chars
            )

            if should_break:
                flush()
                cur_start, cur_end, cur_words = start, end, [text]
            else:
                cur_words.append(text)
                cur_end = end

        flush()
        return segments

    def transcribe_audio(
        self,
        audio_path: str,
        enable_timestamps: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with optional progress tracking.

        Args:
            audio_path: Path to audio file
            enable_timestamps: Include timestamps in output
            start_time: Start time in seconds (for segment selection)
            end_time: End time in seconds (for segment selection)
            progress_callback: Optional callback function for progress updates
                              Receives dict with keys: stage, progress, message, etc.

        Returns:
            Dictionary containing:
                - text: Full transcript text (with or without timestamps)
                - chunks: List of chunks with start/end times and text (if timestamps enabled)
                - duration: Audio duration in seconds
                - processing_time: Time taken to process
                - success: Boolean indicating success
                - error: Error message if failed
        """
        self.log(f"Transcribing: {audio_path}")
        start_processing_time = time.time()

        result = {
            'success': False,
            'text': '',
            'chunks': [],
            'duration': 0,
            'processing_time': 0,
            'error': None
        }

        try:
            if self.model is None:
                raise RuntimeError("Transcriber model is not initialized. Call load_model() first.")

            # Stage 1: Loading audio
            if progress_callback:
                progress_callback({
                    'stage': 'loading',
                    'progress': 0.1,
                    'message': 'Loading audio file...'
                })

            audio, duration = self.load_audio_segment(audio_path, start_time, end_time)
            result['duration'] = duration

            # Stage 2: Transcribing
            if progress_callback:
                progress_callback({
                    'stage': 'transcribing',
                    'progress': 0.2,
                    'message': 'Starting transcription...'
                })

            self.log("Running Qwen3-ASR transcription (internal energy-based chunking for long audio)")

            want_timestamps = enable_timestamps
            if want_timestamps:
                try:
                    self._ensure_aligner()
                except Exception as e:
                    self.log(f"Failed to load forced aligner, falling back to plain text: {e}")
                    want_timestamps = False

            asr_result = self._run_transcribe(audio, want_timestamps)

            # Stage 3: Processing results
            if progress_callback:
                progress_callback({
                    'stage': 'processing',
                    'progress': 0.9,
                    'message': 'Processing transcription results...'
                })

            if want_timestamps and asr_result.time_stamps is not None and len(asr_result.time_stamps.items) > 0:
                segments = self._group_words_into_segments(list(asr_result.time_stamps.items))

                transcript_lines = []
                chunks_data = []

                for seg in segments:
                    start = seg['start'] if seg['start'] is not None else 0
                    end = seg['end'] if seg['end'] is not None else duration
                    text = seg['text']

                    if start_time is not None:
                        start += start_time
                        end += start_time

                    start_ts = format_timestamp(start)
                    end_ts = format_timestamp(end)

                    transcript_lines.append(f"[{start_ts} - {end_ts}] {text}")
                    chunks_data.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })

                result['text'] = "\n".join(transcript_lines)
                result['chunks'] = chunks_data
            else:
                result['text'] = asr_result.text

            result['success'] = True
            result['processing_time'] = time.time() - start_processing_time

            # Stage 4: Complete
            if progress_callback:
                progress_callback({
                    'stage': 'complete',
                    'progress': 1.0,
                    'message': 'Transcription completed successfully'
                })

            self.log(f"Transcription completed in {result['processing_time']:.1f}s")
            return result

        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_processing_time
            self.log(f"Error transcribing {audio_path}: {str(e)}")
            logger.exception("Transcription failed for %s", audio_path)

            if progress_callback:
                progress_callback({
                    'stage': 'error',
                    'progress': 0,
                    'message': f'Error: {str(e)}'
                })

            return result

    def _run_transcribe(self, audio: np.ndarray, want_timestamps: bool):
        """
        Call Qwen3ASRModel.transcribe(), retrying without timestamps if the
        forced aligner rejects the audio/language (e.g. an unsupported
        script) rather than failing the whole job.
        """
        try:
            return self.model.transcribe(
                audio=(audio, 16000),
                context=self.context,
                language=self.language,
                return_time_stamps=want_timestamps,
            )[0]
        except Exception as e:
            if want_timestamps:
                self.log(f"Timestamp alignment failed, retrying without timestamps: {e}")
                return self.model.transcribe(
                    audio=(audio, 16000),
                    context=self.context,
                    language=self.language,
                    return_time_stamps=False,
                )[0]
            raise

    def save_transcript(
        self,
        result: Dict[str, Any],
        audio_path: str,
        output_dir: str,
        output_format: str = 'markdown'
    ) -> str:
        """
        Save transcription result to file.

        Args:
            result: Transcription result from transcribe_audio()
            audio_path: Original audio file path
            output_dir: Output directory
            output_format: 'markdown' or 'json'

        Returns:
            Path to saved file
        """
        audio_name = Path(audio_path).stem

        if output_format == 'json':
            output_file = Path(output_dir) / f"{audio_name}.json"
            self.log(f"Saving transcript to JSON: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'audio_file': Path(audio_path).name,
                    'duration': result['duration'],
                    'processing_time': result['processing_time'],
                    'text': result['text'],
                    'chunks': result['chunks']
                }, f, ensure_ascii=False, indent=2)

        else:  # markdown (default)
            output_file = Path(output_dir) / f"{audio_name}.md"
            self.log(f"Saving transcript to Markdown: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Transcript: {audio_name}\n\n")
                f.write(f"**Source:** {Path(audio_path).name}\n\n")
                f.write("## Content\n\n")
                f.write(result['text'])
                f.write("\n")

        self.log(f"Transcript saved successfully")
        return str(output_file)
