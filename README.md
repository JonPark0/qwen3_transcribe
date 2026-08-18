# Qwen3 Transcribe

A Python tool that transcribes audio files using Alibaba's Qwen3-ASR-1.7B model. Supports multiple file formats, timestamp functionality (via Qwen3-ForcedAligner-0.6B), and AI-powered transcript enhancement.

This is a sibling project of [`whisper_transcribe`](https://github.com/JonPark0/whisper_transcribe), mirroring its structure and public API (`core.Qwen3ASRTranscriber` is a drop-in replacement for `core.WhisperTranscriber`) so that [`whisper_webui`](https://github.com/JonPark0/whisper_webui) can switch STT engines with minimal code changes.

## Related Projects

- **[whisper_transcribe](https://github.com/JonPark0/whisper_transcribe):** The Whisper large-v3-turbo counterpart to this project.
- **[whisper_webui](https://github.com/JonPark0/whisper_webui):** A web-based user interface, used with this project via its `qwen3-asr` branch.

---

## Features

- Uses Qwen3-ASR-1.7B for high-quality, multilingual transcription (30 languages, 22 Chinese dialects)
- Automatic language identification, or force a specific recognition language
- Optional context/hotwords to bias transcription (domain vocabulary, names, etc.)
- Supports multiple audio formats (MP3, WAV, FLAC, AAC, OGG, M4A, WMA)
- Optional word-level timestamps via Qwen3-ForcedAligner-0.6B, reassembled into
  Whisper-style `[HH:MM:SS - HH:MM:SS] text` segments
- Internal energy-based chunking for long audio (handled by the `qwen-asr` package)
- Batch processing with wildcard support
- Outputs transcriptions in Markdown or JSON format

## Differences from whisper_transcribe

Qwen3-ASR is an audio-LLM, not a Seq2Seq encoder-decoder like Whisper, so a few
things work differently:

- **No manual chunk length.** `qwen-asr` splits long audio internally at
  low-energy boundaries (up to 20 minutes per ASR call, 3 minutes per forced-
  alignment call). There's no `-ch/--chunked` flag.
- **Timestamps require a second model.** Qwen3-ASR itself does not emit
  timestamps; `Qwen3ASRTranscriber` lazily loads `Qwen3-ForcedAligner-0.6B`
  only when `enable_timestamps=True`, and only keeps it loaded once loaded.
- **Timestamps are word-level, reassembled into segments.** The forced
  aligner returns per-word (or per-Han-character) spans; this library groups
  them into sentence-ish segments (bounded by punctuation, ~15s duration,
  ~80 chars, and silence gaps) to match Whisper's segment-level output shape.
- **`-lang/--language` forces recognition language** (Qwen3-ASR canonical
  names like `"Korean"`, `"English"`, `"Chinese"` — not ISO codes), separate
  from `-tr/--translate` which only affects the optional Gemini enhancement
  step.

## Setup

1. Clone or download this repository
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
source venv/bin/activate
python3 convert.py -i input_audio.mp3 -o output_folder/
```

### Advanced Usage Examples
```bash
# With timestamps and verbose output
python3 convert.py -i "audio file.mp3" -o "./output folder/" -ts -v

# Multiple files with wildcards
python3 convert.py -i *.mp3 -o ./output/ -ts

# Force the recognition language
python3 convert.py -i korean_audio.mp3 -o ./output/ -lang Korean -ts

# Bias transcription with domain vocabulary
python3 convert.py -i audio.mp3 -o ./output/ -ctx "Vocabulary: Quilter, apostle, gospel"

# With automatic enhancement using Gemini API
python3 convert.py -i audio.mp3 -o ./output/ -ts -e
```

## System Requirements

- Python 3.9 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Sufficient disk space for model downloads (~3.5GB for Qwen3-ASR-1.7B, ~1.2GB for the forced aligner)

### For M4A/AAC Support (Optional)
If you need to process M4A/AAC files, install system ffmpeg:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

## Dependencies

### Core Dependencies
- qwen-asr>=0.0.6 (pins `transformers==4.57.6`, `accelerate==1.12.0`)
- torch>=2.2.0
- librosa>=0.10.0
- pydub>=0.25.0
- ffmpeg-python>=0.2.0
- soundfile>=0.12.0

### Optional Performance Dependencies
- ninja>=1.11.0 (build system for Flash Attention)
- psutil>=7.0.0 (system utilities for Flash Attention)
- flash-attn>=2.7.4 (Flash Attention 2 for GPU acceleration)

### Optional Enhancement Dependencies
- google-genai>=1.0.0 (Google Gemini API for transcript enhancement)

### Flash Attention 2 Requirements (Optional)
Flash Attention 2 provides significant performance improvements for GPU processing:

**GPU Requirements:**
- NVIDIA GPUs: Ampere, Ada, or Hopper architecture (RTX 3090, RTX 4070/4090, A100, H100, etc.)
- CUDA >= 12.3 (recommended: CUDA 12.8)
- AMD GPUs: MI200 or MI300 with ROCm 6.0+

**Installation (Method 1: Using setup.py extras):**
```bash
# Recommended: Install with optional flash-attn extras
pip install -e .[flash-attn]
```

**Installation (Method 2: Manual installation):**
```bash
# Step 1: Install core dependencies first (includes torch)
pip install -r requirements.txt

# Step 2: Install build tools
pip install ninja psutil

# Step 3: Install Flash Attention 2 (requires torch to be already installed)
pip install flash-attn --no-build-isolation
```

**Important Notes:**
- Flash Attention 2 installation may take 5-10 minutes as it compiles from source
- **torch must be installed before installing flash-attn** (flash-attn needs torch during build)
- Not required for basic functionality - only provides performance optimization

### Transcript Enhancement with Gemini API (Optional)
The enhancement feature uses Google's Gemini API to improve transcript quality:

**Setup:**
1. Get a Google AI API key from [Google AI Studio](https://aistudio.google.com/)
2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```
3. Install the dependency:
   ```bash
   pip install google-genai>=1.0.0
   ```

**Enhancement Features:**
- Grammar and punctuation correction
- Improved sentence structure and readability
- Technical term correction based on context
- Removal of excessive filler words
- Better formatting with headings and structure
- Optional translation to target language (`-tr/--translate`)

**Usage:**
```bash
# Basic enhancement (uses gemini-flash-latest by default)
python3 convert.py -i audio.mp3 -o ./output/ -e

# Enhancement with custom prompt
python3 convert.py -i lecture.mp3 -o ./output/ -e "Focus on technical accuracy"

# Enhancement with translation
python3 convert.py -i spanish_audio.mp3 -o ./output/ -tr en -e
```

**Standalone Enhancement:**
You can also enhance existing transcripts using `enhance.py` directly:
```bash
python3 enhance.py -i transcript.md -o enhanced.md -v
python3 enhance.py -i transcript.md -o enhanced.md -tr es
python3 enhance.py -i *.md -o enhanced/ -m gemini-pro-latest -v
```

## Notes

- First run will download Qwen3-ASR-1.7B (~3.5GB); enabling `-ts/--timestamp`
  additionally downloads Qwen3-ForcedAligner-0.6B (~1.2GB)
- GPU acceleration is automatically used if available
- For M4A/AAC support, system ffmpeg installation is required
- Supported recognition languages (`-lang/--language`): Chinese, English,
  Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian,
  Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch,
  Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian,
  Hungarian, Macedonian
