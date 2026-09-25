#!/usr/bin/env python3

import argparse
import os
import sys
import glob
import time
import platform
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Reconfigure stdio to UTF-8 so the emoji status markers below don't crash
# with UnicodeEncodeError on Windows consoles using a non-UTF-8 code page
# (e.g. cp949 for Korean locales). No-op on platforms that are already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Import from core modules
from core import Qwen3ASRTranscriber, TranscriptEnhancer, format_duration


# Timeout exception (always defined for compatibility)
class TimeoutException(Exception):
    pass


# Platform-specific timeout handling
TIMEOUT_SUPPORTED = False
if platform.system() != 'Windows':
    try:
        import signal
        TIMEOUT_SUPPORTED = True

        def timeout_handler(signum, frame):
            raise TimeoutException("Timeout occurred")
    except (ImportError, AttributeError):
        pass


def expand_file_paths(input_paths: List[str]) -> List[str]:
    """Expand wildcards and validate file paths."""
    expanded_paths = []
    for path in input_paths:
        if '*' in path or '?' in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)

    valid_paths = []
    for path in expanded_paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        else:
            print(f"Warning: File not found: {path}")

    return valid_paths


def is_audio_file(file_path: str) -> bool:
    """Check if file is a supported audio format."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions


def enhance_file(audio_file: str, output_dir: str, custom_prompt: str, verbose: bool, translate: str):
    """Run enhancement on the generated transcript using the core library."""
    # Get the generated markdown file path
    audio_name = Path(audio_file).stem
    markdown_file = Path(output_dir) / f"{audio_name}.md"
    enhanced_file = Path(output_dir) / f"{audio_name}_enhanced.md"

    # Check if markdown file exists
    if not markdown_file.exists():
        raise Exception(f"Transcript file not found: {markdown_file}")

    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception(
            "GEMINI_API_KEY not found. Please set your Google AI API key either:\n"
            "1. Create a .env file with: GEMINI_API_KEY=your-api-key-here\n"
            "2. Or export as environment variable: export GEMINI_API_KEY='your-api-key-here'"
        )

    # Create enhancer
    enhancer = TranscriptEnhancer(verbose=verbose, target_language=translate)

    # Setup Gemini API
    enhancer.setup_gemini(api_key)

    # Read the transcript
    with open(markdown_file, 'r', encoding='utf-8') as f:
        input_content = f.read()

    # Enhance the transcript
    result = enhancer.enhance_transcript(
        input_content=input_content,
        custom_prompt=custom_prompt if custom_prompt else None
    )

    if result['success']:
        # Save enhanced content
        enhancer.save_enhanced_transcript(
            enhanced_text=result['enhanced_text'],
            original_file=str(markdown_file),
            output_file=str(enhanced_file),
            processing_time=result['processing_time'],
            output_tokens=result['output_tokens']
        )
        print(f"📄 Enhanced transcript saved: {enhanced_file}")
    else:
        raise Exception(f"Enhancement failed: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Qwen3-ASR-1.7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert.py -i audio.mp3 -o ./output/
  python3 convert.py -i "audio file.mp3" -o "./output folder/" -ts -v
  python3 convert.py -i *.mp3 -o ./output/ -ts --timeout 300
  python3 convert.py -i file1.wav file2.mp3 -o ./output/
  python3 convert.py -i audio.mp3 -o ./output/ --flash-attn
  python3 convert.py -i audio.mp3 -o ./output/ -lang Korean -ts -v
  python3 convert.py -i audio.mp3 -o ./output/ -ctx "Vocabulary: Quilter, apostle"
  python3 convert.py -i audio.mp3 -o ./output/ -ts -e
  python3 convert.py -i audio.mp3 -o ./output/ -e "Focus on technical terms"
  python3 convert.py -i audio.mp3 -o ./output/ -ts --json  # JSON output
  python3 convert.py -i audio.mp3 -o ./output/ -ts --start 10 --end 60  # Segment selection
        """
    )

    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input audio file path(s). Supports wildcards and multiple files.')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory path for markdown files.')
    parser.add_argument('-ts', '--timestamp', action='store_true',
                       help='Enable timestamp feature in transcription (loads the forced aligner).')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output to see processing details.')
    parser.add_argument('-to', '--timeout', type=int,
                       help='Set timeout in seconds for each file processing.')
    parser.add_argument('--flash-attn', action='store_true',
                       help='Enable Flash Attention 2 for faster processing on compatible GPUs.')
    parser.add_argument('-lang', '--language', type=str, metavar='LANGUAGE',
                       help='Force the recognition language using Qwen3-ASR canonical names '
                            '(e.g. "Korean", "English", "Chinese"). Default: auto-detect.')
    parser.add_argument('-ctx', '--context', type=str, default='',
                       help='Free-form context/hotwords to bias transcription '
                            '(domain vocabulary, names, etc.)')
    parser.add_argument('--max-chunk-sec', type=float, default=None, metavar='SECONDS',
                       help='Split long audio at the quietest point near every N seconds and '
                            'decode the pieces in batches (default: the library default). '
                            'Lower it if long files run out of GPU memory.')
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                       help='Audio pieces decoded per generate() call (default: 4). '
                            'Lower values reduce VRAM usage.')
    parser.add_argument('-tr', '--translate', type=str, metavar='LANGUAGE',
                       help='Target language for the (optional) Gemini enhancement/translation '
                            'step using ISO 639-1 two-letter codes (e.g., "en", "es", "fr"). '
                            'Note: this affects -e/--enhance only, not the ASR recognition '
                            'language (use -lang/--language for that).')
    parser.add_argument('-e', '--enhance', nargs='?', const='', type=str, metavar='PROMPT',
                       help='Automatically execute enhancement process using Gemini API. Optional additional prompt can be provided.')
    parser.add_argument('--json', action='store_true',
                       help='Output transcription in JSON format instead of Markdown.')
    parser.add_argument('--start', type=float, metavar='SECONDS',
                       help='Start time in seconds for audio segment selection.')
    parser.add_argument('--end', type=float, metavar='SECONDS',
                       help='End time in seconds for audio segment selection.')

    args = parser.parse_args()

    input_files = expand_file_paths(args.input)
    if not input_files:
        print("Error: No valid input files found.")
        sys.exit(1)

    audio_files = [f for f in input_files if is_audio_file(f)]
    if not audio_files:
        print("Error: No valid audio files found.")
        sys.exit(1)

    if len(audio_files) != len(input_files):
        print(f"Warning: {len(input_files) - len(audio_files)} non-audio files will be skipped.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        if args.verbose:
            print(f"Created output directory: {args.output}")

    # Check timeout support
    if args.timeout and not TIMEOUT_SUPPORTED:
        print("Warning: Timeout feature is not supported on Windows. Timeout will be ignored.")
        args.timeout = None

    tuning = {}
    if args.max_chunk_sec:
        tuning['max_chunk_sec'] = args.max_chunk_sec
    if args.batch_size:
        tuning['batch_size'] = args.batch_size
    transcriber = Qwen3ASRTranscriber(
        verbose=args.verbose,
        use_flash_attn=args.flash_attn,
        language=args.language,
        context=args.context,
        **tuning,
    )
    transcriber.load_model()

    total_files = len(audio_files)
    processed_files = 0
    total_start_time = time.time()
    processing_times = []

    print(f"Processing {total_files} audio file(s)...")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {Path(audio_file).name}")

        try:
            if args.timeout and TIMEOUT_SUPPORTED:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(args.timeout)

            # Transcribe audio with segment selection support
            result = transcriber.transcribe_audio(
                audio_file,
                enable_timestamps=args.timestamp,
                start_time=args.start,
                end_time=args.end
            )

            if result['success']:
                # Determine output format
                output_format = 'json' if args.json else 'markdown'

                # Save transcript
                transcriber.save_transcript(result, audio_file, args.output, output_format)
                processed_files += 1
                processing_times.append(result['processing_time'])

                print(f"✅ Completed in {format_duration(result['processing_time'])}")

                # Run enhancement if requested
                if args.enhance is not None:
                    try:
                        print("🔧 Running enhancement...")
                        enhance_file(audio_file, args.output, args.enhance, args.verbose, args.translate)
                    except Exception as e:
                        print(f"⚠️  Enhancement failed: {str(e)}")
            else:
                print(f"❌ Error: {result['error']}")

            if args.timeout and TIMEOUT_SUPPORTED:
                signal.alarm(0)

        except TimeoutException:
            print(f"⏰ Timeout: Processing of {audio_file} exceeded {args.timeout} seconds")
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {str(e)}")

    total_elapsed_time = time.time() - total_start_time

    print(f"\n{'='*50}")
    print(f"📊 Processing Summary")
    print(f"{'='*50}")
    print(f"✅ Processed: {processed_files}/{total_files} files")
    if processed_files < total_files:
        print(f"❌ Failed: {total_files - processed_files} files")

    print(f"⏱️  Total time: {format_duration(total_elapsed_time)}")

    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        print(f"📈 Average per file: {format_duration(avg_time)}")
        print(f"⚡ Fastest: {format_duration(min_time)}")
        print(f"🐌 Slowest: {format_duration(max_time)}")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
