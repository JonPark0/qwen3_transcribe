#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Import from core modules
from core import TranscriptEnhancer, format_duration


def main():
    parser = argparse.ArgumentParser(
        description="Enhance audio transcripts using Google Gemini 2.5 Pro API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python3 enhance.py -i transcript.md -o enhanced.md
  python3 enhance.py -i transcript.md -o enhanced.md -v -tr es

  # Multiple files (batch processing)
  python3 enhance.py -i *.md -o enhanced/ -v
  python3 enhance.py -i file1.md file2.md file3.md -o output_dir/ -tr en

  # Using different models
  python3 enhance.py -i transcript.md -o enhanced.md -m gemini-flash-latest
  python3 enhance.py -i transcript.md -o enhanced.md -m gemini-pro-latest

Requirements:
  - Google AI API key (set as GEMINI_API_KEY environment variable)
  - Install: pip install google-generativeai
        """
    )

    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input markdown file path(s). Supports multiple files.')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory path for enhanced files (when multiple inputs) or output file path (when single input).')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output to see processing details.')
    parser.add_argument('-tr', '--translate', type=str, metavar='LANGUAGE',
                       help='Set target language for translation using ISO 639-1 two-letter codes (e.g., "en", "es", "fr").')
    parser.add_argument('-p', '--prompt', type=str,
                       help='Custom enhancement prompt (replaces default prompt).')
    parser.add_argument('-m', '--model', type=str, default='gemini-flash-latest',
                       help='Gemini model to use (default: gemini-flash-latest). Examples: gemini-2.5-flash, gemini-2.0-flash-exp, gemini-1.5-pro')

    args = parser.parse_args()

    # Validate input files
    input_files = []
    for input_path in args.input:
        if os.path.isfile(input_path):
            input_files.append(input_path)
        else:
            print(f"Error: Input file '{input_path}' not found.")
            sys.exit(1)

    # Determine if processing single file or multiple files
    is_batch = len(input_files) > 1

    # Validate output path
    if is_batch:
        # Multiple inputs: output should be a directory
        output_dir = args.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        elif not os.path.isdir(output_dir):
            print(f"Error: For multiple inputs, output must be a directory: {output_dir}")
            sys.exit(1)
    else:
        # Single input: output can be a file
        output_file = args.output
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Load environment variables from .env file in project directory
    load_dotenv()

    # Suppress warnings (done through python warnings module instead of env modification)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='google')

    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        print("Please set your Google AI API key either:")
        print("1. Create a .env file in the project directory with: GEMINI_API_KEY=your-api-key-here")
        print("2. Or export as environment variable: export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Create enhancer
    enhancer = TranscriptEnhancer(verbose=args.verbose, target_language=args.translate, model_name=args.model)

    try:
        total_start_time = time.time()

        # Setup Gemini API
        enhancer.setup_gemini(api_key)

        if is_batch:
            print(f"Processing {len(input_files)} files in batch mode...")
            successful_files = 0

            for i, input_file in enumerate(input_files, 1):
                print(f"\n[{i}/{len(input_files)}] Processing: {Path(input_file).name}")

                # Generate output filename
                input_name = Path(input_file).stem
                output_file_path = Path(output_dir) / f"{input_name}_enhanced.md"

                try:
                    # Read input file
                    with open(input_file, 'r', encoding='utf-8') as f:
                        input_content = f.read()

                    # Enhance the transcript
                    result = enhancer.enhance_transcript(
                        input_content=input_content,
                        custom_prompt=args.prompt
                    )

                    if result['success']:
                        # Save enhanced content
                        enhancer.save_enhanced_transcript(
                            enhanced_text=result['enhanced_text'],
                            original_file=input_file,
                            output_file=str(output_file_path),
                            processing_time=result['processing_time'],
                            output_tokens=result['output_tokens']
                        )
                        successful_files += 1
                        print(f"✅ Saved: {output_file_path}")
                    else:
                        print(f"❌ Error: {result['error']}")

                    # Rate limiting between files (Gemini API: 5 requests/minute)
                    if i < len(input_files):
                        print("⏳ Waiting 12 seconds before next file (rate limiting)...")
                        time.sleep(12)

                except Exception as e:
                    print(f"❌ Error processing {input_file}: {str(e)}")
                    continue

            total_time = time.time() - total_start_time

            print(f"\n{'='*50}")
            print(f"✅ Batch Enhancement Complete")
            print(f"{'='*50}")
            print(f"📁 Input files: {len(input_files)}")
            print(f"✅ Successfully processed: {successful_files}")
            print(f"❌ Failed: {len(input_files) - successful_files}")
            print(f"📁 Output directory: {output_dir}")
            print(f"⏱️  Total time: {format_duration(total_time)}")
            if args.translate:
                print(f"🌍 Target language: {args.translate}")
            print(f"{'='*50}")

        else:
            # Single file processing
            input_file = input_files[0]
            output_file_path = args.output

            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                input_content = f.read()

            # Enhance the transcript
            result = enhancer.enhance_transcript(
                input_content=input_content,
                custom_prompt=args.prompt
            )

            if result['success']:
                # Save enhanced content
                enhancer.save_enhanced_transcript(
                    enhanced_text=result['enhanced_text'],
                    original_file=input_file,
                    output_file=output_file_path,
                    processing_time=result['processing_time'],
                    output_tokens=result['output_tokens']
                )

                total_time = time.time() - total_start_time

                print(f"\n{'='*50}")
                print(f"✅ Enhancement Complete")
                print(f"{'='*50}")
                print(f"📄 Input: {input_file}")
                print(f"📄 Output: {output_file_path}")
                print(f"⏱️  Total time: {format_duration(total_time)}")
                if args.translate:
                    print(f"🌍 Target language: {args.translate}")
                print(f"📊 Character count: {len(result['enhanced_text']):,}")
                print(f"{'='*50}")
            else:
                print(f"\n❌ Enhancement Failed")
                print(f"Error: {result['error']}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nEnhancement interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
