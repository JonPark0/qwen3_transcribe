"""Core enhancement module with API support."""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from google import genai


class TranscriptEnhancer:
    """
    TranscriptEnhancer provides transcript enhancement using Google Gemini API.

    This class can be used as a library (with progress callbacks) or through CLI.
    """

    def __init__(self, verbose: bool = False, target_language: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        """
        Initialize TranscriptEnhancer.

        Args:
            verbose: Enable detailed logging
            target_language: Target language for translation (ISO 639-1 code)
            model_name: Gemini model name to use (default: gemini-flash-latest)
        """
        self.verbose = verbose
        self.target_language = target_language
        self.model_name = model_name
        self._client = None

        # Gemini 2.5 Pro token limits
        self.MAX_INPUT_TOKENS = 1_048_576  # ~4.2M characters
        self.MAX_OUTPUT_TOKENS = 65_536    # ~262K characters

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using the 4-character approximation.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def format_token_count(self, tokens: int) -> str:
        """
        Format token count with commas.

        Args:
            tokens: Token count

        Returns:
            Formatted token count string
        """
        return f"{tokens:,}"

    def validate_input_size(self, content: str, prompt: str) -> Tuple[bool, str]:
        """
        Validate if input content + prompt fits within token limits.

        Args:
            content: Original transcript content
            prompt: Enhancement prompt

        Returns:
            Tuple of (is_valid, error_message)
        """
        content_tokens = self.estimate_tokens(content)
        prompt_tokens = self.estimate_tokens(prompt)
        total_input_tokens = content_tokens + prompt_tokens

        self.log(f"Token analysis:")
        self.log(f"  Original content: {self.format_token_count(content_tokens)} tokens")
        self.log(f"  Enhancement prompt: {self.format_token_count(prompt_tokens)} tokens")
        self.log(f"  Total input: {self.format_token_count(total_input_tokens)} tokens")
        self.log(f"  Input limit: {self.format_token_count(self.MAX_INPUT_TOKENS)} tokens")
        self.log(f"  Input usage: {(total_input_tokens / self.MAX_INPUT_TOKENS * 100):.1f}%")

        if total_input_tokens > self.MAX_INPUT_TOKENS:
            excess_tokens = total_input_tokens - self.MAX_INPUT_TOKENS
            error_msg = (
                f"Input exceeds token limit by {self.format_token_count(excess_tokens)} tokens.\n"
                f"Total input: {self.format_token_count(total_input_tokens)} tokens\n"
                f"Maximum allowed: {self.format_token_count(self.MAX_INPUT_TOKENS)} tokens\n"
                f"Consider reducing the transcript size or splitting into smaller files."
            )
            return False, error_msg

        return True, ""

    def estimate_output_requirements(self, input_tokens: int) -> int:
        """
        Estimate required output tokens based on input size.

        Args:
            input_tokens: Estimated input token count

        Returns:
            Estimated output token count
        """
        # Enhanced content is typically 1.2-1.5x the original size
        # due to improved grammar, structure, and formatting
        return int(input_tokens * 1.3)

    def validate_output_capacity(self, input_tokens: int) -> Tuple[bool, str]:
        """
        Check if estimated output will fit within output token limits.

        Args:
            input_tokens: Estimated input token count

        Returns:
            Tuple of (is_valid, warning_message)
        """
        estimated_output = self.estimate_output_requirements(input_tokens)

        self.log(f"Output analysis:")
        self.log(f"  Estimated output: {self.format_token_count(estimated_output)} tokens")
        self.log(f"  Output limit: {self.format_token_count(self.MAX_OUTPUT_TOKENS)} tokens")
        self.log(f"  Output usage: {(estimated_output / self.MAX_OUTPUT_TOKENS * 100):.1f}%")

        if estimated_output > self.MAX_OUTPUT_TOKENS:
            excess_tokens = estimated_output - self.MAX_OUTPUT_TOKENS
            error_msg = (
                f"Estimated output may exceed token limit by {self.format_token_count(excess_tokens)} tokens.\n"
                f"Estimated output: {self.format_token_count(estimated_output)} tokens\n"
                f"Maximum allowed: {self.format_token_count(self.MAX_OUTPUT_TOKENS)} tokens\n"
                f"Consider splitting the transcript into smaller sections."
            )
            return False, error_msg

        return True, ""

    def setup_gemini(self, api_key: str):
        """
        Setup Gemini API with the provided API key.

        Args:
            api_key: Google AI API key

        Raises:
            Exception: If API configuration fails
        """
        try:
            self._client = genai.Client(api_key=api_key)
            self.log(f"Gemini API configured successfully with model: {self.model_name}")
        except Exception as e:
            raise Exception(f"Failed to configure Gemini API: {str(e)}")

    def get_default_prompt(self) -> str:
        """
        Get the default enhancement prompt.

        Returns:
            Default prompt string
        """
        base_prompt = """You are an expert transcript editor and enhancer. Please improve this audio transcript by:

1. **Grammar & Structure**: Fix grammatical errors, improve sentence structure, and ensure proper punctuation
2. **Clarity & Readability**: Make the text more readable while preserving the original meaning and speaker's intent
3. **Formatting**: Organize content with appropriate headings, bullet points, and paragraphs where suitable
4. **Coherence**: Ensure logical flow and smooth transitions between ideas
5. **Technical Terms**: Correct any misheard technical terms or jargon based on context
6. **Filler Words**: Remove excessive "um," "uh," "you know," etc. while keeping natural speech patterns
7. **Speaker Intent**: Maintain the original tone and meaning - don't add new information

Important guidelines:
- Preserve all factual content and key information
- Keep the original speaker's voice and style
- Don't add information that wasn't in the original transcript
- Maintain timestamps if present
- Focus on enhancement, not rewriting

Please provide the enhanced transcript:"""

        if self.target_language and self.target_language != "en":
            base_prompt += f"\n\nAdditionally, translate the enhanced transcript to {self.target_language} (ISO 639-1: {self.target_language})."

        return base_prompt

    def enhance_transcript(
        self,
        input_content: str,
        custom_prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Enhance transcript content using Gemini API.

        Args:
            input_content: Original transcript content
            custom_prompt: Optional custom enhancement prompt (replaces default)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - enhanced_text: Enhanced transcript text
                - processing_time: Time taken to process
                - input_tokens: Estimated input token count
                - output_tokens: Estimated output token count
                - error: Error message if failed
        """
        start_time = time.time()

        result = {
            'success': False,
            'enhanced_text': '',
            'processing_time': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'error': None
        }

        try:
            # Stage 1: Validation
            if progress_callback:
                progress_callback({
                    'stage': 'validating',
                    'progress': 0.1,
                    'message': 'Validating token limits...'
                })

            # Prepare the prompt
            if custom_prompt:
                prompt = f"{custom_prompt}\n\nOriginal transcript:\n{input_content}"
            else:
                prompt = f"{self.get_default_prompt()}\n\nOriginal transcript:\n{input_content}"

            # Validate input size
            self.log("Validating token limits...")
            input_valid, input_error = self.validate_input_size(input_content, prompt)
            if not input_valid:
                result['error'] = f"Input validation failed:\n{input_error}"
                return result

            # Validate output capacity
            content_tokens = self.estimate_tokens(input_content)
            result['input_tokens'] = content_tokens

            output_valid, output_error = self.validate_output_capacity(content_tokens)
            if not output_valid:
                self.log(f"⚠️  Warning: {output_error}")
                self.log("Proceeding anyway, but output may be truncated.")

            # Stage 2: Enhancing
            if progress_callback:
                progress_callback({
                    'stage': 'enhancing',
                    'progress': 0.3,
                    'message': 'Sending to Gemini API for enhancement...'
                })

            self.log("✅ Token validation passed. Sending to Gemini for enhancement...")

            if self._client is None:
                raise RuntimeError("Gemini client is not initialized. Call setup_gemini() first.")

            # Generate enhanced content
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent, factual enhancement
                    max_output_tokens=self.MAX_OUTPUT_TOKENS,  # Use configured limit
                )
            )

            enhanced_content = response.text

            # Stage 3: Processing results
            if progress_callback:
                progress_callback({
                    'stage': 'processing',
                    'progress': 0.9,
                    'message': 'Processing enhanced content...'
                })

            # Validate actual output size
            actual_output_tokens = self.estimate_tokens(enhanced_content)
            result['output_tokens'] = actual_output_tokens

            self.log(f"Actual output: {self.format_token_count(actual_output_tokens)} tokens")

            if actual_output_tokens >= self.MAX_OUTPUT_TOKENS * 0.95:  # 95% threshold
                self.log("⚠️  Warning: Output is close to token limit. Content may be truncated.")

            result['success'] = True
            result['enhanced_text'] = enhanced_content
            result['processing_time'] = time.time() - start_time

            # Stage 4: Complete
            if progress_callback:
                progress_callback({
                    'stage': 'complete',
                    'progress': 1.0,
                    'message': 'Enhancement completed successfully'
                })

            self.log(f"Enhancement completed in {result['processing_time']:.1f}s")
            return result

        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.log(f"Error during enhancement: {str(e)}")

            if progress_callback:
                progress_callback({
                    'stage': 'error',
                    'progress': 0,
                    'message': f'Error: {str(e)}'
                })

            return result

    def save_enhanced_transcript(
        self,
        enhanced_text: str,
        original_file: str,
        output_file: str,
        processing_time: float,
        output_tokens: int
    ):
        """
        Save enhanced transcript with metadata.

        Args:
            enhanced_text: Enhanced transcript text
            original_file: Path to original transcript file
            output_file: Path to output file
            processing_time: Time taken to process
            output_tokens: Estimated output token count
        """
        self.log(f"Saving enhanced transcript to: {output_file}")

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add enhancement metadata with token information
        metadata = f"""# Enhanced Transcript

**Original File:** {Path(original_file).name}
**Enhanced:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Processing Time:** {processing_time:.1f}s
**Model:** {self.model_name}
**Output Tokens:** {self.format_token_count(output_tokens)}
{'**Target Language:** ' + self.target_language if self.target_language else ''}

---

"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(metadata + enhanced_text)

        self.log("Enhanced transcript saved successfully")
