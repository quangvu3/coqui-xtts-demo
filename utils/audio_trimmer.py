"""
Audio trimming utilities for XTTS model inference.

This module provides intelligent audio trimming to remove excess silence
and over-generated audio, especially for short sentences. It uses a hybrid
approach combining text-based duration prediction with audio energy analysis.

The trimming system helps prevent the common issue where short sentences
generate unnecessarily long audio with trailing silence or artifacts.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from utils.logger import setup_logger

# Try to import torch for tensor handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = setup_logger(__file__)

# Vietnamese audio duration: 0.35 seconds per word (word-based prediction)
VI_AUDIO_PER_WORD = 0.35


def _ensure_1d_array(audio):
    """Convert audio to 1D numpy array, handling torch tensors and 2D arrays."""
    # Convert torch tensor to numpy if needed
    if HAS_TORCH and hasattr(audio, 'cpu'):  # torch tensor
        audio = audio.cpu().numpy()

    # Handle 2D array input (shape: [1, samples] or [channels, samples])
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.squeeze()

    return audio


@dataclass
class TrimConfig:
    """Configuration for audio trimming behavior.

    This configuration allows fine-tuning of the trimming algorithm for
    different use cases and languages. All duration-related values are
    in seconds unless otherwise specified.
    """

    # Text-based prediction parameters
    base_duration_per_char: float = 0.025  # 25ms per character (~40 chars/sec)
    pause_per_punct: float = 0.5  # 500ms pause per punctuation mark
    text_safety_margin: float = 1.15  # 15% safety buffer for short text
    min_length_ratio: float = 0.85  # Never trim more than 15% below prediction

    # Length-aware trimming parameters
    long_text_threshold: int = 100  # chars beyond which safety margin increases
    long_text_safety_margin: float = 1.4  # 40% buffer for long text (> threshold)

    # Language-specific multipliers for different speech patterns
    language_multipliers: dict = field(default_factory=lambda: {
        'ja': 1.3,      # Japanese: denser information per character
        'zh-cn': 1.3,   # Chinese: similar to Japanese
        'ko': 1.2,      # Korean: moderately dense
        'vi': 1.0,      # Vietnamese: uses word-based prediction instead
        'en': 1.0,      # English: baseline
        'es': 1.0,      # Spanish
        'fr': 1.0,      # French
        'de': 1.0,      # German
        'it': 1.0,      # Italian
        'pt': 1.0,      # Portuguese
        'pl': 1.0,      # Polish
        'tr': 1.0,      # Turkish
        'ru': 1.0,      # Russian
        'nl': 1.0,      # Dutch
        'cs': 1.0,      # Czech
        'ar': 1.1,      # Arabic: slightly denser
        'hu': 1.0,      # Hungarian
    })

    # Vietnamese word-based prediction: seconds per word
    vi_audio_per_word: float = 0.6

    # Word count adjustments for very short sentences
    short_sentence_thresholds: dict = field(default_factory=lambda: {
        3: 0.5,   # < 3 words: 50% more aggressive
        5: 0.7,   # < 5 words: 30% more aggressive
    })

    # Energy analysis parameters
    energy_threshold: float = 0.01  # RMS threshold for speech detection
    window_ms: int = 50  # Window size for energy calculation (milliseconds)
    min_silence_ms: int = 100  # Safety margin after detected speech (milliseconds)

    # Strategy selection
    default_strategy: str = 'hybrid'  # 'hybrid', 'text_only', 'energy_only', 'none'

    # Language-specific strategy overrides
    language_strategies: dict = field(default_factory=dict)


def predict_vietnamese_audio_length(
    text: str,
    sample_rate: int = 24000,
    config: Optional[TrimConfig] = None
) -> int:
    """
    Predict Vietnamese audio length using word-based calculation.

    Vietnamese text uses spaces between words, making word count a reliable
    predictor of audio duration. Uses 0.6 seconds per word as the standard.

    Args:
        text: Input Vietnamese text to synthesize
        sample_rate: Audio sample rate (default: 24000)
        config: Optional TrimConfig for customization

    Returns:
        int: Predicted audio length in samples

    Examples:
        >>> predict_vietnamese_audio_length("xin chào", 24000)
        28800  # 2 words * 0.6s * 24000 samples = 28800

        >>> predict_vietnamese_audio_length("tôi tên là minh", 24000)
        43200  # 4 words * 0.6s * 24000 samples = 57600
    """
    config = config or TrimConfig()

    word_count = len(text.split())
    expected_seconds = word_count * config.vi_audio_per_word

    return int(expected_seconds * sample_rate)


def predict_audio_length(
    text: str,
    language: str,
    sample_rate: int = 24000,
    config: Optional[TrimConfig] = None
) -> int:
    """
    Predict expected audio duration from text characteristics.

    Uses character-based calculation for most languages, but Vietnamese
    uses word-based calculation since Vietnamese has spaces between words
    and character count doesn't map well to speech duration.

    Args:
        text: Input text to synthesize
        language: Language code (en, vi, ja, zh-cn, etc.)
        sample_rate: Audio sample rate (default: 24000)
        config: Optional TrimConfig for customization

    Returns:
        int: Predicted audio length in samples

    Examples:
        >>> predict_audio_length("Hello", "en", 24000)
        3600  # ~0.15 seconds for 5 characters

        >>> predict_audio_length("こんにちは", "ja", 24000)
        4680  # Longer due to 1.3x Japanese multiplier

        >>> predict_audio_length("xin chào", "vi", 24000)
        28800  # Word-based: 2 words * 0.6s * 24000
    """
    config = config or TrimConfig()

    # Vietnamese uses word-based prediction
    if language == 'vi':
        return predict_vietnamese_audio_length(text, sample_rate, config)

    # Base calculation using character count
    text_length = len(text)
    base_samples = text_length * config.base_duration_per_char * sample_rate

    # Punctuation adjustment (pauses)
    punctuation_marks = '.!?,;:'
    punct_count = sum(text.count(p) for p in punctuation_marks)
    punct_samples = punct_count * config.pause_per_punct * sample_rate

    # Apply language multiplier
    lang_multiplier = config.language_multipliers.get(language, 1.0)

    # Word count factor for very short sentences
    word_count = len(text.split())
    word_multiplier = 1.0
    for threshold, multiplier in sorted(config.short_sentence_thresholds.items()):
        if word_count < threshold:
            word_multiplier = multiplier
            break

    # Final prediction
    predicted_samples = (base_samples + punct_samples) * lang_multiplier * word_multiplier

    return int(predicted_samples)


def detect_speech_endpoint(
    audio_array: np.ndarray,
    sample_rate: int = 24000,
    energy_threshold: float = 0.01,
    window_ms: int = 50,
    min_silence_ms: int = 100
) -> int:
    """
    Detect where speech actually ends using energy analysis.

    Works backwards from the end of the audio to find the last point
    where speech energy is detected above the threshold.

    Args:
        audio_array: Audio samples as numpy array
        sample_rate: Audio sample rate (default: 24000)
        energy_threshold: RMS energy threshold for speech detection (default: 0.01)
        window_ms: Window size for energy calculation in milliseconds (default: 50)
        min_silence_ms: Safety margin after detected speech in milliseconds (default: 100)

    Returns:
        int: Sample index where speech ends (including safety margin)

    Examples:
        >>> # Audio with 1s speech + 1s silence
        >>> speech = np.random.randn(24000) * 0.1
        >>> silence = np.zeros(24000)
        >>> audio = np.concatenate([speech, silence])
        >>> endpoint = detect_speech_endpoint(audio, 24000)
        >>> # endpoint will be around 24000-26400 (1.0-1.1 seconds)
    """
    audio_array = _ensure_1d_array(audio_array)

    if len(audio_array) == 0:
        return 0

    window_samples = int(window_ms * sample_rate / 1000)
    min_silence_samples = int(min_silence_ms * sample_rate / 1000)

    # Edge case: audio shorter than one window
    if len(audio_array) < window_samples:
        return len(audio_array)

    # Start from the end and work backwards
    # Find first window above threshold (working backwards)
    for i in range(len(audio_array) - window_samples, 0, -window_samples):
        window = audio_array[i:i + window_samples]

        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(window ** 2))

        if rms_energy > energy_threshold:
            # Found speech, add safety margin and return
            endpoint = i + window_samples + min_silence_samples
            return min(endpoint, len(audio_array))

    # Fallback: if no speech detected, return original length
    # This prevents over-aggressive trimming
    return len(audio_array)


def trim_audio(
    audio_array: np.ndarray,
    text: str,
    language: str,
    sample_rate: int = 24000,
    strategy: str = 'text_only',
    config: Optional[TrimConfig] = None
) -> np.ndarray:
    """
    Trim generated audio to remove excess silence and over-generation.

    Uses a hybrid approach combining text-based prediction with audio
    energy analysis to intelligently trim audio while preventing
    accidental speech cutoff.

    Args:
        audio_array: Generated audio numpy array
        text: Source text that was synthesized
        language: Language code (en, vi, ja, zh-cn, etc.)
        sample_rate: Audio sample rate (default: 24000)
        strategy: Trimming strategy - 'hybrid', 'text_only', 'energy_only', 'none'
        config: Optional TrimConfig object for fine-tuning

    Returns:
        np.ndarray: Trimmed audio array

    Strategy descriptions:
        - 'hybrid' (recommended): Combines text prediction and energy analysis,
          takes minimum of both for aggressive but safe trimming
        - 'text_only': Uses only text-based duration prediction
        - 'energy_only': Uses only audio energy analysis
        - 'none': No trimming, returns original audio

    Examples:
        >>> # Generate audio for short text
        >>> audio = xtts_model.inference(text="Hi", ...)["wav"]
        >>> # Trim excess audio
        >>> trimmed = trim_audio(audio, "Hi", "en", 24000, strategy='hybrid')
        >>> # trimmed will be significantly shorter than audio
    """
    config = config or TrimConfig()

    audio_array = _ensure_1d_array(audio_array)

    # Check for strategy override
    if language in config.language_strategies:
        strategy = config.language_strategies[language]
    elif strategy == 'hybrid':
        # Use default_strategy from config if not explicitly specified
        strategy = config.default_strategy

    # Strategy: none - bypass trimming
    if strategy == 'none':
        return audio_array

    # Strategy: text_only - trim based on text prediction
    if strategy == 'text_only':
        predicted_length = predict_audio_length(text, language, sample_rate, config)

        # Dynamic safety margin based on text length
        # Long text gets more conservative trimming to prevent speech cutoff
        text_length = len(text)
        if text_length > config.long_text_threshold:
            safety_margin = config.long_text_safety_margin  # 40% buffer for long text
        else:
            safety_margin = config.text_safety_margin  # 15% buffer for short text

        max_length = int(predicted_length * safety_margin)

        logger.debug(f"Trimming (text_only): '{text[:50]}...' ({text_length} chars)")
        logger.debug(f"  Original: {len(audio_array)} samples ({len(audio_array)/sample_rate:.2f}s)")
        logger.debug(f"  Predicted: {predicted_length} samples ({predicted_length/sample_rate:.2f}s)")
        logger.debug(f"  Safety margin: {safety_margin:.2f} ({'long' if text_length > config.long_text_threshold else 'short'} text)")
        logger.debug(f"  Max allowed: {max_length} samples ({max_length/sample_rate:.2f}s)")

        # Skip trimming if audio is already within reasonable range
        # This prevents unnecessary cutting when generation is close to expected
        if len(audio_array) <= max_length * 1.1:  # within 10% of max allowed
            logger.debug(f"  Skipping trim: audio length is reasonable")
            return audio_array

        logger.debug(f"  Final: {max_length} samples ({max_length/sample_rate:.2f}s)")
        return audio_array[:max_length]

    # Strategy: energy_only - trim based on energy analysis
    if strategy == 'energy_only':
        detected_endpoint = detect_speech_endpoint(
            audio_array=audio_array,
            sample_rate=sample_rate,
            energy_threshold=config.energy_threshold,
            window_ms=config.window_ms,
            min_silence_ms=config.min_silence_ms
        )

        logger.debug(f"Trimming (energy_only): '{text[:50]}...' ({len(text)} chars)")
        logger.debug(f"  Original: {len(audio_array)} samples ({len(audio_array)/sample_rate:.2f}s)")
        logger.debug(f"  Detected endpoint: {detected_endpoint} samples ({detected_endpoint/sample_rate:.2f}s)")

        return audio_array[:detected_endpoint]

    # Strategy: hybrid - combine both approaches
    if strategy == 'hybrid':
        # Get prediction from text
        predicted_length = predict_audio_length(text, language, sample_rate, config)

        # Dynamic safety margin based on text length
        text_length = len(text)
        if text_length > config.long_text_threshold:
            safety_margin = config.long_text_safety_margin  # 40% buffer for long text
        else:
            safety_margin = config.text_safety_margin  # 15% buffer for short text

        max_from_prediction = int(predicted_length * safety_margin)

        # Get detection from audio
        detected_endpoint = detect_speech_endpoint(
            audio_array=audio_array,
            sample_rate=sample_rate,
            energy_threshold=config.energy_threshold,
            window_ms=config.window_ms,
            min_silence_ms=config.min_silence_ms
        )

        # Use the minimum of both (more aggressive trimming)
        # But ensure we don't go below minimum based on text
        min_length = int(predicted_length * config.min_length_ratio)
        final_length = max(min_length, min(max_from_prediction, detected_endpoint))

        # Skip trimming if audio is already within reasonable range
        if len(audio_array) <= final_length * 1.1:  # within 10% of calculated max
            logger.debug(f"Trimming (hybrid): '{text[:50]}...' ({text_length} chars)")
            logger.debug(f"  Skipping trim: audio length ({len(audio_array)/sample_rate:.2f}s) is reasonable")
            return audio_array

        # Calculate reduction percentage
        reduction_pct = 100 * (1 - final_length / len(audio_array)) if len(audio_array) > 0 else 0

        logger.debug(f"Trimming (hybrid): '{text[:50]}...' ({text_length} chars)")
        logger.debug(f"  Original: {len(audio_array)} samples ({len(audio_array)/sample_rate:.2f}s)")
        logger.debug(f"  Predicted: {predicted_length} samples ({predicted_length/sample_rate:.2f}s)")
        logger.debug(f"  Safety margin: {safety_margin:.2f} ({'long' if text_length > config.long_text_threshold else 'short'} text)")
        logger.debug(f"  Max from prediction: {max_from_prediction} samples ({max_from_prediction/sample_rate:.2f}s)")
        logger.debug(f"  Detected endpoint: {detected_endpoint} samples ({detected_endpoint/sample_rate:.2f}s)")
        logger.debug(f"  Min allowed: {min_length} samples ({min_length/sample_rate:.2f}s)")
        logger.debug(f"  Final: {final_length} samples ({final_length/sample_rate:.2f}s)")
        logger.debug(f"  Reduction: {len(audio_array) - final_length} samples ({reduction_pct:.1f}%)")

        return audio_array[:final_length]

    # Unknown strategy - log warning and return original
    logger.warning(f"Unknown trimming strategy '{strategy}', returning original audio")
    return audio_array


def validate_audio_length(
    audio: np.ndarray,
    text: str,
    language: str,
    sample_rate: int = 24000,
    inference_fn: Optional[Callable] = None,
    word_threshold: int = 15,
    length_tolerance: float = 1.5,
    max_retries: int = 5,
    config: Optional[TrimConfig] = None,
    **inference_kwargs
) -> np.ndarray:
    """
    Validate audio length for short text segments and retry with adjusted length_penalty if needed.

    For text segments with fewer than `word_threshold` words, this function estimates
    the expected audio length using `predict_audio_length()` and compares it to the
    generated audio. If the actual audio is significantly longer than expected
    (exceeding `length_tolerance`), it retries inference with more aggressive
    length_penalty values in the range [-0.2, -0.1].

    Args:
        audio: Generated audio numpy array from initial inference
        text: Source text that was synthesized
        language: Language code (en, vi, ja, zh-cn, etc.)
        sample_rate: Audio sample rate (default: 24000)
        inference_fn: Optional function to call for retry inference with signature:
                      fn(text, language, length_penalty, **kwargs) -> dict with 'wav' key
                      Must accept length_penalty as keyword argument.
        word_threshold: Words below this trigger validation (default: 15)
        length_tolerance: Retry if actual > expected * tolerance (default: 1.5)
        max_retries: Number of retry attempts (default: 5)
        config: Optional TrimConfig for prediction parameters
        **inference_kwargs: Additional arguments passed to inference_fn on retry

    Returns:
        np.ndarray: Audio array (possibly from retry inference) trimmed using text_only strategy

    Examples:
        >>> # During inference loop
        >>> out = xtts_model.inference(text=txt, ..., length_penalty=lp)
        >>> audio = validate_audio_length(
        ...     audio=out["wav"],
        ...     text=txt,
        ...     language=lang,
        ...     sample_rate=24000,
        ...     inference_fn=xtts_model.inference,
        ...     word_threshold=15,
        ...     length_tolerance=1.5,
        ...     max_retries=5,
        ...     gpt_cond_latent=gpt_cond_latent,
        ...     speaker_embedding=speaker_embedding,
        ...     temperature=temperature,
        ...     top_p=top_p,
        ...     top_k=top_k,
        ...     repetition_penalty=repetition_penalty,
        ...     enable_text_splitting=True,
        ... )
    """
    config = config or TrimConfig()

    audio = _ensure_1d_array(audio)

    # Skip validation for longer text segments
    word_count = len(text.split())
    if word_count >= word_threshold:
        logger.debug(f"Skipping length validation: {word_count} words >= {word_threshold} threshold")
        return trim_audio(
            audio_array=audio,
            text=text,
            language=language,
            sample_rate=sample_rate,
            strategy='text_only',
            config=config
        )

    # Estimate expected audio length
    expected_length = predict_audio_length(text, language, sample_rate, config)
    actual_length = len(audio)

    logger.debug(f"Validating short text ({word_count} words): '{text[:50]}...'")
    logger.debug(f"  Expected: {expected_length} samples ({expected_length/sample_rate:.2f}s)")
    logger.debug(f"  Actual: {actual_length} samples ({actual_length/sample_rate:.2f}s)")
    logger.debug(f"  Ratio: {actual_length/expected_length:.2f}x (tolerance: {length_tolerance}x)")

    # Check if audio is within tolerance
    max_allowed = int(expected_length * length_tolerance)
    if actual_length <= max_allowed:
        logger.debug(f"  Result: PASS - audio within tolerance")
        return trim_audio(
            audio_array=audio,
            text=text,
            language=language,
            sample_rate=sample_rate,
            strategy='text_only',
            config=config
        )

    # Audio is too long - need to retry with adjusted length_penalty
    logger.info(f"  Result: OVER-GENERATED ({actual_length/max_allowed:.1f}x) - retrying with adjusted length_penalty")
    logger.info(f"  Text segment: '{text[:100]}{'...' if len(text) > 100 else ''}'")

    if inference_fn is None:
        logger.warning("No inference_fn provided, cannot retry - returning trimmed audio")
        return trim_audio(
            audio_array=audio,
            text=text,
            language=language,
            sample_rate=sample_rate,
            strategy='text_only',
            config=config
        )

    # Track all retry results and best result (shortest within tolerance)
    all_attempts = []
    best_audio = None
    best_length = float('inf')

    for retry in range(max_retries):
        length_penalty = random.uniform(-2.0, -1.5)
        temperature = random.uniform(1.0, 2.0)

        try:
            logger.debug(f"  Retry {retry + 1}/{max_retries}: length_penalty={length_penalty:.3f}, temperature={temperature:.3f}")
            # Override temperature in kwargs (avoid duplicate keyword error)
            retry_kwargs = {**inference_kwargs, 'length_penalty': length_penalty, 'temperature': temperature}
            out = inference_fn(
                text=text,
                language=language,
                **retry_kwargs
            )
            retry_audio = _ensure_1d_array(out["wav"])
            retry_length = len(retry_audio)

            # Store all attempts
            all_attempts.append({
                'length_penalty': length_penalty,
                'temperature': temperature,
                'audio': retry_audio,
                'length': retry_length,
            })

            logger.debug(f"    Length: {retry_length} samples ({retry_length/sample_rate:.2f}s)")

            # Check if this result is within tolerance (prefer shorter)
            if retry_length <= max_allowed:
                if best_audio is None or retry_length < best_length:
                    best_audio = retry_audio
                    best_length = retry_length
                    logger.debug(f"    Result: NEW BEST - within tolerance, shorter than previous")

        except Exception as e:
            logger.error(f"  Retry {retry + 1} failed: {e}")
            continue

    # Log all attempts summary
    if all_attempts:
        lengths = [a['length'] for a in all_attempts]
        logger.debug(f"  All attempts lengths: {lengths}")
        logger.debug(f"  Best attempt: {best_length} samples ({best_length/sample_rate:.2f}s)" if best_audio is not None else f"  No valid attempts")

    # If we have a valid result, return the best (shortest) one
    if best_audio is not None:
        logger.debug(f"  Returning best attempt (shortest valid: {best_length} samples)")
        return trim_audio(
            audio_array=best_audio,
            text=text,
            language=language,
            sample_rate=sample_rate,
            strategy='text_only',
            config=config
        )

    # No valid result - select shortest from all attempts
    if all_attempts:
        shortest = min(all_attempts, key=lambda x: x['length'])
        logger.warning(f"  No valid attempts after {max_retries} retries - using shortest: {shortest['length']} samples")
        return trim_audio(
            audio_array=shortest['audio'],
            text=text,
            language=language,
            sample_rate=sample_rate,
            strategy='text_only',
            config=config
        )

    # Fallback: return original audio trimmed
    logger.warning("  All retries failed - returning original trimmed audio")
    return trim_audio(
        audio_array=audio,
        text=text,
        language=language,
        sample_rate=sample_rate,
        strategy='text_only',
        config=config
    )
