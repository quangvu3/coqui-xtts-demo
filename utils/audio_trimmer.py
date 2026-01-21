"""
Audio validation utilities for XTTS model inference.

This module provides audio length validation and retry logic for short text segments.
It helps prevent the common issue where short sentences generate unnecessarily long
audio with trailing silence or artifacts.

Additionally provides audio end trimming functionality to:
- Remove excessive trailing silence (keeping max 0.5s)
- Detect and remove click/pop artifacts at audio segment ends

Per-Speaker Learning:
    When a SpeakerStatsTracker is provided to the prediction functions, speaker-specific
    learned rates will be used instead of global defaults. This allows the system to
    adapt to individual speaker speaking patterns.
"""

import random
import numpy as np
from typing import Optional, Callable, Tuple
from utils.logger import setup_logger
from utils.speaker_stats import SpeakerStatsTracker

# Try to import torch for tensor handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = setup_logger(__file__)


def _ensure_1d_array(audio):
    """Convert audio to 1D numpy array, handling torch tensors and 2D arrays."""
    # Convert torch tensor to numpy if needed
    if HAS_TORCH and hasattr(audio, 'cpu'):  # torch tensor
        audio = audio.cpu().numpy()

    # Handle 2D array input (shape: [1, samples] or [channels, samples])
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.squeeze()

    return audio


def predict_vietnamese_audio_length(
    text: str,
    sample_rate: int = 24000,
    speaker_stats_tracker: Optional[SpeakerStatsTracker] = None,
    speaker_id: Optional[str] = None,
    fallback_audio_per_word: float = 0.6
) -> int:
    """
    Predict Vietnamese audio length using word-based calculation.

    Vietnamese text uses spaces between words, making word count a reliable
    predictor of audio duration. Uses speaker-specific learned rate if available,
    otherwise falls back to the configured default (0.6 seconds per word).

    Args:
        text: Input Vietnamese text to synthesize
        sample_rate: Audio sample rate (default: 24000)
        speaker_stats_tracker: Optional tracker for per-speaker learned rates
        speaker_id: Optional speaker ID for using learned rates
        fallback_audio_per_word: Fallback rate when no learned data available (default: 0.6)

    Returns:
        int: Predicted audio length in samples

    Examples:
        >>> predict_vietnamese_audio_length("xin chào", 24000)
        28800  # 2 words * 0.6s * 24000 samples = 28800

        >>> predict_vietnamese_audio_length("tôi tên là minh", 24000)
        57600  # 4 words * 0.6s * 24000 samples = 57600
    """
    word_count = len(text.split())

    # Try to use learned rate from speaker stats
    audio_per_word = None
    if speaker_id and speaker_stats_tracker:
        audio_per_word = speaker_stats_tracker.get_audio_per_word(
            speaker_id, 'vi', fallback_audio_per_word
        )
        logger.debug(f"Using learned rate for {speaker_id}/vi: {audio_per_word:.4f}s/word")

    # Fall back to default if no learned rate available
    if audio_per_word is None:
        audio_per_word = fallback_audio_per_word

    expected_seconds = word_count * audio_per_word

    return int(expected_seconds * sample_rate)


def predict_audio_length(
    text: str,
    language: str,
    sample_rate: int = 24000,
    speaker_stats_tracker: Optional[SpeakerStatsTracker] = None,
    speaker_id: Optional[str] = None,
    fallback_audio_per_word: float = 0.6,
    fallback_audio_per_char: float = 0.025,
    language_multipliers: Optional[dict] = None,
    base_duration_per_char: float = 0.025,
    pause_per_punct: float = 0.5,
    short_sentence_thresholds: Optional[dict] = None
) -> int:
    """
    Predict expected audio duration from text characteristics.

    Uses character-based calculation for most languages, but Vietnamese
    uses word-based calculation since Vietnamese has spaces between words
    and character count doesn't map well to speech duration.

    When speaker_stats_tracker and speaker_id are provided, uses speaker-specific
    learned rates for better prediction accuracy.

    Args:
        text: Input text to synthesize
        language: Language code (en, vi, ja, zh-cn, etc.)
        sample_rate: Audio sample rate (default: 24000)
        speaker_stats_tracker: Optional tracker for per-speaker learned rates
        speaker_id: Optional speaker ID for using learned rates
        fallback_audio_per_word: Fallback rate for Vietnamese (default: 0.6)
        fallback_audio_per_char: Fallback rate for other languages (default: 0.025)
        language_multipliers: Optional dict of language code to multiplier
        base_duration_per_char: Base duration per character in seconds (default: 0.025)
        pause_per_punct: Pause duration per punctuation mark (default: 0.5)
        short_sentence_thresholds: Optional dict of word count to multiplier

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
    # Default language multipliers
    if language_multipliers is None:
        language_multipliers = {
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
        }

    # Default short sentence thresholds
    if short_sentence_thresholds is None:
        short_sentence_thresholds = {
            3: 0.5,   # < 3 words: 50% more aggressive
            5: 0.7,   # < 5 words: 30% more aggressive
        }

    # Vietnamese uses word-based prediction
    if language == 'vi':
        return predict_vietnamese_audio_length(
            text, sample_rate, speaker_stats_tracker, speaker_id, fallback_audio_per_word
        )

    # Try to use learned rate for character-based languages
    if speaker_id and speaker_stats_tracker:
        audio_per_char = speaker_stats_tracker.get_audio_per_char(
            speaker_id, language, fallback_audio_per_char
        )
        char_count = len(text)
        expected_seconds = char_count * audio_per_char
        logger.debug(f"Using learned rate for {speaker_id}/{language}: {audio_per_char:.4f}s/char")
        return int(expected_seconds * sample_rate)

    # Fall back to default character-based calculation
    text_length = len(text)
    base_samples = text_length * base_duration_per_char * sample_rate

    # Punctuation adjustment (pauses)
    punctuation_marks = '.!?,;:'
    punct_count = sum(text.count(p) for p in punctuation_marks)
    punct_samples = punct_count * pause_per_punct * sample_rate

    # Apply language multiplier
    lang_multiplier = language_multipliers.get(language, 1.0)

    # Word count factor for very short sentences
    word_count = len(text.split())
    word_multiplier = 1.0
    for threshold, multiplier in sorted(short_sentence_thresholds.items()):
        if word_count < threshold:
            word_multiplier = multiplier
            break

    # Final prediction
    predicted_samples = (base_samples + punct_samples) * lang_multiplier * word_multiplier

    return int(predicted_samples)


def validate_audio_length(
    audio: np.ndarray,
    text: str,
    language: str,
    sample_rate: int = 24000,
    inference_fn: Optional[Callable] = None,
    word_threshold: int = 15,
    length_tolerance: float = 1.3,
    max_retries: int = 5,
    speaker_stats_tracker: Optional[SpeakerStatsTracker] = None,
    speaker_id: Optional[str] = None,
    **inference_kwargs
) -> np.ndarray:
    """
    Validate audio length for short text segments and retry with adjusted length_penalty if needed.

    For text segments with fewer than `word_threshold` words, this function estimates
    the expected audio length using `predict_audio_length()` and compares it to the
    generated audio. If the actual audio is significantly longer than expected
    (exceeding `length_tolerance`), it retries inference with more aggressive
    length_penalty values in the range [-2.0, -1.5].

    Args:
        audio: Generated audio numpy array from initial inference
        text: Source text that was synthesized
        language: Language code (en, vi, ja, zh-cn, etc.)
        sample_rate: Audio sample rate (default: 24000)
        inference_fn: Optional function to call for retry inference with signature:
                      fn(text, language, length_penalty, **kwargs) -> dict with 'wav' key
                      Must accept length_penalty as keyword argument.
        word_threshold: Words below this trigger validation (default: 15)
        length_tolerance: Retry if actual > expected * tolerance (default: 1.3)
        max_retries: Number of retry attempts (default: 5)
        speaker_stats_tracker: Optional tracker for per-speaker learned rates
        speaker_id: Optional speaker ID for using learned rates
        **inference_kwargs: Additional arguments passed to inference_fn on retry

    Returns:
        np.ndarray: Audio array (possibly from retry inference)

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
        ...     length_tolerance=1.3,
        ...     max_retries=5,
        ...     speaker_stats_tracker=tracker,
        ...     speaker_id=speaker_id,
        ...     gpt_cond_latent=gpt_cond_latent,
        ...     speaker_embedding=speaker_embedding,
        ...     temperature=temperature,
        ...     top_p=top_p,
        ...     top_k=top_k,
        ...     repetition_penalty=repetition_penalty,
        ...     enable_text_splitting=True,
        ... )
    """
    audio = _ensure_1d_array(audio)

    # Skip validation for longer text segments
    word_count = len(text.split())
    if word_count >= word_threshold:
        logger.debug(f"Skipping length validation: {word_count} words >= {word_threshold} threshold")
        return audio

    # Estimate expected audio length
    expected_length = predict_audio_length(
        text, language, sample_rate,
        speaker_stats_tracker=speaker_stats_tracker,
        speaker_id=speaker_id
    )
    actual_length = len(audio)

    logger.debug(f"Validating short text ({word_count} words): '{text[:50]}...'")
    logger.debug(f"  Expected: {expected_length} samples ({expected_length/sample_rate:.2f}s)")
    logger.debug(f"  Actual: {actual_length} samples ({actual_length/sample_rate:.2f}s)")
    logger.debug(f"  Ratio: {actual_length/expected_length:.2f}x (tolerance: {length_tolerance}x)")

    # Check if audio is within tolerance
    max_allowed = int(expected_length * length_tolerance)
    if actual_length <= max_allowed:
        logger.debug(f"  Result: PASS - audio within tolerance")
        return audio

    # Audio is too long - need to retry with adjusted length_penalty
    logger.info(f"  Result: OVER-GENERATED ({actual_length/max_allowed:.1f}x) - retrying with adjusted length_penalty")
    logger.info(f"  Text segment: '{text[:100]}{'...' if len(text) > 100 else ''}'")

    if inference_fn is None:
        logger.warning("No inference_fn provided, cannot retry - returning original audio")
        return audio

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
        return best_audio

    # No valid result - select shortest from all attempts
    if all_attempts:
        shortest = min(all_attempts, key=lambda x: x['length'])
        logger.warning(f"  No valid attempts after {max_retries} retries - using shortest: {shortest['length']} samples")
        return shortest['audio']

    # Fallback: return original audio
    logger.warning("  All retries failed - returning original audio")
    return audio


def detect_trailing_silence_start(
    audio: np.ndarray,
    sample_rate: int = 24000,
    silence_threshold_db: float = -40.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0
) -> int:
    """
    Detect where trailing silence begins using RMS energy analysis.

    Works backwards from the end of audio to find where continuous silence starts.
    Uses frame-based RMS analysis to detect when audio energy drops below threshold.

    Args:
        audio: 1D numpy array of audio samples
        sample_rate: Audio sample rate (default: 24000)
        silence_threshold_db: RMS below this (in dB) is considered silence (default: -40.0)
        frame_length_ms: Analysis frame length in milliseconds (default: 25.0)
        hop_length_ms: Hop between frames in milliseconds (default: 10.0)

    Returns:
        int: Sample index where trailing silence begins (or len(audio) if no silence)

    Examples:
        >>> audio = np.concatenate([np.sin(np.linspace(0, 100, 24000)), np.zeros(12000)])
        >>> detect_trailing_silence_start(audio)
        24000  # Silence starts at sample 24000
    """
    audio = _ensure_1d_array(audio)

    if len(audio) == 0:
        return 0

    # Convert parameters to samples
    frame_length = int((frame_length_ms / 1000.0) * sample_rate)
    hop_length = int((hop_length_ms / 1000.0) * sample_rate)

    # Convert dB threshold to linear amplitude
    # RMS threshold: 10^(dB/20)
    silence_threshold = 10 ** (silence_threshold_db / 20.0)

    # Calculate number of frames
    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)

    # Work backwards through frames
    silence_start_frame = num_frames  # Default: no silence detected

    for frame_idx in range(num_frames - 1, -1, -1):
        start_sample = frame_idx * hop_length
        end_sample = min(start_sample + frame_length, len(audio))
        frame = audio[start_sample:end_sample]

        # Calculate RMS energy
        rms = np.sqrt(np.mean(frame ** 2))

        if rms > silence_threshold:
            # Found non-silent frame, silence starts after this frame
            silence_start_frame = frame_idx + 1
            break
    else:
        # All frames are silent
        silence_start_frame = 0

    # Convert frame index to sample index
    silence_start_sample = silence_start_frame * hop_length

    return min(silence_start_sample, len(audio))


def detect_end_clicks(
    audio: np.ndarray,
    sample_rate: int = 24000,
    search_region_ms: float = 200.0,
    click_threshold_factor: float = 3.0,
    window_ms: float = 20.0
) -> int:
    """
    Detect clicks/pops near the end of audio using amplitude spike detection.

    Looks for sudden amplitude spikes in the final portion of the audio that
    indicate click or pop artifacts (common in TTS synthesis).

    Args:
        audio: 1D numpy array of audio samples
        sample_rate: Audio sample rate (default: 24000)
        search_region_ms: How far back from end to search in ms (default: 200.0)
        click_threshold_factor: Peak/RMS ratio to detect click (default: 3.0)
        window_ms: Sliding window size in ms for local RMS (default: 20.0)

    Returns:
        int: Sample index where click begins, or len(audio) if no click detected

    Examples:
        >>> # Audio with click at end
        >>> audio = np.concatenate([np.sin(np.linspace(0, 100, 24000)), np.array([0.9, -0.8])])
        >>> click_start = detect_end_clicks(audio)
        >>> click_start < len(audio)
        True
    """
    audio = _ensure_1d_array(audio)

    if len(audio) == 0:
        return 0

    # Convert parameters to samples
    search_region_samples = int((search_region_ms / 1000.0) * sample_rate)
    window_samples = int((window_ms / 1000.0) * sample_rate)

    # Limit search region to audio length
    search_region_samples = min(search_region_samples, len(audio))
    window_samples = max(1, min(window_samples, search_region_samples // 4))

    # Get the search region (end portion of audio)
    search_start = len(audio) - search_region_samples
    search_region = audio[search_start:]

    # Calculate overall RMS of the search region
    overall_rms = np.sqrt(np.mean(search_region ** 2))

    if overall_rms < 1e-10:
        # Audio is essentially silent, no clicks to detect
        return len(audio)

    # Slide through search region looking for spikes
    click_position = len(audio)  # Default: no click detected

    for i in range(len(search_region) - window_samples):
        window = search_region[i:i + window_samples]
        window_rms = np.sqrt(np.mean(window ** 2))
        peak = np.max(np.abs(window))

        # Check for spike: peak significantly higher than local RMS
        if window_rms > 1e-10 and peak / window_rms > click_threshold_factor:
            # Also verify this is actually higher than surrounding context
            context_start = max(0, i - window_samples)
            context_end = min(len(search_region), i + 2 * window_samples)
            context = search_region[context_start:context_end]
            context_rms = np.sqrt(np.mean(context ** 2))

            if peak > context_rms * click_threshold_factor:
                # Found a click - mark position and continue searching
                # (we want the first click in the region)
                click_position = search_start + i
                break

    return click_position


def trim_audio_end(
    audio: np.ndarray,
    sample_rate: int = 24000,
    max_trailing_silence_ms: float = 500.0,
    silence_threshold_db: float = -40.0,
    click_detection_enabled: bool = True,
    click_threshold_factor: float = 3.0,
    min_audio_ms: float = 100.0,
    fade_out_ms: float = 10.0
) -> np.ndarray:
    """
    Trim trailing silence and remove end-of-audio clicks/pops.

    This function:
    1. Detects and removes click artifacts at the end (if enabled)
    2. Trims excessive trailing silence, keeping up to max_trailing_silence_ms
    3. Applies a short fade-out to prevent new clicks from trimming

    Args:
        audio: Audio numpy array (will be converted to 1D if needed)
        sample_rate: Audio sample rate (default: 24000)
        max_trailing_silence_ms: Maximum trailing silence to keep in ms (default: 500.0)
        silence_threshold_db: RMS below this (in dB) is considered silence (default: -40.0)
        click_detection_enabled: Whether to detect/remove end clicks (default: True)
        click_threshold_factor: Peak/RMS ratio to detect click (default: 3.0)
        min_audio_ms: Never trim below this duration in ms (default: 100.0)
        fade_out_ms: Fade-out duration to apply at trim point in ms (default: 10.0)

    Returns:
        np.ndarray: Trimmed audio array

    Examples:
        >>> # Audio with 1 second of trailing silence
        >>> speech = np.sin(np.linspace(0, 100, 24000))  # 1 second
        >>> silence = np.zeros(24000)  # 1 second silence
        >>> audio = np.concatenate([speech, silence])
        >>> trimmed = trim_audio_end(audio, max_trailing_silence_ms=500)
        >>> len(trimmed) < len(audio)  # Trimmed to ~1.5s (1s speech + 0.5s silence)
        True
    """
    audio = _ensure_1d_array(audio)

    if len(audio) == 0:
        return audio

    # Calculate minimum samples to keep
    min_samples = int((min_audio_ms / 1000.0) * sample_rate)
    max_silence_samples = int((max_trailing_silence_ms / 1000.0) * sample_rate)
    fade_samples = int((fade_out_ms / 1000.0) * sample_rate)

    original_length = len(audio)
    trim_point = original_length

    # Step 1: Detect clicks at the end (if enabled)
    if click_detection_enabled:
        click_start = detect_end_clicks(
            audio,
            sample_rate=sample_rate,
            click_threshold_factor=click_threshold_factor
        )
        if click_start < original_length:
            trim_point = click_start
            logger.debug(f"Click detected at sample {click_start} ({click_start/sample_rate:.3f}s)")

    # Step 2: Detect trailing silence
    silence_start = detect_trailing_silence_start(
        audio[:trim_point],  # Only analyze up to click point
        sample_rate=sample_rate,
        silence_threshold_db=silence_threshold_db
    )

    # Step 3: Calculate final trim point
    # Keep at most max_trailing_silence_ms of silence after speech ends
    if silence_start < trim_point:
        allowed_silence_end = silence_start + max_silence_samples
        trim_point = min(trim_point, allowed_silence_end)
        logger.debug(f"Silence starts at sample {silence_start} ({silence_start/sample_rate:.3f}s)")

    # Step 4: Enforce minimum length
    trim_point = max(trim_point, min_samples)

    # Step 5: Apply fade-out if we're trimming
    if trim_point < original_length:
        trimmed = audio[:trim_point].copy()

        # Apply fade-out at the end to prevent new clicks
        if fade_samples > 0 and len(trimmed) > fade_samples:
            fade_curve = np.linspace(1.0, 0.0, fade_samples)
            trimmed[-fade_samples:] = trimmed[-fade_samples:] * fade_curve

        logger.debug(
            f"Trimmed audio: {original_length} -> {trim_point} samples "
            f"({original_length/sample_rate:.3f}s -> {trim_point/sample_rate:.3f}s)"
        )
        return trimmed

    return audio
