"""
Audio trimming utilities for XTTS model inference.

This module provides intelligent audio trimming to remove excess silence
and over-generated audio, especially for short sentences. It uses a hybrid
approach combining text-based duration prediction with audio energy analysis.

The trimming system helps prevent the common issue where short sentences
generate unnecessarily long audio with trailing silence or artifacts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__file__)


@dataclass
class TrimConfig:
    """Configuration for audio trimming behavior.

    This configuration allows fine-tuning of the trimming algorithm for
    different use cases and languages. All duration-related values are
    in seconds unless otherwise specified.
    """

    # Text-based prediction parameters
    base_duration_per_char: float = 0.025  # 25ms per character (~40 chars/sec)
    pause_per_punct: float = 0.1  # 100ms pause per punctuation mark
    text_safety_margin: float = 1.15  # 15% safety buffer on text prediction
    min_length_ratio: float = 0.85  # Never trim more than 15% below prediction

    # Language-specific multipliers for different speech patterns
    language_multipliers: dict = field(default_factory=lambda: {
        'ja': 1.3,      # Japanese: denser information per character
        'zh-cn': 1.3,   # Chinese: similar to Japanese
        'ko': 1.2,      # Korean: moderately dense
        'vi': 2.7,      # Vietnamese: normal after normalization
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


def predict_audio_length(
    text: str,
    language: str,
    sample_rate: int = 24000,
    config: Optional[TrimConfig] = None
) -> int:
    """
    Predict expected audio duration from text characteristics.

    Uses character-based calculation which works universally across all
    languages including CJK (Japanese, Chinese, Korean) that don't use
    spaces between words.

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
    """
    config = config or TrimConfig()

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
    strategy: str = 'hybrid',
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
        max_length = int(predicted_length * config.text_safety_margin)

        logger.debug(f"Trimming (text_only): '{text[:50]}...' ({len(text)} chars)")
        logger.debug(f"  Original: {len(audio_array)} samples ({len(audio_array)/sample_rate:.2f}s)")
        logger.debug(f"  Predicted: {predicted_length} samples ({predicted_length/sample_rate:.2f}s)")
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
        max_from_prediction = int(predicted_length * config.text_safety_margin)

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

        # Calculate reduction percentage
        reduction_pct = 100 * (1 - final_length / len(audio_array)) if len(audio_array) > 0 else 0

        logger.debug(f"Trimming (hybrid): '{text[:50]}...' ({len(text)} chars)")
        logger.debug(f"  Original: {len(audio_array)} samples ({len(audio_array)/sample_rate:.2f}s)")
        logger.debug(f"  Predicted: {predicted_length} samples ({predicted_length/sample_rate:.2f}s)")
        logger.debug(f"  Max from prediction: {max_from_prediction} samples ({max_from_prediction/sample_rate:.2f}s)")
        logger.debug(f"  Detected endpoint: {detected_endpoint} samples ({detected_endpoint/sample_rate:.2f}s)")
        logger.debug(f"  Min allowed: {min_length} samples ({min_length/sample_rate:.2f}s)")
        logger.debug(f"  Final: {final_length} samples ({final_length/sample_rate:.2f}s)")
        logger.debug(f"  Reduction: {len(audio_array) - final_length} samples ({reduction_pct:.1f}%)")

        return audio_array[:final_length]

    # Unknown strategy - log warning and return original
    logger.warning(f"Unknown trimming strategy '{strategy}', returning original audio")
    return audio_array
