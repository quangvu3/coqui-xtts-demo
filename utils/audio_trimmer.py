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
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
from utils.logger import setup_logger
from utils.speaker_stats import SpeakerStatsTracker

# Try to import torch for tensor handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = setup_logger(__file__)

# Filler sentences for text augmentation strategy
# Used to help generate appropriate audio length for very short text segments
FILLER_SENTENCES = {
    'vi': [
        "Đây là một đoạn văn bản mẫu để thử nghiệm giọng nói tổng hợp.",
        "Văn bản này được thêm vào để giúp tạo ra âm thanh tự nhiên hơn.",
        "Thử nghiệm hệ thống tạo giọng nói với các đoạn văn bản mẫu.",
    ],
    'en': [
        "This is a sample text to test the synthesized voice output.",
        "Additional text is added here to help generate more natural speech.",
        "Testing the voice synthesis system with sample content.",
    ],
    'es': [
        "Este es un texto de ejemplo para probar la voz sintetizada.",
        "Se añade texto adicional para ayudar a generar un habla más natural.",
    ],
    'fr': [
        "Ceci est un texte d'exemple pour tester la voix synthétisée.",
        "Du texte supplémentaire est ajouté pour aider à générer une parole plus naturelle.",
    ],
    'de': [
        "Dies ist ein Beispieltext zur Prüfung der synthetisierten Stimme.",
        "Zusätzlicher Text wird hier eingefügt, um natürlichere Sprache zu erzeugen.",
    ],
    'it': [
        "Questo è un testo di esempio per testare la voce sintetizzata.",
        "Viene aggiunto del testo aggiuntivo per aiutare a generare un discorso più naturale.",
    ],
    'pt': [
        "Este é um texto de exemplo para testar a voz sintetizada.",
        "Texto adicional é adicionado aqui para ajudar a gerar fala mais natural.",
    ],
    'ja': [
        "これは合成音声をテストするためのサンプルテキストです。",
        "より自然な音声を生成するために追加のテキストが記載されています。",
    ],
    'zh-cn': [
        "这是用于测试合成语音的示例文本。",
        "此处添加了额外的文本，以帮助生成更自然的语音。",
    ],
    'ko': [
        "합성 음성을 테스트하기 위한 샘플 텍스트입니다.",
        "더 자연스러운 음성을 생성하기 위해 추가 텍스트가 포함되어 있습니다.",
    ],
}


# Configuration for multi-signal boundary detection
# Used by extract_original_audio() to find speech boundaries in augmented audio
BOUNDARY_DETECTION_CONFIG = {
    # Search window (balanced: search around prediction point)
    'search_before_ms': 800,       # 0.8 seconds before prediction
    'search_after_ms': 1200,       # 1.2 seconds after prediction (allow for late boundaries)

    # Scoring weights for candidate selection (must sum to 1.0)
    'weight_energy_drop': 0.35,    # Weight for energy drop detection
    'weight_low_energy': 0.40,     # Weight for being in low energy region
    'weight_zcr_change': 0.10,     # Weight for zero-crossing rate change
    'weight_proximity': 0.15,      # Weight for proximity to prediction (reduced bias)

    # Detection thresholds
    'energy_drop_threshold': 0.25, # Minimum energy drop ratio to consider
    'low_energy_percentile': 15,   # Percentile for low energy threshold
    'min_candidate_gap_ms': 50,    # Minimum ms between candidates to avoid duplicates

    # Frame analysis parameters
    'frame_length_ms': 25.0,       # Analysis frame length
    'hop_length_ms': 10.0,         # Hop between frames

    # Fallback behavior
    'fallback_bias_percent': 0,    # No bias - use prediction directly if no candidates
}


def _ensure_1d_array(audio):
    """Convert audio to 1D numpy array, handling torch tensors and 2D arrays."""
    # Convert torch tensor to numpy if needed
    if HAS_TORCH and hasattr(audio, 'cpu'):  # torch tensor
        audio = audio.cpu().numpy()

    # Handle 2D array input (shape: [1, samples] or [channels, samples])
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.squeeze()

    return audio


# =============================================================================
# Acoustic Feature Functions for Boundary Detection
# =============================================================================

def compute_short_term_energy(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0
) -> Tuple[np.ndarray, int, int]:
    """
    Calculate RMS energy per frame for the audio signal.

    Args:
        audio: 1D numpy array of audio samples
        sample_rate: Audio sample rate (default: 24000)
        frame_length_ms: Analysis frame length in milliseconds (default: 25.0)
        hop_length_ms: Hop between frames in milliseconds (default: 10.0)

    Returns:
        Tuple of (energy_array, frame_length_samples, hop_length_samples)
    """
    frame_length = int((frame_length_ms / 1000.0) * sample_rate)
    hop_length = int((hop_length_ms / 1000.0) * sample_rate)

    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0

    return energy, frame_length, hop_length


def compute_energy_derivative(energy: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of energy to detect rapid transitions.

    Negative values indicate energy drops (potential speech boundaries).

    Args:
        energy: Array of frame energy values

    Returns:
        Array of energy derivatives (same length as input, first value is 0)
    """
    if len(energy) < 2:
        return np.zeros_like(energy)

    derivative = np.zeros_like(energy)
    derivative[1:] = energy[1:] - energy[:-1]
    return derivative


def compute_zero_crossing_rate(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0
) -> np.ndarray:
    """
    Compute zero-crossing rate per frame.

    ZCR is useful for detecting voiced/unvoiced transitions. Unvoiced segments
    (silence, fricatives) typically have higher ZCR than voiced speech.

    Args:
        audio: 1D numpy array of audio samples
        sample_rate: Audio sample rate (default: 24000)
        frame_length_ms: Analysis frame length in milliseconds (default: 25.0)
        hop_length_ms: Hop between frames in milliseconds (default: 10.0)

    Returns:
        Array of zero-crossing rates per frame
    """
    frame_length = int((frame_length_ms / 1000.0) * sample_rate)
    hop_length = int((hop_length_ms / 1000.0) * sample_rate)

    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    zcr = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        frame = audio[start:end]

        if len(frame) > 1:
            # Count sign changes
            signs = np.sign(frame)
            sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
            zcr[i] = sign_changes / len(frame)

    return zcr


def find_low_energy_regions(
    energy: np.ndarray,
    percentile: float = 20
) -> np.ndarray:
    """
    Find regions where energy is below the specified percentile.

    Args:
        energy: Array of frame energy values
        percentile: Percentile threshold for "low" energy (default: 20)

    Returns:
        Boolean array where True indicates low energy frame
    """
    if len(energy) == 0:
        return np.array([], dtype=bool)

    threshold = np.percentile(energy, percentile)
    return energy < threshold


# =============================================================================
# Boundary Candidate System
# =============================================================================

@dataclass
class BoundaryCandidate:
    """Represents a potential speech boundary with scoring metrics."""
    frame_idx: int           # Frame index in the analysis window
    sample_idx: int          # Sample index in the original audio
    energy_drop_score: float # Score based on energy drop magnitude
    low_energy_score: float  # Score based on being in low energy region
    zcr_change_score: float  # Score based on ZCR change
    proximity_score: float   # Score based on proximity to prediction
    total_score: float       # Weighted combination of all scores


def find_boundary_candidates(
    audio: np.ndarray,
    prediction_sample: int,
    sample_rate: int = 24000,
    config: Optional[dict] = None
) -> List[BoundaryCandidate]:
    """
    Find potential speech boundary candidates using multi-signal analysis.

    Uses energy drops, low energy regions, and zero-crossing rate changes
    to identify likely speech boundaries. Scores candidates with conservative
    bias toward earlier cuts (to avoid filler bleed).

    Args:
        audio: Full audio array to analyze
        prediction_sample: Predicted split point (sample index)
        sample_rate: Audio sample rate (default: 24000)
        config: Optional configuration dict (uses BOUNDARY_DETECTION_CONFIG if None)

    Returns:
        List of BoundaryCandidate objects, sorted by total_score descending
    """
    if config is None:
        config = BOUNDARY_DETECTION_CONFIG

    # Extract config values
    search_before_ms = config['search_before_ms']
    search_after_ms = config['search_after_ms']
    frame_length_ms = config['frame_length_ms']
    hop_length_ms = config['hop_length_ms']
    energy_drop_threshold = config['energy_drop_threshold']
    low_energy_percentile = config['low_energy_percentile']
    min_candidate_gap_ms = config['min_candidate_gap_ms']

    # Weights
    w_energy_drop = config['weight_energy_drop']
    w_low_energy = config['weight_low_energy']
    w_zcr_change = config['weight_zcr_change']
    w_proximity = config['weight_proximity']

    # Convert to samples
    search_before_samples = int((search_before_ms / 1000.0) * sample_rate)
    search_after_samples = int((search_after_ms / 1000.0) * sample_rate)
    min_candidate_gap_samples = int((min_candidate_gap_ms / 1000.0) * sample_rate)

    # Define asymmetric search window
    window_start = max(0, prediction_sample - search_before_samples)
    window_end = min(len(audio), prediction_sample + search_after_samples)

    if window_end <= window_start:
        return []

    # Extract search window
    window_audio = audio[window_start:window_end]

    # Compute acoustic features
    energy, frame_length, hop_length = compute_short_term_energy(
        window_audio, sample_rate, frame_length_ms, hop_length_ms
    )
    energy_derivative = compute_energy_derivative(energy)
    zcr = compute_zero_crossing_rate(
        window_audio, sample_rate, frame_length_ms, hop_length_ms
    )
    zcr_derivative = compute_energy_derivative(zcr)  # Reuse for ZCR changes
    low_energy_mask = find_low_energy_regions(energy, low_energy_percentile)

    # Find candidates at energy drops
    candidates = []
    min_gap_frames = max(1, min_candidate_gap_samples // hop_length)

    # Normalize energy for consistent scoring
    max_energy = np.max(energy) if np.max(energy) > 0 else 1.0

    for i in range(1, len(energy)):
        # Check for significant energy drop
        if energy_derivative[i] < 0:
            drop_magnitude = abs(energy_derivative[i]) / max_energy

            if drop_magnitude >= energy_drop_threshold or low_energy_mask[i]:
                # Calculate sample position in original audio
                frame_sample_in_window = i * hop_length
                sample_in_audio = window_start + frame_sample_in_window

                # Check minimum gap from existing candidates
                too_close = any(
                    abs(c.frame_idx - i) < min_gap_frames for c in candidates
                )
                if too_close:
                    continue

                # Calculate scores

                # 1. Energy drop score (larger drops = higher score)
                energy_drop_score = min(1.0, drop_magnitude / 0.5)

                # 2. Low energy score (in low energy region = 1.0)
                low_energy_score = 1.0 if low_energy_mask[i] else 0.0

                # 3. ZCR change score (larger changes = higher score)
                max_zcr_change = np.max(np.abs(zcr_derivative)) if np.max(np.abs(zcr_derivative)) > 0 else 1.0
                zcr_change_score = min(1.0, abs(zcr_derivative[i]) / max_zcr_change) if i < len(zcr_derivative) else 0.0

                # 4. Proximity score (balanced: prefer boundaries close to prediction)
                # Both early and late cuts get penalized equally based on distance
                distance_from_prediction = sample_in_audio - prediction_sample
                if distance_from_prediction <= 0:
                    # Before prediction: penalize based on distance
                    normalized_dist = abs(distance_from_prediction) / search_before_samples
                    proximity_score = max(0.0, 1.0 - normalized_dist)
                else:
                    # After prediction: same treatment, penalize based on distance
                    normalized_dist = distance_from_prediction / search_after_samples
                    proximity_score = max(0.0, 1.0 - normalized_dist)

                # Calculate total weighted score
                total_score = (
                    w_energy_drop * energy_drop_score +
                    w_low_energy * low_energy_score +
                    w_zcr_change * zcr_change_score +
                    w_proximity * proximity_score
                )

                candidates.append(BoundaryCandidate(
                    frame_idx=i,
                    sample_idx=sample_in_audio,
                    energy_drop_score=energy_drop_score,
                    low_energy_score=low_energy_score,
                    zcr_change_score=zcr_change_score,
                    proximity_score=proximity_score,
                    total_score=total_score
                ))

    # Sort by total score descending
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    logger.debug(f"  Found {len(candidates)} boundary candidates in search window")
    if candidates:
        best = candidates[0]
        logger.debug(
            f"    Best candidate: sample {best.sample_idx} "
            f"(score={best.total_score:.3f}, energy_drop={best.energy_drop_score:.3f}, "
            f"low_energy={best.low_energy_score:.3f}, zcr={best.zcr_change_score:.3f}, "
            f"proximity={best.proximity_score:.3f})"
        )

    return candidates


def create_augmented_text(
    text: str,
    language: str = 'en',
    max_chars: int = 180
) -> str:
    """
    Combine original short text with a filler sentence.

    This helps generate appropriate audio length for very short text segments
    that would otherwise cause the TTS model to over-generate audio.

    Args:
        text: Original short text to augment
        language: Language code for selecting appropriate filler (default: 'en')
        max_chars: Maximum characters for combined text (default: 180)

    Returns:
        str: Combined text with original + filler, truncated to max_chars

    Examples:
        >>> create_augmented_text("Xin chào", "vi", 180)
        'Xin chào. Đây là một đoạn văn bản mẫu để thử nghiệm giọng nói tổng hợp.'
    """
    # Get filler sentence for language
    fillers = FILLER_SENTENCES.get(language, FILLER_SENTENCES.get('en', []))
    filler = random.choice(fillers)

    # Add period if text doesn't end with punctuation
    punctuation_marks = '.!?,;:。！？，'
    if text and text[-1] not in punctuation_marks:
        text = text.rstrip() + '.'

    # Combine and truncate
    combined = f"{text} {filler}"
    if len(combined) > max_chars:
        combined = combined[:max_chars - 3] + "..."

    return combined


def extract_original_audio(
    full_audio: np.ndarray,
    original_text: str,
    augmented_text: str,
    language: str,
    sample_rate: int = 24000,
    speaker_stats_tracker: Optional[SpeakerStatsTracker] = None,
    speaker_id: Optional[str] = None,
    silence_threshold_db: float = -40.0  # Deprecated, kept for API compatibility
) -> np.ndarray:
    """
    Extract audio corresponding to original_text from full audio.

    Uses prediction-based ratio to estimate the split point, then applies
    multi-signal boundary detection to find the optimal cut point. This approach
    uses energy drops, low energy regions, and zero-crossing rate changes to
    identify speech boundaries, with conservative bias toward earlier cuts to
    prevent filler speech bleed.

    Args:
        full_audio: Audio generated from augmented text
        original_text: The original short text
        augmented_text: The combined text (original + filler)
        language: Language code for prediction
        sample_rate: Audio sample rate (default: 24000)
        speaker_stats_tracker: Optional tracker for per-speaker rates
        speaker_id: Optional speaker ID for using learned rates
        silence_threshold_db: Deprecated, kept for API compatibility

    Returns:
        np.ndarray: Audio corresponding to original text with 10ms fade-out

    Examples:
        >>> # After generating augmented audio
        >>> original_audio = extract_original_audio(
        ...     full_audio=augmented_audio,
        ...     original_text="Xin chào",
        ...     augmented_text="Xin chào Đây là một đoạn văn bản...",
        ...     language="vi",
        ...     sample_rate=24000
        ... )
    """
    # Use prediction functions for more accurate split estimation
    # This accounts for language rates, punctuation, and word count adjustments
    predicted_original = predict_audio_length(
        original_text, language, sample_rate,
        speaker_stats_tracker=speaker_stats_tracker,
        speaker_id=speaker_id
    )

    # Extract filler portion by removing original text
    filler_only = augmented_text.replace(original_text, '').strip()
    if filler_only:
        predicted_filler = predict_audio_length(
            filler_only, language, sample_rate,
            speaker_stats_tracker=speaker_stats_tracker,
            speaker_id=speaker_id
        )
    else:
        predicted_filler = 0

    # Calculate split ratio using predicted lengths
    total_predicted = predicted_original + predicted_filler
    if total_predicted > 0:
        ratio = predicted_original / total_predicted
    else:
        # Fallback to raw text length ratio
        ratio = len(original_text) / len(augmented_text)

    logger.debug(f"    Prediction ratio: {ratio:.3f} (original={predicted_original}, filler={predicted_filler})")

    # Estimate split point using prediction ratio
    estimated_split = int(len(full_audio) * ratio)
    logger.debug(f"    Estimated split: sample {estimated_split} ({estimated_split/sample_rate:.3f}s)")

    # Use multi-signal boundary detection to find best cut point
    candidates = find_boundary_candidates(
        audio=full_audio,
        prediction_sample=estimated_split,
        sample_rate=sample_rate,
        config=BOUNDARY_DETECTION_CONFIG
    )

    # Select best candidate with conservative bias
    if candidates:
        # Use the highest-scoring candidate
        best_candidate = candidates[0]
        trim_point = best_candidate.sample_idx
        logger.debug(
            f"    Using boundary candidate at sample {trim_point} "
            f"({trim_point/sample_rate:.3f}s, score={best_candidate.total_score:.3f})"
        )
    else:
        # Fallback: use prediction directly (no bias)
        trim_point = estimated_split
        logger.debug(
            f"    No candidates found, using prediction directly: sample {trim_point} "
            f"({trim_point/sample_rate:.3f}s)"
        )

    # Ensure trim point is valid
    trim_point = max(0, min(trim_point, len(full_audio)))

    # Extract audio with fade-out to prevent artifacts
    extracted = full_audio[:trim_point].copy()

    # Apply 10ms fade-out at the end
    fade_samples = int(0.010 * sample_rate)  # 10ms
    if len(extracted) > fade_samples:
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        extracted[-fade_samples:] = extracted[-fade_samples:] * fade_curve

    logger.debug(f"    Final extracted length: {len(extracted)} samples ({len(extracted)/sample_rate:.3f}s)")

    return extracted


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
    fallback_audio_per_word: float = 0.4,
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

    # Audio is too long - need to retry with adjusted parameters
    short_text_word_threshold = 11
    if word_count < short_text_word_threshold:
        logger.info(f"  Result: OVER-GENERATED ({actual_length/max_allowed:.1f}x) - will use text augmentation")
    else:
        logger.info(f"  Result: OVER-GENERATED ({actual_length/max_allowed:.1f}x) - retrying with adjusted length_penalty")
    # logger.info(f"  Text segment: '{text[:100]}{'...' if len(text) > 100 else ''}'")

    if inference_fn is None:
        logger.warning("No inference_fn provided, cannot retry - returning original audio")
        return audio

    # Track all retry results and best result (shortest within tolerance)
    all_attempts = []
    best_audio = None
    best_length = float('inf')

    # Strategy 1: Text augmentation for very short text
    if word_count < short_text_word_threshold and inference_fn is not None:
        logger.debug(f"  Strategy: Text augmentation for very short text ({word_count} words)")

        # Create augmented text
        augmented_text = create_augmented_text(text, language, max_chars=180)

        logger.debug(f"    Original: '{text[:50]}...' ({len(text)} chars)")
        logger.debug(f"    Augmented: '{augmented_text[:50]}...' ({len(augmented_text)} chars)")

        try:
            # Generate audio for augmented text
            retry_kwargs = {**inference_kwargs, 'length_penalty': -1.7, 'temperature': 1.0}
            out = inference_fn(
                text=augmented_text,
                language=language,
                **retry_kwargs
            )
            full_audio = _ensure_1d_array(out["wav"])

            # Extract original portion using prediction-based split
            original_audio = extract_original_audio(
                full_audio=full_audio,
                original_text=text,
                augmented_text=augmented_text,
                language=language,
                sample_rate=sample_rate,
                speaker_stats_tracker=speaker_stats_tracker,
                speaker_id=speaker_id
            )

            # Check if within tolerance
            extracted_length = len(original_audio)

            logger.debug(f"    Full audio: {len(full_audio)} samples ({len(full_audio)/sample_rate:.2f}s)")
            logger.debug(f"    Extracted audio: {extracted_length} samples ({extracted_length/sample_rate:.2f}s)")

            if extracted_length <= max_allowed:
                logger.debug(f"    Result: SUCCESS - extracted audio within tolerance")
            else:
                logger.debug(f"    Result: Extracted audio still too long ({extracted_length/max_allowed:.1f}x)")
            return original_audio
        except Exception as e:
            logger.error(f"    Text augmentation failed: {e}")
            # Fall through to random retry
    
    # Strategy 2: Random parameter tuning retry
    for retry in range(max_retries):
        length_penalty = random.uniform(1.0, 2.0)
        temperature = random.uniform(2.0, 3.0)

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


def trim_audio_end(
    audio: np.ndarray,
    sample_rate: int = 24000,
    max_trailing_silence_ms: float = 500.0,
    silence_threshold_db: float = -40.0,
    min_audio_ms: float = 100.0,
    fade_out_ms: float = 10.0
) -> np.ndarray:
    """
    Trim trailing silence from audio.

    This function:
    1. Detects where trailing silence begins using RMS energy analysis
    2. Trims excessive trailing silence, keeping up to max_trailing_silence_ms
    3. Applies a short fade-out to prevent clicks from abrupt trimming

    Args:
        audio: Audio numpy array (will be converted to 1D if needed)
        sample_rate: Audio sample rate (default: 24000)
        max_trailing_silence_ms: Maximum trailing silence to keep in ms (default: 500.0)
        silence_threshold_db: RMS below this (in dB) is considered silence (default: -40.0)
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

    # Detect trailing silence
    silence_start = detect_trailing_silence_start(
        audio,
        sample_rate=sample_rate,
        silence_threshold_db=silence_threshold_db
    )

    # Calculate final trim point
    # Keep at most max_trailing_silence_ms of silence after speech ends
    trim_point = original_length
    if silence_start < original_length:
        allowed_silence_end = silence_start + max_silence_samples
        trim_point = min(original_length, allowed_silence_end)
        logger.debug(f"Silence starts at sample {silence_start} ({silence_start/sample_rate:.3f}s)")

    # Enforce minimum length
    trim_point = max(trim_point, min_samples)

    # Apply fade-out if we're trimming
    if trim_point < original_length:
        trimmed = audio[:trim_point].copy()

        # Apply fade-out at the end to prevent clicks from abrupt cut
        if fade_samples > 0 and len(trimmed) > fade_samples:
            fade_curve = np.linspace(1.0, 0.0, fade_samples)
            trimmed[-fade_samples:] = trimmed[-fade_samples:] * fade_curve

        logger.debug(
            f"Trimmed audio: {original_length} -> {trim_point} samples "
            f"({original_length/sample_rate:.3f}s -> {trim_point/sample_rate:.3f}s)"
        )
        return trimmed

    return audio
