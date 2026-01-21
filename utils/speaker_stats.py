"""
Per-speaker audio generation statistics tracking.

This module tracks actual audio generation metrics per speaker and language,
allowing the system to learn speaker-specific speech rates over time.
It uses exponential weighted moving average (EWMA) for stable statistics
that adapt to speaker patterns while remaining smooth.
"""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__file__)

# EWMA alpha factor - 0.3 means 30% weight for new observation
EWMA_ALPHA = 0.3

# Minimum generations before using learned rates
MIN_GENERATIONS = 5


@dataclass
class LanguageStats:
    """Statistics for a specific speaker-language combination."""
    total_words: int = 0
    total_chars: int = 0
    total_samples: int = 0
    generation_count: int = 0

    # Learned rates (seconds per unit)
    audio_per_word: float = 0.0  # For Vietnamese and other word-based languages
    audio_per_char: float = 0.0  # For character-based languages

    # EWMA state for continuous learning
    _ewma_samples_per_word: float = 0.0
    _ewma_samples_per_char: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_words': self.total_words,
            'total_chars': self.total_chars,
            'total_samples': self.total_samples,
            'generation_count': self.generation_count,
            'audio_per_word': self.audio_per_word,
            'audio_per_char': self.audio_per_char,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LanguageStats':
        """Create from dictionary (loaded from JSON)."""
        stats = cls()
        stats.total_words = data.get('total_words', 0)
        stats.total_chars = data.get('total_chars', 0)
        stats.total_samples = data.get('total_samples', 0)
        stats.generation_count = data.get('generation_count', 0)
        stats.audio_per_word = data.get('audio_per_word', 0.0)
        stats.audio_per_char = data.get('audio_per_char', 0.0)
        return stats


@dataclass
class SpeakerStats:
    """Statistics for a specific speaker across all languages."""
    generation_count: int = 0
    last_used: Optional[str] = None
    languages: Dict[str, LanguageStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'generation_count': self.generation_count,
            'last_used': self.last_used,
            'languages': {
                lang: stats.to_dict()
                for lang, stats in self.languages.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpeakerStats':
        """Create from dictionary (loaded from JSON)."""
        stats = cls()
        stats.generation_count = data.get('generation_count', 0)
        stats.last_used = data.get('last_used')
        for lang, lang_data in data.get('languages', {}).items():
            stats.languages[lang] = LanguageStats.from_dict(lang_data)
        return stats


class SpeakerStatsTracker:
    """Track per-speaker audio generation statistics with EWMA learning."""

    def __init__(self, stats_path: str = "cache/speaker_stats.json"):
        """Initialize the tracker.

        Args:
            stats_path: Path to JSON file for persisting statistics.
        """
        self.stats_path = stats_path
        self.stats: Dict[str, SpeakerStats] = {}
        self._load_stats()

    def _load_stats(self) -> None:
        """Load statistics from disk if exists."""
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for speaker_id, speaker_data in data.items():
                        self.stats[speaker_id] = SpeakerStats.from_dict(speaker_data)

                        # Restore EWMA state from loaded values so stats accumulate correctly
                        for lang, lang_stats in self.stats[speaker_id].languages.items():
                            if lang_stats.audio_per_word > 0:
                                lang_stats._ewma_samples_per_word = lang_stats.audio_per_word * 24000
                            if lang_stats.audio_per_char > 0:
                                lang_stats._ewma_samples_per_char = lang_stats.audio_per_char * 24000
            except (json.JSONDecodeError, KeyError, Exception):
                self.stats = {}
        else:
            self.stats = {}

    def _save_stats(self) -> None:
        """Save statistics to disk."""
        if not self.stats:
            return
        try:
            dir_path = os.path.dirname(self.stats_path)
            os.makedirs(dir_path or '.', exist_ok=True)

            data = {
                speaker_id: speaker_stats.to_dict()
                for speaker_id, speaker_stats in self.stats.items()
            }
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _get_or_create_speaker(self, speaker_id: str) -> SpeakerStats:
        """Get or create stats for a speaker."""
        if speaker_id not in self.stats:
            self.stats[speaker_id] = SpeakerStats()
        return self.stats[speaker_id]

    def _get_or_create_language(self, speaker_id: str, language: str) -> LanguageStats:
        """Get or create stats for a speaker-language combination."""
        speaker = self._get_or_create_speaker(speaker_id)
        if language not in speaker.languages:
            speaker.languages[language] = LanguageStats()
        return speaker.languages[language]

    def record_generation(
        self,
        speaker_id: str,
        language: str,
        word_count: int,
        char_count: int,
        audio_samples: int,
        sample_rate: int = 24000
    ) -> None:
        """Record a generation event for statistics learning.

        Uses exponential weighted moving average (EWMA) to update the learned
        rates, giving 30% weight to new observations.

        Args:
            speaker_id: Unique identifier for the speaker.
            language: Language code (e.g., 'vi', 'en', 'ja').
            word_count: Number of words in the generated text.
            char_count: Number of characters in the generated text.
            audio_samples: Number of audio samples generated.
            sample_rate: Audio sample rate (default: 24000).
        """
        if word_count <= 0 and char_count <= 0:
            return

        lang_stats = self._get_or_create_language(speaker_id, language)
        speaker = self._get_or_create_speaker(speaker_id)

        # Calculate actual rate for this generation
        samples_per_word = audio_samples / word_count if word_count > 0 else 0
        samples_per_char = audio_samples / char_count if char_count > 0 else 0

        # Update EWMA for samples per word (used for Vietnamese and similar)
        if word_count > 0:
            if lang_stats._ewma_samples_per_word == 0:
                # First observation - initialize
                lang_stats._ewma_samples_per_word = samples_per_word
            else:
                # Apply EWMA update
                lang_stats._ewma_samples_per_word = (
                    EWMA_ALPHA * samples_per_word +
                    (1 - EWMA_ALPHA) * lang_stats._ewma_samples_per_word
                )
            # Update learned rate in seconds per word
            lang_stats.audio_per_word = lang_stats._ewma_samples_per_word / sample_rate

        # Update EWMA for samples per char (used for other languages)
        if char_count > 0:
            if lang_stats._ewma_samples_per_char == 0:
                # First observation - initialize
                lang_stats._ewma_samples_per_char = samples_per_char
            else:
                # Apply EWMA update
                lang_stats._ewma_samples_per_char = (
                    EWMA_ALPHA * samples_per_char +
                    (1 - EWMA_ALPHA) * lang_stats._ewma_samples_per_char
                )
            # Update learned rate in seconds per character
            lang_stats.audio_per_char = lang_stats._ewma_samples_per_char / sample_rate

        # Update totals and counters
        lang_stats.total_words += word_count
        lang_stats.total_chars += char_count
        lang_stats.total_samples += audio_samples
        lang_stats.generation_count += 1

        speaker.generation_count += 1
        speaker.last_used = datetime.now(timezone.utc).isoformat()

        # Persist to disk
        self._save_stats()

    def get_audio_per_word(self, speaker_id: str, language: str = 'vi',
                           default_rate: float = 0.6) -> float:
        """Get learned seconds per word for Vietnamese or similar languages.

        Args:
            speaker_id: Unique identifier for the speaker.
            language: Language code (default: 'vi').
            default_rate: Fallback rate if no stats available or too few samples.

        Returns:
            Seconds per word (learned if available, otherwise default).
        """
        if speaker_id in self.stats:
            lang_stats = self.stats[speaker_id].languages.get(language)
            if lang_stats and lang_stats.generation_count >= MIN_GENERATIONS:
                return lang_stats.audio_per_word

        logger.debug(f"Using default rate {default_rate} for {speaker_id}/{language}")
        return default_rate

    def get_audio_per_char(self, speaker_id: str, language: str,
                           default_rate: float = 0.025) -> float:
        """Get learned seconds per character for character-based languages.

        Args:
            speaker_id: Unique identifier for the speaker.
            language: Language code.
            default_rate: Fallback rate if no stats available or too few samples.

        Returns:
            Seconds per character (learned if available, otherwise default).
        """
        if speaker_id in self.stats:
            lang_stats = self.stats[speaker_id].languages.get(language)
            if lang_stats and lang_stats.generation_count >= MIN_GENERATIONS:
                return lang_stats.audio_per_char

        logger.debug(f"Using default rate {default_rate} for {speaker_id}/{language}")
        return default_rate

    def get_speaker_stats(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific speaker.

        Args:
            speaker_id: Unique identifier for the speaker.

        Returns:
            Dictionary with speaker stats or None if not found.
        """
        if speaker_id in self.stats:
            speaker = self.stats[speaker_id]
            return {
                'speaker_id': speaker_id,
                'generation_count': speaker.generation_count,
                'last_used': speaker.last_used,
                'languages': {
                    lang: {
                        'generation_count': stats.generation_count,
                        'total_words': stats.total_words,
                        'total_chars': stats.total_chars,
                        'total_samples': stats.total_samples,
                        'audio_per_word': stats.audio_per_word,
                        'audio_per_char': stats.audio_per_char,
                    }
                    for lang, stats in speaker.languages.items()
                }
            }
        return None

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all speakers.

        Returns:
            Dictionary mapping speaker IDs to their statistics.
        """
        return {
            speaker_id: self.get_speaker_stats(speaker_id)
            for speaker_id in self.stats
        }

    def has_sufficient_data(self, speaker_id: str, language: str) -> bool:
        """Check if a speaker has sufficient data for learned rates.

        Args:
            speaker_id: Unique identifier for the speaker.
            language: Language code.

        Returns:
            True if speaker has at least MIN_GENERATIONS for this language.
        """
        if speaker_id in self.stats:
            lang_stats = self.stats[speaker_id].languages.get(language)
            if lang_stats:
                return lang_stats.generation_count >= MIN_GENERATIONS
        return False

    def reset_speaker(self, speaker_id: str) -> bool:
        """Reset statistics for a specific speaker.

        Args:
            speaker_id: Unique identifier for the speaker.

        Returns:
            True if speaker was found and reset.
        """
        if speaker_id in self.stats:
            del self.stats[speaker_id]
            self._save_stats()
            logger.info(f"Reset stats for speaker {speaker_id}")
            return True
        return False

    def reset_all(self) -> int:
        """Reset statistics for all speakers.

        Returns:
            Number of speakers that were reset.
        """
        count = len(self.stats)
        self.stats = {}
        self._save_stats()
        logger.info(f"Reset stats for all {count} speakers")
        return count
