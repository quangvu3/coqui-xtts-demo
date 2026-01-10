import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__file__)


class MultiSpeakerInference:
    """
    Handle multi-speaker inference with automatic speaker switching and silence insertion.
    """

    def __init__(self, xtts_model, speaker_registry, inference_fn):
        """
        Initialize the multi-speaker inference engine.

        Args:
            xtts_model: The XTTS model instance
            speaker_registry: UnifiedSpeakerRegistry instance
            inference_fn: The inference function to use for TTS generation
        """
        self.xtts_model = xtts_model
        self.speaker_registry = speaker_registry
        self.inference_fn = inference_fn

    def synthesize_segments(
        self,
        segments,
        language='Auto',
        temperature=0.2,
        top_p=0.85,
        top_k=70,
        repetition_penalty=9.0,
        sentence_silence_ms=500
    ):
        """
        Synthesize audio from parsed segments with multiple speakers.
        Automatically adds 1 second of silence between different speakers.

        Args:
            segments: List of parsed segments from TextParser
            language: Target language
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
            sentence_silence_ms: Silence to add between sentences (milliseconds)

        Returns:
            tuple: (final_wav_array, total_tokens)
        """
        out_wavs = []
        total_tokens = 0
        previous_speaker = None

        logger.info(f"Synthesizing {len(segments)} segments")

        for i, segment in enumerate(segments):
            if segment['type'] == 'silence':
                # Generate silence
                duration = segment['duration']
                logger.info(f"Segment {i+1}: Silence ({duration}s)")
                silence_wav = self._generate_silence(duration)
                out_wavs.append(silence_wav)

            elif segment['type'] == 'speech':
                # Get speaker embeddings
                speaker_id = segment['speaker_id']
                text = segment['text']

                # Add 1 second silence between different speakers
                if previous_speaker is not None and previous_speaker != speaker_id:
                    logger.info(f"Speaker change detected: [{previous_speaker}] -> [{speaker_id}], adding 1s silence")
                    silence_wav = self._generate_silence(1.0)
                    out_wavs.append(silence_wav)

                logger.info(f"Segment {i+1}: Speaker [{speaker_id}] - {text[:50]}...")

                # Get speaker embeddings from registry
                gpt_cond_latent, speaker_embedding = self.speaker_registry.get_speaker(speaker_id)

                if gpt_cond_latent is None or speaker_embedding is None:
                    logger.error(f"Failed to get embeddings for speaker: {speaker_id}")
                    continue

                try:
                    # Call the inference function
                    wav, tokens = self.inference_fn(
                        input_text=text,
                        language=language,
                        speaker_id=None,  # We provide embeddings directly
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        sentence_silence_ms=sentence_silence_ms
                    )

                    out_wavs.append(wav)
                    total_tokens += tokens
                    previous_speaker = speaker_id  # Update previous speaker

                except Exception as e:
                    logger.error(f"Error synthesizing segment {i+1}: {e}")
                    continue

        # Concatenate all audio segments
        if out_wavs:
            final_wav = np.concatenate(out_wavs)
            logger.info(f"Successfully synthesized {len(out_wavs)} segments, {total_tokens} total tokens")
        else:
            logger.warning("No audio segments generated")
            final_wav = np.array([])

        return final_wav, total_tokens

    def _generate_silence(self, duration_seconds):
        """
        Generate silence array.

        Args:
            duration_seconds: Duration of silence in seconds

        Returns:
            numpy.ndarray: Array of zeros representing silence
        """
        sample_rate = self.xtts_model.config.audio.sample_rate
        num_samples = int(duration_seconds * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    def estimate_duration(self, segments):
        """
        Estimate total audio duration from segments.

        This is a rough estimate based on text length and silence durations.

        Args:
            segments: List of parsed segments

        Returns:
            float: Estimated duration in seconds
        """
        total_duration = 0.0

        for segment in segments:
            if segment['type'] == 'silence':
                total_duration += segment['duration']
            elif segment['type'] == 'speech':
                # Rough estimate: ~150 words per minute for speech
                # That's 2.5 words per second, or 0.4 seconds per word
                word_count = len(segment['text'].split())
                total_duration += word_count * 0.4

        return total_duration
