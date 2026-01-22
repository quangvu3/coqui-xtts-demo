"""
Length penalty calculation for XTTS model inference.

The length_penalty parameter controls how much the model should match
the input text length when generating audio:
- Values closer to 1.0: For shorter text
- Values closer to 2.0: For longer text
"""


def calculate_length_penalty(
    text_length: int,
    max_length: int = 180
) -> float:
    """
    Calculate dynamic length penalty using linear scaling.

    Maps text length linearly to penalty range [1.0, 2.0] without bias.

    Args:
        text_length: The character count of the text being synthesized
        max_length: The maximum text length for normalization (default: 180)

    Returns:
        float: Length penalty in range [1.0, 2.0]
    """
    # Clamp text_length to max_length
    clamped_length = min(text_length, max_length)

    # Normalize to [0, 1] range
    normalized = clamped_length / max_length

    # Scale to [1.0, 2.0] range
    penalty = 1.0 + normalized

    return penalty
