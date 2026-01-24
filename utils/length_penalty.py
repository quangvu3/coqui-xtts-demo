"""
Length penalty calculation for XTTS model inference.

The length_penalty parameter controls how much the model should match
the input text length when generating audio:
- Values closer to -0.5: For shorter text
- Values closer to 1.5: For longer text
"""


def calculate_length_penalty(
    text_length: int,
    max_length: int = 180
) -> float:
    """
    Calculate dynamic length penalty using linear scaling.

    Maps text length linearly to penalty range [-0.5, 1.5] without bias.

    Args:
        text_length: The character count of the text being synthesized
        max_length: The maximum text length for normalization (default: 180)

    Returns:
        float: Length penalty in range [-0.5, 1.5]
    """
    # Clamp text_length to max_length
    clamped_length = min(text_length, max_length)

    # Normalize to [0, 1] range
    normalized = clamped_length / max_length

    # Scale to [-0.5, 1.5] range
    penalty = -0.5 + (2.0 * normalized)

    return penalty
