"""
Length penalty calculation for XTTS model inference.

The length_penalty parameter controls how much the model should match
the input text length when generating audio:
- Values closer to -1.0: Encourage shorter audio (prevent over-generation)
- Values closer to +0.0: Allow longer audio (for longer text)
"""


def calculate_length_penalty(
    text_length: int,
    max_length: int = 180,
    exponent: float = 1.0
) -> float:
    """
    Calculate dynamic length penalty using a linear function.

    Uses linear scaling (exponent=1.0) to bias toward shorter sentences.
    This ensures short text generates appropriately brief audio while
    longer text can expand more naturally.

    Args:
        text_length: The character count of the text being synthesized
        max_length: The maximum text length for normalization (default: 180)
        exponent: The power exponent (default: 1.0 for linear)

    Returns:
        float: Length penalty in range [-1.5, -0.5]
    """
    # Clamp text_length to max_length
    clamped_length = min(text_length, max_length)

    # Normalize to [0, 1] range
    normalized = clamped_length / max_length

    # Apply power function and scale to [-1.5, -0.5] range
    penalty = (normalized ** exponent) - 1.5

    return penalty
