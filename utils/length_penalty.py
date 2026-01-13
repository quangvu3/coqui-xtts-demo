"""
Length penalty calculation for XTTS model inference.

The length_penalty parameter controls how much the model should match
the input text length when generating audio:
- Values closer to -1.0: Encourage shorter audio (prevent over-generation)
- Values closer to +1.0: Allow longer audio (for longer text)
"""


def calculate_length_penalty(
    text_length: int,
    max_length: int = 180,
    exponent: float = 2.0
) -> float:
    """
    Calculate dynamic length penalty using a non-linear power function.

    Uses quadratic scaling by default (exponent=2.0) to strongly bias
    toward short sentences. This ensures short text generates appropriately
    brief audio while longer text can expand more naturally.

    Args:
        text_length: The character count of the text being synthesized
        max_length: The maximum text length for normalization (default: 180)
        exponent: The power exponent (default: 2.0 for quadratic)

    Returns:
        float: Length penalty in range [-1.0, 1.0]

    Examples:
        >>> calculate_length_penalty(45)   # Short
        -1.0
        >>> calculate_length_penalty(90)   # Medium
        -0.500
        >>> calculate_length_penalty(180)  # Long
        1.0
    """
    if text_length <= 45:
        return -1.0
    
    # Clamp text_length to max_length
    clamped_length = min(text_length, max_length)

    # Normalize to [0, 1] range
    normalized = clamped_length / max_length

    # Apply power function and scale to [-1, 1] range
    penalty = (2 * (normalized ** exponent)) - 1

    return penalty
