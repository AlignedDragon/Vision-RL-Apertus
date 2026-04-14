"""Counting tool — count objects matching a description."""

from PIL import Image
from tools.grounding_tool import grounding


def counting(image: Image.Image, description: str) -> int:
    """Count objects matching description in the image."""
    boxes = grounding(image, description)
    return len(boxes)
