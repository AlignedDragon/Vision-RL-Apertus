"""CropZoomIn tool — crop a region and resize for detailed inspection."""

from PIL import Image


def crop_zoom_in(image: Image.Image, bbox: list[int], ratio: float = 2.0) -> Image.Image:
    """Crop bbox=[x1,y1,x2,y2] from image and resize by ratio.

    Returns the cropped and zoomed image.
    """
    x1, y1, x2, y2 = bbox
    w, h = image.size
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = image.crop((x1, y1, x2, y2))

    new_w = int(cropped.width * ratio)
    new_h = int(cropped.height * ratio)
    return cropped.resize((new_w, new_h), Image.LANCZOS)
