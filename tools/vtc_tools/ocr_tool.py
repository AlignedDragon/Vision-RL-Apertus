"""OCR tool — extract text from image or region using EasyOCR."""

import numpy as np
from PIL import Image

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def ocr(image: Image.Image, bbox: list[int] | None = None) -> str:
    """Extract text from image. If bbox=[x1,y1,x2,y2], crop first.

    Returns concatenated text found in the region.
    """
    if bbox:
        x1, y1, x2, y2 = bbox
        image = image.crop((x1, y1, x2, y2))

    reader = _get_reader()
    results = reader.readtext(np.array(image))
    texts = [text for _, text, conf in results if conf > 0.3]
    return " ".join(texts) if texts else "(no text found)"
