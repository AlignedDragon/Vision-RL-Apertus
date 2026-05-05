"""Grounding tool — locate objects by description using GroundingDINO."""

import numpy as np
from PIL import Image

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch

        model_id = "IDEA-Research/grounding-dino-tiny"
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if torch.cuda.is_available():
            _model = _model.to("cuda")
    return _model, _processor


def grounding(image: Image.Image, description: str, threshold: float = 0.25) -> list[list[int]]:
    """Find objects matching description. Returns list of [x1, y1, x2, y2] bounding boxes.

    Coordinates are in pixel values relative to the input image.
    """
    model, processor = _load_model()

    inputs = processor(images=image, text=description, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    boxes = []
    for box in results["boxes"]:
        x1, y1, x2, y2 = box.cpu().tolist()
        boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes
