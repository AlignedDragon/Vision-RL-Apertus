"""Apertus image_zoom_in_tool for verl rollouts.

Differs from verl/tools/image_zoom_in_tool.py:
- Returns the cropped region as IBQ token text (string), not as a PIL image.
  Apertus consumes images via inline IBQ token strings, so the tool message
  body is a string starting with <|img_start|> and ending with <|img_end|>.
- Source image is loaded from `image_path` provided by the dataset row via
  tools_kwargs.image_zoom_in_tool.create_kwargs.image_path. The model passes
  only `bbox_2d` in the tool call.
- Schema exposes only `bbox_2d` (no label, no ratio).
"""

import logging
import os
import sys
import threading
from math import ceil, floor
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.vision import encode_image, load_vq_model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageZoomInEmuTool(BaseTool):
    """Crop a bbox from a source image and return its IBQ token string."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.vq_model_path: str = config["vq_model_path"]
        self.vq_device: str = config.get("vq_device", "cuda:0")
        self.target_area: int = int(config.get("target_area", 512 * 512))
        self.min_dimension: int = int(config.get("min_dimension", 28))

        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._vq_model = None
        self._vq_lock = threading.Lock()

    def _ensure_vq_model(self):
        if self._vq_model is None:
            with self._vq_lock:
                if self._vq_model is None:
                    logger.info(f"Loading IBQ vision tokenizer from {self.vq_model_path}")
                    self._vq_model = load_vq_model(self.vq_model_path, device=self.vq_device)

    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        if not (left < right and top < bottom):
            return False
        h = bottom - top
        w = right - left
        if min(h, w) == 0:
            return False
        if max(h, w) / min(h, w) > 100:
            return False
        return True

    def _maybe_resize_bbox(
        self, bbox_2d: list[float], image_width: int, image_height: int
    ) -> Optional[list[int]]:
        """Clamp, validate, and (if too small) recenter-expand the bbox.

        Returns int coordinates [x1, y1, x2, y2] inside the image, with both
        sides >= self.min_dimension. Returns None if no valid box can be made.
        """
        left = max(0.0, float(bbox_2d[0]))
        top = max(0.0, float(bbox_2d[1]))
        right = min(float(image_width), float(bbox_2d[2]))
        bottom = min(float(image_height), float(bbox_2d[3]))

        if not self._validate_bbox(left, top, right, bottom):
            return None

        h = bottom - top
        w = right - left

        if h < self.min_dimension or w < self.min_dimension:
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            min_side = min(h, w)
            if min_side == 0:
                return None
            ratio = self.min_dimension / min_side
            target_w = w * ratio
            target_h = h * ratio
            if target_w > image_width:
                target_h *= image_width / target_w
                target_w = image_width
            if target_h > image_height:
                target_w *= image_height / target_h
                target_h = image_height
            left = cx - target_w / 2.0
            top = cy - target_h / 2.0
            if left < 0:
                left = 0.0
            if top < 0:
                top = 0.0
            if left + target_w > image_width:
                left = image_width - target_w
            if top + target_h > image_height:
                top = image_height - target_h
            right = left + target_w
            bottom = top + target_h

        l, t, r, b = floor(left), floor(top), ceil(right), ceil(bottom)
        if not self._validate_bbox(l, t, r, b):
            return None
        if (b - t) < self.min_dimension or (r - l) < self.min_dimension:
            return None
        return [l, t, r, b]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        image_path = create_kwargs.get("image_path")

        entry: dict[str, Any] = {"image": None, "error": None, "image_path": image_path}
        if not image_path:
            entry["error"] = "tools_kwargs.image_zoom_in_tool.create_kwargs.image_path is missing"
        else:
            try:
                entry["image"] = Image.open(image_path).convert("RGB")
            except Exception as e:
                entry["error"] = f"failed to open image_path={image_path!r}: {e}"

        self._instance_dict[instance_id] = entry
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return (
                ToolResponse(text="Error: tool instance not found."),
                0.0,
                {"success": False},
            )

        if entry["error"]:
            return (
                ToolResponse(text=f"Error: {entry['error']}"),
                0.0,
                {"success": False},
            )

        bbox_2d = parameters.get("bbox_2d")
        if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
            return (
                ToolResponse(text="Error: bbox_2d must be a list of 4 numbers."),
                0.0,
                {"success": False},
            )
        try:
            bbox_floats = [float(v) for v in bbox_2d]
        except (TypeError, ValueError):
            return (
                ToolResponse(text="Error: bbox_2d entries must be numeric."),
                0.0,
                {"success": False},
            )

        image: Image.Image = entry["image"]
        sanitized = self._maybe_resize_bbox(bbox_floats, image.width, image.height)
        if sanitized is None:
            return (
                ToolResponse(
                    text=(
                        f"Error: bbox {bbox_2d} is invalid or smaller than the minimum "
                        f"size of {self.min_dimension}x{self.min_dimension} after clamping."
                    )
                ),
                0.0,
                {"success": False},
            )

        try:
            self._ensure_vq_model()
            cropped = image.crop(tuple(sanitized))
            token_str = encode_image(cropped, self._vq_model, target_area=self.target_area)
        except Exception as e:
            logger.warning(f"image_zoom_in_tool encoding failed: {e}")
            return (
                ToolResponse(text=f"Error: failed to encode cropped region: {e}"),
                0.0,
                {"success": False},
            )

        return (
            ToolResponse(text=token_str),
            0.0,
            {"success": True, "bbox": sanitized},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        entry = self._instance_dict.pop(instance_id, None)
        if entry is not None and entry.get("image") is not None:
            try:
                entry["image"].close()
            except Exception:
                pass
