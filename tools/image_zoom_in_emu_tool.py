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

import torch
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.vision import encode_image, load_vq_model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VQ_MODELS: dict[tuple[str, str], Any] = {}
_VQ_MODELS_LOCK = threading.Lock()


class ImageZoomInEmuTool(BaseTool):
    """Crop a bbox from a source image and return its IBQ token string."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.vq_model_path: str = config["vq_model_path"]
        self.vq_device: str = config.get("vq_device", "cuda:0")
        self.min_dimension: int = int(config.get("min_dimension", 16))

        self._instance_dict: dict[str, dict[str, Any]] = {}

    def _ensure_vq_model(self):
        device = self._resolve_vq_device()
        key = (self.vq_model_path, device)
        vq_model = _VQ_MODELS.get(key)
        if vq_model is None:
            with _VQ_MODELS_LOCK:
                vq_model = _VQ_MODELS.get(key)
                if vq_model is None:
                    msg = (
                        f"image_zoom_in_tool loading IBQ vision tokenizer on {device}; "
                        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
                    )
                    print(msg, flush=True)
                    logger.warning(msg)
                    vq_model = load_vq_model(self.vq_model_path, device=device)
                    _VQ_MODELS[key] = vq_model
        return vq_model

    def _resolve_vq_device(self) -> str:
        if self.vq_device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            raise RuntimeError(
                "image_zoom_in_tool requires a GPU, but CUDA is not visible in this Ray worker. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        if self.vq_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Configured vq_device={self.vq_device}, but CUDA is not visible in this Ray worker. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        if self.vq_device == "cpu":
            raise RuntimeError("image_zoom_in_tool is configured for CPU, but this run requires GPU.")
        return self.vq_device

    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        if not (left < right and top < bottom):
            return False
        h = bottom - top
        w = right - left
        if h < self.min_dimension or w < self.min_dimension:
            return False
        if max(h, w) / min(h, w) > 100:
            return False
        return True

    def _sanitize_bbox(
        self, bbox_2d: list[float], image_width: int, image_height: int
    ) -> Optional[list[int]]:
        """Clamp the bbox to image bounds, validate, and snap to int coords.

        Returns int [x1, y1, x2, y2] inside the image, with both sides
        >= self.min_dimension, or None if the bbox is invalid.
        """
        left = max(0.0, float(bbox_2d[0]))
        top = max(0.0, float(bbox_2d[1]))
        right = min(float(image_width), float(bbox_2d[2]))
        bottom = min(float(image_height), float(bbox_2d[3]))

        if not self._validate_bbox(left, top, right, bottom):
            return None

        return [floor(left), floor(top), ceil(right), ceil(bottom)]

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
        sanitized = self._sanitize_bbox(bbox_floats, image.width, image.height)
        if sanitized is None:
            return (
                ToolResponse(
                    text=(
                        f"Error: bbox {bbox_2d} is invalid or has a side smaller "
                        f"than {self.min_dimension}px after clamping to the image."
                    )
                ),
                0.0,
                {"success": False},
            )

        try:
            vq_model = self._ensure_vq_model()
            cropped = image.crop(tuple(sanitized))
            token_str = encode_image(cropped, vq_model)
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
