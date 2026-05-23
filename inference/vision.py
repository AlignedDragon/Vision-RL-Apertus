"""IBQ image encoding for Apertus.

Converts PIL images into discrete token strings that Apertus can consume.
The pipeline: PIL Image → resize → normalize → IBQ encode → format as token string.

Apertus encodes images as discrete tokens (131,072 codebook entries) via the
Emu3.5 VisionTokenizer. The resulting token string is inserted directly into
the text prompt, replacing the <|image|> placeholder.

Token convention (Apertus, NOT Emu3.5):
    <|img_start|>      token 131073
    <|img_end|>        token 131074
    <|img_token_start|> token 131075
    <|img_end_of_row|>  token 131076
    <|visual token N|>  tokens 131272+  (no zero-padding)

Requires: ~/Emu3.5/src on PYTHONPATH (for vision_tokenizer.build_vision_tokenizer)
"""

import math

import numpy as np
import torch
from PIL import Image



# Apertus special token strings (different from Emu3.5's naming)
IMG_START = "<|img_start|>"
IMG_END = "<|img_end|>"
IMG_TOKEN_START = "<|img_token_start|>"
IMG_END_OF_ROW = "<|img_end_of_row|>"


def load_vq_model(path: str, device: str = "cuda:0"):
    """Load the Emu3.5 IBQ vision tokenizer.

    The checkpoint uses config.yaml + model.ckpt format, which requires
    Emu3.5's build_vision_tokenizer loader (no HuggingFace AutoModel alternative).

    Args:
        path: Path to the Emu3.5-VisionTokenizer directory.
        device: Device to load the model on.

    Returns:
        IBQ model ready for encode().
    """
    from vision_tokenizer import build_vision_tokenizer

    return build_vision_tokenizer("ibq", path, device=device)


def smart_resize(
    image: Image.Image,
    min_patches: int = 16,
    max_patches: int = 2048,
    ds_factor: int = 16,
) -> Image.Image:
    """Resize image so its IBQ token count lies in [min_patches, max_patches].

    Each "patch" is one IBQ token (a ds_factor x ds_factor pixel block, since
    the encoder has stride ds_factor). Pixel dimensions are snapped to multiples
    of ds_factor so the encoder produces an exact (h/ds_factor, w/ds_factor)
    token grid; if the resulting grid would have too many or too few tokens,
    the image is rescaled (aspect-preserving) before snapping.

    Args:
        image: Input PIL image.
        min_patches: Minimum total tokens. Image is upscaled if it would emit fewer.
        max_patches: Maximum total tokens. Image is downscaled if it would emit more.
        ds_factor: IBQ encoder stride.
    """
    w, h = image.size
    if h < ds_factor or w < ds_factor:
        raise ValueError(f"height:{h} or width:{w} must be >= ds_factor:{ds_factor}")

    new_h = round(h / ds_factor) * ds_factor
    new_w = round(w / ds_factor) * ds_factor
    patches = (new_h // ds_factor) * (new_w // ds_factor)

    if patches > max_patches:
        scale = math.sqrt(patches / max_patches)
        new_h = max(ds_factor, math.floor(h / scale / ds_factor) * ds_factor)
        new_w = max(ds_factor, math.floor(w / scale / ds_factor) * ds_factor)
    elif patches < min_patches:
        scale = math.sqrt(min_patches / patches)
        new_h = math.ceil(h * scale / ds_factor) * ds_factor
        new_w = math.ceil(w * scale / ds_factor) * ds_factor

    return image.resize((new_w, new_h), Image.BICUBIC)

def format_image_tokens(token_grid: torch.Tensor) -> str:
    """Format IBQ codebook indices into Apertus image token string.

    Args:
        token_grid: 2D tensor of shape (H_tokens, W_tokens) with codebook indices.

    Returns:
        String like: <|img_start|>32*32<|img_token_start|><|visual token 0|>...<|img_end|>
    """
    h, w = token_grid.shape
    rows = []
    for row_idx in range(h):
        row = "".join(f"<|visual token {int(token_grid[row_idx, col])}|>" for col in range(w))
        rows.append(row)
    token_str = IMG_END_OF_ROW.join(rows)
    return f"{IMG_START}{h}*{w}{IMG_TOKEN_START}{token_str}{IMG_END}"


@torch.no_grad()
def encode_image(
    image: Image.Image,
    vq_model,
    min_patches: int = 16,
    max_patches: int = 2048,
) -> str:
    """Encode a PIL image to an Apertus image token string.

    Full pipeline:
    1. Convert to RGB
    2. Resize so token count is in [min_patches, max_patches] (aspect-preserving, dims divisible by 16)
    3. Normalize pixels to [-1, 1]
    4. Run IBQ encoder → codebook indices
    5. Reshape to 2D grid (h/16, w/16)
    6. Format as Apertus token string

    Args:
        image: Input PIL image.
        vq_model: Loaded IBQ vision tokenizer.
        min_patches: Minimum number of IBQ tokens.
        max_patches: Maximum number of IBQ tokens.

    Returns:
        Formatted image token string ready for prompt insertion.
    """
    image = image.convert("RGB")
    image = smart_resize(image, min_patches=min_patches, max_patches=max_patches)
    w, h = image.size

    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype

    # Normalize to [-1, 1] and convert to tensor
    pixel_values = torch.tensor(np.array(image) / 127.5 - 1.0)
    pixel_values = pixel_values.to(device, dtype).permute(2, 0, 1)  # HWC → CHW

    # IBQ encode: returns (quant, diff, indices) — we need indices
    _, _, indices = vq_model.encode(pixel_values[None])  # add batch dim
    token_grid = indices[-1].view(h // 16, w // 16)

    return format_image_tokens(token_grid)
