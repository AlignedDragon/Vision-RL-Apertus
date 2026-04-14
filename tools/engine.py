"""Tool execution engine — dispatches tool calls and returns results."""

from PIL import Image

from tools.ocr_tool import ocr
from tools.grounding_tool import grounding
from tools.crop_zoom_tool import crop_zoom_in
from tools.counting_tool import counting
from tools.calculate_tool import calculate
from tools.line_tool import draw_line


def execute_tool(tool_name: str, args: dict, image: Image.Image) -> dict:
    """Execute a tool call and return the result.

    Returns:
        {"text": str|None, "image": Image|None}
        - text tools (OCR, Grounding, Counting, Calculate) return text
        - image tools (CropZoomIn, Line) return a new image
    """
    name = tool_name.lower()

    if name == "ocr":
        text = ocr(image, bbox=args.get("bbox"))
        return {"text": text, "image": None}

    elif name == "grounding":
        boxes = grounding(image, description=args["description"])
        text = str(boxes) if boxes else "(no objects found)"
        return {"text": text, "image": None}

    elif name == "cropzoomin":
        new_img = crop_zoom_in(image, bbox=args["bbox"], ratio=args.get("ratio", 2.0))
        return {"text": None, "image": new_img}

    elif name == "counting":
        count = counting(image, description=args["description"])
        return {"text": str(count), "image": None}

    elif name == "calculate":
        result = calculate(expression=args["expression"])
        return {"text": result, "image": None}

    elif name == "line":
        new_img = draw_line(image, points=args["points"],
                            color=args.get("color", "red"),
                            width=args.get("width", 3))
        return {"text": None, "image": new_img}

    else:
        return {"text": f"Unknown tool: {tool_name}", "image": None}


TOOL_DESCRIPTIONS = """Available tools:
- OCR(bbox?) — Read text from the image. Optional bbox=[x1,y1,x2,y2] to read a specific region.
- Grounding(description) — Find objects matching a text description. Returns bounding boxes [[x1,y1,x2,y2], ...].
- CropZoomIn(bbox, ratio?) — Crop and zoom into a region. bbox=[x1,y1,x2,y2], ratio defaults to 2.0.
- Counting(description) — Count objects matching a description.
- Calculate(expression) — Evaluate an arithmetic expression (e.g., "3 * 4 + 2").
- Line(points) — Draw lines connecting points [[x1,y1], [x2,y2], ...] on the image.

To use a tool, output exactly:
TOOL: {"name": "<tool_name>", "args": {<arguments>}}

Example:
TOOL: {"name": "OCR", "args": {"bbox": [100, 50, 400, 150]}}"""
