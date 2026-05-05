"""Line tool — draw annotation lines on image."""

from PIL import Image, ImageDraw


def draw_line(image: Image.Image, points: list[list[int]], color: str = "red", width: int = 3) -> Image.Image:
    """Draw lines connecting the given points on the image.

    points: list of [x, y] coordinates. Lines connect consecutive points.
    Returns a new image with the lines drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    if len(points) < 2:
        return img

    flat_points = [(p[0], p[1]) for p in points]
    draw.line(flat_points, fill=color, width=width)

    return img
