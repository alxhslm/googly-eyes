from io import BytesIO
import base64
from typing import Any
from PIL import Image


def put_image_into_buffer(image: Image.Image, **kwargs: Any) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image.format, **kwargs)
    buffer.seek(0)
    return buffer


def get_image_from_bytes(content: bytes) -> Image.Image:
    buffer = BytesIO()
    buffer.write(content)
    buffer.seek(0)
    return Image.open(buffer)


def serialize_image(image: Image.Image, **kwargs: Any) -> str:
    return base64.b64encode(put_image_into_buffer(image, **kwargs).read()).decode("utf-8")


def serialize_exif(exif) -> str:
    return base64.b64encode(exif).decode("utf-8")


def deserialize_image(data: str) -> Image.Image:
    return get_image_from_bytes(base64.b64decode(data.encode("utf-8")))


def deserialize_exif(data: Any) -> str | None:
    if not data:
        return None
    return base64.b64decode(data.encode("utf-8"))
