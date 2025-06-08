from io import BytesIO
import base64
from PIL import Image


def put_image_into_buffer(image: Image.Image) -> BytesIO:
    buffer = BytesIO()
    exif = image.info.get("exif", None)
    kwargs = {"exif": exif} if exif else {}
    image.save(buffer, format=image.format, **kwargs)
    buffer.seek(0)
    return buffer


def get_image_from_bytes(content: bytes) -> Image.Image:
    buffer = BytesIO()
    buffer.write(content)
    buffer.seek(0)
    return Image.open(buffer)


def serialize_image(image: Image.Image) -> str:
    return base64.b64encode(put_image_into_buffer(image).read()).decode("utf-8")


def deserialize_image(data: str) -> Image.Image:
    return get_image_from_bytes(base64.b64decode(data.encode("utf-8")))
