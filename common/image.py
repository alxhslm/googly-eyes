from io import BytesIO

from PIL import Image


def put_image_into_buffer(image: Image) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    buffer.seek(0)
    return buffer


def get_image_from_bytes(content: bytes) -> Image:
    buffer = BytesIO()
    buffer.write(content)
    buffer.seek(0)
    return Image.open(buffer)
