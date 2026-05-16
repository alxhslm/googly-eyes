import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from common.image import (
    deserialize_image,
    get_image_from_bytes,
    put_image_into_buffer,
    serialize_image,
)


def _make_jpeg_image() -> Image.Image:
    buf = BytesIO()
    Image.new("RGB", (20, 20), color=(100, 150, 200)).save(buf, format="JPEG")
    buf.seek(0)
    return Image.open(buf)


def _make_png_image() -> Image.Image:
    buf = BytesIO()
    Image.new("RGB", (20, 20), color=(10, 20, 30)).save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


class TestGetImageFromBytes:
    def test_returns_pil_image(self):
        buf = BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = get_image_from_bytes(buf.getvalue())
        assert isinstance(result, Image.Image)

    def test_preserves_size(self):
        buf = BytesIO()
        Image.new("RGB", (30, 50)).save(buf, format="PNG")
        result = get_image_from_bytes(buf.getvalue())
        assert result.size == (30, 50)

    def test_preserves_mode(self):
        buf = BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = get_image_from_bytes(buf.getvalue())
        assert result.mode == "RGB"


class TestPutImageIntoBuffer:
    def test_returns_non_empty_buffer(self):
        img = _make_jpeg_image()
        buf = put_image_into_buffer(img)
        assert buf.read() != b""

    def test_buffer_is_valid_image(self):
        img = _make_png_image()
        buf = put_image_into_buffer(img)
        recovered = Image.open(buf)
        assert recovered.size == img.size

    def test_roundtrip_preserves_size(self):
        img = _make_jpeg_image()
        buf = put_image_into_buffer(img)
        recovered = Image.open(buf)
        assert recovered.size == img.size


class TestSerializeDeserialize:
    def test_serialize_returns_string(self):
        img = _make_jpeg_image()
        result = serialize_image(img)
        assert isinstance(result, str)

    def test_serialize_is_valid_base64(self):
        img = _make_jpeg_image()
        result = serialize_image(img)
        # Should not raise
        base64.b64decode(result)

    def test_roundtrip_preserves_size(self):
        img = _make_jpeg_image()
        recovered = deserialize_image(serialize_image(img))
        assert recovered.size == img.size

    def test_roundtrip_preserves_mode(self):
        img = _make_png_image()
        recovered = deserialize_image(serialize_image(img))
        assert recovered.mode == img.mode

    def test_roundtrip_pixel_values_close(self):
        img = _make_png_image()
        recovered = deserialize_image(serialize_image(img))
        np.testing.assert_array_equal(np.array(recovered), np.array(img))

    def test_deserialize_invalid_base64_raises(self):
        with pytest.raises(Exception):
            deserialize_image("not-valid-base64!!!")
