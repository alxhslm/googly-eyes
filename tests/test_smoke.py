"""
End-to-end smoke test for the Lambda handler.

The tflite model is mocked (via conftest.py), and face detection is patched to return
controlled outputs so we can exercise the full serialise → detect → draw → serialise
pipeline without the actual model.
"""

import json
import sys
import os
from unittest.mock import patch
from io import BytesIO

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda"))


def _make_payload(width: int = 100, height: int = 100, mode: str = "RGB") -> dict:
    buf = BytesIO()
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    color = (200, 200, 200, 128) if mode == "RGBA" else (200, 200, 200)
    Image.new(mode, (width, height), color=color).save(buf, format=fmt)
    buf.seek(0)
    import base64

    return {"image": base64.b64encode(buf.read()).decode()}


def _fake_face():
    from common.face import Face

    return Face(
        score=0.99,
        bounding_box=[20.0, 20.0, 80.0, 80.0],
        landmarks={
            "left_eye": [35.0, 40.0],
            "right_eye": [65.0, 40.0],
            "nose": [50.0, 55.0],
            "mouth_right": [35.0, 65.0],
            "mouth_left": [65.0, 65.0],
        },
    )


class TestLambdaSmoke:
    def test_no_faces_returns_valid_image(self):
        import predict

        with patch("common.googlify.detect_faces", return_value=[]):
            result = predict.lambda_handler({"body": json.dumps(_make_payload())}, None)

        assert "image" in result
        assert "faces" in result
        assert result["faces"] == []
        # Returned image must be a valid decodable image
        from common.image import deserialize_image

        img = deserialize_image(result["image"])
        assert img.size == (100, 100)

    def test_with_face_returns_modified_image_and_face_data(self):
        import predict

        with patch("common.googlify.detect_faces", return_value=[_fake_face()]):
            result = predict.lambda_handler({"body": json.dumps(_make_payload())}, None)

        assert "image" in result
        assert len(result["faces"]) == 1
        assert result["faces"][0]["score"] == 0.99
        # Returned image must still be valid and same dimensions
        from common.image import deserialize_image

        img = deserialize_image(result["image"])
        assert img.size == (100, 100)

    def test_custom_eye_size_accepted(self):
        import predict

        payload = {**_make_payload(), "eye_size": 0.3}
        with patch("common.googlify.detect_faces", return_value=[_fake_face()]):
            result = predict.lambda_handler({"body": json.dumps(payload)}, None)

        assert "image" in result

    def test_rgba_input_preserves_alpha_in_output(self):
        import predict

        with patch("common.googlify.detect_faces", return_value=[]):
            result = predict.lambda_handler({"body": json.dumps(_make_payload(mode="RGBA"))}, None)

        from common.image import deserialize_image

        assert deserialize_image(result["image"]).mode == "RGBA"

    def test_rgb_input_does_not_gain_alpha(self):
        import predict

        with patch("common.googlify.detect_faces", return_value=[]):
            result = predict.lambda_handler({"body": json.dumps(_make_payload(mode="RGB"))}, None)

        from common.image import deserialize_image

        assert deserialize_image(result["image"]).mode == "RGB"

    def test_response_image_differs_from_input_when_face_detected(self):
        import predict

        payload = _make_payload()
        with patch("common.googlify.detect_faces", return_value=[_fake_face()]):
            result = predict.lambda_handler({"body": json.dumps(payload)}, None)

        # Drawing googly eyes should change at least some pixels
        from common.image import deserialize_image

        original = np.array(deserialize_image(payload["image"]))
        modified = np.array(deserialize_image(result["image"]))
        assert not np.array_equal(original, modified)
