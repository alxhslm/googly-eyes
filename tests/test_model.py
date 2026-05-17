"""
Integration tests against the real ONNX model using a known test image.

These tests require the compiled retinaface.onnx model, so they are skipped
in the unit-test CI environment where it is unavailable.

Run manually (inside the server container) with:
    poetry run pytest tests/test_model.py -v
"""

import os

import numpy as np
import pytest
from PIL import Image

ONNX_MODEL_AVAILABLE = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "retinaface", "retinaface.onnx"))

pytestmark = pytest.mark.skipif(not ONNX_MODEL_AVAILABLE, reason="retinaface.onnx not present")

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "group_of_people.jpg")

# Gold-standard predictions captured from the ONNX model on 2026-05-17.
# Each entry: (right_eye, left_eye) in image pixel coordinates.
EXPECTED_FACES = [
    {
        "score": 0.9990,
        "bbox": [1007, 2483, 1234, 2770],
        "right_eye": [1070.376, 2584.4277],
        "left_eye": [1176.9109, 2592.7446],
    },
    {
        "score": 0.9988,
        "bbox": [454, 2426, 670, 2698],
        "right_eye": [527.3534, 2519.5051],
        "left_eye": [627.3571, 2534.4487],
    },
    {
        "score": 0.9978,
        "bbox": [2278, 2395, 2496, 2661],
        "right_eye": [2317.5776, 2500.7954],
        "left_eye": [2419.0125, 2491.7854],
    },
    {
        "score": 0.9974,
        "bbox": [2884, 2459, 3133, 2752],
        "right_eye": [2929.3679, 2579.1292],
        "left_eye": [3042.5022, 2564.491],
    },
    {
        "score": 0.9963,
        "bbox": [1724, 2546, 1949, 2816],
        "right_eye": [1776.3665, 2655.4387],
        "left_eye": [1869.9397, 2643.5427],
    },
]

EYE_TOLERANCE_PX = 5.0


def _closest_face(detected, expected):
    """Return the detected face whose right_eye is nearest to expected right_eye."""
    ex, ey = expected["right_eye"]
    dists = [
        np.linalg.norm(np.array(d.landmarks["right_eye"]) - np.array([ex, ey]))
        for d in detected
    ]
    return detected[int(np.argmin(dists))]


@pytest.fixture(scope="module")
def detected_faces():
    from common.googlify import detect_faces

    img = np.array(Image.open(TEST_IMAGE))
    return detect_faces(img)


class TestModelDetection:
    def test_detects_five_faces(self, detected_faces):
        assert len(detected_faces) == 5

    def test_all_scores_above_threshold(self, detected_faces):
        for face in detected_faces:
            assert face.score >= 0.9

    @pytest.mark.parametrize("expected", EXPECTED_FACES)
    def test_eye_positions(self, detected_faces, expected):
        face = _closest_face(detected_faces, expected)

        right_err = np.linalg.norm(
            np.array(face.landmarks["right_eye"]) - np.array(expected["right_eye"])
        )
        left_err = np.linalg.norm(
            np.array(face.landmarks["left_eye"]) - np.array(expected["left_eye"])
        )

        assert right_err < EYE_TOLERANCE_PX, (
            f"right_eye off by {right_err:.1f}px (got {face.landmarks['right_eye']}, "
            f"expected {expected['right_eye']})"
        )
        assert left_err < EYE_TOLERANCE_PX, (
            f"left_eye off by {left_err:.1f}px (got {face.landmarks['left_eye']}, "
            f"expected {expected['left_eye']})"
        )

    @pytest.mark.parametrize("expected", EXPECTED_FACES)
    def test_bbox_positions(self, detected_faces, expected):
        face = _closest_face(detected_faces, expected)
        for got, exp in zip(face.bounding_box, expected["bbox"]):
            assert abs(got - exp) < 10, f"bbox mismatch: got {face.bounding_box}, expected {expected['bbox']}"
