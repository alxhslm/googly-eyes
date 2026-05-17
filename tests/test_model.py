"""
Integration tests against the real TFLite model using a known test image.

These tests require tflite_runtime and the compiled model, so they are skipped
in the unit-test CI environment where those are unavailable.

Run manually (inside the server container) with:
    poetry run pytest tests/test_model.py -v
"""

import os

import numpy as np
import pytest
from PIL import Image

TFLITE_AVAILABLE = True
try:
    import tflite_runtime.interpreter  # noqa: F401
except ImportError:
    TFLITE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TFLITE_AVAILABLE, reason="tflite_runtime not installed")

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "group_of_people.jpg")

# Gold-standard predictions captured from the current model on 2026-05-17.
# Each entry: (right_eye, left_eye) in image pixel coordinates.
EXPECTED_FACES = [
    {
        "score": 0.9991,
        "bbox": [1010, 2482, 1230, 2771],
        "right_eye": [1072.2521, 2585.7686],
        "left_eye": [1175.6461, 2593.8928],
    },
    {
        "score": 0.9987,
        "bbox": [1728, 2540, 1957, 2824],
        "right_eye": [1770.6294, 2665.843],
        "left_eye": [1875.1309, 2633.7722],
    },
    {
        "score": 0.9985,
        "bbox": [463, 2421, 671, 2713],
        "right_eye": [524.5632, 2518.8127],
        "left_eye": [621.5748, 2537.2546],
    },
    {
        "score": 0.9984,
        "bbox": [2285, 2376, 2505, 2678],
        "right_eye": [2327.4814, 2513.7825],
        "left_eye": [2432.2595, 2486.8306],
    },
    {
        "score": 0.9979,
        "bbox": [2886, 2458, 3129, 2768],
        "right_eye": [2935.7007, 2587.1938],
        "left_eye": [3048.3145, 2563.0161],
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
