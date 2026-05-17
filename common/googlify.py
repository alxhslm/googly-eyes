import os
import typing as t
from dataclasses import asdict

import numpy as np
import onnxruntime

from common.drawing import add_googly_eyes
from common.face import Face
from common.image import deserialize_image, serialize_image
from retinaface import detect

_session = onnxruntime.InferenceSession(
    os.path.join(os.path.dirname(detect.__file__), "retinaface.onnx")
)


def _model(X: np.ndarray) -> list[np.ndarray]:
    return _session.run(["bbox", "cls", "ldm"], {"input": X})


def detect_faces(image: np.ndarray) -> list[Face]:
    return [
        Face(
            score=face["score"],
            bounding_box=face["facial_area"],
            landmarks=face["landmarks"],
        )
        for face in detect.detect_faces(image, model=_model)
    ]


def googlify(data: dict[str, t.Any]) -> dict[str, t.Any]:
    image = deserialize_image(data["image"])
    eye_size = data.get("eye_size", 0.5)
    pupil_size_range_raw = data.get("pupil_size_range", None)
    pupil_size_range = tuple(pupil_size_range_raw) if pupil_size_range_raw else (0.4, 0.6)
    faces = detect_faces(np.array(image))
    for face in faces:
        add_googly_eyes(image, face, eye_size=eye_size, pupil_size_range=pupil_size_range)
    return {
        "image": serialize_image(image),
        "faces": [asdict(face) for face in faces],
    }
