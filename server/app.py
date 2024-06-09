import os
from dataclasses import asdict

import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request

from common.drawing import add_googly_eyes
from common.face import Face
from retinaface import detect
from common.image import serialize_image, deserialize_image

app = Flask(__name__)

interpreter = tflite.Interpreter(model_path=os.path.join(os.path.dirname(detect.__file__), "retinaface.tflite"))
# We need this list of output names to ensure that the outputs from the tflite model are in the same order as the
# original Keras model
output_names = [
    "tf.compat.v1.transpose_1",
    "face_rpn_bbox_pred_stride32",
    "face_rpn_landmark_pred_stride32",
    "tf.compat.v1.transpose_3",
    "face_rpn_bbox_pred_stride16",
    "face_rpn_landmark_pred_stride16",
    "tf.compat.v1.transpose_5",
    "face_rpn_bbox_pred_stride8",
    "face_rpn_landmark_pred_stride8",
]


def _model(X: np.ndarray) -> list[np.ndarray]:
    model = interpreter.get_signature_runner("serving_default")
    result = model(data=X)
    return [result[o] for o in output_names]


def detect_faces(image: np.ndarray) -> list[Face]:
    return [
        Face(
            score=face["score"],
            bounding_box=face["facial_area"],
            landmarks=face["landmarks"],
        )
        for face in detect.detect_faces(image, model=_model)
    ]


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    data = request.get_json()
    image = deserialize_image(data["image"])
    eye_size = data.get("eye_size", 0.5)
    pupil_size_range = data.get("pupil_size_range", [])
    pupil_size_range = tuple(pupil_size_range) if pupil_size_range else (0.4, 0.6)
    for face in detect_faces(np.array(image)):
        add_googly_eyes(image, face, eye_size=eye_size, pupil_size_range=pupil_size_range)
    return jsonify(
        {
            "image": serialize_image(image),
            "faces": [asdict(face) for face in detect_faces(np.array(image))],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
