from io import BytesIO

import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request, send_file
from PIL import Image

from common.drawing import add_googly_eyes
from common.face import Face
from retinaface import RetinaFace

app = Flask(__name__)

interpreter = tflite.Interpreter(model_path="model.tflite")
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


def model(X: np.ndarray) -> list[np.ndarray]:
    model = interpreter.get_signature_runner("serving_default")
    result = model(data=X)
    return [result[o] for o in output_names]


def _detect_faces(image: np.ndarray) -> list[Face]:
    return [
        Face(
            score=face["score"],
            bounding_box=face["facial_area"],
            landmarks=face["landmarks"],
        )
        for face in RetinaFace.detect_faces(image, model=model)
    ]


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    image = Image.open(request.files["image"])
    for face in _detect_faces(np.array(image)):
        add_googly_eyes(
            image,
            face,
            request.form.get("eye_size", type=float),
            request.form.getlist("pupil_size_range", float),
        )

    buffer = BytesIO()
    image.save(buffer, format=image.format)
    buffer.seek(0)

    return send_file(buffer, mimetype=request.files["image"].mimetype)


@app.route("/identify_faces", methods=["POST"])
def identify_faces():
    image = Image.open(request.files["image"])
    return jsonify([face.asdict() for face in _detect_faces(np.array(image))])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8502")
