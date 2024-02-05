from io import BytesIO

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

from common.drawing import add_googly_eyes
from common.face import Face
from retinaface import RetinaFace
from retinaface.model import build_model

app = Flask(__name__)

MODEL = build_model()


def model(X: np.ndarray) -> list[np.ndarray]:
    return [o.numpy() for o in MODEL(X)]


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
