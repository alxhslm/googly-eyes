import typing as t

import numpy as np
import streamlit as st
from attr import dataclass
from keras.models import Model
from PIL import Image, ImageDraw
from retinaface import RetinaFace


@st.cache_resource
def model() -> Model:
    return RetinaFace.build_model()


@dataclass
class Face:
    score: float
    bounding_box: list[float]
    landmarks: dict[str, list[float]]


def identify_faces(image: Image) -> list[Face]:
    return [
        Face(score=face["score"], bounding_box=face["facial_area"], landmarks=face["landmarks"])
        for face in RetinaFace.detect_faces(np.array(image), model=model()).values()
    ]


def plot_circle(draw: ImageDraw, xy: t.Sequence[float], radius: float, **kwargs: t.Any) -> None:
    draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), **kwargs)


def draw_face(image: Image, face: Face) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(face.bounding_box, width=10, outline=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["right_eye"], radius=10, fill=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["left_eye"], radius=10, fill=(255, 0, 0, 0))


st.title(":eyes: Goofy eyes")

uploaded_file = st.file_uploader("Upload photo", type=["png", "jpg"])
if uploaded_file is None:
    st.error("No image uploaded")
    st.stop()

image = Image.open(uploaded_file)

for face in identify_faces(image):
    draw_face(image, face)

st.image(image)
