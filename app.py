import typing as t
from io import BytesIO

import numpy as np
import streamlit as st
from attr import dataclass
from keras.models import Model
from PIL import Image, ImageDraw

from retinaface import RetinaFace
from retinaface.model import build_model


@st.cache_resource
def model() -> Model:
    return build_model()


@dataclass
class Face:
    score: float
    bounding_box: list[float]
    landmarks: dict[str, list[float]]


@st.cache_data
def identify_faces(image: np.ndarray) -> list[Face]:
    return [
        Face(score=face["score"], bounding_box=face["facial_area"], landmarks=face["landmarks"])
        for face in RetinaFace.detect_faces(image, model=model())
    ]


def plot_circle(draw: ImageDraw, xy: t.Sequence[float], radius: float, **kwargs: t.Any) -> None:
    draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), **kwargs)


def draw_face(image: Image, face: Face) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(face.bounding_box, width=10, outline=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["right_eye"], radius=10, fill=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["left_eye"], radius=10, fill=(255, 0, 0, 0))


def add_googly_eyes(image: Image, face: Face, eye_size: float, pupil_size_range: tuple[float, float]) -> None:
    draw = ImageDraw.Draw(image)
    radius = 0.5 * float(np.linalg.norm(np.array(face.landmarks["right_eye"]) - np.array(face.landmarks["left_eye"])))

    def plot_googly_eye(eye: list[float]) -> None:
        pupil_size = eye_size * np.random.uniform(pupil_size_range[0], pupil_size_range[1])
        plot_circle(draw, eye, radius=eye_size * radius, fill=(255, 255, 255, 0), outline=(0, 0, 0, 0))
        orientation = np.random.uniform(0, np.pi)
        plot_circle(
            draw,
            np.array(eye) + (eye_size - pupil_size) * radius * np.array([np.sin(orientation), np.cos(orientation)]),
            radius=pupil_size * radius,
            fill=(0, 0, 0, 0),
        )

    plot_googly_eye(face.landmarks["right_eye"])
    plot_googly_eye(face.landmarks["left_eye"])


st.title(":eyes: Googly eyes")

uploaded_file = st.file_uploader("Upload photo", type=["png", "jpg"])
if uploaded_file is None:
    st.error("No image uploaded")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    googly_eyes_enabled = st.checkbox("Add googly eyes", value=True)
with col2:
    highlight_faces = st.checkbox("Highlight identified faces")

with st.form("googly_eye_options"):
    eye_size = st.slider("Eye size", min_value=0.0, max_value=1.0, value=0.5, disabled=not googly_eyes_enabled)
    pupil_size_range = st.slider(
        "Pupil size", min_value=0.0, max_value=1.0, value=[0.4, 0.6], disabled=not googly_eyes_enabled
    )
    st.form_submit_button("Update")

image = Image.open(uploaded_file)

for face in identify_faces(np.array(image)):
    if highlight_faces:
        draw_face(image, face)
    if googly_eyes_enabled:
        add_googly_eyes(image, face, eye_size, pupil_size_range)

st.image(image)

buf = BytesIO()
image.save(buf, format=image.format)
filename, ext = uploaded_file.name.split(".")
st.download_button(
    label="Download Image", data=buf.getvalue(), file_name=f"{filename}_googly_eyes.{ext}", mime=uploaded_file.type
)
