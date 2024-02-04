import os
import typing as t
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw

URL = os.environ["SERVER_URL"]
from dataclasses import dataclass


@dataclass
class Face:
    score: float
    bounding_box: list[float]
    landmarks: dict[str, list[float]]


def plot_circle(
    draw: ImageDraw, xy: t.Sequence[float], radius: float, **kwargs: t.Any
) -> None:
    draw.ellipse(
        (xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), **kwargs
    )


def draw_face(image: Image, face: Face) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(face.bounding_box, width=10, outline=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["right_eye"], radius=10, fill=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["left_eye"], radius=10, fill=(255, 0, 0, 0))


def add_googly_eyes(
    image: Image, face: Face, eye_size: float, pupil_size_range: tuple[float, float]
) -> None:
    draw = ImageDraw.Draw(image)
    radius = 0.5 * float(
        np.linalg.norm(
            np.array(face.landmarks["right_eye"]) - np.array(face.landmarks["left_eye"])
        )
    )

    def plot_googly_eye(eye: list[float]) -> None:
        pupil_size = eye_size * np.random.uniform(
            pupil_size_range[0], pupil_size_range[1]
        )
        plot_circle(
            draw,
            eye,
            radius=eye_size * radius,
            fill=(255, 255, 255, 0),
            outline=(0, 0, 0, 0),
        )
        orientation = np.random.uniform(0, np.pi)
        plot_circle(
            draw,
            np.array(eye)
            + (eye_size - pupil_size)
            * radius
            * np.array([np.sin(orientation), np.cos(orientation)]),
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
    eye_size = st.slider(
        "Eye size",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        disabled=not googly_eyes_enabled,
    )
    pupil_size_range = st.slider(
        "Pupil size",
        min_value=0.0,
        max_value=1.0,
        value=[0.4, 0.6],
        disabled=not googly_eyes_enabled,
    )
    st.form_submit_button("Update")

image = Image.open(uploaded_file)


def save_image(image: Image) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    buffer.seek(0)
    return buffer


def download_image(r: requests.Response) -> Image:
    buffer = BytesIO()
    buffer.write(r.content)
    buffer.seek(0)
    return Image.open(buffer)


if googly_eyes_enabled:
    files = {"image": save_image(image)}
    settings = {"eye_size": eye_size, "pupil_size_range": pupil_size_range}
    response = requests.post(f"{URL}/googly_eyes", files=files, data=settings)
    response.raise_for_status()
    image = download_image(response)


if highlight_faces:
    files = {"image": save_image(image)}
    response = requests.post(f"{URL}/identify_faces", files=files)
    response.raise_for_status()
    for face in response.json():
        draw_face(image, Face(**face))

st.image(image)

buffer = save_image(image)
filename, ext = uploaded_file.name.split(".")
st.download_button(
    label="Download Image",
    data=buffer.getvalue(),
    file_name=f"{filename}_googly_eyes.{ext}",
    mime=uploaded_file.type,
)
