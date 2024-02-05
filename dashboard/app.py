import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

from common.drawing import plot_circle
from common.face import Face

URL = os.environ.get("SERVER_URL", "http://localhost:8000")


def _save_image(image: Image) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    buffer.seek(0)
    return buffer


def _download_image(r: requests.Response) -> Image:
    buffer = BytesIO()
    buffer.write(r.content)
    buffer.seek(0)
    return Image.open(buffer)


def _draw_face(image: Image, face: Face) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(face.bounding_box, width=10, outline=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["right_eye"], radius=10, fill=(255, 0, 0, 0))
    plot_circle(draw, face.landmarks["left_eye"], radius=10, fill=(255, 0, 0, 0))


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


if googly_eyes_enabled:
    files = {"image": _save_image(image)}
    settings = {"eye_size": eye_size, "pupil_size_range": pupil_size_range}
    response = requests.post(f"{URL}/googly_eyes", files=files, data=settings)
    response.raise_for_status()
    image = _download_image(response)


if highlight_faces:
    files = {"image": _save_image(image)}
    response = requests.post(f"{URL}/identify_faces", files=files)
    response.raise_for_status()
    for face in response.json():
        _draw_face(image, Face(**face))

st.image(image)

buffer = _save_image(image)
filename, ext = uploaded_file.name.split(".")
st.download_button(
    label="Download Image",
    data=buffer.getvalue(),
    file_name=f"{filename}_googly_eyes.{ext}",
    mime=uploaded_file.type,
)
