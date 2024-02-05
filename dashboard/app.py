import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

from common.drawing import plot_circle
from common.face import Face

URL = os.environ.get("SERVER_URL", "http://localhost:8000")


def _put_image_in_buffer(image: Image) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    buffer.seek(0)
    return buffer


def _get_image_from_bytes(content: bytes) -> Image:
    buffer = BytesIO()
    buffer.write(content)
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
    googly_eyes_enabled = st.checkbox(
        "Add googly eyes", value=True, help="Put google eyes on each face in the image"
    )
with col2:
    highlight_faces = st.checkbox(
        "Highlight identified faces",
        help="Adding bounding boxes around identifed faces",
    )

with st.form("googly_eye_options"):
    eye_size = st.slider(
        "Eye size",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        disabled=not googly_eyes_enabled,
        help="Eye radius relative to eye-to-eye distance",
    )
    pupil_size_range = st.slider(
        "Pupil size",
        min_value=0.0,
        max_value=1.0,
        value=[0.4, 0.6],
        disabled=not googly_eyes_enabled,
        help="Pupil radius relative to eye radius",
    )
    st.form_submit_button("Update")

image = Image.open(uploaded_file)


if googly_eyes_enabled:
    files = {"image": _put_image_in_buffer(image)}
    settings = {"eye_size": eye_size, "pupil_size_range": pupil_size_range}
    response = requests.post(f"{URL}/googly_eyes", files=files, data=settings)
    response.raise_for_status()
    image = _get_image_from_bytes(response.content)


if highlight_faces:
    files = {"image": _put_image_in_buffer(image)}
    response = requests.post(f"{URL}/identify_faces", files=files)
    response.raise_for_status()
    for face in response.json():
        _draw_face(image, Face(**face))

st.image(image)

buffer = _put_image_in_buffer(image)
filename, ext = uploaded_file.name.split(".")
st.download_button(
    label="Download Image",
    data=buffer.getvalue(),
    file_name=f"{filename}_googly_eyes.{ext}",
    mime=uploaded_file.type,
    help="Download modified image",
)
