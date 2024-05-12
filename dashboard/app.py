import os

from aws_requests_auth.aws_auth import AWSRequestsAuth
import requests
import streamlit as st
from PIL import Image, ImageDraw
import typing as t
from common.drawing import plot_circle
from common.face import Face
from common.image import put_image_into_buffer, deserialize_image, serialize_image

resource = "dzdzxpkrlmjj74daububqcowty0fbiev"
region = "eu-west-2"
host = f"{resource}.lambda-url.{region}.on.aws"
url = f"https://{host}/"

auth = AWSRequestsAuth(
    aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_host=host,
    aws_region=region,
    aws_service="lambda",
)


@st.cache_data
def _add_googly_eyes(body: dict[str, t.Any]) -> dict[str, t.Any]:
    response = requests.post(url, data=body, auth=auth)
    response.raise_for_status()
    return response.json()


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
    googly_eyes_enabled = st.checkbox("Add googly eyes", value=True, help="Put google eyes on each face in the image")
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

body = {
    "image": serialize_image(image),
    "eye_size": eye_size,
    "pupil_size_range": pupil_size_range,
}

data = _add_googly_eyes(body)
if googly_eyes_enabled:
    image = deserialize_image(data["image"])


if highlight_faces:
    for face in data["faces"]:
        _draw_face(image, Face(**face))

st.image(image)

buffer = put_image_into_buffer(image)
filename, ext = uploaded_file.name.split(".")
st.download_button(
    label="Download Image",
    data=buffer.getvalue(),
    file_name=f"{filename}_googly_eyes.{ext}",
    mime=uploaded_file.type,
    help="Download modified image",
)
