import typing as t

import numpy as np
from face import Face
from PIL import Image, ImageDraw


def plot_circle(draw: ImageDraw, xy: t.Sequence[float], radius: float, **kwargs: t.Any) -> None:
    draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), **kwargs)


def add_googly_eyes(image: Image, face: Face, eye_size: float, pupil_size_range: tuple[float, float]) -> None:
    draw = ImageDraw.Draw(image)
    radius = 0.5 * float(np.linalg.norm(np.array(face.landmarks["right_eye"]) - np.array(face.landmarks["left_eye"])))

    def plot_googly_eye(eye: list[float]) -> None:
        pupil_size = eye_size * np.random.uniform(pupil_size_range[0], pupil_size_range[1])
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
            np.array(eye) + (eye_size - pupil_size) * radius * np.array([np.sin(orientation), np.cos(orientation)]),
            radius=pupil_size * radius,
            fill=(0, 0, 0, 0),
        )

    plot_googly_eye(face.landmarks["right_eye"])
    plot_googly_eye(face.landmarks["left_eye"])
