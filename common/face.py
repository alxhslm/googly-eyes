from dataclasses import dataclass


@dataclass
class Face:
    score: float
    bounding_box: list[float]
    landmarks: dict[str, list[float]]
