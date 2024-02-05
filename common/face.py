import typing as t
from dataclasses import dataclass

import numpy as np


@dataclass
class Face:
    score: float
    bounding_box: list[float]
    landmarks: dict[str, list[float]]

    def asdict(self) -> dict[str, t.Any]:
        return {
            "score": self.score,
            "bounding_box": np.array(self.bounding_box).tolist(),
            "landmarks": {k: np.array(v).tolist() for k, v in self.landmarks.items()},
        }
