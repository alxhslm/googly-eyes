import math
from typing import Any, Callable

import numpy as np

from retinaface.commons import postprocess, preprocess

# MobileNet0.25 anchor config (biubug6/Pytorch_Retinaface)
_MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
_STEPS = [8, 16, 32]
_VARIANCE = [0.1, 0.2]
_IMAGE_SIZE = 640


def _generate_priors() -> np.ndarray:
    priors = []
    for min_sizes, step in zip(_MIN_SIZES, _STEPS):
        feat_h = math.ceil(_IMAGE_SIZE / step)
        feat_w = math.ceil(_IMAGE_SIZE / step)
        for i in range(feat_h):
            for j in range(feat_w):
                for min_size in min_sizes:
                    cx = (j + 0.5) * step / _IMAGE_SIZE
                    cy = (i + 0.5) * step / _IMAGE_SIZE
                    w = min_size / _IMAGE_SIZE
                    h = min_size / _IMAGE_SIZE
                    priors.append([cx, cy, w, h])
    return np.array(priors, dtype=np.float32)


_PRIORS = _generate_priors()


def _decode_boxes(loc: np.ndarray) -> np.ndarray:
    boxes = np.concatenate(
        [
            _PRIORS[:, :2] + loc[:, :2] * _VARIANCE[0] * _PRIORS[:, 2:],
            _PRIORS[:, 2:] * np.exp(loc[:, 2:] * _VARIANCE[1]),
        ],
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2  # center → corner
    boxes[:, 2:] += boxes[:, :2]
    return boxes * _IMAGE_SIZE


def _decode_landmarks(pre: np.ndarray) -> np.ndarray:
    return (
        np.concatenate(
            [
                _PRIORS[:, :2] + pre[:, 0:2] * _VARIANCE[0] * _PRIORS[:, 2:],
                _PRIORS[:, :2] + pre[:, 2:4] * _VARIANCE[0] * _PRIORS[:, 2:],
                _PRIORS[:, :2] + pre[:, 4:6] * _VARIANCE[0] * _PRIORS[:, 2:],
                _PRIORS[:, :2] + pre[:, 6:8] * _VARIANCE[0] * _PRIORS[:, 2:],
                _PRIORS[:, :2] + pre[:, 8:10] * _VARIANCE[0] * _PRIORS[:, 2:],
            ],
            axis=1,
        )
        * _IMAGE_SIZE
    )


def detect_faces(
    img_path: str | np.ndarray,
    model: Callable[[np.ndarray], list[np.ndarray]],
    threshold: float = 0.9,
    allow_upscaling: bool = True,
) -> list[dict[str, Any]]:
    img = preprocess.get_image(img_path)
    im_tensor, im_shape, im_scale, im_offset = preprocess.preprocess_image(img, allow_upscaling)

    bbox_raw, cls_raw, ldm_raw = model(im_tensor)

    bbox = bbox_raw[0]  # (N, 4)
    cls = cls_raw[0]    # (N, 2)
    ldm = ldm_raw[0]    # (N, 10)

    boxes = _decode_boxes(bbox)        # pixels in 640×640 space
    landmarks = _decode_landmarks(ldm)  # pixels in 640×640 space
    scores = cls[:, 1]

    keep = np.where(scores >= threshold)[0]
    if len(keep) == 0:
        return []
    boxes, landmarks, scores = boxes[keep], landmarks[keep], scores[keep]

    # transform back to original image coordinates
    boxes[:, 0], boxes[:, 1] = postprocess.transform_bbox(boxes[:, 0], boxes[:, 1], im_scale, im_offset)
    boxes[:, 2], boxes[:, 3] = postprocess.transform_bbox(boxes[:, 2], boxes[:, 3], im_scale, im_offset)

    lm_x, lm_y = postprocess.transform_bbox(landmarks[:, 0::2], landmarks[:, 1::2], im_scale, im_offset)
    landmarks_out = np.stack([lm_x, lm_y], axis=2)  # (N, 5, 2)

    pre_det = np.hstack([boxes[:, :4], scores[:, np.newaxis]]).astype(np.float32)
    keep = postprocess.cpu_nms(pre_det, 0.4)
    boxes, scores, landmarks_out = boxes[keep], scores[keep], landmarks_out[keep]

    return [
        {
            "score": float(scores[i]),
            "facial_area": boxes[i, :4].astype(int).tolist(),
            "landmarks": {
                "right_eye": landmarks_out[i][0].tolist(),
                "left_eye": landmarks_out[i][1].tolist(),
                "nose": landmarks_out[i][2].tolist(),
                "mouth_right": landmarks_out[i][3].tolist(),
                "mouth_left": landmarks_out[i][4].tolist(),
            },
        }
        for i in range(len(boxes))
    ]
