import os
import warnings
from typing import Any, Callable

import numpy as np

from retinaface.commons import postprocess, preprocess

# ---------------------------

# configurations
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Limit the amount of reserved VRAM so that other scripts can be run in the same GPU as well
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ---------------------------


def detect_faces(
    img_path: str | np.ndarray,
    model: Callable[[np.ndarray], list[np.ndarray]],
    threshold: float = 0.9,
    allow_upscaling: bool = True,
) -> list[dict[str, Any]]:
    """
    Detect the facial area for a given image
    Args:
        img_path (str or numpy array): given image
        threshold (float): threshold for detection
        model (Model): pre-trained model can be given
        allow_upscaling (bool): allowing up-scaling
    Returns:
        detected faces as:
        [
            {
                "score": 0.9993440508842468,
                "facial_area": [155, 81, 434, 443],
                "landmarks": {
                    "right_eye": [257.82974, 209.64787],
                    "left_eye": [374.93427, 251.78687],
                    "nose": [303.4773, 299.91144],
                    "mouth_right": [228.37329, 338.73193],
                    "mouth_left": [320.21982, 374.58798]
                }
            }
        ]
    """
    img = preprocess.get_image(img_path)

    # ---------------------------

    # if model is None:
    #     model = build_model()

    # ---------------------------

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        "stride32": np.array([[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32),
        "stride16": np.array([[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32),
        "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }

    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    # ---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_shape, im_scale, im_offset = preprocess.preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    sym_idx = 0

    for _, s in enumerate(_feat_stride_fpn):
        # _key = f"stride{s}"
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{s}"] :]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{s}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{s}"]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_shape)

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0], proposals[:, 1] = postprocess.transform_bbox(
            proposals[:, 0], proposals[:, 1], im_scale, im_offset
        )
        proposals[:, 2], proposals[:, 3] = postprocess.transform_bbox(
            proposals[:, 2], proposals[:, 3], im_scale, im_offset
        )
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0], landmarks[:, :, 1] = postprocess.transform_bbox(
            landmarks[:, :, 0], landmarks[:, :, 1], im_scale, im_offset
        )
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)

    if proposals.shape[0] == 0:
        return []

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    return [
        {
            "score": float(face[4]),
            "facial_area": face[0:4].astype(int).tolist(),
            "landmarks": {
                "right_eye": landmarks[idx][0].tolist(),
                "left_eye": landmarks[idx][1].tolist(),
                "nose": landmarks[idx][2].tolist(),
                "mouth_right": landmarks[idx][3].tolist(),
                "mouth_left": landmarks[idx][4].tolist(),
            },
        }
        for idx, face in enumerate(det)
    ]
