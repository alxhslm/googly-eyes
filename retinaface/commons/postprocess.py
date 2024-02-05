import numpy as np

# pylint: disable=unused-argument


def findEuclideanDistance(source_representation: np.ndarray | list, test_representation: np.ndarray | list) -> float:
    """
    Find euclidean distance between 2 vectors
    Args:
        source_representation (numpy array or list)
        test_representation (numpy array or list)
    Returns
        distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return float(euclidean_distance)


def bbox_pred(boxes: np.ndarray, box_deltas: np.ndarray) -> np.ndarray:
    """
    This function is copied from the following code snippet:
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes


def landmark_pred(boxes: np.ndarray, landmark_deltas: np.ndarray) -> np.ndarray:
    """
    This function copied from the following code snippet
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
    return pred


def clip_boxes(boxes: np.ndarray, im_shape: tuple[int, int]) -> np.ndarray:
    """
    This function copied from rcnn module of retinaface-tf2 project
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/processing/bbox_transform.py
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def anchors_plane(height: int, width: int, stride: int, base_anchors: np.ndarray) -> np.ndarray:
    """
    This function is mainly based on the following code snippet
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/anchors.pyx
    """
    A = base_anchors.shape[0]
    c_0_2 = np.tile(np.arange(0, width)[np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
    c_1_3 = np.tile(np.arange(0, height)[:, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
    return np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(
        base_anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1)
    )  # type: ignore


def cpu_nms(dets: np.ndarray, threshold: float) -> list[int]:
    """
    This function is mainly based on the following code snippet
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx
    Fast R-CNN by Ross Girshick
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep


def transform_bbox(
    x: np.ndarray,
    y: np.ndarray,
    im_scale: float,
    im_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return (x / im_scale) - im_offset[0], (y / im_scale) - im_offset[1]
