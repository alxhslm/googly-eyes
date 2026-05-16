import numpy as np
import pytest
from retinaface.commons.postprocess import (
    findEuclideanDistance,
    bbox_pred,
    landmark_pred,
    clip_boxes,
    anchors_plane,
    cpu_nms,
    transform_bbox,
)


class TestFindEuclideanDistance:
    def test_identical_vectors_returns_zero(self):
        assert findEuclideanDistance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self):
        # 3-4-5 right triangle
        assert findEuclideanDistance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_accepts_numpy_arrays(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert findEuclideanDistance(a, b) == pytest.approx(np.sqrt(2))

    def test_accepts_mixed_list_and_array(self):
        assert findEuclideanDistance([0, 0, 0], np.array([1, 1, 1])) == pytest.approx(np.sqrt(3))


class TestBboxPred:
    def test_empty_boxes_returns_zeros(self):
        boxes = np.zeros((0, 4))
        deltas = np.zeros((0, 4))
        result = bbox_pred(boxes, deltas)
        assert result.shape == (0, 4)

    def test_zero_deltas_preserves_boxes(self):
        # With zero deltas, the predicted box should equal the input box
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        deltas = np.zeros((1, 4))
        result = bbox_pred(boxes, deltas)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0, 0], boxes[0, 0], rtol=1e-5)
        np.testing.assert_allclose(result[0, 1], boxes[0, 1], rtol=1e-5)

    def test_extra_columns_passed_through(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        deltas = np.array([[0.0, 0.0, 0.0, 0.0, 99.0]])
        result = bbox_pred(boxes, deltas)
        assert result.shape == (1, 5)
        assert result[0, 4] == 99.0


class TestLandmarkPred:
    def test_empty_boxes_returns_zeros(self):
        boxes = np.zeros((0, 4))
        deltas = np.zeros((0, 5, 2))
        result = landmark_pred(boxes, deltas)
        # shape[1] comes from landmark_deltas.shape[1] which is 5 for a (0,5,2) input
        assert result.shape == (0, 5)

    def test_output_shape(self):
        boxes = np.array([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 50.0, 50.0]])
        deltas = np.zeros((2, 5, 2))
        result = landmark_pred(boxes, deltas)
        assert result.shape == (2, 5, 2)


class TestClipBoxes:
    def test_clips_to_image_bounds(self):
        boxes = np.array([[-10.0, -5.0, 200.0, 300.0]])
        result = clip_boxes(boxes, im_shape=(100, 150))
        assert result[0, 0] >= 0          # x1
        assert result[0, 1] >= 0          # y1
        assert result[0, 2] <= 149        # x2 < width
        assert result[0, 3] <= 99         # y2 < height

    def test_in_bounds_boxes_unchanged(self):
        boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
        result = clip_boxes(boxes.copy(), im_shape=(100, 100))
        np.testing.assert_array_equal(result, boxes)


class TestAnchorsPlane:
    def test_output_shape(self):
        base_anchors = np.array([[-8, -8, 8, 8], [-16, -16, 16, 16]])  # 2 anchors
        result = anchors_plane(height=4, width=6, stride=8, base_anchors=base_anchors)
        assert result.shape == (4, 6, 2, 4)

    def test_stride_applied(self):
        base_anchors = np.array([[0, 0, 0, 0]])
        result = anchors_plane(height=2, width=2, stride=16, base_anchors=base_anchors)
        # First anchor at (0,0): offset should be 0*16 = 0
        assert result[0, 0, 0, 0] == 0
        # Anchor at grid position (0,1): x offset = 1*16 = 16
        assert result[0, 1, 0, 0] == 16


class TestCpuNms:
    def test_non_overlapping_boxes_all_kept(self):
        dets = np.array([
            [0, 0, 10, 10, 0.9],
            [20, 20, 30, 30, 0.8],
            [40, 40, 50, 50, 0.7],
        ])
        kept = cpu_nms(dets, threshold=0.5)
        assert len(kept) == 3

    def test_fully_overlapping_keeps_highest_score(self):
        dets = np.array([
            [0, 0, 10, 10, 0.9],
            [0, 0, 10, 10, 0.5],
        ])
        kept = cpu_nms(dets, threshold=0.5)
        assert len(kept) == 1
        assert dets[kept[0], 4] == pytest.approx(0.9)

    def test_partial_overlap_below_threshold_both_kept(self):
        # Small overlap — both should survive at threshold=0.5
        dets = np.array([
            [0, 0, 10, 10, 0.9],
            [8, 8, 18, 18, 0.8],
        ])
        kept = cpu_nms(dets, threshold=0.5)
        assert len(kept) == 2

    def test_empty_input(self):
        dets = np.zeros((0, 5))
        kept = cpu_nms(dets, threshold=0.5)
        assert kept == []


class TestTransformBbox:
    def test_identity_transform(self):
        x = np.array([100.0, 200.0])
        y = np.array([50.0, 150.0])
        tx, ty = transform_bbox(x, y, im_scale=1.0, im_offset=(0.0, 0.0))
        np.testing.assert_allclose(tx, x)
        np.testing.assert_allclose(ty, y)

    def test_scale_and_offset(self):
        x = np.array([200.0])
        y = np.array([100.0])
        tx, ty = transform_bbox(x, y, im_scale=2.0, im_offset=(10.0, 5.0))
        np.testing.assert_allclose(tx, [90.0])   # 200/2 - 10
        np.testing.assert_allclose(ty, [45.0])   # 100/2 - 5
