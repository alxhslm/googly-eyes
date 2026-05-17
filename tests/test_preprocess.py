import numpy as np
import pytest

from retinaface.commons.preprocess import get_image, preprocess_image, resize_image


def _solid_image(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), fill_value=128, dtype=np.uint8)


class TestGetImage:
    def test_passthrough_numpy_array(self):
        img = _solid_image(50, 60)
        result = get_image(img)
        np.testing.assert_array_equal(result, img)

    def test_returns_copy_not_reference(self):
        img = _solid_image(10, 10)
        result = get_image(img)
        result[0, 0, 0] = 0
        assert img[0, 0, 0] == 128

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            get_image("/nonexistent/path/image.jpg")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_image(12345)  # type: ignore[arg-type]

    def test_rgba_stripped_to_rgb(self):
        img = np.full((50, 60, 4), fill_value=128, dtype=np.uint8)
        result = get_image(img)
        assert result.shape[2] == 3
        np.testing.assert_array_equal(result, img[:, :, :3])


class TestResizeImage:
    def test_output_is_target_size(self):
        img = _solid_image(200, 400)
        resized, _, _ = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert resized.shape[:2] == (1024, 1024)

    def test_returns_three_channels(self):
        img = _solid_image(100, 100)
        resized, _, _ = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert resized.shape[2] == 3

    def test_scale_landscape(self):
        # 500×1000 (h×w) image — height is limiting dimension
        img = _solid_image(500, 1000)
        _, scale, _ = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert scale == pytest.approx(1024 / 1000)

    def test_scale_portrait(self):
        # 1000×500 (h×w) image — width is limiting dimension
        img = _solid_image(1000, 500)
        _, scale, _ = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert scale == pytest.approx(1024 / 1000)

    def test_scale_square(self):
        img = _solid_image(512, 512)
        _, scale, _ = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert scale == pytest.approx(2.0)

    def test_offset_landscape_image(self):
        # Wide image (200h×400w): width is the constraining dimension, padding goes to height
        # im_offset = (0.0, (w-h)/2) = (0.0, 100.0)
        img = _solid_image(200, 400)
        _, _, offset = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert offset[0] == pytest.approx(0.0)  # no row offset
        assert offset[1] == pytest.approx((400 - 200) / 2)  # column offset from padding

    def test_offset_square_image_is_zero(self):
        img = _solid_image(100, 100)
        _, _, offset = resize_image(img, target_size=(1024, 1024), allow_upscaling=True)
        assert offset == (0.0, 0.0)


class TestPreprocessImage:
    def test_output_tensor_shape(self):
        img = _solid_image(200, 300)
        tensor, im_shape, _, _ = preprocess_image(img, allow_upscaling=True)
        assert tensor.shape == (1, 3, 640, 640)
        assert im_shape == (640, 640)

    def test_output_dtype_is_float32(self):
        img = _solid_image(100, 100)
        tensor, _, _, _ = preprocess_image(img, allow_upscaling=True)
        assert tensor.dtype == np.float32

    def test_pixel_values_mean_subtracted(self):
        # BGR means (104, 117, 123); darkest possible pixel after subtraction is 0 - 123 = -123
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        tensor, _, _, _ = preprocess_image(img, allow_upscaling=True)
        assert tensor.min() >= -123.0
        assert tensor.max() <= 255.0 - 104.0
