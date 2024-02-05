import os
from pathlib import Path

import numpy as np
from PIL import Image


def get_image(img_uri: str | np.ndarray) -> np.ndarray:
    """
    Load the given image
    Args:
        img_path (str or numpy array): exact image path or pre-loaded numpy array (RGB format)
    Returns:
        image itself
    """
    # if it is pre-loaded numpy array
    if isinstance(img_uri, np.ndarray):  # Use given NumPy array
        img = img_uri.copy()

    # then it has to be a path on filesystem
    elif isinstance(img_uri, str):
        if isinstance(img_uri, Path):
            img_uri = str(img_uri)

        if not os.path.isfile(img_uri):
            raise ValueError(f"Input image file path ({img_uri}) does not exist.")

        img = np.array(Image.open(img_uri))

    else:
        raise ValueError(
            f"Invalid image input - {img_uri}."
            "Exact paths, pre-loaded numpy arrays, base64 encoded "
            "strings and urls are welcome."
        )

    # Validate image shape
    if len(img.shape) != 3 or np.prod(img.shape) == 0:
        raise ValueError("Input image needs to have 3 channels at must not be empty.")

    return img


def resize_image(img: np.ndarray, target_size: int, max_size: int, allow_upscaling: bool) -> tuple[np.ndarray, float]:
    """
    This function is modified from the following code snippet:
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L49-L74

    Args:
        img (numpy array): given image
        target_size: int
        max_size: int
        allow_upscaling (bool)
    Returns
        resized image, im_scale
    """
    img_h, img_w = img.shape[0:2]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        im = Image.fromarray(img)
        im = im.resize(size=(int(im_scale * img_w), int(im_scale * img_h)), resample=Image.Resampling.BILINEAR)
        img = np.array(im)

    return img, im_scale


def preprocess_image(img: np.ndarray, allow_upscaling: bool) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    This function is modified from the following code snippet:
    https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L76-L96
    Args:
        img (numpy array): given image
        allow_upscaling (bool)
    Returns:
        tensor, image shape, im_scale
    """
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)

    img, im_scale = resize_image(img, target_size=1024, max_size=1980, allow_upscaling=allow_upscaling)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    im_shape = (img.shape[0], img.shape[1])

    # Make image scaling + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, i] / pixel_scale - pixel_means[i]) / pixel_stds[i]

    return im_tensor, im_shape, im_scale
