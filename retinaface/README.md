# RetinaFace
This is a modified version of this [Tensorflow implementation](https://github.com/serengil/retinaface) of the RetinaFace model.

The major changes are as follows:

1. Removed all unneeded functionality related to extraction of faces
2. Updated to use fixed input image dimensions to allow conversion to TF-lite

## Converting a model to tflite
Run the following:
``` bash
    poetry run python retinaface/convert_model.py
```
which will generate a `retinaface.tflite` file, and commit the result. Rebuild the server for changes to take affect.
