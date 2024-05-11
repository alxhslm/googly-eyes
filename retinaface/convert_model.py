# Convert the model
import os

import tensorflow as tf
import urllib.request

from retinaface.model import build_model

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"
weights_file = os.path.join(os.path.dirname(__file__), "retinaface.h5")
if not os.path.exists(weights_file):
    urllib.request.urlretrieve(WEIGHTS_URL, weights_file)


model = build_model(weights_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(os.path.dirname(__file__), "retinaface.tflite"), "wb") as f:
    f.write(tflite_model)
