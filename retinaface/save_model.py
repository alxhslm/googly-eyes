# Convert the model
import os

import tensorflow as tf

from retinaface.model import build_model

model = build_model()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(os.path.dirname(__file__), "retinaface.tflite"), "wb") as f:
    f.write(tflite_model)
