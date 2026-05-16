import sys
from unittest.mock import MagicMock

# Prevent tflite_runtime from being imported — it's not installed in the test environment
sys.modules["tflite_runtime"] = MagicMock()
sys.modules["tflite_runtime.interpreter"] = MagicMock()
