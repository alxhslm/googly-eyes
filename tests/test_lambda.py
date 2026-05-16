import json
import sys
import os
from unittest.mock import patch

# lambda/ is not importable as a package (reserved keyword), so add it to path directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda"))


class TestLambdaHandler:
    def test_parses_body_and_delegates_to_googlify(self):
        import predict

        payload = {"image": "base64encodeddata", "eye_size": 0.4}
        event = {"body": json.dumps(payload)}

        with patch("predict.googlify", return_value={"image": "out", "faces": []}) as mock:
            result = predict.lambda_handler(event, None)

        mock.assert_called_once_with(payload)
        assert result == {"image": "out", "faces": []}

    def test_returns_googlify_result_directly(self):
        import predict

        expected = {"image": "result_image", "faces": [{"score": 0.99}]}
        event = {"body": json.dumps({"image": "test"})}

        with patch("predict.googlify", return_value=expected):
            result = predict.lambda_handler(event, None)

        assert result == expected
