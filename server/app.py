from flask import Flask, jsonify, request

from common.googlify import googlify

app = Flask(__name__)


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    return jsonify(googlify(request.get_json()))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
