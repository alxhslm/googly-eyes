"""
Convert the biubug6 MobileNet0.25 RetinaFace PyTorch model to ONNX.

Weights are downloaded automatically from HuggingFace (py-feat/retinaface).
Run from the repo root:

    poetry run python retinaface/convert_model.py
"""

import os
import sys

import torch
from huggingface_hub import hf_hub_download

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "pytorch"))

from pytorch.data.config import cfg_mnet  # noqa: E402
from pytorch.models.retinaface import RetinaFace  # noqa: E402

WEIGHTS_REPO = "py-feat/retinaface"
WEIGHTS_FILE = "mobilenet0.25_Final.pth"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "retinaface.onnx")

# Disable pretrain weight loading — we supply the final weights ourselves
_cfg = {**cfg_mnet, "pretrain": False}
IMAGE_SIZE = _cfg["image_size"]


def load_model(weights_path: str) -> torch.nn.Module:
    net = RetinaFace(cfg=_cfg, phase="test")
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def export(net: torch.nn.Module, output_path: str) -> None:
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        net,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["bbox", "cls", "ldm"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    weights_path = hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE)
    net = load_model(weights_path)
    export(net, OUTPUT_PATH)
