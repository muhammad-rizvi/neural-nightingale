"""Re-create model.onnx and class_names.json from model.pth.

This mirrors the last cell of neural_nightingale.ipynb, so you can rebuild the
web app's model files on any computer (no GPU or Colab needed):

    pip install torch torchvision onnx onnxruntime
    python scripts/export_onnx.py
"""
import json

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models

# 1. Rebuild the architecture and load the trained weights (same as app.py)
ckpt = torch.load("model.pth", map_location="cpu")
pretty = [c.replace("_", " ") for c in ckpt["class_names"]]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(pretty))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 2. Export: one 1x3x224x224 float32 input called "image", one output called "logits"
torch.onnx.export(model, torch.randn(1, 3, 224, 224), "model.onnx",
                  input_names=["image"], output_names=["logits"])
with open("class_names.json", "w") as f:
    json.dump(pretty, f, indent=2)

# 3. Sanity check: ONNX Runtime must give the same logits as PyTorch
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    torch_logits = model(x).numpy()
onnx_logits = ort.InferenceSession("model.onnx").run(None, {"image": x.numpy()})[0]
print("Max difference PyTorch vs ONNX:", float(np.abs(torch_logits - onnx_logits).max()))
print("Wrote model.onnx and class_names.json")
