from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# DEVICE
# ==========================================
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# PATHS
# ==========================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
META_PATH    = os.path.join(MODELS_DIR, "resnet50_meta.json")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "resnet50_best.pth")

# ==========================================
# LOAD META
# ==========================================
with open(META_PATH) as f:
    meta = json.load(f)

print(f"Classes: {meta['class_names']}")

# ==========================================
# BUILD MODEL
# ==========================================
def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = build_resnet50(meta["num_classes"])

# ==========================================
# LOAD WEIGHTS
# ==========================================
state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

# Strip 'module.' prefix if saved with DataParallel
if isinstance(state, dict):
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[new_key] = v
    state = new_state

model.load_state_dict(state)
model.to(DEVICE).eval()
print("Model loaded successfully!")

# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((meta["input_size"], meta["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=meta["normalize_mean"],
        std=meta["normalize_std"]
    )
])

# ==========================================
# GRAD-CAM HOOKS
# ==========================================
activations_store = {}
gradients_store   = {}

def forward_hook(module, inp, out):
    activations_store["value"] = out

def backward_hook(module, grad_in, grad_out):
    gradients_store["value"] = grad_out[0]

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ==========================================
# GRAD-CAM GENERATOR
# ==========================================
def generate_gradcam(img_tensor, class_idx):
    activations_store.clear()
    gradients_store.clear()

    x      = img_tensor.unsqueeze(0).to(DEVICE)
    logits = model(x)
    model.zero_grad()
    logits[0, class_idx].backward()

    weights = gradients_store["value"].mean(dim=(2, 3), keepdim=True)
    cam     = (weights * activations_store["value"]).sum(dim=1, keepdim=True)
    cam     = torch.relu(cam)
    cam     = torch.nn.functional.interpolate(
                  cam,
                  size=(meta["input_size"], meta["input_size"]),
                  mode="bilinear",
                  align_corners=False
              )
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# ==========================================
# HELPERS
# ==========================================
def denormalize(tensor):
    mean = torch.tensor(meta["normalize_mean"]).view(3, 1, 1)
    std  = torch.tensor(meta["normalize_std"]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1)

def array_to_base64(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ==========================================
# ROUTES
# ==========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_bytes = request.files["image"].read()
        image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor  = transform(image)

        # Inference
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(DEVICE))
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_label = meta["class_names"][pred_idx]
        confidence = float(probs[pred_idx])

        fake_idx  = meta["class_to_idx"]["Fake"]
        real_idx  = meta["class_to_idx"]["Real"]
        fake_prob = float(probs[fake_idx])
        real_prob = float(probs[real_idx])

        # Grad-CAM
        cam        = generate_gradcam(img_tensor, pred_idx)
        img_np     = denormalize(img_tensor).permute(1, 2, 0).numpy()
        heatmap_np = plt.get_cmap("jet")(cam)[..., :3]
        overlay    = np.clip(0.55 * img_np + 0.45 * heatmap_np, 0, 1)

        return jsonify({
            "label":        pred_label,
            "confidence":   round(confidence * 100, 2),
            "fake_prob":    round(fake_prob * 100, 2),
            "real_prob":    round(real_prob * 100, 2),
            "original_b64": array_to_base64(img_np),
            "heatmap_b64":  array_to_base64(heatmap_np),
            "overlay_b64":  array_to_base64(overlay)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)