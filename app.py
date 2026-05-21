import json
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml_models" / "plant_disease_checkpoint.pth"
CLASS_PATH = BASE_DIR / "ml_models" / "class_indices.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)

# -----------------------------
# LOAD CLASS MAPPING
# -----------------------------
if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
    class_indices = checkpoint["class_to_idx"]

elif isinstance(checkpoint, dict) and "class_names" in checkpoint:
    class_indices = {
        class_name: idx
        for idx, class_name in enumerate(checkpoint["class_names"])
    }

else:
    with open(CLASS_PATH, "r") as f:
        class_indices = json.load(f)

index_to_class = {
    int(v): k
    for k, v in class_indices.items()
}

num_classes = len(class_indices)

# -----------------------------
# BUILD MODEL
# -----------------------------
model = models.mobilenet_v2(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, num_classes),
)

# -----------------------------
# LOAD MODEL WEIGHTS
# -----------------------------
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])

else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "AgroGuide AI API running",
        "model": "MobileNetV2",
        "classes": num_classes,
        "device": str(device)
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, 1)

        predicted_index = int(predicted.item())

        predicted_class = index_to_class.get(
            predicted_index,
            "unknown"
        )

        confidence_score = round(
            float(confidence.item()) * 100,
            2
        )

        return jsonify({
            "success": True,
            "disease_class": predicted_class,
            "class_name": predicted_class,
            "confidence": confidence_score,
            "confidence_percent": confidence_score
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)