from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

# CONFIG

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml_models" / "plant_disease_checkpoint_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD CHECKPOINT

checkpoint = torch.load(MODEL_PATH, map_location=device)

if not isinstance(checkpoint, dict):
    raise RuntimeError("Invalid checkpoint format.")

if "model_state_dict" not in checkpoint:
    raise RuntimeError("Checkpoint does not contain model_state_dict.")

if "class_names" not in checkpoint:
    raise RuntimeError("Checkpoint does not contain class_names.")

class_names = checkpoint["class_names"]

class_indices = {
    class_name: idx
    for idx, class_name in enumerate(class_names)
}

index_to_class = {
    idx: class_name
    for class_name, idx in class_indices.items()
}

num_classes = len(class_names)

# BUILD MODEL

model = models.mobilenet_v2(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, num_classes),
)

# LOAD MODEL WEIGHTS

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

# IMAGE TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ROUTES

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
            "error": "No image uploaded."
        }), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, dim=1)

        predicted_index = predicted.item()

        predicted_class = index_to_class[predicted_index]

        confidence_score = round(confidence.item() * 100, 2)

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

# MAIN

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)