import json
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml_models" / "plant_disease_checkpoint.pth"
CLASS_PATH = BASE_DIR / "ml_models" / "class_indices.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
num_classes = len(class_indices)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

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
        "classes": num_classes
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_index = predicted.item()
        predicted_class = index_to_class[predicted_index]
        confidence_score = round(confidence.item() * 100, 2)

        return jsonify({
            "success": True,
            "disease_class": predicted_class,
            "confidence": confidence_score
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)