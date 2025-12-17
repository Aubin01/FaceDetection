# Author: Aubin Mugisha & Copilot
# Description: Flask UI for face verification, detection, and embeddings.
import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from facenet_pytorch import MTCNN
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model import FaceEncoder, cosine_distance  # noqa: E402
from src.train.preparedata import load_manifest  # noqa: E402


def get_device() -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

APP_ROOT = PROJECT_ROOT / "src" / "UI_demo"

app = Flask(__name__, template_folder=str(APP_ROOT / "templates"), static_folder=str(APP_ROOT / "static"))
CORS(app)

model = None
mtcnn = None
device = None
config = None
best_threshold = 0.45


def load_model():
    """Load MTCNN and the trained encoder checkpoint."""
    global model, mtcnn, device, config, best_threshold

    config_path = PROJECT_ROOT / "src" / "model" / "config.yaml"
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    device = get_device()

    mtcnn = MTCNN(
        image_size=config["data"]["image_size"],
        margin=config["mtcnn"]["margin"],
        keep_all=False,
        thresholds=config["mtcnn"]["thresholds"],
        post_process=config["mtcnn"]["post_process"],
        device=device,
    )

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_metric.pt"
    fallback = PROJECT_ROOT / "outputs" / "checkpoints" / "best_classification.pt"
    if not checkpoint_path.exists():
        checkpoint_path = fallback
        print("Warning: Using classification model. Triplet model preferred for better accuracy.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    processed_root = PROJECT_ROOT / config["paths"]["processed_root"]
    manifest = load_manifest(processed_root)
    num_classes = len(manifest["train_identities"])

    model = FaceEncoder(
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"],
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    best_threshold = float(ckpt.get("best_threshold", 0.45))
    print(f"Model loaded successfully on {device}")
    print(f"Using threshold: {best_threshold} (from checkpoint if available)")


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess_image(image: Image.Image):
    """Detect face and get embedding."""
    try:
        img_np = np.array(image)
        face_tensor = mtcnn(img_np)
        if face_tensor is None:
            return None, "No face detected"
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode(face_tensor)
        return embedding, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect_face():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        image = decode_base64_image(image_data)
        img_np = np.array(image)
        boxes, probs = mtcnn.detect(img_np)

        if boxes is None:
            return jsonify({"success": False, "message": "No face detected", "faces": 0})

        import cv2

        img_with_boxes = img_np.copy()
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Face {i+1}: {prob*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(img_with_boxes, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        img_pil = Image.fromarray(img_with_boxes)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "faces": len(boxes),
                "boxes": boxes.tolist(),
                "confidences": probs.tolist(),
                "image_with_boxes": f"data:image/jpeg;base64,{img_base64}",
            }
        )

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@app.route("/api/embed", methods=["POST"])
def get_embedding():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        image = decode_base64_image(image_data)
        embedding, error = preprocess_image(image)

        if error:
            return jsonify({"success": False, "error": error})

        return jsonify({"success": True, "embedding": embedding.cpu().numpy().tolist()[0], "dimension": embedding.shape[1]})

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_faces():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        image1_data = data.get("image1")
        image2_data = data.get("image2")
        if not image1_data or not image2_data:
            return jsonify({"error": "Two images required"}), 400

        image1 = decode_base64_image(image1_data)
        emb1, error1 = preprocess_image(image1)
        if error1:
            return jsonify({"success": False, "error": f"Image 1: {error1}"})

        image2 = decode_base64_image(image2_data)
        emb2, error2 = preprocess_image(image2)
        if error2:
            return jsonify({"success": False, "error": f"Image 2: {error2}"})

        similarity = F.cosine_similarity(emb1, emb2).item()
        distance = float(cosine_distance(emb1, emb2).item())
        threshold = best_threshold
        is_same = distance < threshold
        confidence = (1 - distance) * 100

        return jsonify(
            {
                "success": True,
                "is_same_person": bool(is_same),
                "similarity": float(similarity),
                "distance": distance,
                "confidence": float(confidence),
                "threshold": float(threshold),
            }
        )

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(
        {
            "model_loaded": model is not None,
            "device": str(device) if device else None,
            "embedding_dim": config["model"]["embedding_dim"] if config else None,
        }
    )


if __name__ == "__main__":
    print("Loading face recognition model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
