"""
Face Recognition UI - Flask Backend
"""
import sys
from pathlib import Path
import io
import base64
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yaml

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbone import FaceEncoder
from src.utils.device import get_device
from facenet_pytorch import MTCNN

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
mtcnn = None
device = None
config = None
best_threshold = 0.45  # Conservative threshold for better precision


def load_model():
    """Load the trained face recognition model"""
    global model, mtcnn, device, config, best_threshold
    
    # Load config
    config_path = ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device(config.get("device", "auto"))
    
    # Load MTCNN for face detection
    mtcnn = MTCNN(
        image_size=config["data"]["image_size"],
        margin=config["mtcnn"]["margin"],
        keep_all=False,
        thresholds=config["mtcnn"]["thresholds"],
        post_process=config["mtcnn"]["post_process"],
        device=device
    )
    
    # Load face recognition model - prefer triplet-trained model
    checkpoint_path = ROOT / "outputs" / "checkpoints" / "best_metric.pt"
    if not checkpoint_path.exists():
        checkpoint_path = ROOT / "outputs" / "checkpoints" / "best_classification.pt"
        print("Warning: Using classification model. Triplet model preferred for better accuracy.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get num_classes from checkpoint config or manifest
    processed_root = ROOT / config["paths"]["processed_root"]
    from src.data.datasets import load_manifest
    manifest = load_manifest(processed_root)
    num_classes = len(manifest["train_identities"])
    
    model = FaceEncoder(
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"],
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    
    # Load state dict with strict=False to ignore extra keys from pretrained model
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    # Load best threshold from checkpoint but use more conservative value
    saved_threshold = ckpt.get('best_threshold', 0.45)
    # Use stricter threshold for better precision (reduce false positives)
    best_threshold = min(saved_threshold * 0.85, 0.45)  # 15% stricter
    
    print(f"Model loaded successfully on {device}")
    print(f"Saved threshold: {saved_threshold}, Using: {best_threshold}")


def decode_base64_image(base64_str):
    """Decode base64 image string to PIL Image"""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def preprocess_image(image):
    """Detect face and get embedding"""
    try:
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Detect face with MTCNN
        face_tensor = mtcnn(img_np)
        
        if face_tensor is None:
            return None, "No face detected"
        
        # Get embedding
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode(face_tensor)
        
        return embedding, None
    except Exception as e:
        return None, str(e)


def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings"""
    return F.cosine_similarity(emb1, emb2).item()


def cosine_distance(emb1, emb2):
    """Calculate cosine distance between two embeddings"""
    return 1.0 - cosine_similarity(emb1, emb2)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_face():
    """Detect face in uploaded image"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(image_data)
        
        # Detect face
        img_np = np.array(image)
        boxes, probs = mtcnn.detect(img_np)
        
        if boxes is None:
            return jsonify({
                'success': False,
                'message': 'No face detected',
                'faces': 0
            })
        
        return jsonify({
            'success': True,
            'faces': len(boxes),
            'boxes': boxes.tolist(),
            'confidences': probs.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/embed', methods=['POST'])
def get_embedding():
    """Get face embedding from image"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode and process image
        image = decode_base64_image(image_data)
        embedding, error = preprocess_image(image)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        
        return jsonify({
            'success': True,
            'embedding': embedding.cpu().numpy().tolist()[0],
            'dimension': embedding.shape[1]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify_faces():
    """Verify if two faces belong to the same person"""
    try:
        print("Received verify request")
        data = request.get_json()
        
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
        
        image1_data = data.get('image1')
        image2_data = data.get('image2')
        
        print(f"Image1 data length: {len(image1_data) if image1_data else 0}")
        print(f"Image2 data length: {len(image2_data) if image2_data else 0}")
        
        if not image1_data or not image2_data:
            return jsonify({'error': 'Two images required'}), 400
        
        # Process first image
        print("Processing image 1...")
        image1 = decode_base64_image(image1_data)
        emb1, error1 = preprocess_image(image1)
        
        if error1:
            print(f"Error with image 1: {error1}")
            return jsonify({
                'success': False,
                'error': f'Image 1: {error1}'
            })
        
        # Process second image
        print("Processing image 2...")
        image2 = decode_base64_image(image2_data)
        emb2, error2 = preprocess_image(image2)
        
        if error2:
            print(f"Error with image 2: {error2}")
            return jsonify({
                'success': False,
                'error': f'Image 2: {error2}'
            })
        
        # Calculate similarity
        print("Calculating similarity...")
        similarity = cosine_similarity(emb1, emb2)
        distance = cosine_distance(emb1, emb2)
        
        # Use threshold from loaded checkpoint
        threshold = best_threshold
        is_same = distance < threshold
        
        confidence = (1 - distance) * 100  # Convert to percentage
        
        print(f"Similarity: {similarity:.4f}, Distance: {distance:.4f}, Same person: {is_same}")
        
        return jsonify({
            'success': True,
            'is_same_person': bool(is_same),
            'similarity': float(similarity),
            'distance': float(distance),
            'confidence': float(confidence),
            'threshold': float(threshold)
        })
    
    except Exception as e:
        print(f"Error in verify_faces: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if model is loaded"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'embedding_dim': config["model"]["embedding_dim"] if config else None
    })


if __name__ == '__main__':
    print("Loading face recognition model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
