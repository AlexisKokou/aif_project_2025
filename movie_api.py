import argparse
import os
import json

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

from model import MoviePosterNet
from logit_scores import entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"API running on device: {device}")

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights/movie_poster_net.pth', help='model path')
args = parser.parse_args()

print(f"Loading model from: {args.model_path}")
try:
    checkpoint = torch.load(args.model_path, map_location=device)

    model = MoviePosterNet(checkpoint['num_genres']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    genres = checkpoint['classes']
    class_to_idx = checkpoint['class_to_idx']

    print("Model loaded successfully!")
    print(f"Genres available: {list(genres)}")
    print(f"Class mapping: {class_to_idx}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ---- OOD config (Part 2) ----
OOD_CONFIG_PATH = os.getenv("OOD_CONFIG_PATH", "ood_config.json")

ood_config = {
    "method": "entropy",
    "threshold": 1.05,
    "tpr_target": 0.90,
    "decision_rule": "valid_if_score <= threshold"
}

if os.path.exists(OOD_CONFIG_PATH):
    try:
        with open(OOD_CONFIG_PATH, "r") as f:
            ood_config.update(json.load(f))
        print(f"OOD config loaded from {OOD_CONFIG_PATH}: {ood_config}")
    except Exception as e:
        print(f"Failed to load OOD config ({OOD_CONFIG_PATH}): {e}")
else:
    print(f"OOD config file not found at {OOD_CONFIG_PATH}. Using defaults: {ood_config}")

OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", ood_config["threshold"]))
OOD_METHOD = ood_config.get("method", "entropy")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_pil = Image.open(file.stream).convert('RGB')
        tensor = transform(img_pil).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities[0][predicted_idx].item()

        all_predictions = {genre: float(probabilities[0][i]) for i, genre in enumerate(genres)}
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'prediction': genres[predicted_idx],
            'confidence': round(confidence, 4),
            'top_3_predictions': [
                {'genre': genre, 'confidence': round(conf, 4)}
                for genre, conf in sorted_predictions[:3]
            ],
            'all_predictions': all_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    results = []
    try:
        for file in files:
            if file.filename == '':
                continue

            img_pil = Image.open(file.stream).convert('RGB')
            tensor = transform(img_pil).to(device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = outputs.argmax(1).item()
                confidence = probabilities[0][predicted_idx].item()

            results.append({
                'filename': file.filename,
                'prediction': genres[predicted_idx],
                'confidence': round(confidence, 4)
            })

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/validate_poster', methods=['POST'])
def validate_poster():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_pil = Image.open(file.stream).convert('RGB')
        tensor = transform(img_pil).to(device).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)

            if OOD_METHOD == "entropy":
                score = float(entropy(logits)[0])
            else:
                return jsonify({'error': f'Unsupported OOD method: {OOD_METHOD}'}), 500

            valid = bool(score <= OOD_THRESHOLD)

            probas = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(1).item()
            confidence = float(probas[0][pred_idx].item())

        return jsonify({
            "valid": valid,
            "ood_method": OOD_METHOD,
            "ood_score": round(score, 6),
            "threshold": round(OOD_THRESHOLD, 6),
            "rule": "valid_if_score <= threshold",
            "prediction_if_valid": genres[pred_idx] if valid else None,
            "confidence_if_valid": round(confidence, 4) if valid else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True,
        'available_genres': list(genres),
        'num_genres': len(genres),
        'ood_method': OOD_METHOD,
        'ood_threshold': OOD_THRESHOLD
    })

@app.route('/genres', methods=['GET'])
def get_genres():
    return jsonify({
        'genres': list(genres),
        'class_mapping': class_to_idx
    })

if __name__ == "__main__":
    print("\nStarting Flask API server...")
    print("Endpoints:")
    print(" - POST /validate_poster")
    print(" - POST /predict")
    print(" - POST /batch_predict")
    print(" - GET /health")
    print(" - GET /genres")
    print("\nServer running on http://0.0.0.0:5075")
    app.run(host='0.0.0.0', port=5075, debug=False)
