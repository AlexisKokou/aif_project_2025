import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import json
from models.model_poster import MoviePosterNet
import os
from logit_scores import entropy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"API running on device: {device}")

from models.model_plot import PlotClassification
from annoy import AnnoyIndex
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import yaml
app = Flask(__name__)
# =========================
# Load models
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--model_poster_path', type=str, default='weights/weights_poster/movie_poster_net.pth', help='model poster path')
parser.add_argument('--model_plot_path', type=str, default='weights/weights_plot', help='model plot path')
args = parser.parse_args()

print(f"Loading model from: {args.model_poster_path}")
try:
    # Charger le checkpoint
    checkpoint = torch.load(args.model_poster_path, map_location=device)
    
    # Créer le modèle
    model_poster = MoviePosterNet(checkpoint['num_genres']).to(device)
    model_poster.load_state_dict(checkpoint['model_state_dict'])
    model_poster.eval()
    
    # Récupérer les classes
    genres = checkpoint['classes']
    class_to_idx = checkpoint['class_to_idx']
    
    print(f"Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Transformations (doivent correspondre à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
model_plot_path = args.model_plot_path
import pandas as pd
plots_path = "data/movie_plots.csv"
df = pd.read_csv(plots_path)

print(f"Loading model from {model_plot_path}")
try:
    model_plot = DistilBertForSequenceClassification.from_pretrained(model_plot_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_plot_path)
    model_plot.to(device)
    model_plot.eval()
    print(f"Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_plot = None
# =========================
# Load unified config.yaml
# =========================
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ---- OOD config ----
OOD_METHOD = cfg["odd"]["method"]
OOD_THRESHOLD = float(cfg["odd"]["threshold"])
OOD_RULE = cfg["odd"]["decision_rule"]

# ---- Model config ----
EMBEDDING_DIM = int(cfg["model"]["embedding_dim"])
NUM_GENRES = int(cfg["model"]["num_genres"])

# ---- Genres ----
GENRES_MAPPING = cfg["genres"]["mapping"]
GENRES_MAPPING_INV = {v: k for k, v in GENRES_MAPPING.items()}

print("Loaded config.yaml:")
print(f"  OOD method     : {OOD_METHOD}")
print(f"  OOD threshold  : {OOD_THRESHOLD}")
print(f"  Num genres     : {NUM_GENRES}")
print(f"  Embedding dim  : {EMBEDDING_DIM}")

# ---- ANN index ----
ann_index = AnnoyIndex(EMBEDDING_DIM, "angular")
ann_index.load("plot_embeddings.ann")


print(f"Everything is loaded successfully!")
# =========================
# API
# =========================
@app.route('/predict_poster', methods=['POST'])
def predict_poster():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Ouvrir l'image
        img_pil = Image.open(file.stream).convert('RGB')
        
        # Appliquer les transformations
        tensor = transform(img_pil).to(device)
        tensor = tensor.unsqueeze(0)  # Ajouter dimension batch
        
        # Prédiction
        with torch.no_grad():
            outputs = model_poster(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Préparer les résultats
        all_predictions = {}
        for i, genre in enumerate(genres):
            all_predictions[genre] = float(probabilities[0][i])
        
        # Trier par confidence décroissante
        sorted_predictions = sorted(
            all_predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
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
@app.route('/predict_plot', methods=['POST'])
def predict_plot():
    if model_plot is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    plot = data.get('plot', '')

    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    inputs = tokenizer(plot, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model_plot(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
    nearest_indices = ann_index.get_nns_by_vector(cls_embedding[0], 5)
    paths = df.iloc[nearest_indices]['movie_poster_path'].tolist()
    genres = df.iloc[nearest_indices]['movie_category'].tolist()
    return jsonify({"predicted_class": predicted_class,
                    "similar_posters": paths,
                    "genres": genres})


@app.route('/batch_predict_poster', methods=['POST'])
def batch_predict_poster():
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    results = []
    try:
        for file in files:
            if file.filename == '':
                continue
                
            # Traiter chaque image
            img_pil = Image.open(file.stream).convert('RGB')
            tensor = transform(img_pil).to(device)
            tensor = tensor.unsqueeze(0)
            
            with torch.no_grad():
                outputs = model_poster(tensor)
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
        
@app.route('/batch_predict_plot', methods=['POST'])
def batch_predict_plot():
    if model_plot is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    plots = data.get('plots', [])

    if not plots:
        return jsonify({"error": "No plots provided"}), 400

    inputs = tokenizer(plots, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model_plot(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1).tolist()

    return jsonify({"predicted_classes": predicted_classes})

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
            logits = model_poster(tensor)

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
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True,
        'available_genres': list(genres),
        'num_genres': len(genres)
    })

@app.route('/genres', methods=['GET'])
def get_genres():
    """Retourne la liste des genres disponibles"""
    return jsonify({
        'genres': list(genres),
        'class_mapping': class_to_idx
    })

if __name__ == "__main__":
    print("\nStarting Flask API server...")
    print(f"\nServer running on http://0.0.0.0:5075")
    app.run(host='0.0.0.0', port=5075, debug=False)


